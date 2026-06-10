#!/usr/bin/env python3
"""
scripts/cache_edge_features.py

Pre-computes pairwise OSM walking-network edge features for every sensor pair and
caches them to JSON. Implements Phase 1 (Component 1) of the Spatial Kriging & Edge
Enrichment architecture (docs/spatial_estimation_architecture.md §3).

Per pair (i -> j): Dijkstra shortest-path distance, haversine euclidean distance,
path directness, and a 6-way road-type composition vector.

Run INSIDE the container (Strict Container Isolation, NSTAR_MLOPS.md §7):

    # Hand-verifiable 5-sensor desk check on a synthetic sub-graph (no network access):
    docker compose exec cpu-runner python scripts/cache_edge_features.py --self-test

    # Full pre-computation against the real Newcastle walking network (~10 min, one-time):
    docker compose exec cpu-runner python scripts/cache_edge_features.py

    # Debug subset:
    docker compose exec cpu-runner python scripts/cache_edge_features.py --limit 5
"""
import argparse
import json
import logging
import os
import sys

# Allow `import src.features` when invoked from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import osmnx as ox

from src.features import (
    compute_pairwise_edge_features,
    EDGE_FEATURE_CACHE_PATH,
    ROAD_CATEGORIES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SENSORS_PATH = 'data/traffic_sensors.json'
# Mirror the OSMnx graph parameters used in src/features.py::_build_osm_cache.
GRAPH_CENTER = (54.978, -1.610)
GRAPH_DIST = 5000
MOCK_OUTPUT_PATH = 'data/external/edge_features_mock.json'


def load_sensor_coords(limit: int = None) -> list:
    """Load (name, lat, lon) tuples from sensor metadata, excluding simulated nodes."""
    with open(SENSORS_PATH, 'r') as f:
        sensors = json.load(f)
    coords = [(s['location'], s['lat'], s['lon'])
              for s in sensors if not s['location'].startswith('Simulated')]
    assert coords, f"CRITICAL: no real sensors found in {SENSORS_PATH}"
    if limit is not None:
        coords = coords[:limit]
    return coords


def build_walking_graph():
    """Download the Newcastle walking network via OSMnx (matches features.py params)."""
    logger.info(f"Downloading walking network (center={GRAPH_CENTER}, dist={GRAPH_DIST}m)...")
    G = ox.graph_from_point(GRAPH_CENTER, dist=GRAPH_DIST, network_type='walk')
    logger.info(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    return G


def build_mock_graph():
    """
    Synthetic 5-sensor walking sub-graph with hand-chosen segment lengths and road types,
    used by the desk check. Linear chain S1-S2-S3-S4-S5 with known per-segment lengths so
    the resulting distances and road-type fractions are verifiable by hand.

    Segments (bidirectional):
        S1-S2: 100m footway      -> pedestrian
        S2-S3: 200m residential  -> residential
        S3-S4: 300m primary      -> primary
        S4-S5: 400m secondary    -> secondary
    """
    segs = [
        (1, 2, 100.0, 'footway'),      # pedestrian
        (2, 3, 200.0, 'residential'),  # residential
        (3, 4, 300.0, 'primary'),      # primary
        (4, 5, 400.0, 'secondary'),    # secondary
    ]
    # Tight (lon, lat) spacing so euclidean << network distance (keeps directness in (0,1]).
    node_xy = {
        1: (-1.6100, 54.9780), 2: (-1.6099, 54.9781), 3: (-1.6098, 54.9782),
        4: (-1.6097, 54.9783), 5: (-1.6096, 54.9784),
    }
    G = nx.MultiDiGraph(crs='epsg:4326')
    for nid, (x, y) in node_xy.items():
        G.add_node(nid, x=x, y=y)
    for u, v, length, hw in segs:
        G.add_edge(u, v, key=0, length=length, highway=hw)
        G.add_edge(v, u, key=0, length=length, highway=hw)

    sensor_coords = [(f"S{nid}", node_xy[nid][1], node_xy[nid][0]) for nid in (1, 2, 3, 4, 5)]
    node_ids = [1, 2, 3, 4, 5]
    return G, sensor_coords, node_ids


def run_self_test():
    """Dummy-data desk check: compute features on the mock graph and assert exact values."""
    G, sensor_coords, node_ids = build_mock_graph()
    feats = compute_pairwise_edge_features(
        G, sensor_coords, cache_path=MOCK_OUTPUT_PATH, nearest_nodes=node_ids
    )

    # --- Hand-computed expectations ---
    # S1 -> S5 traverses all four segments: 100+200+300+400 = 1000m.
    f15 = feats['S1']['S5']
    assert abs(f15['shortest_path_distance'] - 1000.0) < 1e-6, f15
    assert abs(f15['pedestrian'] - 0.1) < 1e-9, f15   # 100/1000
    assert abs(f15['residential'] - 0.2) < 1e-9, f15  # 200/1000
    assert abs(f15['primary'] - 0.3) < 1e-9, f15      # 300/1000
    assert abs(f15['secondary'] - 0.4) < 1e-9, f15    # 400/1000
    assert abs(f15['tertiary'] - 0.0) < 1e-9, f15
    assert abs(f15['service'] - 0.0) < 1e-9, f15

    # S2 -> S3 is a single residential segment.
    f23 = feats['S2']['S3']
    assert abs(f23['shortest_path_distance'] - 200.0) < 1e-6, f23
    assert abs(f23['residential'] - 1.0) < 1e-9, f23

    # Symmetry: walking distance is direction-independent.
    assert abs(feats['S1']['S5']['shortest_path_distance']
               - feats['S5']['S1']['shortest_path_distance']) < 1e-6

    # Global invariants across every directed pair.
    n_pairs = 0
    for a in feats:
        for b, r in feats[a].items():
            n_pairs += 1
            comp_sum = sum(r[c] for c in ROAD_CATEGORIES)
            assert abs(comp_sum - 1.0) < 1e-6, (a, b, comp_sum, r)
            assert 0.0 < r['directness'] <= 1.0 + 1e-9, (a, b, r['directness'])
            assert r['euclidean_distance'] > 0.0, (a, b, r)
            assert r['shortest_path_distance'] > 0.0, (a, b, r)

    assert n_pairs == 20, f"CRITICAL: expected 20 directed pairs (5x4), got {n_pairs}"

    logger.info(f"SELF-TEST PASSED. {n_pairs} directed pairs written to {MOCK_OUTPUT_PATH}.")
    print("\n--- S1 outgoing edge features (desk check sample) ---")
    print(json.dumps(feats['S1'], indent=2))


def main():
    ap = argparse.ArgumentParser(description="Pre-compute pairwise OSM edge features.")
    ap.add_argument('--self-test', action='store_true',
                    help="Run the hand-verifiable 5-sensor synthetic desk check.")
    ap.add_argument('--limit', type=int, default=None,
                    help="Limit the number of real sensors (debugging only).")
    ap.add_argument('--output', default=EDGE_FEATURE_CACHE_PATH,
                    help="Output JSON path for the real cache.")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    coords = load_sensor_coords(limit=args.limit)
    n = len(coords)
    logger.info(f"Computing edge features for {n} sensors ({n * (n - 1)} directed pairs)...")
    G = build_walking_graph()
    compute_pairwise_edge_features(G, coords, cache_path=args.output)
    logger.info("Done.")


if __name__ == '__main__':
    main()
