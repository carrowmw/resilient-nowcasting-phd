#!/usr/bin/env python3
"""
scripts/cache_simulated_edge_features.py

Pre-computes pairwise OSM walking-network edge features for every ordered pair that
touches at least one simulated node, and caches them to JSON. These features complement
data/external/edge_features_cache.json (which covers only real sensor pairs) and are
merged at DynamicGraphLearner init time.

Run inside the container:

    # Hand-verifiable synthetic desk check (no network access):
    docker compose exec cpu-runner python scripts/cache_simulated_edge_features.py --self-test

    # Full pre-computation against the real Newcastle walking network (~minutes, one-time):
    docker compose exec cpu-runner python scripts/cache_simulated_edge_features.py
"""
import argparse
import json
import logging
import os
import sys

# Allow `import src.*` and sibling scripts when invoked from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

import networkx as nx
import yaml

from src.features import compute_pairwise_edge_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = 'data/external/simulated_edge_features_cache.json'
DEFAULT_SENSORS_PATH = 'data/traffic_sensors.json'
DEFAULT_NUM_SIMULATED = 5

def _load_geo_config():
    path = os.path.join(_REPO_ROOT, "configs", "geo.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["osm"]

_GEO = _load_geo_config()
GRAPH_CENTER = tuple(_GEO["center"])
GRAPH_DIST = int(_GEO["dist"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_coords_in_graph_bbox(G, sim_coords):
    """Fail-fast: raise ValueError if any simulated coordinate lies outside the graph bbox."""
    lon_min = min(data['x'] for _, data in G.nodes(data=True))
    lon_max = max(data['x'] for _, data in G.nodes(data=True))
    lat_min = min(data['y'] for _, data in G.nodes(data=True))
    lat_max = max(data['y'] for _, data in G.nodes(data=True))
    for name, lat, lon in sim_coords:
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            raise ValueError(
                f"CRITICAL: Simulated node coord ({lat}, {lon}) outside OSM graph bbox "
                f"[lat {lat_min:.6f}..{lat_max:.6f}, lon {lon_min:.6f}..{lon_max:.6f}]"
            )


def filter_simulated_pairs(all_pairs: dict, is_simulated) -> dict:
    """Return a new dict containing only directed pairs where at least one endpoint is simulated."""
    result = {}
    for src, dst_dict in all_pairs.items():
        for dst, feats in dst_dict.items():
            if is_simulated(src) or is_simulated(dst):
                if src not in result:
                    result[src] = {}
                result[src][dst] = feats
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test():
    """
    Desk check on the synthetic mock graph from cache_edge_features.py.

    S1, S2, S3 are treated as real sensors; S4, S5 as simulated nodes.
    Asserts that the filtered output contains exactly the 14 directed pairs that
    touch at least one of {S4, S5}, and no real→real pairs.
    """
    from cache_edge_features import build_mock_graph

    G, all_coords, node_ids = build_mock_graph()

    is_simulated = lambda name: name in {"S4", "S5"}
    sim_coords = [(name, lat, lon) for name, lat, lon in all_coords if is_simulated(name)]

    # Bounding box check passes for coords inside the mock graph.
    assert_coords_in_graph_bbox(G, sim_coords)

    all_pairs = compute_pairwise_edge_features(
        G, all_coords, cache_path=None, nearest_nodes=node_ids
    )
    filtered = filter_simulated_pairs(all_pairs, is_simulated)

    real_nodes = {"S1", "S2", "S3"}
    sim_nodes = {"S4", "S5"}

    # Every real↔sim directed pair must be present.
    for real in real_nodes:
        for sim in sim_nodes:
            assert real in filtered and sim in filtered[real], \
                f"SELF-TEST FAIL: missing {real}→{sim}"
            assert sim in filtered and real in filtered[sim], \
                f"SELF-TEST FAIL: missing {sim}→{real}"

    # Sim↔sim directed pairs must be present.
    assert "S4" in filtered and "S5" in filtered["S4"], "SELF-TEST FAIL: missing S4→S5"
    assert "S5" in filtered and "S4" in filtered["S5"], "SELF-TEST FAIL: missing S5→S4"

    # No real→real pairs should appear.
    for src in real_nodes:
        for dst in real_nodes:
            if src != dst:
                if src in filtered:
                    assert dst not in filtered[src], \
                        f"SELF-TEST FAIL: unexpected real→real pair {src}→{dst}"

    total_pairs = sum(len(dsts) for dsts in filtered.values())
    assert total_pairs == 14, f"SELF-TEST FAIL: expected 14 directed pairs, got {total_pairs}"

    logger.info(f"SELF-TEST PASSED. {total_pairs} simulated-touching directed pairs verified.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pre-compute pairwise OSM edge features for simulated-node pairs."
    )
    ap.add_argument(
        '--self-test', action='store_true',
        help="Run the synthetic desk check (no network access required).",
    )
    ap.add_argument(
        '--output', default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    ap.add_argument(
        '--num_simulated', type=int, default=DEFAULT_NUM_SIMULATED,
        help="Number of simulated nodes to include (default: 5).",
    )
    ap.add_argument(
        '--sensors_path', default=DEFAULT_SENSORS_PATH,
        help=f"Path to traffic_sensors.json (default: {DEFAULT_SENSORS_PATH}).",
    )
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    # Lazy import: avoid dragging export_ui_payload's heavy deps into self-test / tests.
    from src.export_ui_payload import SIMULATED_NODE_COORDS
    import osmnx as ox
    from cache_edge_features import load_sensor_coords

    assert args.num_simulated <= len(SIMULATED_NODE_COORDS), (
        f"--num_simulated={args.num_simulated} exceeds available coords "
        f"(max {len(SIMULATED_NODE_COORDS)})"
    )

    real_coords = load_sensor_coords()
    sim_coords = [
        (f"Simulated_Node_{j}", lat, lon)
        for j, (lat, lon) in enumerate(SIMULATED_NODE_COORDS[:args.num_simulated])
    ]

    logger.info(f"Real sensors: {len(real_coords)}, Simulated nodes: {len(sim_coords)}")

    logger.info(f"Downloading walking network (center={GRAPH_CENTER}, dist={GRAPH_DIST}m)...")
    G = ox.graph_from_point(GRAPH_CENTER, dist=GRAPH_DIST, network_type='walk')
    logger.info(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges.")

    assert_coords_in_graph_bbox(G, sim_coords)
    logger.info("All simulated coordinates lie within the OSM graph bounding box.")

    all_coords = real_coords + sim_coords
    logger.info(
        f"Computing pairwise features for {len(all_coords)} nodes "
        f"({len(all_coords) * (len(all_coords) - 1)} directed pairs)..."
    )
    all_pairs = compute_pairwise_edge_features(G, all_coords, cache_path=None)

    is_simulated = lambda name: name.startswith("Simulated_Node_")
    filtered = filter_simulated_pairs(all_pairs, is_simulated)

    total_pairs = sum(len(dsts) for dsts in filtered.values())
    total_sources = len(filtered)
    logger.info(
        f"Filtered to {total_pairs} simulated-touching directed pairs across "
        f"{total_sources} source nodes."
    )

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(filtered, f)
    logger.info(f"Written to {out_path}.")


if __name__ == '__main__':
    main()
