import logging
import math
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os
import numpy as np
import random
import networkx as nx
import osmnx as ox
import yaml

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NEWCASTLE_CENTRE = (54.9738, -1.6131)

def _load_geo_config() -> dict:
    """Load OSMnx constants from configs/geo.yaml (searched relative to this file and repo root)."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "configs", "geo.yaml"),
        os.path.join(os.getcwd(), "configs", "geo.yaml"),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)["osm"]
    raise FileNotFoundError(
        "configs/geo.yaml not found. Expected at repo-root/configs/geo.yaml."
    )

_GEO = _load_geo_config()
OSM_CENTER = tuple(_GEO["center"])  # (lat, lon)
OSM_DIST = int(_GEO["dist"])
OSM_CACHE_PATH = 'data/external/osm_features_cache.json'

REQUIRED_KEYS = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking'
]

# --- Edge feature enrichment (Spatial Kriging architecture, Component 1) ---
# Cached pairwise OSM walking-network features used to build the GATv2 edge_attr tensor.
EDGE_FEATURE_CACHE_PATH = 'data/external/edge_features_cache.json'

# K=6 collapsed road-type categories (docs/spatial_estimation_architecture.md §3.3).
ROAD_CATEGORIES = ['pedestrian', 'residential', 'tertiary', 'secondary', 'primary', 'service']

# OSM `highway` tag -> collapsed category. Unknown tags fall through to 'service'
# (the residual minor-way bucket) so that road-type fractions always sum to 1.0.
HIGHWAY_TO_CATEGORY = {
    'pedestrian': 'pedestrian', 'footway': 'pedestrian', 'path': 'pedestrian',
    'steps': 'pedestrian', 'corridor': 'pedestrian',
    'residential': 'residential', 'living_street': 'residential',
    'tertiary': 'tertiary', 'tertiary_link': 'tertiary', 'unclassified': 'tertiary',
    'secondary': 'secondary', 'secondary_link': 'secondary',
    'primary': 'primary', 'primary_link': 'primary', 'trunk': 'primary', 'trunk_link': 'primary',
    'service': 'service', 'track': 'service', 'bridleway': 'service', 'cycleway': 'service',
}

# Per-pair cached feature schema. Order is the contract for the runtime [E, edge_dim]
# tensor: these 9 cached dims + a 10th `learned_similarity` dim supplied at runtime by
# DynamicGraphLearner (see Phase 5) => edge_dim=10.
EDGE_FEATURE_KEYS = ['shortest_path_distance', 'euclidean_distance', 'directness'] + ROAD_CATEGORIES


def classify_highway(hw) -> str:
    """Collapse a raw OSM `highway` tag (str or list) into one of ROAD_CATEGORIES."""
    if isinstance(hw, (list, tuple)):
        hw = hw[0] if hw else None
    return HIGHWAY_TO_CATEGORY.get(hw, 'service')


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns distance in metres between two WGS-84 lat/lon points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _shannon_entropy(counts: list) -> float:
    """Normalised Shannon entropy in [0,1] across N categories."""
    n = len(counts)
    if n <= 1:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    return h / math.log2(n)


class StaticFeatureLoader:
    def __init__(self, cache_path: str = 'data/external/places_cache.json'):
        self.api_key = os.environ.get('GOOGLE_PLACES_API_KEY')
        if self.api_key is None:
            raise ValueError("GOOGLE_PLACES_API_KEY not found in environment.")

        self.cache_path = cache_path
        self.cache = self._load_cache()

        # Session with automatic retry on transient SSL/network errors.
        # Does NOT retry on 4xx (quota/auth failures crash hard instead).
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session = requests.Session()
        self._session.mount("https://", adapter)

        logger.info(f"Initialized StaticFeatureLoader with cache: {self.cache_path}")

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Error loading cache: {e}. Starting fresh.")
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _nearby_search_count(self, lat: float, lon: float, place_types: list, radius: float) -> int:
        """Single-page Nearby Search (New). Crashes hard on 403/429."""
        url = "https://places.googleapis.com/v1/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id"
        }
        payload = {
            "includedTypes": place_types,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lon},
                    "radius": float(radius)
                }
            }
        }
        response = self._session.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code in (403, 429):
            raise RuntimeError(
                f"CRITICAL: Google Places API Quota/Auth Error: {response.status_code} - {response.text}"
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"CRITICAL: Nearby Search API returned {response.status_code}: {response.text}. "
                "An API outage must not be cached as zero POIs."
            )
        return len(response.json().get('places', []))

    def _text_search_paginated(self, lat: float, lon: float, text_query: str,
                                included_type: str, radius: float) -> int:
        """
        Text Search (New) with nextPageToken pagination — up to 3 pages (max 60 results).

        Uses locationRestriction with a rectangle bounding box (Text Search does not
        support a circle for locationRestriction; locationBias is non-restrictive and
        returns results far outside the target radius).

        Crashes hard on 403/429. Requires 2s delay between pages per API spec.
        """
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,nextPageToken"
        }

        # Approximate bounding box for the given radius (metres → degrees)
        dlat = radius / 111_000.0
        dlon = radius / (111_000.0 * math.cos(math.radians(lat)))

        total = 0
        page_token = None

        for page_num in range(1, 4):
            payload = {
                "textQuery": text_query,
                "includedType": included_type,
                "locationRestriction": {
                    "rectangle": {
                        "low":  {"latitude": lat - dlat, "longitude": lon - dlon},
                        "high": {"latitude": lat + dlat, "longitude": lon + dlon}
                    }
                },
                "maxResultCount": 20
            }
            if page_token:
                payload["pageToken"] = page_token

            response = self._session.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code in (403, 429):
                raise RuntimeError(
                    f"CRITICAL: Google Places API Quota/Auth Error (page {page_num}): "
                    f"{response.status_code} - {response.text}"
                )
            if response.status_code != 200:
                raise RuntimeError(
                    f"CRITICAL: Text Search API (page {page_num}) returned {response.status_code}: "
                    f"{response.text}. An API outage must not be cached as zero POIs."
                )

            data = response.json()
            page_results = len(data.get('places', []))
            total += page_results
            logger.debug(f"  Text Search '{included_type}' page {page_num}: {page_results} results")

            page_token = data.get('nextPageToken')
            if not page_token or page_results == 0:
                break

            time.sleep(2.0)

        return total

    def _build_osm_cache(self, sensors_data: list) -> dict:
        """
        Downloads Newcastle walking network via OSMnx, computes betweenness/closeness/degree
        centrality for each sensor location. Caches to OSM_CACHE_PATH.
        """
        osm_cache_path = OSM_CACHE_PATH
        os.makedirs(os.path.dirname(osm_cache_path), exist_ok=True)

        # Load existing OSM cache if present
        existing = {}
        if os.path.exists(osm_cache_path):
            try:
                with open(osm_cache_path, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        sensor_names = [s['location'] for s in sensors_data]
        if all(loc in existing for loc in sensor_names):
            logger.info("OSM cache fully populated. Skipping OSMnx download.")
            return existing

        logger.info("Downloading Newcastle walking network via OSMnx (this runs once)...")
        G = ox.graph_from_point(OSM_CENTER, dist=OSM_DIST, network_type='walk')
        logger.info(f"OSMnx graph: {len(G.nodes)} nodes, {len(G.edges)} edges. Computing centralities...")

        # Convert to undirected — walking is symmetric and dramatically speeds up centrality.
        G_u = G.to_undirected()

        # Approximate betweenness with k=500 random-pivot samples (seconds vs. hours for exact).
        betweenness = nx.betweenness_centrality(G_u, k=500, normalized=True, seed=SEED)
        logger.info("Betweenness centrality done.")

        closeness = nx.closeness_centrality(G_u)
        logger.info("Closeness centrality done.")

        degree = nx.degree_centrality(G_u)
        logger.info("Centrality computation complete.")

        result = dict(existing)
        for sensor in sensors_data:
            loc = sensor['location']
            lat, lon = sensor['lat'], sensor['lon']
            nearest_node = ox.nearest_nodes(G, lon, lat)
            result[loc] = {
                'betweenness_centrality': float(betweenness.get(nearest_node, 0.0)),
                'closeness_centrality': float(closeness.get(nearest_node, 0.0)),
                'degree_centrality': float(degree.get(nearest_node, 0.0))
            }

        # Shape invariant: every sensor must be represented
        missing = [s['location'] for s in sensors_data if s['location'] not in result]
        if missing:
            raise RuntimeError(f"CRITICAL: OSM cache missing entries for: {missing}")

        with open(osm_cache_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"OSM cache saved to {osm_cache_path} ({len(result)} entries).")
        return result

    def _fetch_location_data(self, lat: float, lon: float) -> dict:
        """
        Issues 9 Google API calls per location to populate the 12 non-OSM features.
        Text Search (paginated) for high-density features; Nearby Search for low-count features.
        Crashes hard on 403/429.
        """
        # --- Text Search (paginated) for density-critical features ---
        transit_count = self._text_search_paginated(lat, lon, "transit stations", "transit_station", 500)
        time.sleep(0.1)

        retail_count = self._text_search_paginated(lat, lon, "stores shops retail", "store", 500)
        time.sleep(0.1)

        # food_drink: Text Search for restaurant + Nearby Search for cafe (two calls, sum)
        restaurant_count = self._text_search_paginated(lat, lon, "restaurants food", "restaurant", 500)
        time.sleep(0.1)
        cafe_count = self._nearby_search_count(lat, lon, ['cafe'], 500)
        time.sleep(0.1)
        food_drink_count = restaurant_count + cafe_count

        # --- Nearby Search for low-count features ---
        nightlife_count = self._nearby_search_count(lat, lon, ['bar', 'night_club'], 500)
        time.sleep(0.1)

        education_count = self._nearby_search_count(lat, lon, ['school', 'university'], 500)
        time.sleep(0.1)

        recreation_count = self._nearby_search_count(lat, lon, ['park', 'tourist_attraction', 'museum'], 500)
        time.sleep(0.1)

        school_zone_count = self._nearby_search_count(lat, lon, ['school'], 250)
        time.sleep(0.1)

        nearby_bus_stops = self._nearby_search_count(lat, lon, ['bus_station'], 300)
        time.sleep(0.1)

        nearby_parking = self._nearby_search_count(lat, lon, ['parking'], 500)

        # --- Derived features ---
        category_counts = [transit_count, retail_count, food_drink_count,
                           nightlife_count, education_count, recreation_count]
        total_poi_count = sum(category_counts)
        landuse_entropy = _shannon_entropy(category_counts)
        dist_to_city_centre = _haversine_distance(lat, lon, *NEWCASTLE_CENTRE)

        return {
            'transit_count': int(transit_count),
            'retail_count': int(retail_count),
            'food_drink_count': int(food_drink_count),
            'nightlife_count': int(nightlife_count),
            'education_count': int(education_count),
            'recreation_count': int(recreation_count),
            'total_poi_count': int(total_poi_count),
            'landuse_entropy': float(landuse_entropy),
            'is_school_zone': bool(school_zone_count > 0),
            'nearby_bus_stops': int(nearby_bus_stops),
            'nearby_parking': int(nearby_parking),
            'dist_to_city_centre': float(dist_to_city_centre),
        }

    def join_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches traffic sensor data with 15 static spatial features.
        Merges Google Places API results (12 features) with OSMnx centrality (3 features).
        Crashes on any missing feature or NaN after join.
        """
        logger.info("Starting join_features — 15-feature schema.")

        sensors_path = 'data/traffic_sensors.json'
        try:
            with open(sensors_path, 'r') as f:
                sensors_data = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"CRITICAL: Sensor metadata not found at {sensors_path}")

        location_map = {s['location']: (s['lat'], s['lon']) for s in sensors_data}

        assert 'location' in df.columns, "Traffic dataframe is missing the 'location' identifier."
        unique_locations = df['location'].unique()

        # Build/load OSM cache for all sensors (one-time OSMnx computation)
        osm_cache = self._build_osm_cache(sensors_data)

        google_required_keys = [k for k in REQUIRED_KEYS if k not in
                                 ('betweenness_centrality', 'closeness_centrality', 'degree_centrality')]

        for loc in unique_locations:
            needs_fetch = False
            if loc not in self.cache:
                needs_fetch = True
            else:
                if not all(k in self.cache[loc] for k in google_required_keys):
                    logger.info(f"Cache entry for '{loc}' is incomplete. Re-fetching...")
                    needs_fetch = True

            if needs_fetch:
                if loc not in location_map:
                    raise KeyError(
                        f"CRITICAL: Location '{loc}' not found in sensor metadata — "
                        "cannot fetch Google Places data without coordinates."
                    )
                lat, lon = location_map[loc]
                logger.info(f"Fetching Google Places data for '{loc}'...")
                features = self._fetch_location_data(lat, lon)
                self.cache[loc] = features
                self._save_cache()

        # Build merged metadata DataFrame (Google + OSM)
        cache_data = []
        for loc, feats in self.cache.items():
            row = {'location': loc}
            row.update(feats)
            # Merge in OSM features
            osm_feats = osm_cache.get(loc, {})
            row.update(osm_feats)
            cache_data.append(row)

        metadata_df = pd.DataFrame(cache_data)

        original_len = len(df)
        df = df.merge(metadata_df, on='location', how='left')
        assert len(df) == original_len, \
            f"CRITICAL: Merge altered row count: {original_len} → {len(df)}"

        # Validate all 15 features present and non-null
        for col in REQUIRED_KEYS:
            assert col in df.columns, f"CRITICAL: Expected column '{col}' missing after join."
            assert df[col].notna().all(), \
                f"CRITICAL: Null values detected in static feature '{col}' after join."

        logger.info("Static feature enrichment complete. 15 features validated.")
        return df


def compute_pairwise_edge_features(G_walk, sensor_coords, cache_path: str = None,
                                   nearest_nodes: list = None) -> dict:
    """
    Compute pairwise OSM walking-network edge features for every ordered sensor pair.

    For each directed pair (i -> j) computes, along the Dijkstra shortest path:
      - shortest_path_distance: network distance in metres (edge `length` weight)
      - euclidean_distance:     straight-line haversine distance in metres
      - directness:             euclidean / shortest_path  in (0, 1]; low = winding
      - road-type composition:  fraction of path length per ROAD_CATEGORIES (sums to 1.0)

    Args:
        G_walk:        networkx (Multi)DiGraph walking network with `length` and
                       `highway` edge attributes (e.g. an OSMnx graph).
        sensor_coords: list of (name, lat, lon) tuples.
        cache_path:    if given, the nested result dict is written to this JSON path.
        nearest_nodes: optional list of graph node ids aligned to sensor_coords. When
                       omitted, nodes are resolved via ox.nearest_nodes. (Passing this
                       explicitly lets the desk-check run on a synthetic graph without
                       invoking OSMnx's spatial index.)

    Returns:
        Nested dict: result[name_i][name_j] = {feature_key: value, ...}. Pairs with no
        walking path are omitted (logged). JSON-serialisable (string keys, float values).
    """
    assert len(sensor_coords) >= 2, \
        f"CRITICAL: need >=2 sensors for pairwise features, got {len(sensor_coords)}"

    names = [c[0] for c in sensor_coords]
    coords = {c[0]: (float(c[1]), float(c[2])) for c in sensor_coords}
    assert len(set(names)) == len(names), "CRITICAL: duplicate sensor names in sensor_coords."

    # Resolve each sensor to its nearest graph node.
    if nearest_nodes is None:
        node_ids = [ox.nearest_nodes(G_walk, lon, lat) for (_, lat, lon) in sensor_coords]
    else:
        assert len(nearest_nodes) == len(sensor_coords), \
            f"CRITICAL: nearest_nodes length {len(nearest_nodes)} != sensor_coords {len(sensor_coords)}"
        node_ids = list(nearest_nodes)

    indexed = list(zip(names, node_ids))
    result = {name: {} for name in names}

    for i, (name_i, node_i) in enumerate(indexed):
        lat_i, lon_i = coords[name_i]
        # One Dijkstra sweep from source i reaches every target in O((V+E) log V),
        # avoiding the O(N^2) pair-by-pair shortest-path calls in the spec pseudocode.
        dist_map, path_map = nx.single_source_dijkstra(G_walk, node_i, weight='length')

        for j, (name_j, node_j) in enumerate(indexed):
            if i == j:
                continue
            if node_j not in dist_map:
                logger.warning(f"No walking path between '{name_i}' and '{name_j}'. Skipping pair.")
                continue

            sp_dist = float(dist_map[node_j])
            path = path_map[node_j]

            # Sum path length per road-type category.
            type_lengths = {cat: 0.0 for cat in ROAD_CATEGORIES}
            for u, v in zip(path[:-1], path[1:]):
                edges = G_walk.get_edge_data(u, v)
                assert edges is not None, \
                    f"CRITICAL: path edge ({u},{v}) absent from graph for {name_i}->{name_j}"
                # Dijkstra uses the minimum-length parallel edge; mirror that choice here.
                best = min(edges.values(), key=lambda e: e.get('length', float('inf')))
                cat = classify_highway(best.get('highway'))
                type_lengths[cat] += float(best.get('length', 0.0))

            total = sum(type_lengths.values())
            if total > 0:
                composition = {cat: type_lengths[cat] / total for cat in ROAD_CATEGORIES}
            else:
                # Degenerate: both sensors snap to the same node (zero-length path).
                composition = {cat: 0.0 for cat in ROAD_CATEGORIES}

            lat_j, lon_j = coords[name_j]
            euc_dist = _haversine_distance(lat_i, lon_i, lat_j, lon_j)
            directness = (euc_dist / sp_dist) if sp_dist > 0 else 1.0

            record = {
                'shortest_path_distance': sp_dist,
                'euclidean_distance': float(euc_dist),
                'directness': float(directness),
            }
            record.update(composition)

            # Offensive tripwires: no NaN, and (for non-degenerate paths) fractions sum to 1.0.
            assert not any(math.isnan(val) for val in record.values()), \
                f"CRITICAL: NaN in edge features for {name_i}->{name_j}: {record}"
            if total > 0:
                comp_sum = sum(composition.values())
                assert abs(comp_sum - 1.0) < 1e-6, \
                    f"CRITICAL: road composition {name_i}->{name_j} sums to {comp_sum}, expected 1.0"

            result[name_i][name_j] = record

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)
        n_pairs = sum(len(v) for v in result.values())
        logger.info(f"Edge features cached to {cache_path} ({n_pairs} directed pairs).")

    return result


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv('/workspace/.env')

    logging.basicConfig(level=logging.INFO)

    try:
        with open('data/traffic_sensors.json', 'r') as f:
            sensors = json.load(f)
        test_sensors = [s for s in sensors if not s['location'].startswith('Simulated')][:3]
        test_locations = [s['location'] for s in test_sensors]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Test setup failed: {e}")
        sys.exit(1)

    test_df = pd.DataFrame({'location': test_locations})

    try:
        loader = StaticFeatureLoader(cache_path='data/external/places_cache_test.json')

        print("\n--- RUN 1: API Query / Cache Fill ---")
        df1 = loader.join_features(test_df.copy())
        print(df1[REQUIRED_KEYS])

        print("\n--- RUN 2: Cache Hit ---")
        df2 = loader.join_features(test_df.copy())
        print(df2[REQUIRED_KEYS])

        assert df1[REQUIRED_KEYS].equals(df2[REQUIRED_KEYS]), "Regression: Cache differs from API."
        logger.info("Standalone test PASSED.")
    except (RuntimeError, KeyError, AssertionError) as e:
        logger.error(f"Test FAILED: {e}")
        raise
