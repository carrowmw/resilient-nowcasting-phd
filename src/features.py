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

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NEWCASTLE_CENTRE = (54.9738, -1.6131)
OSM_CACHE_PATH = 'data/external/osm_features_cache.json'

REQUIRED_KEYS = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking'
]


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
            except Exception as e:
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
            logger.warning(f"Nearby Search API Error: {response.status_code} - {response.text}")
            return 0
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
                logger.warning(f"Text Search API Error (page {page_num}): {response.status_code} - {response.text}")
                break

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
            except Exception:
                existing = {}

        sensor_names = [s['location'] for s in sensors_data]
        if all(loc in existing for loc in sensor_names):
            logger.info("OSM cache fully populated. Skipping OSMnx download.")
            return existing

        logger.info("Downloading Newcastle walking network via OSMnx (this runs once)...")
        G = ox.graph_from_point((54.978, -1.610), dist=5000, network_type='walk')
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
                    logger.warning(f"Location '{loc}' not found in sensor metadata. Skipping.")
                    continue
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
    except Exception as e:
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
    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        raise
