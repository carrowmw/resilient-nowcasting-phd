import logging
import pandas as pd
import requests
import json
import os
import numpy as np
import random

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StaticFeatureLoader:
    def __init__(self, cache_path: str = 'data/external/places_cache.json'):
        """
        Initializes the loader with API security and caching.
        Uses Google Places API (New) as legacy APIs are disabled.
        """
        self.api_key = os.environ.get('GOOGLE_PLACES_API_KEY')
        if self.api_key is None:
            raise ValueError("GOOGLE_PLACES_API_KEY not found in environment.")
        
        self.cache_path = cache_path
        self.cache = self._load_cache()
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

    def _fetch_location_data(self, lat: float, lon: float) -> dict:
        """
        Queries Google Places API (New) for location features.
        Implements Offensive Programming: crash on quota or auth failures.
        """
        def get_count(place_type, radius):
            url = "https://places.googleapis.com/v1/places:searchNearby"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "places.id"
            }
            payload = {
                "includedTypes": [place_type],
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat,
                            "longitude": lon
                        },
                        "radius": float(radius)
                    }
                }
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            # OFFENSIVE ASSERTION: Check response status code
            if response.status_code == 403 or response.status_code == 429:
                raise RuntimeError(f"CRITICAL: Google Places API Quota/Auth Error: {response.status_code} - {response.text}")
            
            if response.status_code != 200:
                # Handle other errors gracefully but log them
                logger.warning(f"API Error: {response.status_code} - {response.text}")
                return 0
            
            data = response.json()
            # If no results found, the API (New) returns an empty dictionary {}
            places = data.get('places', [])
            return min(len(places), 50)

        # 1. Commercial Density: Query for type=store within 500m
        comm_count = get_count('store', 500)
        
        # 2. School Zone: Query for type=school within 250m. Return True if count > 0.
        school_count = get_count('school', 250)
        is_school_zone = school_count > 0
        
        # 3. Highway Proximity: Fallback strategy using transit_station within 500m.
        transit_count = get_count('transit_station', 500)
        
        return {
            'commercial_density': int(comm_count),
            'is_school_zone': bool(is_school_zone),
            'highway_proximity': int(transit_count)
        }

    def join_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches traffic sensor data with live Google Places features using a caching layer.
        """
        logger.info("Starting join_features with live Google Places API logic.")
        
        # Load unique location names and their coordinates from traffic_sensors.json
        sensors_path = 'data/traffic_sensors.json'
        try:
            with open(sensors_path, 'r') as f:
                sensors_data = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"CRITICAL: Sensor metadata not found at {sensors_path}")

        # Map location names to lat/lon
        location_map = {s['location']: (s['lat'], s['lon']) for s in sensors_data}
        
        # Identify unique locations in the input dataframe
        assert 'location' in df.columns, "Traffic dataframe is missing the 'location' identifier."
        unique_locations = df['location'].unique()
        
        for loc in unique_locations:
            if loc not in self.cache:
                if loc not in location_map:
                    logger.warning(f"Location {loc} not found in sensor metadata. Skipping.")
                    continue
                
                lat, lon = location_map[loc]
                logger.info(f"Cache miss for {loc}. Fetching live API data...")
                
                features = self._fetch_location_data(lat, lon)
                self.cache[loc] = features
                
                # Immediate save back to disk as requested
                self._save_cache()
        
        # Convert cache dictionary into a pandas DataFrame for merging
        cache_data = []
        for loc, feats in self.cache.items():
            row = {'location': loc}
            row.update(feats)
            cache_data.append(row)
        
        metadata_df = pd.DataFrame(cache_data)
        
        # Execute left merge exactly as before
        original_len = len(df)
        df = df.merge(metadata_df, on='location', how='left')
        assert len(df) == original_len, "CRITICAL: The merge operation unexpectedly altered the row count."
        
        # Validation: Ensure no NaNs in the required columns
        new_feature_cols = ['commercial_density', 'is_school_zone', 'highway_proximity']
        for col in new_feature_cols:
            assert col in df.columns, f"CRITICAL: Expected column '{col}' missing after join."
            assert df[col].notna().all(), f"CRITICAL: Null values detected in static feature '{col}' after join."
            
        logger.info("Static feature enrichment complete.")
        return df

if __name__ == "__main__":
    # STANDALONE TEST
    # Requires GOOGLE_PLACES_API_KEY in environment.
    
    # Load 3 locations for testing
    try:
        with open('data/traffic_sensors.json', 'r') as f:
            sensors = json.load(f)
        test_locations = [s['location'] for s in sensors[:3]]
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        test_locations = []

    test_df = pd.DataFrame({'location': test_locations})
    
    # Instantiate loader (will fail if API key is missing)
    try:
        loader = StaticFeatureLoader(cache_path='data/external/places_cache_test.json')
        
        print("\n--- RUN 1: API Query / Cache Fill ---")
        df1 = loader.join_features(test_df.copy())
        print(df1)
        
        print("\n--- RUN 2: Bypassing API (Cache Hit) ---")
        df2 = loader.join_features(test_df.copy())
        print(df2)
        
        # Verification
        assert df1.equals(df2), "Regression: Cache results differ from API results."
        assert os.path.exists('data/external/places_cache_test.json'), "Cache file was not created."
        
        logger.info("Standalone test successful.")
    except Exception as e:
        logger.error(f"Test Execution Failed: {e}")
        # Clean up test cache if needed
        # if os.path.exists('data/external/places_cache_test.json'):
        #     os.remove('data/external/places_cache_test.json')
        raise
