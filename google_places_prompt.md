Read ARCHITECTURE.md and MLOPS.md. We are upgrading the StaticFeatureLoader in src/features.py to replace the mock CSV with live Google Places API queries. Execute these tasks sequentially:

1. API Security & Setup

Update src/features.py to import requests, json, and os.

The class StaticFeatureLoader must securely pull the API key via os.environ.get('GOOGLE_PLACES_API_KEY').

Tripwire: If the key is None, immediately raise a ValueError("GOOGLE_PLACES_API_KEY not found in environment.").

2. The Caching Layer (Crucial for MLOps)

We cannot hit the API repeatedly during pipeline runs. Add a cache_path parameter to __init__ defaulting to data/external/places_cache.json.

Upon initialization, load this JSON dictionary into memory if it exists; otherwise, initialize an empty dictionary.

3. The Query Logic
Create a new helper method _fetch_location_data(self, lat, lon) that uses the Google Places Nearby Search API (or Text Search).

Commercial Density: Query for type=store or point_of_interest within a 500m radius. Return the integer count of results (cap at 50 if paginated).

School Zone: Query for type=school within a 250m radius. Return True if count > 0, else False.

Highway Proximity: Query for type=route or highway (or calculate distance to nearest major road if possible, otherwise use a default fallback strategy like querying for transit_station).

Offensive Assertion: Check response.status_code == 200. If 403 or 429 (Quota limits), crash the script immediately with a RuntimeError.

4. Refactor join_features

The method must extract the unique location names and their corresponding lat/lon coordinates from data/traffic_sensors.json.

Iterate through the unique locations. If the location is already in the cache, skip the API call.

If not in the cache, call _fetch_location_data, format the 3 required features into a dictionary, update the cache dictionary, and immediately save the cache back to disk (places_cache.json).

Convert the cache dictionary into a pandas DataFrame and execute the left merge exactly as before.

5. The Standalone Test

Add an if __name__ == "__main__": block to src/features.py.

Load a subset of 3 locations from data/traffic_sensors.json, instantiate StaticFeatureLoader, and run the queries.

Ensure the cache file is created and the API is bypassed on the second run.

Remember Rule 6.1: Update EXPERIMENTS.md.