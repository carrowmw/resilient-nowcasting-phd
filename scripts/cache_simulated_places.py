import logging
import random
import osmnx as ox
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/workspace/src')
from src.features import StaticFeatureLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bbox = {
    'lat_min': 54.968, 'lat_max': 55.012,
    'lon_min': -1.660, 'lon_max': -1.576
}

OSM_CENTER = (54.980, -1.600)
OSM_DIST = 4000

def main():
    logging.info("Downloading walking network...")
    G_sample = ox.graph_from_point(OSM_CENTER, dist=OSM_DIST, network_type='walk')
    
    all_nodes = []
    for node, data in G_sample.nodes(data=True):
        lat = data.get('y')
        lon = data.get('x')
        if lat is None or lon is None:
            continue
        if bbox['lat_min'] <= lat <= bbox['lat_max'] and bbox['lon_min'] <= lon <= bbox['lon_max']:
            all_nodes.append((lat, lon))
            
    assert len(all_nodes) >= 50, f"Found only {len(all_nodes)} nodes, but 50 requested."
    
    random.seed(42)
    sampled_coords = random.sample(all_nodes, 50)
    
    nodes_meta = []
    for j, (lat, lon) in enumerate(sampled_coords):
        nodes_meta.append({
            "location_name": f"Simulated_Node_{j}",
            "lat": lat,
            "lon": lon
        })
        
    logging.info("Running Simulated Node injection logic...")
    loader = StaticFeatureLoader(cache_path="data/external/simulated_places_cache.json")

    # Query missing Google features
    for meta in nodes_meta:
        loc = meta["location_name"]
        if loc not in loader.cache:
            logging.info(f"Fetching live Google Places data for simulated node: {loc}")
            loader.cache[loc] = loader._fetch_location_data(meta["lat"], meta["lon"])
            loader._save_cache()

    # Query OSM features
    logging.info("Fetching OSM features...")
    sim_sensors_data = [{"location": m["location_name"], "lat": m["lat"], "lon": m["lon"]} for m in nodes_meta]
    loader._build_osm_cache(sim_sensors_data)
    
    logging.info("Finished populating simulated caches!")

if __name__ == '__main__':
    main()
