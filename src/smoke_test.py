import logging
import pandas as pd
import json
import numpy as np
from preprocess import PreprocessingPipeline
from features import StaticFeatureLoader

# Standard random seed initialization
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_smoke_test():
    # 1. Load Data
    traffic_path = 'data/traffic_20230108_20260108.parquet'
    full_traffic = pd.read_parquet(traffic_path)
    
    target_loc = 'NclCedarRdGowlandAveWest'
    # Force a single consistent group for all test rows
    # Moving time to NOON (12:00) so gaps are NOT filled as structural zeros
    sample_traffic = pd.DataFrame({
        'location': [target_loc, target_loc],
        'category': ['flow', 'flow'],
        'veh_class': ['car', 'car'],
        'dir': ['east_to_west', 'east_to_west'],
        'dt': [pd.Timestamp('2023-03-02 12:00:00'), pd.Timestamp('2023-03-02 14:00:00')],
        'value': [10, 20]
    })
    
    logger.info(f"Created manual test data with 2-hour gap.")

    with open('data/traffic_sensors.json', 'r') as f:
        traffic_sensors = json.load(f)
    with open('data/weather_sensors.json', 'r') as f:
        weather_sensors = json.load(f)

    # 2. Populate the Google Places Template
    unique_locs = sample_traffic['location'].unique()
    metadata_df = pd.DataFrame({
        'location_id': unique_locs,
        'nearby_poi_count': [5] * len(unique_locs),
        'zone_type': ['Residential'] * len(unique_locs),
        'dist_to_nearest_intersection': [150.0] * len(unique_locs)
    })
    metadata_df.to_csv('data/external/google_places_template.csv', index=False)

    # 3. Initialize Pipeline
    pipeline = PreprocessingPipeline(frequency='15min')
    feature_loader = StaticFeatureLoader()

    # 4. Execution
    logger.info(f"Shape BEFORE reindexing: {sample_traffic.shape}")
    resampled_df = pipeline.resample_traffic(sample_traffic)
    logger.info(f"Shape AFTER reindexing: {resampled_df.shape}")
    
    imputed_df = pipeline.impute_missingness(resampled_df)
    joined_df = pipeline.spatial_join(imputed_df, traffic_sensors, weather_sensors)
    final_df = feature_loader.join_features(joined_df)

    # 5. Validation
    logger.info("--- THE BURN TEST: MANUAL INSPECTION ---")
    
    nan_samples = final_df[final_df['value'].isna()]
    zero_samples = final_df[final_df['value'] == 0]

    print("\nSample of TRUE MISSINGNESS (Expected NaNs from gaps):")
    if not nan_samples.empty:
        print(nan_samples[['dt', 'location', 'value']].head(5))
    else:
        print("No NaNs found.")

    print("\nSample of STRUCTURAL ZEROS (Night hours 0-5 AM):")
    if not zero_samples.empty:
        print(zero_samples[['dt', 'location', 'value']].head(5))
    else:
        print("No structural zeros found.")

    logger.info("Smoke test completed successfully.")

if __name__ == "__main__":
    run_smoke_test()
