import logging
import os
import gc
import json
import numpy as np
import pandas as pd
from preprocess import PreprocessingPipeline
from features import StaticFeatureLoader

# Standard random seed initialization
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_weather_data(weather_path: str, frequency: str = '15min') -> pd.DataFrame:
    """Loads, pivots, and resamples weather data to the target frequency."""
    logger.info("Loading weather data...")
    df = pd.read_parquet(weather_path)
    
    # Drop units as per EDA
    if 'units' in df.columns:
        df = df.drop(columns=['units'])
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info("Pivoting weather data...")
    pivoted = df.pivot_table(
        index=['sensor_name', 'timestamp'],
        columns='variable',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    logger.info("Resampling weather data to %s...", frequency)
    # Resample weather data to ensure it aligns with the 15min grid
    # We will set index to timestamp, group by sensor_name, and resample
    def resample_weather(g):
        return g.set_index('timestamp').resample(frequency).mean()
        
    # Apply resampling
    resampled_weather = pivoted.groupby('sensor_name').apply(resample_weather)
    resampled_weather = resampled_weather.reset_index()
    
    logger.info(f"Weather data prepared. Shape: {resampled_weather.shape}")
    return resampled_weather

def process_full_dataset():
    traffic_path = 'data/traffic_20230108_20260608.parquet'
    weather_path = 'data/weather_20230108_20260108.parquet'
    output_dir = 'data/processed'
    output_file = os.path.join(output_dir, 'master_dataset_15T.parquet')
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open('data/traffic_sensors.json', 'r') as f:
        traffic_sensors = json.load(f)
    with open('data/weather_sensors.json', 'r') as f:
        weather_sensors = json.load(f)

    # Prepare external datasets
    weather_df = prepare_weather_data(weather_path, frequency='15min')
    feature_loader = StaticFeatureLoader()
    pipeline = PreprocessingPipeline(frequency='15min')
    
    logger.info("Discovering unique locations in traffic data...")
    # Load just locations to get unique ones
    locs_df = pd.read_parquet(traffic_path, columns=['location'])
    unique_locations = locs_df['location'].unique()
    del locs_df
    gc.collect()
    
    logger.info(f"Found {len(unique_locations)} unique locations to process. Starting chunked processing...")
    
    # Process sequentially by location to be safe with RAM
    first_chunk = True
    processed_rows = 0
    
    # Read full traffic data into memory (21M rows, ~6 cols, fits in RAM easily, we just chunk the processing)
    # Actually, 21M rows is a bit large, let's load it and group by location.
    # To be extremely memory safe, we could read filters with pyarrow, but loading once is faster if it fits.
    # Let's try loading it once and chunking the DataFrame.
    full_traffic = pd.read_parquet(traffic_path)
    
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    writer = None
    
    for loc in unique_locations:
        loc_df = full_traffic[full_traffic['location'] == loc].copy()
        
        # 1. Resample
        resampled_df = pipeline.resample_traffic(loc_df)
        
        # 2. Impute
        imputed_df = pipeline.impute_missingness(resampled_df)
        
        # 3. Spatial Join
        joined_df = pipeline.spatial_join(imputed_df, traffic_sensors, weather_sensors)
        
        # 4. Static Feature Join
        enriched_df = feature_loader.join_features(joined_df)
        
        # 5. Weather Time Series Join
        # Merge on nearest_weather_sensor and dt
        final_chunk = enriched_df.merge(
            weather_df,
            left_on=['nearest_weather_sensor', 'dt'],
            right_on=['sensor_name', 'timestamp'],
            how='left'
        )
        # Drop redundant columns from weather
        if 'sensor_name' in final_chunk.columns:
            final_chunk = final_chunk.drop(columns=['sensor_name'])
        if 'timestamp' in final_chunk.columns:
            final_chunk = final_chunk.drop(columns=['timestamp'])
            
        # Write to parquet file using pyarrow iterative writer
        table = pa.Table.from_pandas(final_chunk)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
            
        processed_rows += len(final_chunk)
        logger.info(f"Processed location {loc}. Total accumulated rows: {processed_rows}")
        
        # Free memory
        del loc_df, resampled_df, imputed_df, joined_df, enriched_df, final_chunk, table
        gc.collect()

    if writer is not None:
        writer.close()

    logger.info(f"Full scale processing complete. Saved to {output_file}")
    logger.info(f"Total Rows: {processed_rows}")

if __name__ == "__main__":
    process_full_dataset()
