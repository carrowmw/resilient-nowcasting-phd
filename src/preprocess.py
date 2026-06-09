import logging
import numpy as np
import pandas as pd
from typing import Dict, List

# Standard random seed initialization
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SensorMappingError(Exception):
    """Custom exception raised when spatial join fails to map a traffic sensor to a weather sensor."""
    pass


class PreprocessingPipeline:
    def __init__(self, frequency: str = '15min'):
        self.frequency = frequency
        logger.info(f"Initialized PreprocessingPipeline with frequency: {self.frequency}")

    def resample_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples the irregular traffic data into a continuous frequency grid.
        Missing timestamps will reappear as explicit NaN rows.
        """
        logger.info(f"Starting resample_traffic. Input shape: {df.shape}")

        # Assertion: Input dataframe is not empty
        assert not df.empty, "Input dataframe is empty."
        
        # Assertion: Contains required columns
        required_cols = ['location', 'category', 'veh_class', 'dir', 'dt', 'value']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        if not pd.api.types.is_datetime64_any_dtype(df['dt']):
            df['dt'] = pd.to_datetime(df['dt'])

        # Handle physical impossibilities (negative values) before resampling
        # Tripwire: Ensure no silent suppression, explicitly convert anomalous negatives to NaN
        df.loc[df['value'] < 0, 'value'] = np.nan

        # Generate the global target grid
        min_dt = df['dt'].min().floor(self.frequency)
        max_dt = df['dt'].max().ceil(self.frequency)
        expected_index = pd.date_range(start=min_dt, end=max_dt, freq=self.frequency, name='dt')

        # Group and resample
        groups = ['location', 'category', 'veh_class', 'dir']
        
        # We'll build the continuous grid for each group explicitly
        resampled_list = []
        unique_groups = df[groups].drop_duplicates().dropna(subset=groups)
        
        for _, row in unique_groups.iterrows():
            logger.info(f"DEBUG: Processing group: {row.to_dict()}")
            group_mask = True
            for col in groups:
                group_mask &= (df[col] == row[col])
            
            group_df = df[group_mask].set_index('dt')
            # Resample and reindex to global grid
            group_resampled = group_df['value'].resample(self.frequency).sum(min_count=1).reindex(expected_index)
            
            # Reconstruct the group columns
            temp_df = group_resampled.reset_index()
            # If the index was named 'dt', it appears as 'dt'. If unnamed, it appears as 'index'.
            if 'index' in temp_df.columns:
                temp_df = temp_df.rename(columns={'index': 'dt'})
            
            # Ensure the value column is correctly named
            potential_value_cols = [c for c in temp_df.columns if c != 'dt']
            if potential_value_cols:
                temp_df = temp_df.rename(columns={potential_value_cols[0]: 'value'})
            for col in groups:
                temp_df[col] = row[col]
            
            resampled_list.append(temp_df)
        
        final_df = pd.concat(resampled_list, ignore_index=True)

        # Grid Validation Tripwire
        observed_groups = df[groups].drop_duplicates()
        expected_total_rows = len(observed_groups) * len(expected_index)
        assert len(final_df) == expected_total_rows, (
            f"Grid validation failed: Expected {expected_total_rows} rows, "
            f"but got {len(final_df)}."
        )

        logger.info(f"Resampling complete. Output shape: {final_df.shape}")
        return final_df

    def impute_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Missingness-Aware Imputation Strategy.
        Prioritizes 'Missing Zero' imputation and leaves true gaps as NaN.
        """
        logger.info(f"Starting impute_missingness. Input shape: {df.shape}")
        assert not df.empty, "Dataframe is empty before imputation."
        assert 'dt' in df.columns and 'value' in df.columns, "Missing 'dt' or 'value' columns."
        
        # Priority 1: Structural Missingness (Zero-Priority)
        # Using a simple contextual heuristic: If it's late night (00:00 to 04:59), assume missing means 0 traffic.
        hour = df['dt'].dt.hour
        is_night = (hour >= 0) & (hour < 5)
        
        structural_zero_mask = df['value'].isna() & is_night
        df.loc[structural_zero_mask, 'value'] = 0.0
        
        # Priority 2: True Gaps
        # All other NaN values remain as NaN to maintain missingness awareness.
        # Explicitly DO NOT use .fillna(method='ffill') or .dropna()
        
        logger.info(f"Imputation complete. Remaining explicit NaNs: {df['value'].isna().sum()}")
        return df

    def spatial_join(self, traffic_df: pd.DataFrame, traffic_sensors: List[Dict], weather_sensors: List[Dict]) -> pd.DataFrame:
        """
        Maps each traffic sensor to the nearest weather sensor using spatial coordinates.
        """
        logger.info("Starting spatial_join.")
        assert not traffic_df.empty, "Traffic dataframe is empty."
        assert len(traffic_sensors) > 0, "No traffic sensors provided."
        assert len(weather_sensors) > 0, "No weather sensors provided."

        # Spatial Edge Case: N=1 weather sensors
        if len(weather_sensors) == 1:
            logger.info("Only 1 weather sensor found. Bypassing distance matrix calculation.")
            weather_sensor_name = weather_sensors[0].get('Sensor_Name')
            assert weather_sensor_name is not None, "Weather sensor missing 'Sensor_Name'."
            nearest_mapping = {ts['location']: weather_sensor_name for ts in traffic_sensors}
        else:
            logger.info(f"Calculating distance matrix for {len(weather_sensors)} weather sensors.")
            nearest_mapping = {}
            for ts in traffic_sensors:
                assert 'lat' in ts and 'lon' in ts, f"Traffic sensor missing lat/lon: {ts}"
                
                min_dist = float('inf')
                best_ws = None
                
                for ws in weather_sensors:
                    assert 'Sensor_Centroid_Latitude' in ws and 'Sensor_Centroid_Longitude' in ws, (
                        f"Weather sensor missing coordinates: {ws}"
                    )
                    
                    dy = ts['lat'] - ws['Sensor_Centroid_Latitude']
                    dx = ts['lon'] - ws['Sensor_Centroid_Longitude']
                    dist = dy**2 + dx**2  # Simplified squared distance (Euclidean approximation of Haversine)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_ws = ws.get('Sensor_Name')
                
                if best_ws is None:
                    raise SensorMappingError(f"Failed to map traffic sensor {ts['location']} to any weather sensor.")
                
                nearest_mapping[ts['location']] = best_ws

        traffic_df['nearest_weather_sensor'] = traffic_df['location'].map(nearest_mapping)

        # No Silent Drops: Crash if spatial join fails for any location
        if traffic_df['nearest_weather_sensor'].isna().any():
            unmapped = traffic_df[traffic_df['nearest_weather_sensor'].isna()]['location'].unique()
            raise SensorMappingError(f"Spatial join failed to find sensor mapping for locations: {unmapped}")

        logger.info(f"Spatial join complete. Output shape: {traffic_df.shape}")
        return traffic_df

