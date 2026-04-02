import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StaticFeatureLoader:
    def __init__(self, metadata_path: str = 'data/external/google_places_template.csv'):
        self.metadata_path = metadata_path
        logger.info(f"Initialized StaticFeatureLoader with metadata path: {self.metadata_path}")

    def join_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads the static external metadata and joins it to the traffic sensor dataframe.
        Implements strict validation to ensure complete feature coverage.
        """
        logger.info(f"Starting join_features. Loading metadata from {self.metadata_path}")
        
        # Load the metadata template
        try:
            metadata_df = pd.read_csv(self.metadata_path)
        except FileNotFoundError:
            raise RuntimeError(f"CRITICAL: Metadata file not found at {self.metadata_path}")

        # Basic schema assertions
        assert not df.empty, "Traffic dataframe is empty prior to static feature join."
        assert 'location' in df.columns, "Traffic dataframe is missing the 'location' identifier."
        assert 'location_id' in metadata_df.columns, "Metadata is missing the 'location_id' column."
        
        # OFFENSIVE RULE: Assert all unique traffic locations exist in metadata
        traffic_locations = set(df['location'].unique())
        metadata_locations = set(metadata_df['location_id'].unique())
        
        missing_locations = traffic_locations - metadata_locations
        
        # This will purposefully crash the pipeline if the CSV has not been fully populated 
        # for every sensor seen in the traffic dataset.
        assert len(missing_locations) == 0, (
            f"CRITICAL ERROR: {len(missing_locations)} traffic sensors are missing static features "
            f"in the Google Places API metadata file. Missing locations: {missing_locations}"
        )
        
        # Perform the join
        logger.info(f"Validation passed. Joining static features for {len(traffic_locations)} unique locations.")
        metadata_df = metadata_df.rename(columns={'location_id': 'location'})
        
        # Use a left merge to attach the features, but strictly validate the row count hasn't changed
        original_len = len(df)
        df = df.merge(metadata_df, on='location', how='left')
        assert len(df) == original_len, "CRITICAL: The merge operation unexpectedly altered the row count."
        
        # Final safety net: Ensure no NaNs were introduced into the new feature columns
        new_feature_cols = ['nearby_poi_count', 'zone_type', 'dist_to_nearest_intersection']
        for col in new_feature_cols:
            assert col in df.columns, f"CRITICAL: Expected column '{col}' missing from metadata."
            assert df[col].notna().all(), f"CRITICAL: Null values detected in static feature '{col}' after join."
            
        logger.info("Static feature enrichment complete.")
        return df
