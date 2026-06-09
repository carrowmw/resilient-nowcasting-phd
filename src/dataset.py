import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

# Standard random seed initialization
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealIoTDataset(Dataset):
    """
    Production PyTorch Dataset for the ResilientIoTPipeline.
    Loads real parquet data efficiently using memory-friendly time filters.
    Groups data by location, applies sliding window extraction, and injects NaN masking.
    """
    def __init__(self, 
                 parquet_path: str, 
                 start_date: str, 
                 end_date: str, 
                 seq_len: int = 12, 
                 mask_prob: float = 0.15,
                 target_veh_class: str = 'person',
                 target_dir: str = 'all'):
        
        logger.info(f"Loading data from {parquet_path} between {start_date} and {end_date}")
        
        # 1. Load Data with Time Filter Constraints
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        df = pd.read_parquet(
            parquet_path,
            filters=[
                ('dt', '>=', start_dt),
                ('dt', '<', end_dt)
            ],
            columns=[
                'dt', 'location', 'value', 'Humidity', 'Solar Radiation',
                'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
                'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
                'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
                'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking',
                'veh_class', 'dir'
            ]
        )
        
        if df.empty:
            raise ValueError(f"No data found for the specified date range: {start_date} to {end_date}")

        available_classes = sorted(df['veh_class'].dropna().astype(str).unique().tolist())
        available_dirs = sorted(df['dir'].dropna().astype(str).unique().tolist())
            
        # 2. Filter target stream to ensure unique (dt, location) indices
        df = df[df['veh_class'] == target_veh_class]
        if df.empty:
            raise ValueError(
                f"No rows after veh_class filter. Requested veh_class='{target_veh_class}' "
                f"for range {start_date} to {end_date}. Available veh_class values: {available_classes}"
            )
        
        if target_dir == 'all':
            # Group by dt and location, sum values, take first for weather/static features
            agg_funcs = {
                'value': 'sum',
                'Humidity': 'first',
                'Solar Radiation': 'first',
                'transit_count': 'first',
                'retail_count': 'first',
                'food_drink_count': 'first',
                'nightlife_count': 'first',
                'education_count': 'first',
                'recreation_count': 'first',
                'total_poi_count': 'first',
                'landuse_entropy': 'first',
                'is_school_zone': 'first',
                'betweenness_centrality': 'first',
                'closeness_centrality': 'first',
                'dist_to_city_centre': 'first',
                'degree_centrality': 'first',
                'nearby_bus_stops': 'first',
                'nearby_parking': 'first',
            }
            df = df.groupby(['dt', 'location'], as_index=False).agg(agg_funcs)
        else:
            df = df[df['dir'] == target_dir]
            if df.empty:
                raise ValueError(
                    f"No rows after dir filter. Requested dir='{target_dir}' with veh_class='{target_veh_class}' "
                    f"for range {start_date} to {end_date}. Available dir values before filtering: {available_dirs}"
                )

        if df.empty:
            raise ValueError(
                f"No rows available after filtering and aggregation for range {start_date} to {end_date}, "
                f"veh_class='{target_veh_class}', dir='{target_dir}'."
            )
            
        # 3. Extract dimensions
        self.locations = sorted(df['location'].unique())
        self.times = sorted(df['dt'].unique())
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.num_nodes = len(self.locations)
        
        logger.info(f"Loaded {len(self.times)} timesteps across {self.num_nodes} locations for class={target_veh_class}, dir={target_dir}.")
        
        # 4. Pivot for Sliding Window Logic
        df = df.set_index(['dt', 'location'])
        
        # Dynamic features
        dyn_df = df[['value', 'Humidity', 'Solar Radiation']].unstack('location')
        locations_idx = pd.Index(self.locations, name='location')
        
        val_arr = dyn_df['value'].reindex(columns=locations_idx).fillna(0.0).values
        hum_arr = dyn_df['Humidity'].reindex(columns=locations_idx).fillna(0.0).values
        sol_arr = dyn_df['Solar Radiation'].reindex(columns=locations_idx).fillna(0.0).values

        # Preserve a raw, unscaled copy of traffic counts BEFORE any scaling.
        # ZINB NLL requires non-negative counts as targets; StandardScaler would
        # convert these to signed floats and corrupt the loss domain.
        self.unscaled_traffic = val_arr.copy()  # [num_times, num_nodes]

        # [num_times, num_nodes, 3]
        self.dynamic_data = np.stack([val_arr, hum_arr, sol_arr], axis=-1)
        self.time_series_dim = self.dynamic_data.shape[-1]

        # Scale dynamic features across the full dataset so the model receives
        # zero-mean, unit-variance inputs — critical for stable gradient flow.
        num_times, num_nodes_d, n_dyn_features = self.dynamic_data.shape
        dyn_flat = self.dynamic_data.reshape(-1, n_dyn_features)
        self.scaler = StandardScaler()
        dyn_scaled = self.scaler.fit_transform(dyn_flat)
        self.dynamic_data = dyn_scaled.reshape(num_times, num_nodes_d, n_dyn_features)
        logger.info(f"Dynamic features scaled: mean≈0, std≈1 over {num_times * num_nodes_d} samples.")

        # Static features
        stat_cols = [
            'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
            'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
            'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
            'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking'
        ]
        stat_df = df[stat_cols].groupby('location').first()
        stat_df['is_school_zone'] = stat_df['is_school_zone'].astype(float).fillna(0.0)
        stat_df = stat_df.reindex(locations_idx).fillna(0.0)

        # Scale static features so DynamicGraphLearner receives normalised coordinates/metadata.
        stat_np = stat_df.values  # [num_nodes, static_feat_dim]
        self.static_scaler = StandardScaler()
        stat_np_scaled = self.static_scaler.fit_transform(stat_np)
        self.static_features = torch.tensor(stat_np_scaled, dtype=torch.float32)
        self.static_feat_dim = self.static_features.shape[-1]
        logger.info(f"Static features scaled: mean≈0, std≈1 over {stat_np.shape[0]} nodes.")
        
        logger.info(f"Feature dimensions established. Static: {self.static_feat_dim}, Dynamic: {self.time_series_dim}")

    def __len__(self) -> int:
        return max(0, len(self.times) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        # Extract sliding window
        # Dynamic: [seq_len, num_nodes, D] -> permute -> [num_nodes, seq_len, D]
        dyn_slice = self.dynamic_data[idx : idx + self.seq_len]
        dyn_slice = np.transpose(dyn_slice, (1, 0, 2))
        
        dyn_tensor = torch.tensor(dyn_slice, dtype=torch.float32)

        # Forecast target: pull from unscaled traffic counts, not scaled dynamic_data.
        # clamp(min=0) is a hard tripwire — raw counts must never be negative,
        # but guards against any residual float precision artifacts from the parquet.
        target = self.unscaled_traffic[idx + self.seq_len]  # [num_nodes]
        target_tensor = torch.tensor(target, dtype=torch.float32).clamp(min=0)
        
        # 4. Dynamic Masking logic (~15% artificial NaN injection)
        node_mask = torch.rand_like(dyn_tensor) < self.mask_prob
        target_impute = dyn_tensor.clone()
        dyn_masked = dyn_tensor.clone()
        dyn_masked[node_mask] = float('nan')

        return {
            'static_features': self.static_features,
            'dynamic_features': dyn_masked,
            'target_impute': target_impute,
            'target_forecast': target_tensor,
            'node_mask': node_mask
        }

def custom_collate_real(batch_list):
    """Collate function to maintain node structural dimensions."""
    return {
        'static_features': batch_list[0]['static_features'], # Static across batch
        'dynamic_features': torch.stack([b['dynamic_features'] for b in batch_list]),
        'target_impute': torch.stack([b['target_impute'] for b in batch_list]),
        'target_forecast': torch.stack([b['target_forecast'] for b in batch_list]),
        'node_mask': torch.stack([b['node_mask'] for b in batch_list])
    }
