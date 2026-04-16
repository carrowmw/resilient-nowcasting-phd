import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging

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
                 mask_prob: float = 0.15):
        
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
            columns=['dt', 'location', 'value', 'Humidity', 'Solar Radiation', 'nearby_poi_count', 'dist_to_nearest_intersection']
        )
        
        if df.empty:
            raise ValueError(f"No data found for the specified date range: {start_date} to {end_date}")
            
        # 2. Extract dimensions
        self.locations = sorted(df['location'].unique())
        self.times = sorted(df['dt'].unique())
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.num_nodes = len(self.locations)
        
        logger.info(f"Loaded {len(self.times)} timesteps across {self.num_nodes} locations.")
        
        # 3. Pivot for Sliding Window Logic
        df = df.set_index(['dt', 'location'])
        
        # Dynamic features
        dyn_df = df[['value', 'Humidity', 'Solar Radiation']].unstack('location')
        locations_idx = pd.Index(self.locations, name='location')
        
        val_arr = dyn_df['value'].reindex(columns=locations_idx).fillna(0.0).values
        hum_arr = dyn_df['Humidity'].reindex(columns=locations_idx).fillna(0.0).values
        sol_arr = dyn_df['Solar Radiation'].reindex(columns=locations_idx).fillna(0.0).values
        
        # [num_times, num_nodes, 3]
        self.dynamic_data = np.stack([val_arr, hum_arr, sol_arr], axis=-1)
        self.time_series_dim = self.dynamic_data.shape[-1]
        
        # Static features
        stat_df = df[['nearby_poi_count', 'dist_to_nearest_intersection']].groupby('location').first()
        stat_df = stat_df.reindex(locations_idx).fillna(0.0)
        self.static_features = torch.tensor(stat_df.values, dtype=torch.float32)
        self.static_feat_dim = self.static_features.shape[-1]
        
        logger.info(f"Feature dimensions established. Static: {self.static_feat_dim}, Dynamic: {self.time_series_dim}")

    def __len__(self) -> int:
        return max(0, len(self.times) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        # Extract sliding window
        # Dynamic: [seq_len, num_nodes, D] -> permute -> [num_nodes, seq_len, D]
        dyn_slice = self.dynamic_data[idx : idx + self.seq_len]
        dyn_slice = np.transpose(dyn_slice, (1, 0, 2))
        
        # Forecast target: next timestep (seq_len), index 0 is 'value'
        target = self.dynamic_data[idx + self.seq_len, :, 0]
        
        dyn_tensor = torch.tensor(dyn_slice, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
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
