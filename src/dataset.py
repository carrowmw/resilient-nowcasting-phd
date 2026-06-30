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
                 target_dir: str = 'all',
                 use_database: bool = False):

        logger.info(f"Loading data from {parquet_path if not use_database else 'DATABASE'} "
                    f"between {start_date} and {end_date}")

        # 1. Load Data with Time Filter Constraints
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        if use_database:
            df = self._load_from_database(start_dt, end_dt, target_veh_class, target_dir)
        else:
            df = self._load_from_parquet(parquet_path, start_dt, end_dt, target_veh_class, target_dir)
        
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

        # PHASE 1: Separate scaling for traffic (ch 0) vs weather (ch 1-2)
        # to enable TRUE MSLE on raw counts via inverse transform in loss computation.

        num_times = len(self.times)
        num_nodes_d = len(self.locations)

        # Traffic channel: Fit dedicated scaler and expose statistics
        traffic_flat = val_arr.reshape(-1, 1)  # [T*N, 1]
        self.traffic_scaler = StandardScaler()
        traffic_scaled_flat = self.traffic_scaler.fit_transform(traffic_flat)
        traffic_scaled = traffic_scaled_flat.reshape(num_times, num_nodes_d)  # [T, N]

        # Extract and expose scaler statistics for pipeline buffers
        self.traffic_mean = float(self.traffic_scaler.mean_[0])
        self.traffic_std = float(self.traffic_scaler.scale_[0])

        # OFFENSIVE ASSERTIONS
        assert self.traffic_std > 0, "CRITICAL: Traffic std is zero (constant values)"
        assert not np.isnan(self.traffic_mean), "CRITICAL: Traffic mean is NaN"
        assert not np.isnan(self.traffic_std), "CRITICAL: Traffic std is NaN"

        # Weather channels: Separate scaler for humidity and solar radiation
        weather_arr = np.stack([hum_arr, sol_arr], axis=-1)  # [T, N, 2]
        weather_flat = weather_arr.reshape(-1, 2)  # [T*N, 2]
        self.weather_scaler = StandardScaler()
        weather_scaled_flat = self.weather_scaler.fit_transform(weather_flat)
        weather_scaled = weather_scaled_flat.reshape(num_times, num_nodes_d, 2)  # [T, N, 2]

        # Stack: [T, N, 3] where dim 0=traffic, dim 1-2=weather
        self.dynamic_data = np.stack([
            traffic_scaled,
            weather_scaled[:, :, 0],  # Humidity
            weather_scaled[:, :, 1]   # Solar Radiation
        ], axis=-1)

        self.time_series_dim = self.dynamic_data.shape[-1]

        logger.info(f"Traffic scaler: mean={self.traffic_mean:.2f}, std={self.traffic_std:.2f}")
        logger.info(f"Dynamic features scaled separately: traffic (ch0) + weather (ch1-2) over {num_times * num_nodes_d} samples.")

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

    def _load_from_parquet(self, parquet_path, start_dt, end_dt, veh_class, direction):
        """Existing Parquet loading logic (unchanged)."""
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
        return df

    def _load_from_database(self, start_dt, end_dt, veh_class, direction):
        """
        Load traffic readings from TimescaleDB and join with static features cache.

        Returns a DataFrame matching the Parquet schema.
        """
        import psycopg2
        import os
        from dotenv import load_dotenv

        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not set — cannot use database mode")

        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        # Query traffic readings with filters
        query = """
            SELECT dt, sensor_id, veh_class, direction, count
            FROM traffic_readings
            WHERE dt >= %s AND dt < %s
              AND veh_class = %s
            ORDER BY dt, sensor_id
        """
        params = [start_dt, end_dt, veh_class]

        if direction != 'all':
            query = query.replace("ORDER BY", "AND direction = %s ORDER BY")
            params.append(direction)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise ValueError(f"No database rows for {start_dt} to {end_dt}, veh_class={veh_class}")

        df = pd.DataFrame(rows, columns=["dt", "location", "veh_class", "dir", "value"])

        # Join with static features cache (weather + spatial metadata)
        # This requires a separate table or JSON cache — for MVP, fall back to Parquet
        logger.warning("Database mode: static features still loaded from Parquet cache")
        static_cache_path = "data/external/static_features_cache.parquet"
        if os.path.exists(static_cache_path):
            static_df = pd.read_parquet(static_cache_path)
            # Ensure timezone alignment for dt column to prevent merge errors
            if df["dt"].dt.tz is not None:
                static_df["dt"] = pd.to_datetime(static_df["dt"]).dt.tz_localize("UTC")
            else:
                static_df["dt"] = pd.to_datetime(static_df["dt"]).dt.tz_localize(None)
            df = df.merge(static_df, on=["dt", "location"], how="left")
        else:
            raise FileNotFoundError(f"Static features cache not found: {static_cache_path}")

        return df

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
            'node_mask': node_mask,
            # PHASE 3: ordered node identifiers for physical edge feature lookup.
            'location_ids': self.locations
        }

def make_spatial_holdout_mask(num_nodes: int,
                              holdout_prob: float,
                              device: str = 'cpu',
                              seed: int = None) -> torch.Tensor:
    """
    Create a boolean mask marking spatially held-out nodes (IGNNK-style node masking,
    Spatial Estimation Architecture Component 2, docs §4).

    Held-out nodes have their dynamic features blanked before the encoder so their
    latent embeddings are shaped purely by message passing from observable neighbours;
    the kriging head must then reconstruct their series.

    Args:
        num_nodes:    Total sensor count.
        holdout_prob: Fraction of nodes to hold out, e.g. 0.15 for 15%.
        device:       Target device for the returned mask.
        seed:         None -> fresh randomness (training, re-rolled per batch);
                      int  -> deterministic mask (validation, stable metric tracking).

    Returns:
        mask: [num_nodes] boolean tensor, True = held out.
    """
    assert 0.0 <= holdout_prob <= 1.0, \
        f"CRITICAL: holdout_prob must be in [0, 1], got {holdout_prob}"
    assert num_nodes > 0, f"CRITICAL: num_nodes must be positive, got {num_nodes}"

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    mask = rng.rand(num_nodes) < holdout_prob
    return torch.from_numpy(mask).bool().to(device)


def apply_spatial_holdout(dynamic_features: torch.Tensor,
                          holdout_mask: torch.Tensor) -> torch.Tensor:
    """
    Zero out dynamic features for held-out nodes across all timesteps and channels.

    Args:
        dynamic_features: [B, N, T, D]
        holdout_mask:     [N] boolean, True = held out.

    Returns:
        masked_features: [B, N, T, D] with held-out nodes zeroed (input left untouched).
    """
    assert dynamic_features.dim() == 4, \
        f"CRITICAL: expected 4D [B, N, T, D], got {tuple(dynamic_features.shape)}"
    assert holdout_mask.dim() == 1, \
        f"CRITICAL: holdout_mask must be 1D, got {tuple(holdout_mask.shape)}"
    assert holdout_mask.shape[0] == dynamic_features.shape[1], \
        f"CRITICAL: mask len {holdout_mask.shape[0]} != num_nodes {dynamic_features.shape[1]}"

    masked = dynamic_features.clone()
    masked[:, holdout_mask, :, :] = 0.0
    return masked


def custom_collate_real(batch_list):
    """Collate function to maintain node structural dimensions."""
    return {
        'static_features': batch_list[0]['static_features'], # Static across batch
        'dynamic_features': torch.stack([b['dynamic_features'] for b in batch_list]),
        'target_impute': torch.stack([b['target_impute'] for b in batch_list]),
        'target_forecast': torch.stack([b['target_forecast'] for b in batch_list]),
        'node_mask': torch.stack([b['node_mask'] for b in batch_list]),
        # PHASE 3: location_ids constant across the batch (same node ordering).
        'location_ids': batch_list[0]['location_ids']
    }
