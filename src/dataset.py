import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
from typing import Optional
from sklearn.preprocessing import StandardScaler

from db_connector import DataConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATIC_FEATURE_COLS = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking',
]


class RealIoTDataset(Dataset):
    """
    Production PyTorch Dataset for the ResilientIoTPipeline.
    Loads real parquet data efficiently using memory-friendly time filters.
    Groups data by location, applies sliding window extraction, and injects NaN masking.
    """
    def __init__(self,
                 connector: DataConnector,
                 start_date: str,
                 end_date: str,
                 seq_len: int = 12,
                 target_veh_class: str = 'person',
                 target_dir: str = 'all',
                 train_split: Optional[float] = None,
                 ablate_masks: bool = False):

        logger.info(f"Loading data via connector {type(connector).__name__} "
                    f"between {start_date} and {end_date}")

        # 1. Load Data with Time Filter Constraints
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        traffic_df = connector.load_traffic(start_dt, end_dt, target_veh_class, target_dir)
        weather_df = connector.load_weather(start_dt, end_dt, traffic_df['location'].unique().tolist())
        static_df = connector.load_static_features(traffic_df['location'].unique().tolist())
        df = traffic_df.merge(weather_df, on=['dt', 'location'], how='left')
        df = df.merge(static_df, on='location', how='left')

        assert df['Humidity'].notna().all(), "CRITICAL: NaN Humidity after weather join"
        assert df['Solar Radiation'].notna().all(), "CRITICAL: NaN Solar Radiation after weather join"
        for _col in STATIC_FEATURE_COLS:
            assert _col in df.columns, f"CRITICAL: Missing static feature column '{_col}'"

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
            # Group by dt and location.
            # Using min_count=1 ensures all-null station slots remain NaN (F-C03 fix)
            agg_funcs = {
                'value': lambda s: np.nan if s.isna().all() else s.sum(min_count=1),
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
        self.num_nodes = len(self.locations)

        logger.info(f"Loaded {len(self.times)} timesteps across {self.num_nodes} locations for class={target_veh_class}, dir={target_dir}.")

        # 4. Pivot for Sliding Window Logic
        df = df.set_index(['dt', 'location'])

        # Dynamic features
        dyn_df = df[['value', 'Humidity', 'Solar Radiation']].unstack('location')
        locations_idx = pd.Index(self.locations, name='location')

        val_df = dyn_df['value'].reindex(columns=locations_idx)
        hum_df = dyn_df['Humidity'].reindex(columns=locations_idx)
        sol_df = dyn_df['Solar Radiation'].reindex(columns=locations_idx)

        # 4b. Extract explicit MNAR masks
        # val_df index is 'dt', columns are 'location'. Shape: [num_times, num_nodes]
        total_mnar_mask = val_df.isna().values.astype(np.float32)
        
        # Extract hours from the DatetimeIndex to build the night heuristic mask
        # Broadcast hour comparison to the shape of the data
        hours = val_df.index.get_level_values('dt').hour.values[:, None]
        is_night_arr = (hours >= 0) & (hours < 5)
        night_mnar_mask = (val_df.isna().values & is_night_arr).astype(np.float32)
        
        # Shape: [num_times, num_nodes, 2]
        self.structural_masks = torch.from_numpy(np.stack([total_mnar_mask, night_mnar_mask], axis=-1))
        
        logger.info(f"MNAR Masks Generated. Total missing: {(total_mnar_mask.mean()*100):.2f}%. Night missing: {(night_mnar_mask.mean()*100):.2f}%")
        
        if ablate_masks:
            self.structural_masks = torch.zeros_like(self.structural_masks)
            logger.info("ABLATION: Explicit MNAR masks zeroed out.")

        # Explicitly apply 0.0 imputation to resolve nulls upstream
        val_df = val_df.fillna(0.0)
        hum_df = hum_df.fillna(0.0)
        sol_df = sol_df.fillna(0.0)

        val_arr = val_df.values
        hum_arr = hum_df.values
        sol_arr = sol_df.values

        # Preserve a raw, unscaled copy of traffic counts BEFORE any scaling.
        # ZINB NLL requires non-negative counts as targets; StandardScaler would
        # convert these to signed floats and corrupt the loss domain.
        self.unscaled_traffic = val_arr.copy()  # [num_times, num_nodes]

        # PHASE 1 & F-C02: Separate scaling for traffic (ch 0) vs weather (ch 1-2)
        # to enable TRUE MSLE on raw counts via inverse transform in loss computation.
        # Scalers are fitted EXCLUSIVELY on the training slice (F-C02 fix).

        num_times = len(self.times)
        num_nodes_d = len(self.locations)

        if train_split is not None and 0.0 < train_split < 1.0:
            train_t_end = int(train_split * num_times)
        else:
            train_t_end = num_times

        # Traffic channel: Fit dedicated scaler on observed training counts only (excluding null fill-zeros)
        val_train_slice = val_arr[:train_t_end]  # [train_T, N]
        mnar_train_slice = total_mnar_mask[:train_t_end]  # [train_T, N]
        observed_mask = (mnar_train_slice == 0.0)

        observed_counts = val_train_slice[observed_mask].reshape(-1, 1)
        assert len(observed_counts) > 0, "CRITICAL: No observed counts in training slice to fit scaler"

        self.traffic_scaler = StandardScaler()
        self.traffic_scaler.fit(observed_counts)

        traffic_flat = val_arr.reshape(-1, 1)  # [T*N, 1]
        traffic_scaled_flat = self.traffic_scaler.transform(traffic_flat)
        traffic_scaled = traffic_scaled_flat.reshape(num_times, num_nodes_d)  # [T, N]

        # Extract and expose scaler statistics for pipeline buffers
        self.traffic_mean = float(self.traffic_scaler.mean_[0])
        self.traffic_std = float(self.traffic_scaler.scale_[0])

        # OFFENSIVE ASSERTIONS
        assert self.traffic_std > 0, "CRITICAL: Traffic std is zero (constant values)"
        assert not np.isnan(self.traffic_mean), "CRITICAL: Traffic mean is NaN"
        assert not np.isnan(self.traffic_std), "CRITICAL: Traffic std is NaN"

        # Weather channels: Fit scaler strictly on training slice
        weather_arr = np.stack([hum_arr, sol_arr], axis=-1)  # [T, N, 2]
        weather_flat = weather_arr.reshape(-1, 2)  # [T*N, 2]
        weather_train = weather_flat[:train_t_end * num_nodes_d]
        self.weather_scaler = StandardScaler()
        self.weather_scaler.fit(weather_train)
        weather_scaled_flat = self.weather_scaler.transform(weather_flat)
        weather_scaled = weather_scaled_flat.reshape(num_times, num_nodes_d, 2)  # [T, N, 2]

        # Stack: [T, N, 3] where dim 0=traffic, dim 1-2=weather
        self.dynamic_data = np.stack([
            traffic_scaled,
            weather_scaled[:, :, 0],  # Humidity
            weather_scaled[:, :, 1]   # Solar Radiation
        ], axis=-1)

        self.dynamic_data = torch.from_numpy(self.dynamic_data.astype(np.float32))
        self.unscaled_traffic = torch.from_numpy(self.unscaled_traffic.astype(np.float32))
        assert not torch.isnan(self.dynamic_data).any(), "CRITICAL: NaN in scaled dynamic data"

        self.time_series_dim = self.dynamic_data.shape[-1]

        logger.info(f"Traffic scaler (train-only): mean={self.traffic_mean:.2f}, std={self.traffic_std:.2f}")
        logger.info(f"Dynamic features scaled separately: traffic (ch0) + weather (ch1-2) over {num_times * num_nodes_d} samples.")

        # Static features
        stat_df = df[STATIC_FEATURE_COLS].groupby('location').first()
        # is_school_zone is boolean in DB — convert to float; NaN here means the column is missing,
        # which is a schema violation, not a gap.
        stat_df['is_school_zone'] = stat_df['is_school_zone'].astype(float)
        stat_df = stat_df.reindex(locations_idx)
        assert not stat_df.isna().any().any(), \
            "CRITICAL: NaN in static features after reindex — sensor IDs are mismatched or data missing"

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
        # Dynamic: [seq_len, num_nodes, D] -> permute -> [num_nodes, seq_len, D] (clean, no masking)
        dyn_tensor = self.dynamic_data[idx : idx + self.seq_len].permute(1, 0, 2)

        # Also slice and permute the structural masks to match
        struct_masks = self.structural_masks[idx : idx + self.seq_len].permute(1, 0, 2)

        # Forecast target: pull from unscaled traffic counts, not scaled dynamic_data.
        target_tensor = self.unscaled_traffic[idx + self.seq_len]
        assert (target_tensor >= -1e-6).all(), \
            f"CRITICAL: negative traffic count at window idx={idx} — expected non-negative raw counts"
        target_tensor = target_tensor.clamp(min=0)  # clamp only float-precision dust (< 1e-6)

        # NaN injection and node_mask are generated on-device in the train/val loops so
        # masking uses CUDA RNG and avoids a CPU→GPU tensor copy of the mask.
        return {
            'dynamic_features': dyn_tensor,
            'target_forecast': target_tensor,
            'structural_masks': struct_masks,
        }

def make_spatial_holdout_mask(num_nodes: int,
                              holdout_prob: float,
                              device: str = 'cpu',
                              seed: Optional[int] = None) -> torch.Tensor:
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

    return dynamic_features.masked_fill(holdout_mask.view(1, -1, 1, 1), 0.0)


def custom_collate_real(batch_list):
    """Collate function to maintain node structural dimensions.
    Note: target_impute and node_mask are NOT returned here — they are generated
    on-device in the train/val loops after .to(device)."""
    return {
        'dynamic_features': torch.stack([b['dynamic_features'] for b in batch_list]),
        'target_forecast': torch.stack([b['target_forecast'] for b in batch_list]),
        'structural_masks': torch.stack([b['structural_masks'] for b in batch_list]),
    }
