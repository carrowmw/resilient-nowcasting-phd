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

# England & Wales bank holidays covering the dataset span (gov.uk/bank-holidays).
ENGLAND_BANK_HOLIDAYS = pd.to_datetime([
    '2023-01-02', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-08', '2023-05-29',
    '2023-08-28', '2023-12-25', '2023-12-26',
    '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-06', '2024-05-27', '2024-08-26',
    '2024-12-25', '2024-12-26',
    '2025-01-01', '2025-04-18', '2025-04-21', '2025-05-05', '2025-05-26', '2025-08-25',
    '2025-12-25', '2025-12-26',
    '2026-01-01', '2026-04-03', '2026-04-06', '2026-05-04', '2026-05-25', '2026-08-31',
    '2026-12-25', '2026-12-28',
]).values

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
                 ablate_masks: bool = False,
                 log1p_inputs: bool = False,
                 complete_weather: bool = False):

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

        # v5 (R1 / C12): un-ablated observation mask. Loss and metric masks are always
        # derived from this copy, so ablating the *input* mask channels can never
        # change which targets are scored.
        self.observed = torch.from_numpy(total_mnar_mask == 0.0)  # [num_times, num_nodes] bool
        self.is_night = torch.from_numpy(is_night_arr[:, 0].copy())  # [num_times] bool
        
        logger.info(f"MNAR Masks Generated. Total missing: {(total_mnar_mask.mean()*100):.2f}%. Night missing: {(night_mnar_mask.mean()*100):.2f}%")
        
        # Recorded so run_training can verify a pre-built dataset matches the run config.
        self.ablate_masks = ablate_masks
        self.complete_weather = complete_weather
        if ablate_masks:
            self.structural_masks = torch.zeros_like(self.structural_masks)
            logger.info("ABLATION: Explicit MNAR masks zeroed out.")

        # v5 (R2): weather is joined onto *traffic rows*, so every missing traffic slot
        # also had NaN -> 0.0 weather, i.e. the weather channels leaked missingness.
        # complete_weather fills those slots from the weather source itself (weather
        # is genuinely available during sensor outages), nearest-in-time otherwise.
        if complete_weather:
            weather_grid = weather_df.drop_duplicates(subset=['dt', 'location']).set_index(['dt', 'location'])

            def _complete(frame, col_name):
                full = weather_grid[col_name].unstack('location').reindex(index=frame.index, columns=locations_idx)
                filled = frame.fillna(full).ffill().bfill()
                assert not filled.isna().any().any(), \
                    f"CRITICAL: complete_weather left NaN in {col_name} (a sensor has no weather at all)"
                return filled

            hum_df = _complete(hum_df, 'Humidity')
            sol_df = _complete(sol_df, 'Solar Radiation')
            logger.info("complete_weather: weather at missing-traffic slots filled from the weather source")

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

        # v5 (R3): optional log1p before standardisation. The scaler then lives in
        # log1p space and inverse_traffic() applies expm1 after de-standardising.
        self.log1p_inputs = log1p_inputs
        traffic_domain = np.log1p(val_arr) if log1p_inputs else val_arr

        self.traffic_scaler = StandardScaler()
        self.traffic_scaler.fit(np.log1p(observed_counts) if log1p_inputs else observed_counts)

        traffic_flat = traffic_domain.reshape(-1, 1)  # [T*N, 1]
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
        # Scaled value of a zero-filled weather reading: how a real traffic gap looks in
        # the weather channels when complete_weather is off (synthetic outages copy it).
        self.weather_zero_scaled = torch.tensor(
            self.weather_scaler.transform(np.zeros((1, 2)))[0], dtype=torch.float32
        )
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

        self._build_v5_tables(val_arr, total_mnar_mask == 0.0, train_t_end)

    # ------------------------------------------------------------------ #
    # v5 auxiliary tables (device-agnostic; moved to GPU by CovariateBuilder) #
    # ------------------------------------------------------------------ #
    def _build_v5_tables(self, counts: np.ndarray, observed: np.ndarray, train_t_end: int) -> None:
        """Timeline tables for calendar/profile/run-length covariates, evaluation
        strata and the structured-masking banks. Everything that is *fitted*
        (profile, quantiles, banks, per-node statistics) uses the training slice
        [0, train_t_end) only — the F-C02 pattern."""
        times = pd.DatetimeIndex(self.times)
        num_times, num_nodes = counts.shape
        self.train_t_end = train_t_end

        steps = np.diff(times.values).astype("timedelta64[s]").astype(np.int64)  # resolution-agnostic
        self.regular_grid = bool(len(steps) == 0 or (steps == steps[0]).all())
        self.step_minutes = int(steps[0] // 60) if len(steps) else 15
        assert self.step_minutes > 0, f"CRITICAL: sub-minute timestep ({steps[0]}s) unsupported"

        # Calendar [T, 6]: sin/cos hour-of-day, sin/cos day-of-week, weekend, bank holiday.
        hour_frac = times.hour.values + times.minute.values / 60.0
        dow = times.dayofweek.values
        bank_holiday = np.isin(times.normalize().values, ENGLAND_BANK_HOLIDAYS)
        self.calendar = torch.from_numpy(np.stack([
            np.sin(2 * np.pi * hour_frac / 24.0),
            np.cos(2 * np.pi * hour_frac / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            (dow >= 5).astype(np.float64),
            bank_holiday.astype(np.float64),
        ], axis=-1).astype(np.float32))
        self.hour_of_day = torch.from_numpy(times.hour.values.astype(np.int64))
        slots_per_hour = max(1, 60 // self.step_minutes)
        slot_of_day = times.hour.values * slots_per_hour + times.minute.values // self.step_minutes
        self.slots_per_week = 7 * 24 * slots_per_hour
        self.slot_of_week = torch.from_numpy((dow * 24 * slots_per_hour + slot_of_day).astype(np.int64))

        # Run-length (steps since last observation; 0 when observed) — causal.
        t_idx = np.arange(num_times)[:, None]
        last_obs = np.maximum.accumulate(np.where(observed, t_idx, -1), axis=0)
        self.run_length = torch.from_numpy((t_idx - last_obs).astype(np.int64))  # [T, N]

        # Total length of the missing run containing t (0 when observed) — NON-causal,
        # evaluation strata only (e.g. night_short_gap_zero_rate). Never a model input.
        gap_len = np.zeros((num_times, num_nodes), dtype=np.int64)
        gap_bank = []
        for n in range(num_nodes):
            miss = ~observed[:, n]
            if not miss.any():
                continue
            edges = np.diff(np.concatenate([[0], miss.astype(np.int8), [0]]))
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
            for a, b in zip(starts, ends):
                gap_len[a:b, n] = b - a
                if b <= train_t_end:
                    gap_bank.append(b - a)
        self.gap_length = torch.from_numpy(gap_len)
        assert len(gap_bank) > 0, "CRITICAL: no missing runs in the training slice to build the gap bank"
        self.gap_length_bank = torch.tensor(gap_bank, dtype=torch.int64)

        # Transplant bank: real missingness patterns over (seq_len + 1)-step windows
        # (input window + target step) from the training slice, CSDI/SAITS style.
        rng = np.random.RandomState(0)
        span = self.seq_len + 1
        n_draw = 20000
        t0 = rng.randint(0, max(1, train_t_end - span), size=n_draw)
        nodes = rng.randint(0, num_nodes, size=n_draw)
        patterns = ~observed[t0[:, None] + np.arange(span)[None, :], nodes[:, None]]
        patterns = patterns[patterns.any(axis=1)]
        assert len(patterns) > 0, "CRITICAL: transplant bank is empty (no missingness in training slice)"
        self.transplant_bank = torch.from_numpy(patterns)  # [P, seq_len + 1] bool, True = missing

        # Train-only count statistics on observed slots.
        train_obs = observed[:train_t_end]
        train_counts = counts[:train_t_end]
        obs_values = train_counts[train_obs]
        self.count_deciles = torch.tensor(
            np.quantile(obs_values, np.linspace(0.1, 0.9, 9)), dtype=torch.float32
        )
        self.count_p90 = float(np.quantile(obs_values, 0.9))
        # Raw-count std of observed training targets: scales so_rmse into the
        # selection score (selection.metric = obs_loss_plus_so).
        self.count_std_raw = float(obs_values.std())
        assert self.count_std_raw > 0, "CRITICAL: observed training counts are constant"
        node_counts = train_obs.sum(axis=0)
        node_sums = np.where(train_obs, train_counts, 0.0).sum(axis=0)
        global_mean = float(obs_values.mean())
        self.node_train_mean = torch.tensor(
            np.where(node_counts > 0, node_sums / np.maximum(node_counts, 1), global_mean),
            dtype=torch.float32,
        )
        self.node_missing_rate = torch.tensor(1.0 - train_obs.mean(axis=0), dtype=torch.float32)

        # Seasonal profile [slots_per_week, N] in log1p space: per (slot-of-week,
        # node) median of observed training counts; fallback chain for sparse cells
        # (< 3 observations): node hour-of-day median -> node median -> global median.
        log_counts = np.log1p(train_counts)
        sow_train = self.slot_of_week.numpy()[:train_t_end]
        hour_train = times.hour.values[:train_t_end]
        global_median = float(np.median(np.log1p(obs_values)))
        profile = np.full((self.slots_per_week, num_nodes), np.nan)
        for n in range(num_nodes):
            obs_n = train_obs[:, n]
            if obs_n.sum() == 0:
                profile[:, n] = global_median
                continue
            frame = pd.DataFrame({"sow": sow_train[obs_n], "hour": hour_train[obs_n], "v": log_counts[obs_n, n]})
            by_slot = frame.groupby("sow")["v"].agg(["median", "count"])
            by_slot = by_slot[by_slot["count"] >= 3]["median"]
            by_hour = frame.groupby("hour")["v"].median()
            node_median = float(frame["v"].median())
            slot_hours = (np.arange(self.slots_per_week) % (24 * slots_per_hour)) // slots_per_hour
            col = by_hour.reindex(slot_hours).values
            col = np.where(np.isnan(col), node_median, col)
            col[by_slot.index.values] = by_slot.values
            profile[:, n] = col
        assert not np.isnan(profile).any(), "CRITICAL: NaN in seasonal profile table"
        self.profile_log1p = torch.tensor(profile, dtype=torch.float32)  # [slots_per_week, N]

        logger.info(
            f"v5 tables: regular_grid={self.regular_grid} ({self.step_minutes}-min), "
            f"gap bank={len(gap_bank)} runs (median {int(np.median(gap_bank))} steps), "
            f"transplant bank={len(patterns)} patterns, train p90={self.count_p90:.1f}"
        )

    def inverse_traffic(self, scaled):
        """Scaled traffic channel -> raw counts (numpy or torch), honouring log1p_inputs."""
        raw = scaled * self.traffic_std + self.traffic_mean
        if self.log1p_inputs:
            raw = np.expm1(raw) if isinstance(raw, np.ndarray) else torch.expm1(raw)
        return raw

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
        # v5: un-ablated observation masks (loss/metric routing) and the absolute
        # window start (covariate / evaluation-stratum lookup on device).
        return {
            'dynamic_features': dyn_tensor,
            'target_forecast': target_tensor,
            'structural_masks': struct_masks,
            'input_observed': self.observed[idx : idx + self.seq_len].T.contiguous(),
            'target_observed': self.observed[idx + self.seq_len],
            'time_idx': torch.tensor(idx, dtype=torch.int64),
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
        key: torch.stack([b[key] for b in batch_list])
        for key in ('dynamic_features', 'target_forecast', 'structural_masks',
                    'input_observed', 'target_observed', 'time_idx')
    }
