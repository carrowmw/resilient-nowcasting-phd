# Detailed Implementation Plan: RIoT v4 & Full Ablation Suite

## Proposed Changes

### 1. Git Branching & Safety Pre-flight
- **Branch:** Create and switch to new branch `feat/riot-v4`.
- **Pre-flight Commit:** Stage and commit any outstanding changes on the previous branch before branching.
- **Rollback point:** Tag previous state as `pre-v4-checkpoint`.

---

### 2. Data Ingestion & Preprocessing (`src/dataset.py`)
#### Proposed Edits:
1. **Fix Direction Aggregation Bug:**
   - In `target_dir == 'all'` aggregation block, change:
     ```python
     agg_funcs = {
         'value': lambda x: np.nan if x.isna().all() else x.sum(min_count=1),
         ...
     }
     ```
   - If all directional channels at a station are `NaN`, the sum is `NaN`.
   - If partial directions report, sum valid directions and flag partial reporting if configured.
2. **True Missingness Mask Generation:**
   - `total_mnar_mask = val_df.isna().values.astype(np.float32)` now captures all true 36.3% nulls instead of 5.5%.
   - Night mask correctly isolates 00:00–05:00 nulls on genuine missing values.
3. **Train-Only Scaler Isolation:**
   - Add parameter `train_cutoff_date: Optional[str] = None` to `RealIoTDataset.__init__`.
   - Split dynamic dataframe into `train_slice = dyn_df.loc[:train_cutoff_date]` before fitting.
   - `traffic_scaler.fit(train_slice['value'].values[~train_mask])` (fit only on observed non-null training counts).
   - Fit `weather_scaler` on `train_slice[['Humidity', 'Solar Radiation']]`.
   - Apply `transform` to the entire timeline using the train-fitted statistics.

---

### 3. Chronological Purged Split (`src/train.py`)
#### Proposed Edits:
1. **Remove `random_split`:**
   - Eliminate `torch.utils.data.random_split(full_dataset, [train_size, val_size])`.
2. **Implement Chronological Sliding-Window Partitions:**
   - Calculate chronological cutoff indices:
     - `train_end_idx = int(train_split * len(full_dataset))`
     - `purge_steps = seq_len + 1` (13 steps = 3.25 hours)
     - `val_start_idx = train_end_idx + purge_steps`
     - `val_end_idx = val_start_idx + int(val_split * len(full_dataset))`
     - `test_start_idx = val_end_idx + purge_steps`
   - Create explicit subset datasets using contiguous index ranges:
     - `train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_end_idx))`
     - `val_dataset = torch.utils.data.Subset(full_dataset, range(val_start_idx, val_end_idx))`
     - `test_dataset = torch.utils.data.Subset(full_dataset, range(test_start_idx, len(full_dataset)))`
   - Propagate scaler and metadata attributes to all subsets.
3. **Evaluation Loop Logging:**
   - Compute and log both point forecast metrics:
     - `rmse_zinb = sqrt(mean((y - mu * (1 - pi))**2))`
     - `mae_zinb = mean(|y - mu * (1 - pi)|)`
     - `rmse_mu = sqrt(mean((y - mu)**2))` (active-state conditional mean)

---

### 4. Architecture & Temporal Positional Encoding (`src/encoder.py`)
#### Proposed Edits:
1. **Implement `SinusoidalPositionalEncoding` Module:**
   ```python
   class SinusoidalPositionalEncoding(nn.Module):
       def __init__(self, hidden_dim: int, max_len: int = 100):
           super().__init__()
           pe = torch.zeros(max_len, hidden_dim)
           position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
           div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
           pe[:, 0::2] = torch.sin(position * div_term)
           pe[:, 1::2] = torch.cos(position * div_term)
           self.register_buffer('pe', pe) # [max_len, hidden_dim]

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # x: [T, B * N, H]
           seq_len = x.size(0)
           return x + self.pe[:seq_len].unsqueeze(1)
   ```
2. **Integrate into `UnifiedSpatioTemporalEncoder`:**
   - Instantiate `self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=32)`.
   - Apply `x_seq = self.pos_encoder(x_seq)` immediately after GATv2 aggregation and reshaping to `[T, B * N, H]`, prior to `self.temporal_transformer(x_seq)`.
   - Gate positional encoding with a boolean flag `use_positional_encoding: bool = True` in config to enable explicit ablation testing (Ablation D).

---

### 5. Topology Masking & Kriging Protection (`src/pipeline.py`)
#### Proposed Edits:
1. **Extend Outgoing Edge Dropping:**
   - In `forward()`, modify edge filtering:
     ```python
     drop_mask = None
     if 'simulated_nodes_mask' in batch:
         drop_mask = batch['simulated_nodes_mask']
     if self.training and 'spatial_holdout_mask' in batch:
         sh_mask = batch['spatial_holdout_mask']
         drop_mask = sh_mask if drop_mask is None else (drop_mask | sh_mask)
     
     if drop_mask is not None:
         sources = edge_index[0]
         valid_edges = ~drop_mask[sources]
         edge_index = edge_index[:, valid_edges]
         edge_weight = edge_weight[valid_edges]
         edge_attr = edge_attr[valid_edges]
     ```
2. **Sanitize Kriging Dynamic Input Channels:**
   - For nodes in `spatial_holdout_mask`, ensure missingness channels (Channel 3 & 4) do not leak ground-truth outage status to the kriging decoder.

---

### 6. Metric Evaluation & UI Export (`src/export_ui_payload.py` & `src/harness/evaluate.py`)
#### Proposed Edits:
1. **`evaluate.py` Point Metrics:**
   - Replace `target_forecast - preds.mu` with `target_forecast - (preds.mu * (1.0 - preds.pi))` as the primary point evaluation.
   - Retain `preds.mu` error as `rmse_active_state` for secondary diagnosis.
2. **`export_ui_payload.py` Calibration & Residuals:**
   - Evaluate `arr_fill_p = np.array(mu_predictions) * (1.0 - np.array(pi_predictions))` for residual calculations.
   - Update zero-reliability calculation:
     $$\hat{P}(Y=0) = \hat{\pi} + (1 - \hat{\pi})\left(\frac{\hat{\theta}}{\hat{\theta} + \hat{\mu}}\right)^{\hat{\theta}}$$
   - Compare $\hat{P}(Y=0)$ against observed zero frequency across confidence bins.

---

### 7. Version 4 Configuration Suite (`configs/train/v4/`)
Create modular configuration files for the full model and all ablation variants:
1. `train_v4_full_inductive.yaml`: Full RIoT v4 model with all audit remediations enabled.
2. `train_v4_ablation_no_impute.yaml`: Set `loss.w_impute: 0.0`.
3. `train_v4_ablation_no_kriging.yaml`: Set `loss.w_kriging: 0.0`.
4. `train_v4_ablation_no_masks.yaml`: Set `data.ablate_masks: true` (zero out Channel 3 & 4).
5. `train_v4_ablation_no_pe.yaml`: Set `model.use_positional_encoding: false`.
6. `train_v4_baseline_lstm.yaml`: Inductive non-graph LSTM comparator.
7. `train_v4_spatial_small_batch.yaml`: Batch size = 32 for spatial integration tuning.
8. `train_v4_spatial_high_dropout.yaml`: Dropout = 0.3 for regularization sweep.

All configs will target:
- `wandb.project: "resilient-iot-v4"`
- `wandb.collection: "resilient-iot-model-v4"`

---

### 8. Multi-GPU Automated Server Execution Suite (`scripts/run_v4_server_suite.py`)
#### Specifications:
- **Dual-GPU Orchestrator:**
  - Manages concurrent or sequential execution across `cuda:0` and `cuda:1`.
  - Queue system assigning jobs with `CUDA_VISIBLE_DEVICES` affinity.
  - Automatically isolates memory and monitors GPU temperatures / memory utilization.
- **Fault-Tolerant Execution:**
  - Saves individual run logs to `logs/v4_suite/<job_name>.log`.
  - Catches process exit codes; on failure, captures stack trace and continues remaining suite without halting.
- **Post-Run Automatic Evaluation & Export:**
  - Automatically invokes `export_ui_payload.py` upon completion of each model training.
  - Evaluates on the held-out test partition.
  - Aggregates final metrics into `artifacts/v4_ablation_results.csv` and renders a formatted LaTeX table ready to paste directly into Chapter 7 Table 7.3.

---

## Desk Checks & Mathematical Verification
1. **Desk Check 1 (MNAR Aggregation):**
   - Create mock DataFrame with 2 locations and 2 directions.
   - Location 1: Direction A = NaN, Direction B = NaN. Verify sum = NaN, mask = 1.0.
   - Location 2: Direction A = 5.0, Direction B = NaN. Verify sum = 5.0, mask = 0.0.
2. **Desk Check 2 (Chronological Purge Boundary):**
   - Verify index bounds: `max(train_indices) + 13 <= min(val_indices)` and `max(val_indices) + 13 <= min(test_indices)`.
   - Verify no timestamp overlap exists across subsets.
3. **Desk Check 3 (Positional Encoding Tensor Shapes):**
   - Feed tensor of shape $[T=12, B\cdot N=128\cdot 81, H=64]$. Verify output shape is identical $[12, 10368, 64]$ and values are non-zero.
4. **Desk Check 4 (ZINB Expected Value Limit Cases):**
   - Case A: $\mu = 100, \pi = 1.0 \implies \hat{y} = 0.0$.
   - Case B: $\mu = 100, \pi = 0.0 \implies \hat{y} = 100.0$.
   - Case C: $\mu = 100, \pi = 0.2 \implies \hat{y} = 80.0$.

---

## Verification & Approval Gate
- **Pre-execution requirement:** This implementation plan must be reviewed and approved by the USER before functional execution code is applied.
- **Git checkpoint:** All changes will be developed on branch `feat/riot-v4`.
