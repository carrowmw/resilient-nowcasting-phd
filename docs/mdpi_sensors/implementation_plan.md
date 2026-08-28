# Add Transformer Model to Quality Experiment Pipeline

## Goal

Integrate Transformer-based forecasters into the existing quality-aware per-sensor experiment pipeline so that re-running the V2 experiment on the server will skip all already-completed jobs and only train the new Transformer models. This enables a direct comparison of how a Transformer responds to sequence fragmentation versus the existing LSTM, and sets the stage for Experiment 3 (missingness-aware architectures).

## Decisions (Resolved)

### Architecture: Both Options A and B
- **Option A (this experiment):** Standard Transformer encoder with learned positional encoding. Same `WindowDataset` as the LSTM (contiguous windows only). The Transformer sees the same data as the LSTM, so any performance difference is purely architectural. This is the fairest comparison for the paper.
- **Option B (bridge to Experiment 3):** Missingness-aware Transformer with a custom dataset that feeds windows *with* NaN gaps, using a missingness mask and time-delta encoding. This directly tests whether a Transformer can learn *through* gaps. This serves as the natural transition into the next thesis experiment.

> [!NOTE]
> Option B requires a new dataset class (`MaskedWindowDataset`) and a modified architecture. It will be built alongside Option A but can be scoped to `none`-only interpolation if compute is tight, since the entire point is that it doesn't need interpolation.

### Model Variants: Both UV and MV
All four variants will be trained:
- `transformer_uv` — univariate (count only)
- `transformer_mv` — multivariate (count + all temporal features)
- `transformer_masked_uv` — missingness-aware univariate (Option B)
- `transformer_masked_mv` — missingness-aware multivariate (Option B)

### Interpolation Variants: Full Grid
All 5 interpolation variants × 2 horizons for the standard Transformer (Options A), keeping the experiment grid complete. The masked Transformer (Option B) runs on `none` only (its raison d'être is avoiding interpolation).

### New Job Count
| Model | Variants | Horizons | Total Jobs |
|---|---|---|---|
| `transformer_uv` | 5 interp | 2 | 10 |
| `transformer_mv` | 5 interp | 2 | 10 |
| `transformer_masked_uv` | 1 (none) | 2 | 2 |
| `transformer_masked_mv` | 1 (none) | 2 | 2 |
| **Total new jobs** | | | **24** |

Each job runs across 110 sensors. Estimated compute: well within 48 hours on the GPU server.

### Registry Strategy: Migrate
After updating the config, a migration script will remap all 84 existing job keys from the old SHA to the new SHA. This preserves the runner's idempotency design.

### Execution: Server Only
> [!CAUTION]
> The experiment runner and aggregation script must be executed on the GPU server, **not locally**. The implementation steps below cover only code creation and config changes.

---

## Proposed Changes

### Standard Transformer Architecture (Option A)

The model will be a standard Transformer encoder with:
- **Learned positional encoding** (not sinusoidal — the window size is fixed at 24, so learnable is appropriate)
- **Multi-head self-attention** over the input window
- **Mean pooling** of the encoder output → linear head → scalar forecast
- Same input interface as the LSTM: `[B, T, D]` feature tensor from `WindowDataset`

Recommended hyperparameters (comparable parameter count to LSTM-MV):
- `d_model`: 64 (matches LSTM hidden_dim)
- `nhead`: 4
- `num_encoder_layers`: 2 (matches LSTM num_layers)
- `dim_feedforward`: 128
- `dropout`: 0.1
- Same optimizer, scheduler, early stopping as LSTM

### Missingness-Aware Transformer Architecture (Option B)

Extends the standard Transformer with:
- **Time-delta encoding**: instead of (or in addition to) learned positional encoding, encode the actual time gap between consecutive observations
- **Missingness mask**: a binary feature indicating whether each timestep is observed or missing
- **Custom dataset** (`MaskedWindowDataset`): windows are extracted from the raw grid *including* NaN gaps; NaN values are replaced with 0.0 and the missingness mask is set to True
- The attention mechanism naturally handles variable-length context through the mask

---

### Source Code Changes

#### [NEW] [`src/quality/transformer_forecaster.py`](file:///Users/administrator/Code/ml-agent-workspace/src/quality/transformer_forecaster.py)

New file containing:

1. **`PerSensorTransformer(nn.Module)`** — Standard Transformer (Option A):
   - Input projection: `nn.Linear(input_dim, d_model)`
   - Learned positional encoding: `nn.Embedding(max_seq_len, d_model)`
   - `nn.TransformerEncoder` with `nn.TransformerEncoderLayer`
   - Mean pooling → `nn.Linear(d_model, 1)` head
   - Forward: `[B, T, D]` → project → add positional → encode → pool → scalar

2. **`PerSensorMaskedTransformer(nn.Module)`** — Missingness-aware (Option B):
   - Extends the standard architecture with time-delta input and missingness mask
   - Input projection includes missingness indicator as an extra feature dimension
   - Uses `src_key_padding_mask` to let attention skip truly missing positions

3. **`TransformerSensorForecaster(SensorForecaster)`** — Training wrapper (Option A):
   - Identical interface to `LSTMSensorForecaster`
   - Same `WindowDataset`, same `_loader()`, same `fit()`/`predict()` pattern
   - Same Adam + MSE + StepLR + early stopping logic

4. **`MaskedTransformerSensorForecaster(SensorForecaster)`** — Training wrapper (Option B):
   - Uses `MaskedWindowDataset` instead of `WindowDataset`
   - Otherwise identical training loop

---

#### [NEW] [`src/quality/masked_dataset.py`](file:///Users/administrator/Code/ml-agent-workspace/src/quality/masked_dataset.py)

New dataset class for Option B:

- **`MaskedWindowDataset(Dataset)`**: extracts windows from the full grid (including NaN gaps)
  - Returns `(features, mask, target)` tuples where:
    - `features`: `[T, D+1]` — original features with NaN replaced by 0.0, plus a missingness indicator column
    - `mask`: `[T]` bool — True where the timestep is missing (for `src_key_padding_mask`)
    - `target`: scalar (must still be non-NaN — windows where the target is NaN are excluded)

---

#### [MODIFY] [`src/quality/registry.py`](file:///Users/administrator/Code/ml-agent-workspace/src/quality/registry.py)

- Add `"transformer"` and `"transformer_masked"` to `KNOWN_TYPES`
- Import `TransformerSensorForecaster` and `MaskedTransformerSensorForecaster`
- Add `elif mtype == "transformer":` and `elif mtype == "transformer_masked":` branches

---

#### [MODIFY] [`configs/quality/quality_full_v2.yaml`](file:///Users/administrator/Code/ml-agent-workspace/configs/quality/quality_full_v2.yaml)

Add model specs under `models:`:

```yaml
  transformer_uv:
    type: transformer
    feature_groups: [count]
    d_model: 64
    nhead: 4
    num_encoder_layers: 2
    dim_feedforward: 128
    dropout: 0.1
    lr: 0.001
    batch_size: 128
    epochs: 30
    patience: 5
    scheduler_step_size: 5
    scheduler_gamma: 0.7
  transformer_mv:
    type: transformer
    feature_groups: [count, half_day, day, week, quarter, year, timestamp]
    d_model: 64
    nhead: 4
    num_encoder_layers: 2
    dim_feedforward: 128
    dropout: 0.1
    lr: 0.0005
    batch_size: 128
    epochs: 30
    patience: 5
    scheduler_step_size: 5
    scheduler_gamma: 0.7
  transformer_masked_uv:
    type: transformer_masked
    feature_groups: [count]
    d_model: 64
    nhead: 4
    num_encoder_layers: 2
    dim_feedforward: 128
    dropout: 0.1
    lr: 0.001
    batch_size: 128
    epochs: 30
    patience: 5
    scheduler_step_size: 5
    scheduler_gamma: 0.7
  transformer_masked_mv:
    type: transformer_masked
    feature_groups: [count, half_day, day, week, quarter, year, timestamp]
    d_model: 64
    nhead: 4
    num_encoder_layers: 2
    dim_feedforward: 128
    dropout: 0.1
    lr: 0.0005
    batch_size: 128
    epochs: 30
    patience: 5
    scheduler_step_size: 5
    scheduler_gamma: 0.7
```

Update `run.models`:

```yaml
  models: [persistence, snaive_daily, snaive_weekly, arima, lstm_uv, lstm_mv,
           transformer_uv, transformer_mv, transformer_masked_uv, transformer_masked_mv]
```

> [!WARNING]
> The masked Transformer models should only run on `none` interpolation. The runner currently runs every model × every interpolation variant. We need to either:
> 1. Add a `interpolation_variants` override per model in the config, or
> 2. Handle it in the runner by skipping masked models on non-`none` variants.
>
> I recommend option 2 — a simple check in the job matrix loop.

---

#### [MODIFY] [`scripts/run_quality_experiments.py`](file:///Users/administrator/Code/ml-agent-workspace/scripts/run_quality_experiments.py)

Add a guard in the job matrix loop (~line 312-318) to skip `transformer_masked_*` models on non-`none` interpolation variants:

```python
for method in variants:
    for model_key in run_cfg["models"]:
        # Masked transformers only run on 'none' — they handle gaps natively
        if model_key.startswith("transformer_masked") and method != "none":
            continue
        ...
```

---

#### [NEW] [`scripts/migrate_registry.py`](file:///Users/administrator/Code/ml-agent-workspace/scripts/migrate_registry.py)

A small one-shot script that:
1. Reads the current `registry.json` from `artifacts/quality_full_v2/`
2. Computes the new `config_sha` from the updated config
3. Remaps all existing job keys from the old SHA (`b49750597c89f06d`) to the new SHA
4. Writes the updated registry

---

## Execution Sequence

1. ✏️ Create `src/quality/transformer_forecaster.py` (both architectures)
2. ✏️ Create `src/quality/masked_dataset.py` (Option B dataset)
3. ✏️ Update `src/quality/registry.py` to register both new types
4. ✏️ Update `configs/quality/quality_full_v2.yaml` with model specs and run list
5. ✏️ Update `scripts/run_quality_experiments.py` with masked model guard
6. ✏️ Create `scripts/migrate_registry.py`
7. 🖥️ **On server:** Run `python scripts/migrate_registry.py --config configs/quality/quality_full_v2.yaml`
8. 🖥️ **On server:** Run `python scripts/run_quality_experiments.py --config configs/quality/quality_full_v2.yaml`
9. 🖥️ **On server:** Run `python scripts/aggregate_quality_results.py`

Steps 1–6 are local code changes. Steps 7–9 are server-only execution.

## Verification Plan

### Automated Tests
- Smoke test with `quality_smoke.yaml` on the server before the full run
- Verify registry migration preserves all 84 existing job keys

### Manual Verification
- After server run completes, compare Transformer vs LSTM-MV median RMSE at both horizons
- Compare masked vs standard Transformer to assess whether native gap handling helps
- Verify all 24 new jobs appear in the aggregated results CSV
