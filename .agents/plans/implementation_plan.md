# Configure Sweeping for Masking Ratio, Sparsity Threshold, and Model Depth

The goal of this task is to expose the missing sparsity threshold (`sparsity_threshold`), the masking ratio (`mask_prob`), and the temporal transformer depth (`num_layers`) so they can be comprehensively tuned using WandB sweeps, along with other architectural and training hyperparameters.

## Background Context

Currently, `sparsity_threshold` is configured in `configs/train.yaml` under `model.sparsity_threshold` but missing from `sweep.yaml`.
The masking ratio is hardcoded as `0.15` in `src/train.py` (within `create_mock_dataloader`) and defaults to `0.15` in `src/dataset.py`.

## Proposed Changes

### 1. Identify Tunable Parameters

For a large, multi-day comprehensive tuning experiment, I recommend sweeping the following parameters in addition to the ones already in `sweep.yaml` (`lr`, `w_forecast`, `w_impute`):

- **Data/Augmentation**:
  - `mask_prob` (Masking Ratio): `0.05` to `0.30`
- **Model Architecture**:
  - `num_layers` (Temporal Transformer Depth): Discrete choices (e.g., `1, 2, 4, 6`)
  - `sparsity_threshold`: `0.1` to `0.9`
  - `embedding_dim`: Discrete choices (e.g., `16, 32, 64`)
  - `hidden_dim`: Discrete choices (e.g., `32, 64, 128`)
- **Training**:
  - `batch_size`: Discrete choices (e.g., `64, 128, 256`)

### 2. File Modifications

#### [MODIFY] `configs/train.yaml`

- Add `mask_prob: 0.15` under the `data:` section to establish a clear default.
- Add `num_layers: 2` under the `model:` section to establish the baseline temporal depth.

#### [MODIFY] `configs/sweep.yaml`

- Add `sparsity_threshold` (e.g., uniform `0.1` to `0.9`).
- Add `mask_prob` (e.g., uniform `0.05` to `0.3`).
- Add `num_layers` (categorical: `[1, 2, 4, 6]`).
- Add `embedding_dim` (categorical: `[16, 32, 64]`).
- Add `hidden_dim` (categorical: `[32, 64, 128]`).
- Add `batch_size` (categorical: `[64, 128, 256]`).

#### [MODIFY] `src/train.py`

- Modify `create_mock_dataloader` to accept `mask_prob: float` as an argument instead of hardcoding `0.15` on line 69.
- Pass `mask_prob=data_cfg.get("mask_prob", 0.15)` into `create_mock_dataloader` when it is called (lines 124-139).
- Pass `mask_prob=data_cfg.get("mask_prob", 0.15)` when initializing `RealIoTDataset` (line 144).
- Update the instantiation of `ResilientIoTPipeline` to read and pass `num_layers=model_cfg.get("num_layers", 2)`.

#### [MODIFY] `src/pipeline.py` (Optional, for consistency)

- Update `ResilientIoTPipeline.__init__` to accept `num_layers: int = 2`.
- Pass `num_layers=num_layers` when instantiating `MaskedSpatiotemporalEncoder`.
- (Optional) Update the hardcoded `0.15` masking logic in `pipeline.py` line 162 used for analysis mode to rely on a passed parameter.

## User Review Required

> [!IMPORTANT]
> Sweeping `batch_size`, `embedding_dim`, and `hidden_dim` drastically increases the search space. Are you comfortable with these discrete choices? Let me know if you want to include `seq_len` as well, though changing `seq_len` might require more careful testing around tensor dimensions.

## Verification Plan

### Automated Tests

- Run `python src/train.py --use_mock_data` to ensure the new configuration parameters don't break standard training.
- Check `wandb` initialization to confirm the nested config correctly captures `mask_prob` and `sparsity_threshold`.

### Implementer Directives

> [!IMPORTANT]
> The implementing agent must adhere to the following directives:
>
> 1. **Desk Check Mandate**: After wiring up `mask_prob`, `sparsity_threshold`, and `num_layers`, execute a quick simulated run (or read through the parameter flow carefully) to ensure that the configurations from the YAML are indeed successfully reaching the instantiation of the models and the dataloaders without being overwritten or defaulting.
> 2. **Update the Ledger**: Once the logic changes are verified, you MUST append a new row to `EXPERIMENTS.md` recording the current date, the active Git branch name, the task initiated (exposing mask_prob and sparsity_threshold for sweeps), and the functional outcome of the update. Do not report completion until this is done.
