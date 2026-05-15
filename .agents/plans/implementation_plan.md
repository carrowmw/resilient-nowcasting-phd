# Implementation Plan: Interactive Study Namespacing & Metadata Stamping

This plan outlines the steps to upgrade the tuning framework to support multiple independent experiments through persistent SQLite namespacing, an interactive selection CLI, and enhanced traceability.

## 1. Environment Protection

- **File**: `.gitignore`
- **Action**: Add patterns to ignore SQLite database files and journals.
- **Content to Add**:
  ```text
  # Optuna Databases
  *.db
  *.db-journal
  *.db-wal
  ```

## 2. Parameter Externalization & Configuration Strategy

- **File**: `configs/train.yaml`
- **Action**: Add new hyperparameters to the YAML configuration.
- **New Parameters**:
  - `model.num_heads`: Number of attention heads (default: 4)
  - `model.dropout`: Dropout rate (default: 0.1)
  - `model.dim_feedforward`: Transformer feedforward dimension (default: 256)
  - `model.zinb_eps`: Epsilon for ZINB stability (default: 1e-6)
  - `training.grad_clip_max_norm`: Gradient clipping threshold (default: 1.0)
  - `training.random_seed`: Seed for reproducibility (default: 42)
  - `training.early_stop_patience`: Epochs to wait for improvement (default: 5)
  - `training.early_stop_min_delta`: Minimum change to qualify as improvement (default: 0.001)

## 3. Core Training Upgrades

### 3.1. Intra-Trial Early Stopping & Parameter Injection
- **File**: `src/train.py`
- **Action**:
  - Update `run_training` to use the new parameters from the config (seed, grad_clip, etc.).
  - Implement an early stopping loop within the epoch loop.
  - Terminate training if `val_loss` does not improve by `early_stop_min_delta` for `early_stop_patience` epochs.

### 3.2. Model Architecture Support
- **Files**: `src/encoder.py`, `src/decoders.py`, `src/pipeline.py`
- **Action**:
  - Update `MaskedSpatiotemporalEncoder` to accept `dim_feedforward`.
  - Update `DecoupledDecoders` to accept `dropout` and `zinb_eps`.
  - Ensure `ResilientIoTPipeline` propagates these parameters to the sub-modules.

## 4. Interactive Sweep Launcher

- **File**: `src/tune.py`
- **Action**:
  - Add CLI arguments: `--study-name`, `--storage`.
  - **Interactive Discovery**:
    - If `--storage` is missing, scan the current directory for `*.db` files and prompt the user.
    - If `--study-name` is missing, list existing studies in the storage for selection or creation.
  - **Metadata Stamping**:
    - Retrieve the current Git commit hash and branch name.
    - Inside the `objective` function, use `trial.set_user_attr` to store `git_commit` and `branch`.
  - **W&B Integration**: Update `WeightsAndBiasesCallback` to use the study name.

## 5. High-Resolution Analysis

- **File**: `src/analyze_sweep.py`
- **Action**:
  - Add CLI arguments: `--study-name`, `--storage`.
  - Refactor `parse_sweep_logs` to optionally pull data directly from the SQLite storage using `optuna.load_study`.
  - Ensure analysis is filtered by the specified study.

## 6. Documentation & Experiment Ledger

- **Action**:
  - Create `docs/tuning.md` with CLI examples and usage instructions.
  - Update `EXPERIMENTS.md` with a new entry for this architectural upgrade.

## 7. Verification Plan

1. **Environment**: Verify `.gitignore` prevents `optuna.db` from being staged.
2. **Interactive Launch**: Run `python src/tune.py` and verify prompts for DB and Study.
3. **Traceability**: Run a 1-trial sweep and verify `git_commit` is visible in `trial.user_attrs` (can check via `analyze_sweep.py` or `optuna-dashboard`).
4. **Early Stopping**: Set `early_stop_patience: 1` and `lr: 10.0` in `configs/train.yaml`, run `src/tune.py`, and verify it stops early.
5. **Parameter Injection**: Change `num_heads` in `configs/train.yaml` and verify the model initializes without errors.
6. **Analysis**: Run `python src/analyze_sweep.py --storage sqlite:///optuna.db --study-name my-experiment` and verify it prints importance/correlations.
