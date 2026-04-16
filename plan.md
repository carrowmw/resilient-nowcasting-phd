# Implementation Plan - Hyperparameter Tuning (src/tune.py)

## Objective
Build an Optuna hyperparameter tuning orchestrator (`src/tune.py`) integrated with Weights & Biases to optimize learning rate and loss weighting parameters.

## Proposed Changes

### 1. Prerequisite Setup
- Install `optuna` and `optuna-integration`.

### 2. Refactoring `src/train.py`
- Extract the core training loop inside `main()` into a standalone function: `def run_training(args, init_wandb=True) -> float`.
- Wrap the `wandb.init()` and `wandb.finish()` calls inside `if init_wandb:` blocks to prevent duplicate run initialization when called by Optuna.
- Make `run_training` return `avg_val_loss`.
- Keep `main()` intact by having it call `args = parse_args()` and `run_training(args)`.

### 3. Module Implementation (`src/tune.py`)
- **Imports**: `optuna`, `WeightsAndBiasesCallback`, `run_training` from `train`, `argparse`.
- **Objective Function**:
  - `def objective(trial)`
  - Suggest `lr`: log-uniform `[1e-5, 1e-2]`.
  - Suggest `w_forecast`: uniform `[0.5, 5.0]`.
  - Suggest `w_impute`: uniform `[0.5, 5.0]`.
  - Construct an `argparse.Namespace` with these suggestions, plus fixed `epochs=2` and `batch_size=4` for the test.
  - Call `val_loss = run_training(args, init_wandb=False)`.
  - Return `val_loss`.
- **Study Setup**:
  - Instantiate `WeightsAndBiasesCallback(metric_name="val_loss_total", wandb_kwargs={"project": "resilient-iot-sweeps"})`.
  - `study = optuna.create_study(direction="minimize")`.
  - `study.optimize(objective, n_trials=2, callbacks=[wandbc])`.

### 4. Verification Plan
- Run `export WANDB_MODE=offline && python src/tune.py` (or online, per prompt constraints, "The Online Local Test" but since the previous step asked for offline, I'll allow it to run and use offline/online as appropriate for testing).
- Validate trials complete, results log, and no crashes occur.
- Update `EXPERIMENTS.md`.
