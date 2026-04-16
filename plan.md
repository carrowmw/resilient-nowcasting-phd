# Plan: Post-hoc Analysis Preparation

## 1. Model Checkpointing (src/train.py)
- Modify `src/train.py` to track `best_val_loss`.
- Ensure `models/` directory exists using `os.makedirs`.
- Save `model.state_dict()` to `models/best_pipeline.pt` when a new best validation loss is achieved.

## 2. Statistical Engine (src/metrics.py)
- Create `src/metrics.py`.
- Implement `calculate_metrics(target_forecast, mu, pi, imputed_features, target_impute, node_mask)` function.
- **RMSE**: `sqrt(mean((target_forecast - mu)^2))`
- **MAE**: `mean(abs(target_forecast - mu))`
- **Zero-Accuracy**: Percentage of `pi > 0.5` when `target_forecast == 0`.
- **Imputation MSE**: `mean((imputed_features - target_impute)^2)` only where `node_mask` is True.
- Use `torch` for calculations and return a dictionary of scalar floats.

## 3. Integration & Refactoring
- **src/pipeline.py**:
    - Modify `ResilientIoTPipeline.forward` to return `(loss, loss_impute, loss_forecast, imputed_features, mu, theta, pi)`.
- **src/train.py**:
    - Import `calculate_metrics` from `metrics`.
    - Update training and validation loops to handle the expanded return values from the model.
    - In the validation loop, call `calculate_metrics` for each batch and aggregate them.
    - Log `val_rmse`, `val_mae`, `val_zero_acc`, `val_impute_mse` to W&B.

## 4. Verification
- Run `python src/train.py --use_mock_data --epochs 2` on CPU.
- Verify `models/best_pipeline.pt` exists.
- Verify console logs show the new metrics.
- Update `EXPERIMENTS.md`.
