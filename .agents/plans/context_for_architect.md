# Context for Architect: Chapter 5 Negative Baseline

**Generated:** 2026-06-15T14:45:00+01:00
**Purpose:** Codebase investigation results for the Ch5 Negative Baseline integration.

---

## Current Architecture Summary

### Pipeline Flow (src/pipeline.py)
```
ResilientIoTPipeline.forward(batch):
  1. DynamicGraphLearner(static_features) → edge_index, edge_weight
  2. MaskedSpatiotemporalEncoder(dynamic_features, edge_index, edge_weight) → latent [B, N, T, H]
  3. DecoupledDecoders(latent) → imputed [B,N,T,D], mu/theta/pi [B,N]
  4. compute_loss(imputed, targets, mu, theta, pi) → total_loss
```

### Tensor Shape Contract
- `dynamic_features`: `[B, N, T, D]` = `[batch, 81 nodes, 12 seq_len, 3 channels]`
- `latent_embeddings`: `[B, N, T, hidden_dim]` = `[batch, 81, 12, 32]`
- `imputed_features`: `[B, N, T, D]` = `[batch, 81, 12, 3]`
- `mu/theta/pi`: `[B, N]` = `[batch, 81]`
- `edge_index`: `[2, E]`
- `edge_weight`: `[E]`

### Key Files
- **Encoder:** `src/encoder.py` — MaskedSpatiotemporalEncoder (GATv2 + Transformer)
- **Decoders:** `src/decoders.py` — DecoupledDecoders (Imputation + ZINB Forecasting)
- **Metrics:** `src/metrics.py` — calculate_metrics, compute_msle, calculate_kriging_metrics
- **Training:** `src/train.py` — run_training(), main()
- **Pipeline:** `src/pipeline.py` — ResilientIoTPipeline
- **Dataset:** `src/dataset.py` — RealIoTDataset
- **Config:** `configs/train/train.yaml`

### Legacy Functions (from digest.txt)
- `create_anomaly_column` (L1848): `timeseries_array[:, 0] > 0.9` → hardcoded boolean flag
- `find_consecutive_sequences` (L2114): Iterates rows, drops sub-sequences shorter than window_size
- `LSTMModel` (L1006): Bidirectional LSTM with `hidden_dim*2` linear output, `x[:, -1, :]` selection

### Config Schema (train.yaml)
```yaml
data: {start_date, end_date, seq_len, target_veh_class, target_dir, mask_prob, spatial_holdout_prob}
model: {num_nodes, static_feat_dim, time_series_dim, embedding_dim, hidden_dim, sparsity_threshold, num_layers, num_heads, dropout, dim_feedforward, zinb_eps}
training: {epochs, batch_size, lr, grad_clip_max_norm, random_seed, early_stop_patience, early_stop_min_delta}
loss: {w_forecast, w_impute, w_kriging}
```

### W&B Logging (current)
```python
wandb.log({
    "train_loss_total", "train_loss_impute", "train_loss_forecast",
    "val_loss_total", "val_rmse", "val_mae", "val_zero_acc",
    "val_zero_true_count", "val_zero_correct_count", "val_impute_mse",
    "epoch"
})
```

### Tripwire Pattern
All modules use `assert` + `raise ValueError` for offensive programming:
- Shape asserts at function entry
- NaN tripwires at function exit
- ZINB eps guards in decoders
