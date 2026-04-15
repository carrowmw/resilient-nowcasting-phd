# Implementation Plan - ResilientIoTPipeline Assembly

## Objective
Assemble the final architecture in `src/pipeline.py` by integrating `DynamicGraphLearner`, `MaskedSpatiotemporalEncoder`, and `DecoupledDecoders`. Write a full integration training loop as the "Burn Test".

## Proposed Architecture
- **Initialization** (`__init__`):
  - Instantiate `DynamicGraphLearner(static_feat_dim, embedding_dim, sparsity_threshold)`
  - Instantiate `MaskedSpatiotemporalEncoder(num_nodes, seq_len, time_series_dim, hidden_dim)`
  - Instantiate `DecoupledDecoders(hidden_dim, time_series_dim)`
- **Forward Pass**:
  - Accept `batch` dict (`static_features`, `dynamic_features`, `target_forecast`, `target_impute`, `node_mask`).
  - Graph Learner -> `edge_index`, `edge_weight`
  - Encoder -> `latent_embeddings` (Masked NaNs should technically be zeroed out before passing into GAT, or handled. The encoder expects floats, but GAT and Transformer will propagate NaNs if not handled. I will handle NaN-to-zero conversion of `dynamic_features` here, as the GAT cannot take NaN values directly).
  - Decoders -> `imputed_features`, `mu`, `theta`, `pi`
  - Compute Loss -> `compute_loss(...)`
  - Return `total_loss` (and optionally the predictions for tracking).

## Handling NaNs in `dynamic_features`
The `ResilientIoTDataset` injects NaNs. Since PyTorch operations on NaNs yield NaNs (triggering our tripwires), we must replace NaNs with zeros (or a specific fill value) *before* passing them into the Encoder.

## Integration Test ("The Burn Test")
1. Mock data for 50 nodes and 12 timesteps.
   - Note: The `ResilientIoTDataset` expects `num_samples` as the first dimension. Since our modules expect `[batch_size, num_nodes, seq_len, time_series_dim]`, we will treat `num_nodes` as the batch dimension for `ResilientIoTDataset` or wrap it to match the expected shapes. Wait, `ResilientIoTDataset` returns `(B, T, D)` for dynamic features, but the encoder expects `(B, N, T, D)`.
   - Ah, looking closely at `ResilientIoTDataset` from the previous script: `dynamic_features` is `(N_samples, T, D)` and `static_features` is `(N_samples, S)`. The batch size yielded is `B`. So the output is `(B, T, D)` and `(B, S)`.
   - BUT the `ARCHITECTURE.md` explicitly says: `static_features: [num_nodes, static_feat_dim]` and `dynamic_features: [batch_size, num_nodes, seq_len, time_series_dim]`.
   - I must adapt the dataset output in the pipeline to match the `num_nodes` requirement. For the burn test, we will generate a batch that exactly matches the expected dictionary keys and shapes.
2. Training loop (2 epochs).
3. Assertions: `assert not torch.isnan(total_loss).any()`
4. Grad Clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
5. Print loss.

## Verification Plan
- Execute `python src/pipeline.py`.
- Update `EXPERIMENTS.md` once the integration test passes.
