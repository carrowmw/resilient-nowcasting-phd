# Implementation Plan - MaskedSpatiotemporalEncoder

## Objective
Implement the `MaskedSpatiotemporalEncoder` in `src/encoder.py` to process dynamic time-series data with spatiotemporal dependencies using GATv2 and Transformers.

## Proposed Changes

### 1. Environment Setup
- Install `torch-geometric` and its dependencies (`torch-scatter`, `torch-sparse`) to support GATv2Conv.

### 2. Module Implementation (`src/encoder.py`)
- **Initialization**:
  - Define `GATv2Conv` layers for spatial processing.
  - Define `TransformerEncoder` for temporal processing.
  - Parameters: `num_nodes`, `seq_len`, `time_series_dim`, `hidden_dim`, `num_heads`, `num_layers`.
- **Forward Pass**:
  - **Spatial Processing**:
    - Input `dynamic_features`: `[batch_size, num_nodes, seq_len, time_series_dim]`.
    - Reshape to `[batch_size * seq_len, num_nodes, time_series_dim]`? No, PyG prefers a flat node format.
    - Better: Iterate over `seq_len` or reshape to `[batch_size * seq_len * num_nodes, time_series_dim]` and use a batch vector for PyG.
    - However, since `edge_index` is likely the same for all time steps and batches (from `graph_learner`), we can treat `batch_size * seq_len` as the "batch dimension" for the graph.
    - Reshape `dynamic_features` to `[batch_size * seq_len * num_nodes, time_series_dim]`.
    - Create a batch vector if needed, but if `edge_index` is for a single graph of `num_nodes`, we can use `edge_index` with offset or repeat it.
    - Actually, `GATv2Conv` can handle batched graphs if we provide a giant `edge_index`.
    - Alternatively, reshape to `(B*T, N, D)` and use `GATv2Conv` if we handle it carefully. PyG layers usually take `(N, D)` and `edge_index`.
    - Strategy: 
      1. Reshape `[B, N, T, D]` -> `[T, B, N, D]`.
      2. For each time step `t`:
         - Apply GAT to `[B*N, D]` with a block-diagonal `edge_index`?
         - Or just loop over `T` and `B`? (Slow).
         - Best PyG way: Reshape `[B, N, T, D]` -> `[B * T * N, D]`. Construct a giant `edge_index` that connects nodes within each `(b, t)` block.
  - **Temporal Processing**:
    - Reshape output of spatial processing back to `[batch_size, num_nodes, seq_len, hidden_dim]`.
    - Permute to `[seq_len, batch_size * num_nodes, hidden_dim]` for `TransformerEncoder`.
    - Apply `TransformerEncoder`.
    - Reshape back to `[batch_size, num_nodes, seq_len, hidden_dim]`.

### 3. Offensive Unit Test
- Instantiate `MaskedSpatiotemporalEncoder`.
- Dummy data: `B=4, N=50, T=12, D=3`.
- Mock `edge_index` and `edge_weight`.
- Assert output shape.
- Assert gradients flow to both GAT and Transformer.

## Verification Plan
- Run `python src/encoder.py` and ensure it passes all assertions.
- Check for NaNs and ensure the computation graph is intact.
