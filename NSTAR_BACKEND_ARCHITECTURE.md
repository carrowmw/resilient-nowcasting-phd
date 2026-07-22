# NSTAR_BACKEND_ARCHITECTURE.md

## Objective

Implement an end-to-end, joint spatiotemporal graph neural network in PyTorch and PyTorch Geometric (PyG). The model is designed for an urban IoT sensor network predicting a configurable target traffic stream (defined by vehicle class and spatial direction) every 15 minutes. It must be self-healing, capable of dynamic graph inference, missing feature imputation, and uncertainty-aware forecasting. **The pipeline must remain strictly univariate per run** to prevent gradient interference across different vehicle classes.

## Core Paradigm

The system abandons the I.I.D. assumption, leveraging Graph Attention Networks (GATs) to pass messages along dynamically learned topological connections, alongside Transformers for temporal sequence modelling.

### 1. Architectural Components

The pipeline (`ResilientIoTPipeline`) consists of three primary modules sharing a latent space:

#### Dynamic Graph Learner (Topology Inference):

- Takes static node features (e.g., coordinates, sensor types).
- Uses a multi-head GAT (or linear embeddings with pairwise similarity) to learn a dynamic adjacency matrix on the fly.
- Applies a ReLU and a sparsity threshold, converting the dense matrix to PyG's `edge_index` and `edge_weight` using `dense_to_sparse`.

#### Masked Spatiotemporal Encoder (Feature Extraction):

- Takes artificially masked dynamic time-series data and the inferred graph structure.
- **Spatial:** Uses PyG `GATv2Conv` layers to aggregate neighbourhood features.
- **Temporal:** Flattens the spatial output and passes it through a standard PyTorch `TransformerEncoder`.

#### Decoupled Decoders (Multi-Task Learning):

- **Imputation Decoder:** An MLP (e.g., Linear -> GELU -> Dropout -> Linear) applied across the entire sequence to reconstruct the masked historical data.
- **Uncertainty-Aware Forecasting Decoder:** An MLP taking the final time step's embedding. Outputs three parameters representing a Zero-Inflated Negative Binomial (ZINB) distribution: $\mu$ (mean), $\theta$ (dispersion), and $\pi$ (zero-inflation probability).
- **Spatial Kriging Decoder (Component 2):** An MLP (mirroring the imputation head's capacity) that reconstructs the full time series for *spatially held-out* nodes from their latent embeddings — embeddings shaped purely by message passing from observable neighbours under IGNNK-style node masking. Active when `w_kriging > 0`.

> **Edge-Enriched Spatial Attention (Component 1).** The `GATv2Conv` layers consume a 10-dimensional `edge_attr` (`edge_dim=10`): 9 physical OSM walking-network features (shortest-path & euclidean distance, directness, and a 6-way road-type composition) precomputed in `data/external/edge_features_cache.json`, plus 1 learned similarity dimension from the graph learner. With no cache the encoder falls back to learned-only attention (`edge_dim=1`). See `docs/model/spatial_estimation_architecture.md` §3.

### 2. Data Pipeline Specifications

The PyTorch Dataset must yield a dictionary for each batch. Target variables must strictly remain as raw, unscaled counts, while input dynamic/static features must be normalized via `StandardScaler` to prevent gradient starvation.

### 3. Joint Training & Loss Logic

The model is trained simultaneously on all tasks. The **unified objective** (Phases 1–3) is:

$$\mathcal{L} = (w_{impute} \times \mathcal{L}_{impute}) + (w_{forecast} \times \mathcal{L}_{ZINB\_NLL}) + (w_{kriging} \times \mathcal{L}_{kriging})$$

Both reconstruction terms use **channel-specific** treatment so that quiet low-count nodes are weighted on par with high-variance city-centre nodes:

- $\mathcal{L}_{impute}$ — **TRUE MSLE** on the *raw* traffic counts (channel 0) **plus** MSE on the *scaled* weather channels (1–2), over temporally masked positions.
- $\mathcal{L}_{kriging}$ — the same channel split, but over the *spatially held-out* nodes (active when $w_{kriging} > 0$).
- $\mathcal{L}_{ZINB\_NLL}$ — ZINB negative log-likelihood on raw counts (forecasting).

**TRUE MSLE, not pseudo-MSLE.** MSLE is defined only on non-negative counts, but the encoder consumes `StandardScaler`-normalized features (signed floats). Rather than clamp the scaled values — which discards ~40–50% of the distribution (the "pseudo-MSLE" debt) — the traffic channel is **inverse-transformed back to raw counts** using scaler statistics (`traffic_mean`, `traffic_std`) registered as `nn.Module` buffers (device-safe; moved to GPU with the model), then clamped to the count domain and passed to `log1p`. Weather channels stay continuous and use MSE on the scaled features. See the ADR [`docs/adr/0001-eliminate-pseudo-msle.md`](docs/adr/0001-eliminate-pseudo-msle.md) and `docs/model/spatial_estimation_architecture.md` §5.

- **ZINB Tripwires:** Apply `F.softplus` to $\mu$ and $\theta$ raw logits, plus an `eps=1e-6` buffer to mathematically guarantee they never equal zero.
- **Loss Tripwires:** every term (`loss_impute_traffic`, `loss_impute_weather`, `loss_forecast`, `loss_kriging`, `total_loss`) is guarded by `CRITICAL:` NaN/Inf assertions in `compute_loss` — divergence fails fast rather than propagating silently.

### 4. Model Development & Ablation Strategy

Before reaching final production architecture, the pipeline must undergo structured evaluation:

- **Baseline Benchmarking:** Before tuning GNN hyperparameters, visualize a "Last Observation Carried Forward" (LOCF) or Historical Average model. The deep learning pipeline must visually and mathematically outperform these dumb baselines.
- **Zero-Inflation ($\pi$) Tuning:** Do not rely on a hardcoded `0.5` threshold to determine structural zeros. Plot the ROC curve for the $\pi$ parameter to find the optimal threshold for traffic shut-offs (e.g., nighttime periods).
- **Component Ablation:** Systematically disable the `DynamicGraphLearner` and swap the `GATv2Conv` for temporal-only baselines to isolate the exact performance delta generated by the graph topology.
