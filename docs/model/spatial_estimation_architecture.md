# Spatial Estimation Architecture Report

> **Status:** Research Investigation  
> **Date:** 2026-06-10  
> **Author:** Automated Research Agent  
> **Related Files:** [pipeline.py](file:///Users/administrator/Code/ml-agent-workspace/src/pipeline.py), [decoders.py](file:///Users/administrator/Code/ml-agent-workspace/src/decoders.py), [encoder.py](file:///Users/administrator/Code/ml-agent-workspace/src/encoder.py), [graph_learner.py](file:///Users/administrator/Code/ml-agent-workspace/src/graph_learner.py), [dataset.py](file:///Users/administrator/Code/ml-agent-workspace/src/dataset.py), [train.py](file:///Users/administrator/Code/ml-agent-workspace/src/train.py)

---

## 1. Problem Diagnosis

The current `ResilientIoTPipeline` trains on two objectives:

| Decoder | Task | What it learns |
|---------|------|----------------|
| **Imputation** | Reconstruct ~15% randomly masked sensor readings | Temporal self-repair at *known* locations |
| **Forecasting (ZINB)** | Predict the next-step count at each sensor | Temporal extrapolation at *known* locations |

Neither objective teaches the model to **reconstruct data at an unknown location given only its spatial context**. When a simulated node is dropped into the graph, it receives messages from its neighbours via GATv2Conv, but the model has never been trained to translate "nearby node patterns + spatial relationship" into a local prediction. The result is a bland, mean-regressed forecast regardless of placement.

### 1.1 Root Causes (Ordered by Impact)

1. **No spatial holdout signal.** The train/val split in [train.py](file:///Users/administrator/Code/ml-agent-workspace/src/train.py#L227-L237) uses `random_split` over *temporal windows*. Every node appears in every batch. The model has zero incentive to infer cross-node spatial transfer.

2. **Edge features are scalar and uninformative.** The `GATv2Conv` layer in [encoder.py](file:///Users/administrator/Code/ml-agent-workspace/src/encoder.py#L37-L43) accepts `edge_dim=1` — a single scalar `edge_weight` derived from the dot-product similarity in `DynamicGraphLearner`. This weight encodes *feature similarity*, not physical distance, road type, or path composition. Two nodes 50m apart on the same street receive the same edge weight as two nodes 5km apart with similar POI profiles.

3. **Scale-coupled imputation loss.** The imputation MSE in [decoders.py](file:///Users/administrator/Code/ml-agent-workspace/src/decoders.py#L145) operates on StandardScaler-normalised features. While this prevents gradient explosion, it still means a node with traffic counts of 200 contributes proportionally more to the loss than a node with counts of 5, biasing the model towards reproducing high-magnitude patterns rather than learning transferable spatial structure.

---

## 2. Proposed Solution: Edge-Enriched Spatial Kriging with Triple Decoders

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STATIC FEATURES (15-dim)                  │
│              ┌──────────────────────────┐                    │
│              │   DynamicGraphLearner    │                    │
│              │   (topology inference)   │                    │
│              └────────────┬─────────────┘                    │
│                           │ edge_index, edge_weight          │
│              ┌────────────▼─────────────┐                    │
│              │   Edge Feature Encoder   │  ← NEW             │
│              │   (distance + road type) │                    │
│              └────────────┬─────────────┘                    │
│                           │ edge_attr [E, edge_dim]          │
│  DYNAMIC    ┌─────────────▼──────────────┐                   │
│  FEATURES → │  MaskedSpatiotemporalEncoder│                  │
│  (3-dim)    │  GATv2Conv(edge_dim=D_e)   │                   │
│             │  TransformerEncoder         │                   │
│             └─────────┬──────────────────┘                   │
│                       │ latent [B, N, T, H]                  │
│         ┌─────────────┼──────────────────┐                   │
│         ▼             ▼                  ▼                   │
│  ┌──────────┐  ┌────────────┐  ┌─────────────────┐          │
│  │ Imputation│  │ Forecast   │  │ Spatial Kriging │ ← NEW   │
│  │ Decoder   │  │ Decoder    │  │ Decoder         │          │
│  │ (MSE/MSLE)│  │ (ZINB NLL) │  │ (MSLE on held  │          │
│  │           │  │            │  │  out nodes)     │          │
│  └──────────┘  └────────────┘  └─────────────────┘          │
│       ↓             ↓                  ↓                     │
│   L_impute      L_forecast        L_kriging                  │
│                                                              │
│   TOTAL = w_i·L_impute + w_f·L_forecast + w_k·L_kriging     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Component 1: Edge Feature Enrichment

### 3.1 The Problem

The current `GATv2Conv` receives a 1-dimensional `edge_weight` — a learned similarity score. This tells the attention mechanism *how similar* two nodes are in latent space, but nothing about:

- **How far apart** they are on the physical walking network
- **What kind of road** connects them (pedestrianised street vs. B-road vs. residential)
- **How direct** the connection is (shortest path vs. Euclidean)

### 3.2 Proposed Edge Feature Vector

For each edge $(i, j)$ in the graph, compute a multi-dimensional feature vector:

| Feature | Source | Dim | Description |
|---------|--------|-----|-------------|
| `shortest_path_distance` | OSMnx | 1 | Network distance in metres along the walking graph |
| `euclidean_distance` | Haversine | 1 | Straight-line distance |
| `path_directness` | Derived | 1 | `euclidean / shortest_path` ratio ∈ (0, 1]. Low = winding path |
| `road_type_composition` | OSMnx edge `highway` tags | K | Fraction of path length along each road type (one-hot-ish) |
| `learned_similarity` | DynamicGraphLearner | 1 | Current dot-product weight (preserved) |

### 3.3 Road Type Categories

OSMnx walking networks contain these `highway` tags on edges. We propose collapsing them into K=6 categories:

| Category | OSM `highway` values |
|----------|---------------------|
| `pedestrian` | `pedestrian`, `footway`, `path`, `steps`, `corridor` |
| `residential` | `residential`, `living_street` |
| `tertiary` | `tertiary`, `tertiary_link`, `unclassified` |
| `secondary` | `secondary`, `secondary_link` |
| `primary` | `primary`, `primary_link`, `trunk`, `trunk_link` |
| `service` | `service`, `track`, `bridleway`, `cycleway` |

For each edge $(i, j)$, compute the shortest path, iterate over the path edges, sum the `length` per category, then normalise by total path length. This yields a 6-dimensional composition vector where values sum to 1.0.

### 3.4 Implementation Strategy

**Pre-computation (offline, cached):**

```python
def compute_pairwise_edge_features(G_walk, sensor_coords, cache_path):
    """
    For each pair of sensors, compute shortest path distance, 
    road type composition, and directness ratio.
    Caches to JSON. Runs once.
    """
    edge_features = {}
    for i, (name_i, lat_i, lon_i) in enumerate(sensor_coords):
        node_i = ox.nearest_nodes(G_walk, lon_i, lat_i)
        for j, (name_j, lat_j, lon_j) in enumerate(sensor_coords):
            if i == j:
                continue
            node_j = ox.nearest_nodes(G_walk, lon_j, lat_j)
            try:
                path = nx.shortest_path(G_walk, node_i, node_j, weight='length')
                sp_dist = nx.shortest_path_length(G_walk, node_i, node_j, weight='length')
            except nx.NetworkXNoPath:
                continue
            
            # Compute road type composition along path
            type_lengths = {cat: 0.0 for cat in ROAD_CATEGORIES}
            for u, v in zip(path[:-1], path[1:]):
                edge_data = G_walk[u][v][0]  # first edge in MultiDiGraph
                hw = edge_data.get('highway', 'unclassified')
                cat = classify_highway(hw)
                type_lengths[cat] += edge_data.get('length', 0.0)
            
            total = sum(type_lengths.values())
            composition = {k: v/total for k, v in type_lengths.items()} if total > 0 else ...
            
            euc_dist = haversine(lat_i, lon_i, lat_j, lon_j)
            directness = euc_dist / sp_dist if sp_dist > 0 else 1.0
            
            edge_features[(name_i, name_j)] = {
                'shortest_path_distance': sp_dist,
                'euclidean_distance': euc_dist,
                'directness': directness,
                **composition
            }
    return edge_features
```

**Runtime integration:**

The `DynamicGraphLearner` produces `edge_index` and `edge_weight`. At runtime, we look up the pre-computed features for each edge in `edge_index` and construct a `[E, edge_dim]` tensor. The `GATv2Conv` layer's `edge_dim` parameter is updated from `1` to `edge_dim` (e.g., 10).

### 3.5 Impact on GATv2Conv Attention

The GATv2 attention coefficient becomes:

$$\alpha_{i,j} \propto \text{LeakyReLU}\left(\mathbf{a}^{\top} \left[\mathbf{\Theta}_s \mathbf{x}_i + \mathbf{\Theta}_t \mathbf{x}_j + \mathbf{\Theta}_e \mathbf{e}_{i,j}\right]\right)$$

Where $\mathbf{e}_{i,j}$ now encodes physical path distance, road composition, and directness. This means the model can learn that:
- Nearby nodes on pedestrianised streets share strong pedestrian patterns
- Nodes connected via primary roads may share vehicle patterns but not pedestrian ones
- Winding, indirect connections carry weaker spatial correlation

### 3.6 Computational Considerations

> [!WARNING]
> Computing all-pairs shortest paths for 66 nodes on a 33k-node OSMnx graph is $O(N_{sensors}^2 \cdot V \log V)$. For 66 sensors this is ~4,356 Dijkstra calls — feasible in ~10 minutes and only needs to run once (cached). However, if sensor count grows to 200+, consider using approximate methods or batched NetworkX shortest paths.

---

## 4. Component 2: Spatial Holdout Decoder (Kriging Head)

### 4.1 The Core Idea: IGNNK-Style Node Masking

Inspired by the IGNNK framework (Wu et al., AAAI 2021), we introduce a **spatial holdout** strategy during training. Instead of only masking individual sensor readings (temporal masking for imputation), we **completely blank out entire nodes** and train a dedicated decoder to reconstruct their full time series from the graph context alone.

### 4.2 Training Protocol

During each training step:

1. **Select holdout nodes.** Randomly sample $k$ nodes (e.g., $k = \lfloor 0.15 \times N \rfloor$, so ~9 of 61 real nodes). These become the "spatial holdout set" for this batch.

2. **Zero their dynamic features.** Replace the selected nodes' entire `[seq_len, time_series_dim]` tensor with zeros (or NaN → 0 via `nan_to_num`, matching the simulated node protocol). Their static features are **preserved** — the model must use spatial context to reconstruct.

3. **Forward pass.** The encoder produces latent embeddings for all nodes. The held-out nodes' embeddings are shaped entirely by message-passing from their observable neighbours.

4. **Kriging decoder.** A new MLP head (matching the architecture of the existing imputation decoder) takes the held-out nodes' latent embeddings and reconstructs their full `[seq_len, time_series_dim]` time series.

5. **Loss.** Compare the reconstructed series against the ground truth (the original, unmasked values) using a scale-invariant loss (see Section 5).

### 4.3 Decoder Architecture

```python
class SpatialKrigingDecoder(nn.Module):
    """
    Reconstructs the full time series for spatially held-out nodes.
    
    Input:  latent_embeddings[:, holdout_indices, :, :]  →  [B, K, T, H]
    Output: reconstructed_series                          →  [B, K, T, D]
    """
    def __init__(self, hidden_dim: int, time_series_dim: int, dropout: float = 0.1):
        super().__init__()
        self.kriging_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, time_series_dim)
        )
    
    def forward(self, latent_embeddings, holdout_mask):
        # Extract only the held-out node embeddings
        # holdout_mask: [num_nodes] boolean
        held_out = latent_embeddings[:, holdout_mask, :, :]  # [B, K, T, H]
        return self.kriging_head(held_out)  # [B, K, T, D]
```

### 4.4 Integration with Existing Pipeline

The `DecoupledDecoders` class becomes `TripleDecoders`:

```python
class TripleDecoders(nn.Module):
    def __init__(self, hidden_dim, time_series_dim, dropout=0.1, zinb_eps=1e-6):
        super().__init__()
        # Existing heads (unchanged)
        self.impute_layer = nn.Sequential(...)   # Temporal imputation
        self.forecast_layer = nn.Sequential(...) # ZINB forecasting
        
        # NEW: Spatial kriging head
        self.kriging_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, time_series_dim)
        )
    
    def forward(self, latent_embeddings, spatial_holdout_mask=None):
        # ... existing imputation and forecast logic ...
        
        # Kriging (only computed during training when holdout_mask is provided)
        kriging_output = None
        if spatial_holdout_mask is not None and spatial_holdout_mask.any():
            held_out_embeddings = latent_embeddings[:, spatial_holdout_mask, :, :]
            kriging_output = self.kriging_layer(held_out_embeddings)
        
        return imputed_features, mu, theta, pi, tripwires, kriging_output
```

### 4.5 Data Flow Changes in `dataset.py` and `train.py`

**`dataset.py`** needs a new method or flag to generate spatial holdout masks:

```python
def __getitem__(self, idx):
    # ... existing logic ...
    
    # Spatial holdout: randomly select ~15% of nodes
    spatial_holdout_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
    k = max(1, int(0.15 * self.num_nodes))
    holdout_indices = torch.randperm(self.num_nodes)[:k]
    spatial_holdout_mask[holdout_indices] = True
    
    # Store the clean target for held-out nodes before zeroing
    kriging_target = dyn_tensor.clone()  # [N, T, D]
    
    # Zero out held-out nodes' dynamic features
    dyn_masked[spatial_holdout_mask] = 0.0  # Not NaN — explicit zero
    
    return {
        'static_features': self.static_features,
        'dynamic_features': dyn_masked,
        'target_impute': target_impute,
        'target_forecast': target_tensor,
        'node_mask': node_mask,
        'spatial_holdout_mask': spatial_holdout_mask,      # NEW
        'kriging_target': kriging_target,                   # NEW
    }
```

### 4.6 Validation Strategy: Spatial Cross-Validation

For validation, use a **fixed** spatial holdout set (not random per batch). This provides a consistent benchmark:

```python
# In train.py, before the training loop:
spatial_val_holdout = torch.zeros(num_nodes, dtype=torch.bool)
# Hold out ~10% of nodes for spatial validation (fixed seed)
rng = torch.Generator().manual_seed(42)
val_holdout_indices = torch.randperm(num_nodes, generator=rng)[:max(1, num_nodes // 10)]
spatial_val_holdout[val_holdout_indices] = True
```

This means the model is evaluated on its ability to predict the full time series at nodes it has *never directly observed* during validation — exactly analogous to the simulated node scenario.

### 4.7 Interaction with Simulated Node Masking

> [!IMPORTANT]
> The `simulated_nodes_mask` we added in `pipeline.py` (which prevents simulated nodes from sending messages) should also be applied to the spatial holdout nodes during training. This ensures the training conditions match inference: held-out nodes cannot pollute their neighbours' embeddings.

```python
# In pipeline.py forward():
if 'simulated_nodes_mask' in batch:
    combined_mask = batch['simulated_nodes_mask']
    if 'spatial_holdout_mask' in batch:
        combined_mask = combined_mask | batch['spatial_holdout_mask']
    sources = edge_index[0]
    valid_edges = ~combined_mask[sources]
    edge_index = edge_index[:, valid_edges]
    edge_weight = edge_weight[valid_edges]
```

---

## 5. Component 3: Scale-Invariant Loss Function

### 5.1 The Problem with Raw MSE

The current imputation loss in [decoders.py:145](file:///Users/administrator/Code/ml-agent-workspace/src/decoders.py#L145):

```python
loss_impute = F.mse_loss(imputed_features, target_impute)
```

Although the features are StandardScaler-normalised, MSE still conflates reconstruction *difficulty* with target *magnitude*. A node with high variance (city centre, counts 0–200) dominates the loss compared to a quiet residential node (counts 0–5), even though the residential node may be harder to predict *relatively*.

### 5.2 Recommended: Mean Squared Logarithmic Error (MSLE)

For count-like data that spans orders of magnitude, **MSLE** is the standard scale-invariant alternative:

$$\mathcal{L}_{\text{MSLE}} = \frac{1}{N} \sum_{i=1}^{N} \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2$$

**Properties:**
- Penalises *relative* error, not absolute error
- The `log(1 + x)` transform compresses high-magnitude values and amplifies small values
- Naturally handles zero counts (no division-by-zero)
- Differentiable everywhere

### 5.3 Implementation

```python
def scale_invariant_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean Squared Logarithmic Error (MSLE).
    
    Operates on the UNSCALED (raw count) domain. If inputs are already scaled,
    inverse-transform first.
    
    Args:
        pred:   [B, N, T, D] or [B, N] predicted values (non-negative)
        target: same shape, ground truth (non-negative)
        eps:    small constant for numerical stability
    Returns:
        Scalar MSLE loss
    """
    # Clamp to ensure non-negative before log
    pred_safe = torch.clamp(pred, min=0.0)
    target_safe = torch.clamp(target, min=0.0)
    
    log_pred = torch.log1p(pred_safe + eps)
    log_target = torch.log1p(target_safe + eps)
    
    return F.mse_loss(log_pred, log_target)
```

### 5.4 Where to Apply MSLE

| Loss Component | Current | Proposed | Rationale |
|----------------|---------|----------|-----------|
| Temporal Imputation | MSE (on scaled features) | **MSLE** (on raw counts) | Scale-invariant, treats all nodes equally |
| Spatial Kriging | N/A | **MSLE** (on raw counts) | Same rationale; kriging targets are full time series |
| Forecasting | ZINB NLL | ZINB NLL (unchanged) | Already distribution-aware and scale-appropriate |

### 5.5 Interaction with StandardScaler

> [!IMPORTANT]
> MSLE must operate on **raw, unscaled count data**, not StandardScaler-normalised features. The `dataset.py` already preserves `self.unscaled_traffic` for exactly this reason. The imputation and kriging targets need a similar unscaled path for the traffic channel.
>
> **Approach:** Apply MSLE only to the **first channel** (traffic counts) of the imputation/kriging outputs. For the auxiliary channels (Humidity, Solar Radiation), standard MSE on scaled values remains appropriate since these are continuous-valued meteorological readings, not count data.

> [!NOTE]
> **Resolution (shipped, Phases 1–3).** The implementation takes the *mathematically clean* inverse-transform path — **not** pseudo-MSLE on clamped scaled features (Open Question #4, now RESOLVED). The traffic channel (ch 0) of both the imputation and kriging predictions/targets is inverse-transformed back to raw counts via `traffic_raw = traffic_scaled * traffic_std + traffic_mean`, clamped to `>= 0`, then passed through `scale_invariant_mse` (`mean((log1p(pred) - log1p(target))^2)`). `traffic_mean`/`traffic_std` are computed once in `RealIoTDataset` and registered as `nn.Module` buffers on `ResilientIoTPipeline` (device-safe — they move to GPU with the model). Weather channels (1–2) use `F.mse_loss` on the scaled values, unchanged. The inverse transform is a constant affine map, so gradients flow through it without special handling. Every loss term carries a `CRITICAL:` NaN tripwire. See `src/decoders.py::compute_loss` / `compute_kriging_loss` and the ADR [`docs/adr/0001-eliminate-pseudo-msle.md`](../adr/0001-eliminate-pseudo-msle.md).

### 5.6 Hybrid Loss Formula

The final joint loss becomes:

$$\mathcal{L}_{\text{total}} = w_f \cdot \mathcal{L}_{\text{ZINB}} + w_i \cdot \mathcal{L}_{\text{impute}} + w_k \cdot \mathcal{L}_{\text{kriging}}$$

Where:
- $\mathcal{L}_{\text{ZINB}}$: Unchanged ZINB NLL on forecast targets
- $\mathcal{L}_{\text{impute}}$: MSLE on the traffic channel + MSE on weather channels, computed over temporally masked positions only
- $\mathcal{L}_{\text{kriging}}$: MSLE on the traffic channel + MSE on weather channels, computed over spatially held-out nodes only

Suggested initial weights: `w_f = 1.0`, `w_i = 1.0`, `w_k = 0.5` (start conservative, tune via Optuna).

---

## 6. Files Requiring Modification

| File | Change Type | Summary |
|------|-------------|---------|
| [features.py](file:///Users/administrator/Code/ml-agent-workspace/src/features.py) | MODIFY | Add `compute_pairwise_edge_features()` with OSMnx shortest path + road type composition |
| [graph_learner.py](file:///Users/administrator/Code/ml-agent-workspace/src/graph_learner.py) | MODIFY | Return enriched `edge_attr` tensor alongside `edge_index` and `edge_weight` |
| [encoder.py](file:///Users/administrator/Code/ml-agent-workspace/src/encoder.py) | MODIFY | Update `GATv2Conv(edge_dim=1)` → `GATv2Conv(edge_dim=D_e)` |
| [decoders.py](file:///Users/administrator/Code/ml-agent-workspace/src/decoders.py) | MODIFY | Add `SpatialKrigingDecoder` head; refactor `compute_loss` for MSLE + triple terms |
| [dataset.py](file:///Users/administrator/Code/ml-agent-workspace/src/dataset.py) | MODIFY | Add spatial holdout mask generation and kriging target to `__getitem__` |
| [pipeline.py](file:///Users/administrator/Code/ml-agent-workspace/src/pipeline.py) | MODIFY | Wire kriging decoder; extend `simulated_nodes_mask` to cover spatial holdout |
| [train.py](file:///Users/administrator/Code/ml-agent-workspace/src/train.py) | MODIFY | Pass spatial holdout to model; log kriging loss; add fixed spatial val holdout |
| [metrics.py](file:///Users/administrator/Code/ml-agent-workspace/src/metrics.py) | MODIFY | Add spatial kriging metrics (MSLE, per-node-type breakdown) |
| [configs/train/train.yaml](file:///Users/administrator/Code/ml-agent-workspace/configs/train/train.yaml) | MODIFY | Add `w_kriging`, `spatial_holdout_prob`, `edge_dim` config keys |
| [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py) | MODIFY | Pass edge features at inference; no spatial holdout at inference time |

---

## 7. Execution Ordering

> [!TIP]
> These three components are largely independent and can be implemented incrementally. The recommended order minimises risk and maximises early signal.

### Phase A: Spatial Holdout Decoder (Highest Impact, Moderate Effort)
This addresses the fundamental training gap. Even without edge features, the model will start learning spatial transfer.

1. Modify `dataset.py` to generate spatial holdout masks
2. Add kriging decoder to `decoders.py`
3. Wire through `pipeline.py` and `train.py`
4. Run ablation: compare val kriging loss with and without the new decoder

### Phase B: Scale-Invariant Loss (Low Effort, Quick Win)
Can be done in parallel with Phase A.

1. Implement `scale_invariant_mse` in `decoders.py`
2. Update `compute_loss` to use MSLE for traffic channel
3. Re-run baseline training and compare convergence curves

### Phase C: Edge Feature Enrichment (High Effort, Requires Retraining)
Depends on the OSMnx graph already being cached.

1. Implement `compute_pairwise_edge_features` in `features.py`
2. Update `graph_learner.py` to emit multi-dim `edge_attr`
3. Update `encoder.py` `GATv2Conv` initialisation
4. Retrain from scratch (edge_dim change invalidates existing checkpoints)

---

## 8. Open Questions

1. **Checkpoint compatibility.** Changing `edge_dim` from 1 to D_e in `GATv2Conv` will break compatibility with the existing `best_model.pt` (v2). A full retrain is required. Should we version this as v3?

2. **Kriging holdout ratio.** The IGNNK paper uses 20-50% node holdout. With only 61 nodes, holding out 15% (~9 nodes) may be a good starting point. Too aggressive and the remaining network may be too sparse for meaningful message passing.

3. **Computational cost of all-pairs shortest paths.** For 66 nodes × 66 nodes = 4,356 Dijkstra calls on a 33k-node walking graph. Estimated ~10 minutes (one-time, cached). Is this acceptable, or should we use approximate methods?

4. **MSLE on scaled vs. unscaled data.** ✅ **RESOLVED (Phase 1).** We inverse-transform the traffic channel back to raw count space (via registered `traffic_mean`/`traffic_std` buffers) before computing MSLE — the mathematically clean path, not pseudo-MSLE on clamped scaled features. The inverse transform is a constant affine map, so the backward pass needs no special handling. See §5.5 *Resolution (shipped)* and the ADR [`docs/adr/0001-eliminate-pseudo-msle.md`](../adr/0001-eliminate-pseudo-msle.md).

5. **Weight tuning.** The three-way loss weighting ($w_f$, $w_i$, $w_k$) introduces a new hyperparameter. Should we add this to the existing Optuna sweep configuration?

---

## 9. References

- Wu, Y., Zhuang, D., Labbe, A., & Sun, L. (2021). *Inductive Graph Neural Networks for Spatiotemporal Kriging.* AAAI 2021. [[arXiv:2006.07527]](https://arxiv.org/abs/2006.07527)
- Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive are Graph Attention Networks?* ICLR 2022. (GATv2 with edge features)
- Boeing, G. (2017). *OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks.* Computers, Environment and Urban Systems.
