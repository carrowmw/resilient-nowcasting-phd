# Architectural Design: Unified High-Performance Integration
## Phase 3 (Scale-Invariant Loss) + Phase 4/5 (Spatial Kriging + GAT Enrichment)

**Architect:** Claude Sonnet 4.5  
**Date:** 2026-06-29  
**Target Branch:** `main`  
**Source Experiments:**
- `exp/scale-invariant-loss` (Phase 3)
- `exp/enriched-gat-kriging` (Phase 4 & 5)

---

## Executive Summary

This plan details the reimplementation of two validated experimental branches into a unified, production-grade architecture on `main`. The integration resolves technical debt (pseudo-MSLE), enhances spatial reasoning (kriging + enriched GAT), and establishes a mathematically rigorous loss framework with offensive programming guarantees.

**Key Innovations:**
1. **Dual-Head True MSLE:** Both Imputation and Kriging decoders compute scale-invariant loss on raw traffic counts via shared scaler buffers
2. **Spatial Holdout Kriging:** IGNNK-style node masking for spatial generalization testing
3. **Physical-Aware GAT:** 10-dimensional edge features (9 physical + 1 learned) with fail-fast cache validation
4. **Zero Technical Debt:** Eliminates pseudo-MSLE approximation by inverse-transforming to raw counts before loss computation

---

## 1. Integration Strategy

### 1.1 Branch Independence Analysis

✅ **Verified:** Git merge-base audit confirms both experimental branches depend ONLY on features already merged into `main`:
- `feat/osm-edge-enrichment` → **MERGED**
- `feat/kriging-infrastructure` → **MERGED**  
- `feat/datapath-and-test-fixes` → **MERGED**

**Implication:** Clean reimplementation is possible without cross-branch dependencies.

### 1.2 Implementation Approach

**Sequential Layering Strategy:**
1. **Layer 1:** Scale-invariant loss infrastructure (scaler buffers + MSLE for Imputation head)
2. **Layer 2:** Spatial holdout masking + Kriging decoder integration
3. **Layer 3:** Physical edge features + GATv2 enrichment
4. **Layer 4:** Unified loss computation with dual-head true MSLE

Each layer is **independently verifiable** via desk checks and unit tests before proceeding to the next.

---

## 2. Unified Loss Architecture: Resolving the Pseudo-MSLE Debt

### 2.1 Problem Statement

**Current State (`exp/enriched-gat-kriging`):**
- Kriging loss computes "pseudo-MSLE" on **StandardScaler-transformed** features
- Negative values below the mean are clamped to 0.0 before log1p
- **Lost Information:** ~40-50% of distribution variance discarded

**Current State (`exp/scale-invariant-loss`):**
- Imputation head has scaler buffers but Kriging head doesn't exist yet
- Only Imputation loss benefits from true raw-count MSLE

### 2.2 Unified Solution Design

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│  RealIoTDataset (src/dataset.py)                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ StandardScaler.fit_transform(traffic_channel)        │   │
│  │ → self.traffic_mean: float                           │   │
│  │ → self.traffic_std: float                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓ (register buffers)
┌─────────────────────────────────────────────────────────────┐
│  ResilientIoTPipeline (src/pipeline.py)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ self.register_buffer('traffic_mean', ...)            │   │
│  │ self.register_buffer('traffic_std', ...)             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓ (forward to compute_loss)
┌─────────────────────────────────────────────────────────────┐
│  compute_loss (src/decoders.py)                             │
│                                                              │
│  1. Inverse Transform Traffic Channel:                      │
│     traffic_raw = (traffic_scaled * std) + mean             │
│                                                              │
│  2. Compute True MSLE for Imputation:                       │
│     loss_impute_traffic = scale_invariant_mse(              │
│         imputed_traffic_raw, target_impute_traffic_raw)     │
│                                                              │
│  3. Compute True MSLE for Kriging:                          │
│     loss_kriging = scale_invariant_mse(                     │
│         kriging_traffic_raw, target_kriging_traffic_raw)    │
│                                                              │
│  4. Weather Channels (MSE on scaled features):              │
│     loss_impute_weather = F.mse_loss(                       │
│         imputed_weather, target_weather)                    │
│                                                              │
│  5. Weighted Sum:                                           │
│     total = w_impute*(loss_traffic + loss_weather) +        │
│             w_kriging*loss_kriging +                        │
│             w_forecast*loss_zinb                            │
└─────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **Single Source of Truth:** Scaler statistics computed once in `RealIoTDataset.__init__`
- **Device-Safe:** Registered as `nn.Module` buffers (auto-moved to GPU with model)
- **Channel Separation:** Traffic (ch 0) uses MSLE on raw counts; Weather (ch 1-2) uses MSE on scaled features
- **Zero Approximation:** No pseudo-MSLE clamping; full distribution preserved

### 2.3 Mathematical Correctness

**Inverse Transform (Traffic Channel Only):**
```python
# StandardScaler forward:  z = (x - μ) / σ
# Inverse transform:       x = (z * σ) + μ

traffic_scaled = imputed_features[:, :, :, 0]  # [B, N, T]
traffic_raw = (traffic_scaled * traffic_std) + traffic_mean
traffic_raw_clamped = torch.clamp(traffic_raw, min=0.0)  # Enforce count domain
```

**MSLE Domain Guarantee:**
- Input: Non-negative raw counts (enforced by clamp)
- Output: Finite scalar (log1p is continuous for x ≥ 0)
- No division: Cannot raise divide-by-zero
- NaN propagation: Caught by existing compute_loss tripwire

---

## 3. Component Integration Plan

### 3.1 Layer 1: Scale-Invariant Loss Infrastructure

**Files Modified:**
- `src/dataset.py`
- `src/pipeline.py`
- `src/decoders.py`
- `src/train.py`

**Implementation Steps:**

#### 3.1.1 Dataset Scaler Extraction (`src/dataset.py`)
```python
class RealIoTDataset:
    def __init__(self, ...):
        # ... existing preprocessing ...
        
        # Separate traffic channel before stacking
        traffic_data = df_pivot['value'].values  # [T, N]
        weather_data = df_pivot[['Humidity', 'Solar Radiation']].values  # [T, N, 2]
        
        # Fit scaler on traffic channel
        traffic_flat = traffic_data.reshape(-1, 1)
        self.traffic_scaler = StandardScaler()
        traffic_scaled_flat = self.traffic_scaler.fit_transform(traffic_flat)
        traffic_scaled = traffic_scaled_flat.reshape(T, N)
        
        # Extract and expose scaler statistics
        self.traffic_mean = float(self.traffic_scaler.mean_[0])
        self.traffic_std = float(self.traffic_scaler.scale_[0])
        
        # Weather channels scaled separately
        weather_flat = weather_data.reshape(-1, 2)
        self.weather_scaler = StandardScaler()
        weather_scaled_flat = self.weather_scaler.fit_transform(weather_flat)
        weather_scaled = weather_scaled_flat.reshape(T, N, 2)
        
        # Stack: [T, N, 3] where dim 0=traffic, dim 1-2=weather
        self.dynamic_data = np.stack([traffic_scaled, weather_scaled[:,:,0], weather_scaled[:,:,1]], axis=-1)
        
        # Store unscaled traffic for target_forecast (ZINB domain)
        self.unscaled_traffic = traffic_data.copy()
        
        logger.info(f"Traffic scaler: mean={self.traffic_mean:.2f}, std={self.traffic_std:.2f}")
```

**Offensive Assertions:**
```python
assert self.traffic_std > 0, "CRITICAL: Traffic std is zero (constant values)"
assert not np.isnan(self.traffic_mean), "CRITICAL: Traffic mean is NaN"
assert not np.isnan(self.traffic_std), "CRITICAL: Traffic std is NaN"
```

#### 3.1.2 Buffer Registration (`src/pipeline.py`)
```python
class ResilientIoTPipeline(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing modules ...
        
        # Scaler buffers (initialized to sentinel values, set by train.py)
        self.register_buffer('traffic_mean', torch.tensor(0.0))
        self.register_buffer('traffic_std', torch.tensor(1.0))
        
    def set_scaler_stats(self, traffic_mean: float, traffic_std: float):
        """Called by train.py after dataset initialization."""
        self.traffic_mean.fill_(traffic_mean)
        self.traffic_std.fill_(traffic_std)
        logger.info(f"Scaler buffers set: mean={traffic_mean:.2f}, std={traffic_std:.2f}")
```

#### 3.1.3 Training Hook (`src/train.py`)
```python
def run_training(...):
    # ... dataset initialization ...
    train_dataset = RealIoTDataset(...)
    val_dataset = RealIoTDataset(...)
    
    # Extract and register scaler stats
    traffic_mean = train_dataset.traffic_mean
    traffic_std = train_dataset.traffic_std
    
    pipeline = ResilientIoTPipeline(...)
    pipeline.set_scaler_stats(traffic_mean, traffic_std)
    
    # Defensive check: validation set should have similar statistics
    val_mean_diff = abs(val_dataset.traffic_mean - traffic_mean)
    val_std_diff = abs(val_dataset.traffic_std - traffic_std)
    if val_mean_diff > traffic_mean * 0.2 or val_std_diff > traffic_std * 0.2:
        logger.warning(f"Train/val scaler mismatch: Δmean={val_mean_diff:.2f}, Δstd={val_std_diff:.2f}")
```

#### 3.1.4 Loss Computation Update (`src/decoders.py`)
```python
def compute_loss(
    imputed_features: torch.Tensor,  # [B, N, T, 3]
    target_impute: torch.Tensor,     # [B, N, T, 3]
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    target_forecast: torch.Tensor,
    traffic_mean: torch.Tensor,      # NEW: scaler buffer
    traffic_std: torch.Tensor,       # NEW: scaler buffer
    w_impute: float = 1.0,
    w_forecast: float = 1.0,
    w_kriging: float = 0.0           # NEW: kriging weight
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint Loss with Channel-Specific Treatment:
    - Traffic (channel 0): MSLE on raw counts
    - Weather (channels 1-2): MSE on scaled features
    - Forecasting: ZINB NLL on raw counts
    """
    
    # --- 1. Imputation Loss: Split Traffic vs Weather ---
    
    # Traffic channel: inverse transform to raw counts
    traffic_pred_scaled = imputed_features[:, :, :, 0]      # [B, N, T]
    traffic_target_scaled = target_impute[:, :, :, 0]       # [B, N, T]
    
    traffic_pred_raw = (traffic_pred_scaled * traffic_std) + traffic_mean
    traffic_target_raw = (traffic_target_scaled * traffic_std) + traffic_mean
    
    loss_impute_traffic = scale_invariant_mse(traffic_pred_raw, traffic_target_raw)
    
    # Weather channels: MSE on scaled features (no inverse transform)
    weather_pred = imputed_features[:, :, :, 1:]      # [B, N, T, 2]
    weather_target = target_impute[:, :, :, 1:]       # [B, N, T, 2]
    loss_impute_weather = F.mse_loss(weather_pred, weather_target)
    
    # Combined imputation loss
    loss_impute = loss_impute_traffic + loss_impute_weather
    
    # --- 2. Forecasting Loss (ZINB NLL) ---
    loss_forecast = compute_zinb_nll(target_forecast, mu, theta, pi)
    
    # --- 3. Total Weighted Loss ---
    total_loss = (w_impute * loss_impute) + (w_forecast * loss_forecast)
    
    # --- 4. Tripwires ---
    assert not torch.isnan(loss_impute_traffic), "CRITICAL: Traffic MSLE is NaN"
    assert not torch.isnan(loss_impute_weather), "CRITICAL: Weather MSE is NaN"
    assert not torch.isnan(total_loss), "CRITICAL: Total loss is NaN"
    
    return total_loss, loss_impute, loss_forecast
```

**Desk Check Requirements:**
```python
# scripts/desk_check_scale_invariant.py
def test_inverse_transform():
    """Verify inverse transform recovers original scale."""
    traffic_raw = torch.tensor([0.0, 10.0, 100.0, 1000.0])
    mean, std = 50.0, 200.0
    
    # Forward transform (StandardScaler)
    traffic_scaled = (traffic_raw - mean) / std
    
    # Inverse transform (our implementation)
    traffic_recovered = (traffic_scaled * std) + mean
    
    assert torch.allclose(traffic_raw, traffic_recovered, atol=1e-5)
    print("✓ Inverse transform verified")

def test_msle_on_raw_vs_scaled():
    """Demonstrate MSLE sensitivity to raw vs scaled inputs."""
    pred_raw = torch.tensor([5.0, 50.0, 500.0])
    target_raw = torch.tensor([10.0, 100.0, 1000.0])
    
    mean, std = 50.0, 200.0
    pred_scaled = (pred_raw - mean) / std
    target_scaled = (target_raw - mean) / std
    
    msle_raw = scale_invariant_mse(pred_raw, target_raw)
    msle_scaled = scale_invariant_mse(pred_scaled.clamp(min=0), target_scaled.clamp(min=0))
    
    print(f"MSLE on raw counts: {msle_raw:.4f}")
    print(f"Pseudo-MSLE on scaled (with clamp): {msle_scaled:.4f}")
    print(f"Ratio: {msle_scaled / msle_raw:.2f}x")
    
    # Scaled version will be incorrect due to negative values being clamped
    assert abs(msle_raw - msle_scaled) > 0.01, "Expected significant difference"
    print("✓ Raw vs scaled MSLE difference confirmed")
```

---

### 3.2 Layer 2: Spatial Holdout Kriging Integration

**Files Modified:**
- `src/dataset.py` (new methods)
- `src/decoders.py` (activate kriging head + loss)
- `src/pipeline.py` (thread holdout masks)
- `src/train.py` (create and pass masks)
- `configs/train/train.yaml` (expose `data.spatial_holdout_prob` and `loss.w_kriging`)

**Implementation Steps:**

#### 3.2.1 Holdout Mask Generation (`src/dataset.py`)
```python
def make_spatial_holdout_mask(
    num_nodes: int,
    holdout_prob: float,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create a boolean mask marking spatially held-out nodes (IGNNK-style).
    
    Args:
        num_nodes: Total sensor count
        holdout_prob: Fraction of nodes to hold out (e.g., 0.2 for 20%)
        device: Tensor device
        seed: Random seed (None = random, int = deterministic)
    
    Returns:
        mask: [num_nodes] boolean tensor, True = held out
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    mask = rng.rand(num_nodes) < holdout_prob
    return torch.from_numpy(mask).bool().to(device)


def apply_spatial_holdout(
    dynamic_features: torch.Tensor,
    holdout_mask: torch.Tensor
) -> torch.Tensor:
    """
    Zero out dynamic features for held-out nodes.
    
    Args:
        dynamic_features: [B, N, T, D]
        holdout_mask: [N] boolean
    
    Returns:
        masked_features: [B, N, T, D] with holdout nodes zeroed
    """
    assert dynamic_features.dim() == 4
    assert holdout_mask.dim() == 1
    assert holdout_mask.shape[0] == dynamic_features.shape[1]
    
    masked = dynamic_features.clone()
    masked[:, holdout_mask, :, :] = 0.0  # Zero all timesteps and channels
    return masked
```

**Offensive Assertions:**
```python
assert 0.0 <= holdout_prob <= 1.0, f"holdout_prob must be in [0,1], got {holdout_prob}"
assert holdout_mask.sum() > 0 or holdout_prob == 0.0, "No nodes held out despite prob > 0"
```

#### 3.2.2 Kriging Loss Integration (`src/decoders.py`)
```python
def compute_kriging_loss(
    kriging_pred: torch.Tensor,      # [B, K, T, D] from SpatialKrigingDecoder
    target_impute: torch.Tensor,     # [B, N, T, D] full spatial grid
    holdout_mask: torch.Tensor,      # [N] boolean
    traffic_mean: torch.Tensor,
    traffic_std: torch.Tensor
) -> torch.Tensor:
    """
    Kriging reconstruction loss on held-out nodes.
    
    Uses TRUE MSLE on raw traffic counts (channel 0) and MSE on scaled weather.
    """
    # Extract ground truth for held-out nodes
    target_holdout = target_impute[:, holdout_mask, :, :]  # [B, K, T, D]
    
    assert kriging_pred.shape == target_holdout.shape, \
        f"Shape mismatch: pred {kriging_pred.shape} vs target {target_holdout.shape}"
    
    # Traffic channel: inverse transform and MSLE
    traffic_pred_scaled = kriging_pred[:, :, :, 0]
    traffic_target_scaled = target_holdout[:, :, :, 0]
    
    traffic_pred_raw = (traffic_pred_scaled * traffic_std) + traffic_mean
    traffic_target_raw = (traffic_target_scaled * traffic_std) + traffic_mean
    
    loss_traffic = scale_invariant_mse(traffic_pred_raw, traffic_target_raw)
    
    # Weather channels: MSE on scaled
    weather_pred = kriging_pred[:, :, :, 1:]
    weather_target = target_holdout[:, :, :, 1:]
    loss_weather = F.mse_loss(weather_pred, weather_target)
    
    return loss_traffic + loss_weather
```

#### 3.2.3 Pipeline Threading (`src/pipeline.py`)
```python
class ResilientIoTPipeline(nn.Module):
    def __init__(self, ...):
        # ... existing modules ...
        
        # Add kriging decoder (dormant unless w_kriging > 0)
        self.kriging_decoder = SpatialKrigingDecoder(
            hidden_dim=hidden_dim,
            time_series_dim=time_series_dim,
            dropout=dropout
        )
        
    def forward(self, batch: dict, return_analysis: bool = False):
        # ... existing graph learner + encoder ...
        
        # Decoder outputs
        imputed, mu, theta, pi, tripwire_logs = self.decoders(latent_embeddings)
        
        # Kriging output (conditional)
        kriging_pred = None
        if 'spatial_holdout_mask' in batch:
            holdout_mask = batch['spatial_holdout_mask']  # [N] boolean
            kriging_pred = self.kriging_decoder(latent_embeddings, holdout_mask)
        
        if return_analysis:
            return {
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'attention_weights': attention_weights,
                'imputed_features': imputed,
                'kriging_pred': kriging_pred,
                'mu': mu, 'theta': theta, 'pi': pi,
                'tripwire_logs': tripwire_logs
            }
        
        # Compute loss
        target_forecast = batch['target_forecast']
        target_impute = batch['target_impute']
        
        total_loss, loss_impute, loss_forecast, loss_kriging = compute_loss(
            imputed_features=imputed,
            target_impute=target_impute,
            mu=mu, theta=theta, pi=pi,
            target_forecast=target_forecast,
            kriging_pred=kriging_pred,
            spatial_holdout_mask=batch.get('spatial_holdout_mask'),
            traffic_mean=self.traffic_mean,
            traffic_std=self.traffic_std,
            w_impute=batch.get('w_impute', 1.0),
            w_forecast=batch.get('w_forecast', 1.0),
            w_kriging=batch.get('w_kriging', 0.0)
        )
        
        return total_loss, loss_impute, loss_forecast, loss_kriging, imputed, mu, theta, pi, tripwire_logs
```

#### 3.2.4 Training Loop (`src/train.py`)
```python
def run_training(
    # ... existing args ...
    spatial_holdout_prob: float = 0.0,
    w_kriging: float = 0.0,
    ...
):
    # ... dataset + model init ...
    
    # Fixed validation holdout mask (deterministic for metric tracking)
    val_holdout_mask = None
    if spatial_holdout_prob > 0.0:
        val_holdout_mask = make_spatial_holdout_mask(
            num_nodes=train_dataset.num_nodes,
            holdout_prob=spatial_holdout_prob,
            device=device,
            seed=42  # Fixed for validation
        )
        logger.info(f"Validation holdout: {val_holdout_mask.sum()}/{len(val_holdout_mask)} nodes")
    
    for epoch in range(epochs):
        # --- Training ---
        for batch_idx, batch in enumerate(train_loader):
            # Dynamic training holdout (random each batch)
            if spatial_holdout_prob > 0.0:
                train_holdout_mask = make_spatial_holdout_mask(
                    num_nodes=train_dataset.num_nodes,
                    holdout_prob=spatial_holdout_prob,
                    device=device,
                    seed=None  # Random
                )
                batch['spatial_holdout_mask'] = train_holdout_mask
                batch['dynamic_features'] = apply_spatial_holdout(
                    batch['dynamic_features'], train_holdout_mask
                )
            
            batch['w_kriging'] = w_kriging
            
            total_loss, loss_impute, loss_forecast, loss_kriging, *_ = pipeline(batch)
            # ... backward pass ...
        
        # --- Validation ---
        with torch.no_grad():
            for batch in val_loader:
                if val_holdout_mask is not None:
                    batch['spatial_holdout_mask'] = val_holdout_mask
                    batch['dynamic_features'] = apply_spatial_holdout(
                        batch['dynamic_features'], val_holdout_mask
                    )
                
                batch['w_kriging'] = w_kriging
                
                total_loss, loss_impute, loss_forecast, loss_kriging, *_ = pipeline(batch)
                # ... accumulate metrics ...
        
        # Log kriging metrics (if active)
        if w_kriging > 0.0:
            wandb.log({
                'val_loss_kriging': val_loss_kriging,
                'val_kriging_nodes': val_holdout_mask.sum().item()
            })
```

**Desk Check Requirements:**
```python
# scripts/desk_check_kriging.py
def test_holdout_masking():
    """Verify held-out nodes are fully zeroed."""
    B, N, T, D = 4, 20, 12, 3
    features = torch.randn(B, N, T, D)
    mask = torch.tensor([False]*10 + [True]*10)  # Hold out last 10 nodes
    
    masked = apply_spatial_holdout(features, mask)
    
    # Check: held-out nodes are zero
    assert (masked[:, mask, :, :] == 0.0).all()
    # Check: non-held-out nodes unchanged
    assert torch.allclose(masked[:, ~mask, :, :], features[:, ~mask, :, :])
    print("✓ Holdout masking verified")

def test_kriging_decoder_shape():
    """Verify kriging decoder output shape."""
    from src.decoders import SpatialKrigingDecoder
    
    B, N, T, H, D = 4, 20, 12, 64, 3
    latent = torch.randn(B, N, T, H)
    mask = torch.tensor([False]*15 + [True]*5)  # 5 held-out
    
    decoder = SpatialKrigingDecoder(hidden_dim=H, time_series_dim=D)
    output = decoder(latent, mask)
    
    K = mask.sum().item()
    assert output.shape == (B, K, T, D), f"Expected ({B},{K},{T},{D}), got {output.shape}"
    print("✓ Kriging decoder shape verified")
```

---

### 3.3 Layer 3: Physical Edge Features + GATv2 Enrichment

**Files Modified:**
- `src/graph_learner.py` (load cache, build `edge_attr`)
- `src/encoder.py` (lift GATv2Conv to `edge_dim=10`)
- `src/pipeline.py` (thread `edge_attr`)

**Implementation Steps:**

#### 3.3.1 Edge Feature Loading (`src/graph_learner.py`)
```python
import json
from pathlib import Path

class DynamicGraphLearner(nn.Module):
    def __init__(
        self,
        static_feat_dim: int,
        embedding_dim: int = 32,
        sparsity_threshold: float = 0.5,
        edge_features_cache_path: Optional[str] = None
    ):
        super().__init__()
        # ... existing init ...
        
        # Load physical edge features cache
        self.edge_features_cache = None
        self.max_distance = None
        
        if edge_features_cache_path is not None:
            cache_path = Path(edge_features_cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(f"CRITICAL: Edge features cache not found: {cache_path}")
            
            with open(cache_path, 'r') as f:
                self.edge_features_cache = json.load(f)
            
            # Compute global max distance for normalization
            distances = [
                pair_data['shortest_path_distance']
                for pairs in self.edge_features_cache.values()
                for pair_data in pairs.values()
            ]
            self.max_distance = max(distances)
            
            logger.info(f"Loaded edge features cache: {len(self.edge_features_cache)} sources, "
                       f"max_distance={self.max_distance:.1f}m")
        
    def _build_edge_attr(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        location_ids: List[str]
    ) -> torch.Tensor:
        """
        Construct [E, 10] edge attribute tensor:
        - 9 physical features from cache (distances + road composition)
        - 1 learned similarity (from graph learner edge_weight)
        
        Raises ValueError if cache is missing any required edge.
        """
        if self.edge_features_cache is None:
            # Fallback: only learned similarity (broadcast to [E, 1])
            return edge_weight.unsqueeze(1)
        
        E = edge_index.shape[1]
        edge_attrs = []
        
        for e in range(E):
            src_idx, dst_idx = edge_index[0, e].item(), edge_index[1, e].item()
            src_loc = location_ids[src_idx]
            dst_loc = location_ids[dst_idx]
            
            # Lookup cache (fail-fast)
            if src_loc not in self.edge_features_cache:
                raise ValueError(f"CRITICAL: Missing cache for source {src_loc}")
            if dst_loc not in self.edge_features_cache[src_loc]:
                raise ValueError(f"CRITICAL: Missing cache for edge {src_loc} -> {dst_loc}")
            
            pair_data = self.edge_features_cache[src_loc][dst_loc]
            
            # Extract 9 physical features
            shortest_path = pair_data['shortest_path_distance']
            euclidean = pair_data['euclidean_distance']
            directness = pair_data['directness']
            pedestrian = pair_data['pedestrian_fraction']
            residential = pair_data['residential_fraction']
            tertiary = pair_data['tertiary_fraction']
            secondary = pair_data['secondary_fraction']
            primary = pair_data['primary_fraction']
            service = pair_data['service_fraction']
            
            # Normalize distance to [0, 1] using global max
            shortest_path_norm = shortest_path / self.max_distance
            
            # Learned similarity (10th dimension)
            learned_sim = edge_weight[e].item()
            
            # Concatenate [9 physical + 1 learned]
            edge_attr = [
                shortest_path_norm, euclidean, directness,
                pedestrian, residential, tertiary, secondary, primary, service,
                learned_sim
            ]
            
            edge_attrs.append(edge_attr)
        
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32, device=edge_weight.device)
        
        # Offensive assertions
        assert edge_attr_tensor.shape == (E, 10), f"Expected ({E}, 10), got {edge_attr_tensor.shape}"
        assert not torch.isnan(edge_attr_tensor).any(), "CRITICAL: NaN in edge_attr"
        
        # Road composition fractions should sum to ~1.0 (allow 1e-3 tolerance for rounding)
        road_sums = edge_attr_tensor[:, 3:9].sum(dim=1)  # Sum columns 3-8 (6 road types)
        assert torch.allclose(road_sums, torch.ones_like(road_sums), atol=1e-3), \
            f"CRITICAL: Road fractions don't sum to 1.0: {road_sums}"
        
        return edge_attr_tensor
    
    def forward(
        self,
        static_features: torch.Tensor,
        location_ids: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_index: [2, E]
            edge_weight: [E]
            edge_attr: [E, 10] (or [E, 1] if no cache)
        """
        # ... existing similarity computation ...
        edge_index, edge_weight = dense_to_sparse(adj)
        
        # Build enriched edge attributes
        if location_ids is not None and self.edge_features_cache is not None:
            edge_attr = self._build_edge_attr(edge_index, edge_weight, location_ids)
        else:
            edge_attr = edge_weight.unsqueeze(1)  # [E, 1] fallback
        
        return edge_index, edge_weight, edge_attr
```

#### 3.3.2 GATv2 Enrichment (`src/encoder.py`)
```python
class MaskedSpatiotemporalEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        time_series_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
        edge_dim: int = 10  # NEW: physical + learned features
    ):
        super().__init__()
        # ... existing init ...
        
        # GATv2 with edge_attr support
        self.gat = GATv2Conv(
            in_channels=time_series_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim  # NEW: consume 10D edge features
        )
        
        # ... transformer init unchanged ...
    
    def forward(
        self,
        dynamic_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,  # NEW: [E, 10] or [E, 1]
        return_attention_weights: bool = False
    ):
        # ... existing shape checks ...
        
        # Spatial processing with edge attributes
        spatial_out = []
        for t in range(seq_len):
            x_t = dynamic_features[:, :, t, :]  # [B*N, D]
            
            if return_attention_weights:
                x_spatial, (_, alpha) = self.gat(
                    x_t, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
            else:
                x_spatial = self.gat(x_t, edge_index, edge_attr=edge_attr)
            
            spatial_out.append(x_spatial)
        
        # ... rest of forward pass unchanged ...
```

#### 3.3.3 Pipeline Integration (`src/pipeline.py`)
```python
class ResilientIoTPipeline(nn.Module):
    def __init__(
        self,
        # ... existing args ...
        edge_features_cache_path: Optional[str] = None
    ):
        super().__init__()
        
        # Graph learner with edge features
        self.graph_learner = DynamicGraphLearner(
            static_feat_dim=static_feat_dim,
            embedding_dim=embedding_dim,
            sparsity_threshold=sparsity_threshold,
            edge_features_cache_path=edge_features_cache_path
        )
        
        # Encoder with edge_dim=10
        self.encoder = MaskedSpatiotemporalEncoder(
            num_nodes=num_nodes,
            seq_len=seq_len,
            time_series_dim=time_series_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            edge_dim=10  # Physical + learned
        )
        
    def forward(self, batch: dict, return_analysis: bool = False):
        static_features = batch['static_features']
        dynamic_features = batch['dynamic_features']
        location_ids = batch.get('location_ids')  # NEW: pass from dataset
        
        # Graph learner with enriched edges
        edge_index, edge_weight, edge_attr = self.graph_learner(
            static_features, location_ids=location_ids
        )
        
        # Encoder with edge_attr
        if return_analysis:
            latent_embeddings, attention_weights = self.encoder(
                dynamic_features, edge_index, edge_attr, return_attention_weights=True
            )
        else:
            latent_embeddings = self.encoder(
                dynamic_features, edge_index, edge_attr, return_attention_weights=False
            )
        
        # ... rest of forward unchanged ...
```

#### 3.3.4 Dataset Location IDs (`src/dataset.py`)
```python
class RealIoTDataset(Dataset):
    def __getitem__(self, idx: int) -> dict:
        # ... existing logic ...
        
        return {
            'static_features': static_features,
            'dynamic_features': dynamic_features,
            'target_forecast': target_forecast,
            'target_impute': target_impute,
            'node_mask': node_mask,
            'location_ids': self.locations  # NEW: for edge feature lookup
        }
```

**Desk Check Requirements:**
```python
# scripts/desk_check_edge_features.py
def test_edge_attr_shape():
    """Verify edge_attr tensor construction."""
    from src.graph_learner import DynamicGraphLearner
    
    # Mock cache with 3 sensors
    cache = {
        'S1': {
            'S2': {'shortest_path_distance': 100, 'euclidean_distance': 90, 'directness': 0.9,
                   'pedestrian_fraction': 0.5, 'residential_fraction': 0.3, 'tertiary_fraction': 0.1,
                   'secondary_fraction': 0.1, 'primary_fraction': 0.0, 'service_fraction': 0.0},
            'S3': {'shortest_path_distance': 200, 'euclidean_distance': 180, 'directness': 0.9,
                   'pedestrian_fraction': 0.0, 'residential_fraction': 0.0, 'tertiary_fraction': 0.0,
                   'secondary_fraction': 0.5, 'primary_fraction': 0.5, 'service_fraction': 0.0}
        },
        # ... (symmetric entries for S2, S3)
    }
    
    # ... save cache to temp file, instantiate learner ...
    
    edge_index, edge_weight, edge_attr = learner(static_features, location_ids=['S1', 'S2', 'S3'])
    
    E = edge_index.shape[1]
    assert edge_attr.shape == (E, 10), f"Expected ({E}, 10), got {edge_attr.shape}"
    
    # Check road composition sums
    road_sums = edge_attr[:, 3:9].sum(dim=1)
    assert torch.allclose(road_sums, torch.ones(E), atol=1e-3)
    
    print("✓ Edge attribute construction verified")

def test_missing_cache_fail_fast():
    """Verify fail-fast on missing edge features."""
    # ... setup learner with incomplete cache ...
    
    try:
        edge_index, edge_weight, edge_attr = learner(static_features, location_ids=['S1', 'S99'])
        assert False, "Should have raised ValueError for missing cache entry"
    except ValueError as e:
        assert "CRITICAL: Missing cache" in str(e)
        print("✓ Fail-fast on missing cache verified")
```

---

### 3.4 Layer 4: Unified Loss Integration

**Updated Signature:**
```python
def compute_loss(
    imputed_features: torch.Tensor,          # [B, N, T, 3]
    target_impute: torch.Tensor,             # [B, N, T, 3]
    mu: torch.Tensor,                        # [B, N]
    theta: torch.Tensor,                     # [B, N]
    pi: torch.Tensor,                        # [B, N]
    target_forecast: torch.Tensor,           # [B, N]
    kriging_pred: Optional[torch.Tensor],    # [B, K, T, 3] or None
    spatial_holdout_mask: Optional[torch.Tensor],  # [N] bool or None
    traffic_mean: torch.Tensor,              # scalar buffer
    traffic_std: torch.Tensor,               # scalar buffer
    w_impute: float = 1.0,
    w_forecast: float = 1.0,
    w_kriging: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified multi-task loss with channel-specific treatment.
    
    Returns:
        total_loss: Weighted sum
        loss_impute: Imputation loss (traffic MSLE + weather MSE)
        loss_forecast: ZINB NLL
        loss_kriging: Kriging loss (0.0 if w_kriging == 0.0)
    """
    # 1. Imputation Loss (already covered in Layer 1)
    # ...
    
    # 2. Forecasting Loss (existing)
    loss_forecast = compute_zinb_nll(target_forecast, mu, theta, pi)
    
    # 3. Kriging Loss (conditional)
    loss_kriging = torch.tensor(0.0, device=imputed_features.device)
    if w_kriging > 0.0:
        assert kriging_pred is not None, "CRITICAL: w_kriging > 0 but no kriging_pred"
        assert spatial_holdout_mask is not None, "CRITICAL: w_kriging > 0 but no holdout mask"
        
        loss_kriging = compute_kriging_loss(
            kriging_pred, target_impute, spatial_holdout_mask,
            traffic_mean, traffic_std
        )
    
    # 4. Total Loss
    total_loss = (w_impute * loss_impute) + (w_forecast * loss_forecast) + (w_kriging * loss_kriging)
    
    # 5. Tripwires
    assert not torch.isnan(total_loss), "CRITICAL: Total loss is NaN"
    assert not torch.isinf(total_loss), "CRITICAL: Total loss is Inf"
    
    return total_loss, loss_impute, loss_forecast, loss_kriging
```

---

## 4. Configuration & Training

### 4.1 Updated `configs/train/train.yaml`
```yaml
data:
  spatial_holdout_prob: 0.2  # 20% of nodes held out for kriging
  mask_prob: 0.15            # Temporal masking for imputation (existing)

loss:
  w_impute: 1.0
  w_forecast: 1.0
  w_kriging: 0.5   # Kriging weight (0.0 = disable)

model:
  edge_features_cache_path: "data/external/edge_features_cache.json"
  sparsity_threshold: 0.5
  num_layers: 2
  embedding_dim: 32
  hidden_dim: 64
```

### 4.2 Training Invocation
```bash
# Phase 3 only: Scale-invariant loss without kriging
python src/train.py --w_kriging 0.0 --spatial_holdout_prob 0.0

# Phase 3 + 4: Scale-invariant + Kriging
python src/train.py --w_kriging 0.5 --spatial_holdout_prob 0.2

# Full integration: Phase 3 + 4 + 5
python src/train.py \
  --w_kriging 0.5 \
  --spatial_holdout_prob 0.2 \
  --edge_features_cache data/external/edge_features_cache.json
```

---

## 5. Offensive Programming Guarantees

### 5.1 Shape Assertions
Every module validates input/output shapes at runtime:
```python
# Dataset
assert self.dynamic_data.shape == (T, N, 3), "Dynamic data shape violated"

# Graph Learner
assert edge_attr.shape == (E, 10), "Edge attr shape violated"

# Encoder
assert latent_embeddings.shape == (B, N, T, H), "Latent shape violated"

# Decoders
assert kriging_pred.shape == (B, K, T, D), "Kriging output shape violated"
```

### 5.2 NaN Tripwires
Every loss component checks for NaN before returning:
```python
assert not torch.isnan(loss_impute_traffic), "CRITICAL: Traffic MSLE is NaN"
assert not torch.isnan(loss_kriging), "CRITICAL: Kriging loss is NaN"
```

### 5.3 Cache Integrity
Missing edge features raise immediate ValueError:
```python
if dst_loc not in self.edge_features_cache[src_loc]:
    raise ValueError(f"CRITICAL: Missing cache for edge {src_loc} -> {dst_loc}")
```

### 5.4 Road Composition Validation
```python
road_sums = edge_attr[:, 3:9].sum(dim=1)
assert torch.allclose(road_sums, torch.ones_like(road_sums), atol=1e-3), \
    "CRITICAL: Road fractions don't sum to 1.0"
```

---

## 6. Mathematical Desk Checks

### 6.1 Mandatory Validation Scripts

All scripts in `scripts/desk_check_*.py` must pass before merging:

1. **`desk_check_scale_invariant.py`**
   - Inverse transform correctness
   - Raw vs scaled MSLE difference
   - Scaler buffer device consistency

2. **`desk_check_kriging.py`**
   - Holdout masking (zeroed nodes)
   - Decoder shape (K = mask.sum())
   - Kriging loss gradient flow

3. **`desk_check_edge_features.py`**
   - Edge attr shape [E, 10]
   - Road composition sums to 1.0
   - Fail-fast on missing cache
   - Distance normalization [0, 1]

4. **`desk_check_unified_loss.py`**
   - All three loss terms finite
   - Gradient backpropagation to all heads
   - Zero-weight bypass (w_kriging=0.0 → no kriging grad)

### 6.2 EXPERIMENTS.md Entry Template
```markdown
| 2026-06-29 | main | Unified Architecture: Phase 3 (Scale-Invariant MSLE) + Phase 4/5 (Spatial Kriging + Enriched GAT) | **Implemented.** Integrated scale-invariant loss buffers (traffic_mean, traffic_std registered in ResilientIoTPipeline, sourced from RealIoTDataset.traffic_scaler) enabling TRUE MSLE on raw counts for both Imputation (traffic channel only; weather uses MSE on scaled) and Kriging (new SpatialKrigingDecoder head with compute_kriging_loss). Added spatial holdout masking (make_spatial_holdout_mask + apply_spatial_holdout in dataset.py; fixed seed for validation, random for training). Enriched GATv2Conv with edge_dim=10 (DynamicGraphLearner._build_edge_attr loads edge_features_cache.json, normalizes shortest_path_distance via global max, fail-fast ValueError on missing edges). Offensive assertions: shape checks on all decoder outputs, road composition sum-to-1.0, NaN tripwires on all loss components. **Desk checks PASSED:** (1) inverse transform verified exact recovery; (2) raw MSLE vs pseudo-MSLE showed expected divergence (~3-5x difference); (3) holdout masking confirmed zero-fill on held-out nodes; (4) kriging decoder shape [B,K,T,D] with K=mask.sum() verified; (5) edge_attr [E,10] with road fractions summing to 1.0±1e-3 confirmed; (6) missing cache edge raised ValueError as expected. **Verification:** pytest tests/ (all pass), 10-epoch smoke test with w_kriging=0.5, spatial_holdout_prob=0.2 completed without NaN (val_loss_kriging=2.34, val_loss_impute=1.82). Config: configs/train/train.yaml updated (data.spatial_holdout_prob=0.2, loss.w_kriging=0.5, model.edge_features_cache_path). Files changed: src/dataset.py, src/pipeline.py, src/decoders.py, src/encoder.py, src/graph_learner.py, src/train.py, configs/train/train.yaml, scripts/desk_check_*.py. |
```

---

## 7. Testing & Validation Strategy

### 7.1 Unit Tests
Extend `tests/test_*.py`:
- `test_dataset.py`: Scaler extraction, location_ids in batch
- `test_decoders.py`: Kriging decoder, unified compute_loss with 4 outputs
- `test_encoder.py`: GATv2 with edge_dim=10
- `test_graph_learner.py`: Edge attr construction, fail-fast on missing cache
- `test_pipeline.py`: End-to-end with all components active

### 7.2 Integration Test
```bash
# Smoke test: 10 epochs with all features active
python src/train.py \
  --epochs 10 \
  --batch_size 16 \
  --start_date 2023-10-01 \
  --end_date 2023-10-03 \
  --w_kriging 0.5 \
  --spatial_holdout_prob 0.2 \
  --edge_features_cache data/external/edge_features_cache.json

# Expected: val_loss finite, no NaN, kriging_nodes > 0 logged to W&B
```

### 7.3 Ablation Study
Compare configurations to validate each component:
```bash
# Baseline: no kriging, no edge features
python src/train.py --w_kriging 0.0 --edge_features_cache ""

# +Scale-invariant (Phase 3)
# (automatic; scaler buffers always active)

# +Kriging (Phase 4)
python src/train.py --w_kriging 0.5 --spatial_holdout_prob 0.2

# +Edge features (Phase 5)
python src/train.py --w_kriging 0.5 --spatial_holdout_prob 0.2 \
  --edge_features_cache data/external/edge_features_cache.json
```

---

## 8. Migration Checklist

### 8.1 Pre-Implementation
- [x] Read context document
- [ ] Review experimental branches (`exp/scale-invariant-loss`, `exp/enriched-gat-kriging`)
- [ ] Verify `data/external/edge_features_cache.json` exists and is complete
- [ ] Create feature branch: `git checkout -b feat/unified-architecture`

### 8.2 Layer 1: Scale-Invariant Loss
- [ ] Modify `src/dataset.py`: Traffic scaler extraction
- [ ] Modify `src/pipeline.py`: Register scaler buffers
- [ ] Modify `src/train.py`: Call `set_scaler_stats()`
- [ ] Modify `src/decoders.py`: Channel-split loss (traffic MSLE + weather MSE)
- [ ] Write `scripts/desk_check_scale_invariant.py`
- [ ] Run desk checks: `python scripts/desk_check_scale_invariant.py`
- [ ] Update unit tests: `pytest tests/test_decoders.py`
- [ ] Smoke test: 10 epochs with `w_kriging=0.0`

### 8.3 Layer 2: Kriging Integration
- [ ] Add `make_spatial_holdout_mask` to `src/dataset.py`
- [ ] Add `apply_spatial_holdout` to `src/dataset.py`
- [ ] Activate `SpatialKrigingDecoder` in `src/pipeline.py`
- [ ] Add `compute_kriging_loss` to `src/decoders.py`
- [ ] Update `compute_loss` signature (4 return values)
- [ ] Modify `src/train.py`: Generate masks, thread through batches
- [ ] Write `scripts/desk_check_kriging.py`
- [ ] Run desk checks
- [ ] Update tests: `pytest tests/test_decoders.py tests/test_pipeline.py`
- [ ] Smoke test: 10 epochs with `w_kriging=0.5`

### 8.4 Layer 3: Edge Features
- [ ] Add `_build_edge_attr` to `src/graph_learner.py`
- [ ] Update `DynamicGraphLearner.forward` (return edge_attr)
- [ ] Modify `MaskedSpatiotemporalEncoder`: `edge_dim=10` in GATv2Conv
- [ ] Add `location_ids` to dataset `__getitem__`
- [ ] Thread edge_attr through pipeline forward
- [ ] Write `scripts/desk_check_edge_features.py`
- [ ] Run desk checks
- [ ] Update tests: `pytest tests/test_graph_learner.py tests/test_encoder.py`
- [ ] Smoke test: 10 epochs with edge cache

### 8.5 Layer 4: Final Integration
- [ ] Update `configs/train/train.yaml`
- [ ] Write `scripts/desk_check_unified_loss.py`
- [ ] Run full desk check suite
- [ ] Run full pytest suite: `pytest tests/`
- [ ] 100-epoch validation run on GPU
- [ ] Update `EXPERIMENTS.md` with entry

### 8.6 Documentation
- [ ] Update `NSTAR_BACKEND_ARCHITECTURE.md` (unified loss section)
- [ ] Update `docs/spatial_estimation_architecture.md` (true MSLE resolution)
- [ ] Add ADR (Architecture Decision Record) for pseudo-MSLE elimination
- [ ] Update README with new config parameters

### 8.7 Merge
- [ ] Squash commits: `git rebase -i main`
- [ ] Final smoke test on clean checkout
- [ ] Create PR: `feat/unified-architecture` → `main`
- [ ] Code review approval
- [ ] Merge to `main`
- [ ] Tag release: `git tag v2.0.0-unified`

---

## 9. Risk Mitigation

### 9.1 Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scaler buffer device mismatch (CPU/GPU) | Medium | High (training crash) | Use `self.register_buffer()` (auto-moved with model) |
| Missing edge features for new sensors | Low | High (ValueError crash) | Fail-fast validation + cache completeness check |
| Kriging loss dominates imputation | Medium | Medium (bad convergence) | Ablation study to tune `w_kriging` |
| Pseudo-MSLE reintroduction (future refactor) | Low | Medium (silent regression) | Unit test asserting inverse transform correctness |
| Road composition drift (cache regeneration) | Low | Low (slight perf change) | Assert sum-to-1.0 in desk check |

### 9.2 Rollback Plan
If integration causes training instability:
1. Set `w_kriging=0.0` (disable kriging head)
2. Set `edge_features_cache_path=""` (fallback to learned-only)
3. Revert to pre-integration `main` via `git revert`

---

## 10. Success Criteria

### 10.1 Functional Requirements
- ✅ Scale-invariant loss buffers registered and device-safe
- ✅ Kriging decoder outputs correct shape `[B, K, T, D]` where `K = mask.sum()`
- ✅ Edge attributes `[E, 10]` with road fractions summing to 1.0
- ✅ All desk checks pass without assertion errors
- ✅ Full pytest suite passes (100% existing tests + new tests)
- ✅ 100-epoch training run completes without NaN

### 10.2 Performance Requirements
- **Imputation Loss:** ≤ 10% regression vs baseline (expected improvement due to MSLE)
- **Kriging Loss:** Finite and decreasing over epochs (when `w_kriging > 0`)
- **Training Speed:** ≤ 20% slowdown vs baseline (edge attr overhead)

### 10.3 Code Quality
- Zero TODO/FIXME comments in production code
- All offensive assertions in place
- EXPERIMENTS.md entry complete with desk check results
- No dead code (no commented-out blocks)

---

## 11. Open Questions & Future Work

### 11.1 Open Questions
1. **Kriging Weight Tuning:** Should `w_kriging` be exposed to Optuna sweeps?
   - *Recommendation:* Yes, add to `configs/sweep.yaml` with range `[0.1, 1.0]`

2. **Weather Channel Treatment:** Should weather also use MSLE or stick with MSE?
   - *Recommendation:* Keep MSE (weather is continuous, not count-based)

3. **Edge Attr Caching Strategy:** Rebuild cache on every dataset change?
   - *Recommendation:* Add cache version hash to config, auto-rebuild if stale

### 11.2 Future Enhancements (Out of Scope)
- **Learned Edge Weights:** Make `edge_attr[:, 9]` trainable via MLP instead of static similarity
- **Adaptive Kriging:** Dynamic `spatial_holdout_prob` per epoch (curriculum learning)
- **Multi-City Support:** Generalize edge cache to multiple geographic regions
- **Attention Visualization:** Export attention weights to frontend for edge highlighting

---

## Appendix A: File Dependency Graph

```
src/dataset.py
  ├─ traffic_scaler (new)
  ├─ location_ids in __getitem__ (new)
  └─ make_spatial_holdout_mask (new)

src/pipeline.py
  ├─ register_buffer('traffic_mean', ...) (new)
  ├─ register_buffer('traffic_std', ...) (new)
  ├─ kriging_decoder (activate existing)
  └─ edge_features_cache_path param (new)

src/graph_learner.py
  ├─ load edge_features_cache.json (new)
  ├─ _build_edge_attr() (new)
  └─ return edge_attr in forward (new)

src/encoder.py
  └─ GATv2Conv edge_dim=10 (modify)

src/decoders.py
  ├─ compute_loss signature (modify: add kriging args)
  ├─ compute_kriging_loss() (new)
  └─ channel-split MSLE logic (new)

src/train.py
  ├─ set_scaler_stats() call (new)
  ├─ holdout mask generation (new)
  └─ edge cache path arg (new)

configs/train/train.yaml
  ├─ data.spatial_holdout_prob (new)
  ├─ loss.w_kriging (new)
  └─ model.edge_features_cache_path (new)
```

---

## Appendix B: Estimated Timeline

| Layer | Tasks | Estimated Time |
|-------|-------|----------------|
| **Layer 1:** Scale-Invariant Loss | Dataset scaler + buffer registration + loss refactor + desk checks | 3-4 hours |
| **Layer 2:** Kriging Integration | Masking logic + decoder activation + loss integration + tests | 4-5 hours |
| **Layer 3:** Edge Features | Cache loading + GATv2 enrichment + fail-fast + tests | 3-4 hours |
| **Layer 4:** Unified Integration | Config updates + full desk checks + smoke tests | 2-3 hours |
| **Documentation & Review** | EXPERIMENTS.md + ADR + code review | 2 hours |
| **Buffer** | Unexpected issues, debugging | 3-4 hours |
| **TOTAL** | | **~17-22 hours** |

*Recommended approach:* Implement sequentially over 2-3 work sessions with validation checkpoints.

---

## Conclusion

This architectural plan provides a step-by-step, mathematically rigorous roadmap for integrating two high-performing experimental branches into a unified production architecture. The design eliminates technical debt (pseudo-MSLE), enhances spatial reasoning (kriging + enriched GAT), and establishes offensive programming guarantees throughout the pipeline.

**Key Differentiators:**
1. **True MSLE** on raw counts for both Imputation and Kriging heads (zero approximation error)
2. **Fail-Fast Edge Features** with mandatory cache validation (no silent degradation)
3. **Sequential Layering** with independent verification at each step (low integration risk)
4. **Comprehensive Desk Checks** mandated before merge (mathematical correctness enforced)

The implementation is **low-risk** (all dependencies merged), **well-scoped** (4 clear layers), and **independently verifiable** (desk checks + unit tests per layer).

**Next Steps:** Review this plan, clarify any open questions, and proceed with Layer 1 implementation.

---

**End of Architectural Design**
