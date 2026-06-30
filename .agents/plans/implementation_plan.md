# PHASE 1 IMPLEMENTATION PLAN: Scale-Invariant Loss
## Date: 2026-06-30
## Implementer: Claude Sonnet 4.5
## Architect Plan Reference: `.agents/plans/architect_plan.md` Section 3.1

---

## PHASE 1 OBJECTIVE
Implement scale-invariant loss infrastructure by:
1. Extracting scaler statistics (mean, std) from the traffic channel in `RealIoTDataset`
2. Registering these statistics as PyTorch buffers in `ResilientIoTPipeline`
3. Modifying `compute_loss()` to compute TRUE MSLE on unscaled traffic counts (inverse transform)
4. Computing MSE on scaled weather variables (channels 1-2)
5. Validating correctness via desk checks and database smoke test

---

## FILE-BY-FILE IMPLEMENTATION PLAN

### FILE 1: `src/dataset.py`

#### Current State Analysis
- Lines 1-99: Dataset initialization loads and filters data
- StandardScaler is already imported (line 5)
- Traffic counts are currently scaled together with all features in a single scaler

#### Required Changes

**Change 1.1: Separate Traffic Channel Scaler (lines ~100-200 area)**

**Location:** In `__init__` method, after data filtering and before creating the final feature tensor

**Action:** 
1. Split traffic data from weather data BEFORE scaling
2. Create dedicated `self.traffic_scaler` for traffic channel only
3. Create separate scaler for weather channels
4. Expose `self.traffic_mean` and `self.traffic_std` as instance attributes
5. Stack channels back together maintaining [T, N, 3] format

**Pseudocode:**
```python
# After line ~150 where df_pivot is created
# BEFORE applying StandardScaler to all features

# Extract traffic channel (dim 0) separately
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

# OFFENSIVE ASSERTION
assert self.traffic_std > 0, "CRITICAL: Traffic std is zero (constant values)"
assert not np.isnan(self.traffic_mean), "CRITICAL: Traffic mean is NaN"
assert not np.isnan(self.traffic_std), "CRITICAL: Traffic std is NaN"

# Weather channels scaled separately
weather_flat = weather_data.reshape(-1, 2)
self.weather_scaler = StandardScaler()
weather_scaled_flat = self.weather_scaler.fit_transform(weather_flat)
weather_scaled = weather_scaled_flat.reshape(T, N, 2)

# Stack: [T, N, 3] where dim 0=traffic, dim 1-2=weather
self.dynamic_data = np.stack([
    traffic_scaled, 
    weather_scaled[:, :, 0], 
    weather_scaled[:, :, 1]
], axis=-1)

logger.info(f"Traffic scaler: mean={self.traffic_mean:.2f}, std={self.traffic_std:.2f}")
```

**Expected Impact:**
- Adds 3 new instance attributes: `self.traffic_scaler`, `self.traffic_mean`, `self.traffic_std`
- Changes how `self.dynamic_data` is constructed (now from two separate scalers)
- No change to output shape: still [T, N, 3]

**Risk Assessment:** 🟡 MEDIUM
- Risk: Breaking existing data flow if weather scaling is implemented differently elsewhere
- Mitigation: Verify no other code depends on a unified scaler object
- Rollback: Revert to single StandardScaler for all channels

---

### FILE 2: `src/pipeline.py`

#### Current State Analysis
- Lines 14-66: `ResilientIoTPipeline.__init__()` initializes graph learner, encoder, decoders
- No scaler buffers currently exist
- Forward pass (lines 68-99+) computes loss via `compute_loss()` function

#### Required Changes

**Change 2.1: Register Scaler Buffers (in `__init__`)**

**Location:** After line 66 (after decoder initialization), before the end of `__init__`

**Action:**
1. Add two buffer registrations for scaler statistics
2. Initialize to sentinel values (0.0 for mean, 1.0 for std)
3. Create `set_scaler_stats()` method to populate buffers from dataset

**Code to Add:**
```python
# In __init__, after self.decoders initialization (after line 66)

# Scaler buffers (initialized to sentinel values, set by train.py)
self.register_buffer('traffic_mean', torch.tensor(0.0))
self.register_buffer('traffic_std', torch.tensor(1.0))

logger.info("Registered traffic scaler buffers (traffic_mean, traffic_std)")
```

**Change 2.2: Add `set_scaler_stats()` Method**

**Location:** After `__init__`, before `forward()` method

**Code to Add:**
```python
def set_scaler_stats(self, traffic_mean: float, traffic_std: float):
    """
    Called by train.py after dataset initialization to populate scaler buffers.
    
    Args:
        traffic_mean: Mean of traffic channel from dataset scaler
        traffic_std: Standard deviation of traffic channel from dataset scaler
    """
    self.traffic_mean.fill_(traffic_mean)
    self.traffic_std.fill_(traffic_std)
    
    # OFFENSIVE ASSERTION
    assert traffic_std > 0, f"CRITICAL: Cannot set traffic_std to {traffic_std} (must be positive)"
    assert not np.isnan(traffic_mean), "CRITICAL: Cannot set traffic_mean to NaN"
    assert not np.isnan(traffic_std), "CRITICAL: Cannot set traffic_std to NaN"
    
    logger.info(f"Scaler buffers set: mean={traffic_mean:.4f}, std={traffic_std:.4f}, device={self.traffic_mean.device}")
```

**Change 2.3: Update `forward()` to Pass Buffers to `compute_loss()`**

**Location:** In `forward()` method, where `compute_loss()` is called (currently around line 110+)

**Action:** Add `traffic_mean` and `traffic_std` as arguments to `compute_loss()`

**Current Code Pattern:**
```python
total_loss, loss_impute, loss_forecast = compute_loss(
    imputed_features=imputed,
    target_impute=target_impute,
    mu=mu, theta=theta, pi=pi,
    target_forecast=target_forecast,
    w_impute=...,
    w_forecast=...
)
```

**New Code Pattern:**
```python
total_loss, loss_impute, loss_forecast = compute_loss(
    imputed_features=imputed,
    target_impute=target_impute,
    mu=mu, theta=theta, pi=pi,
    target_forecast=target_forecast,
    traffic_mean=self.traffic_mean,  # NEW
    traffic_std=self.traffic_std,    # NEW
    w_impute=batch.get('w_impute', 1.0),
    w_forecast=batch.get('w_forecast', 1.0)
)
```

**Expected Impact:**
- Adds 2 new buffers to pipeline state_dict
- Buffers automatically move to GPU when `pipeline.to(device)` is called
- No change to forward pass logic until loss computation

**Risk Assessment:** 🟢 LOW
- Risk: Buffers not initialized before first forward pass
- Mitigation: Add assertion in forward() to check buffers are set
- Rollback: Remove buffer registration and method

---

### FILE 3: `src/decoders.py`

#### Current State Analysis
- Lines 86-150+: `compute_loss()` function signature and implementation
- Currently computes MSE on imputed features (all channels together)
- ZINB NLL for forecasting

#### Required Changes

**Change 3.1: Update `compute_loss()` Signature**

**Location:** Line ~115+ where `compute_loss()` is defined

**Current Signature:**
```python
def compute_loss(
    imputed_features: torch.Tensor,  # [B, N, T, 3]
    target_impute: torch.Tensor,     # [B, N, T, 3]
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    target_forecast: torch.Tensor,
    w_impute: float = 1.0,
    w_forecast: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**New Signature:**
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
    w_forecast: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**Change 3.2: Implement Channel-Split Loss Computation**

**Location:** Inside `compute_loss()`, replace the current imputation loss computation

**Current Pattern (approximate):**
```python
# Current: MSE on all channels together
loss_impute = F.mse_loss(imputed_features, target_impute)
```

**New Implementation:**
```python
"""
Joint Loss with Channel-Specific Treatment:
- Traffic (channel 0): MSLE on raw counts (inverse transform)
- Weather (channels 1-2): MSE on scaled features
- Forecasting: ZINB NLL on raw counts
"""

# --- 1. Imputation Loss: Split Traffic vs Weather ---

# Extract traffic channel (dim 0)
traffic_pred_scaled = imputed_features[:, :, :, 0]      # [B, N, T]
traffic_target_scaled = target_impute[:, :, :, 0]       # [B, N, T]

# Inverse transform to raw counts
traffic_pred_raw = (traffic_pred_scaled * traffic_std) + traffic_mean
traffic_target_raw = (traffic_target_scaled * traffic_std) + traffic_mean

# Clamp to non-negative (count domain)
traffic_pred_raw = torch.clamp(traffic_pred_raw, min=0.0)
traffic_target_raw = torch.clamp(traffic_target_raw, min=0.0)

# Compute TRUE MSLE on raw counts
loss_impute_traffic = scale_invariant_mse(traffic_pred_raw, traffic_target_raw)

# Weather channels: MSE on scaled features (no inverse transform)
weather_pred = imputed_features[:, :, :, 1:]      # [B, N, T, 2]
weather_target = target_impute[:, :, :, 1:]       # [B, N, T, 2]
loss_impute_weather = F.mse_loss(weather_pred, weather_target)

# Combined imputation loss
loss_impute = loss_impute_traffic + loss_impute_weather

# --- 2. Forecasting Loss (ZINB NLL) - UNCHANGED ---
loss_forecast = compute_zinb_nll(target_forecast, mu, theta, pi)

# --- 3. Total Weighted Loss ---
total_loss = (w_impute * loss_impute) + (w_forecast * loss_forecast)

# --- 4. OFFENSIVE ASSERTIONS (Tripwires) ---
assert not torch.isnan(loss_impute_traffic), "CRITICAL: Traffic MSLE is NaN"
assert not torch.isnan(loss_impute_weather), "CRITICAL: Weather MSE is NaN"
assert not torch.isnan(loss_forecast), "CRITICAL: Forecast loss is NaN"
assert not torch.isnan(total_loss), "CRITICAL: Total loss is NaN"
assert not torch.isinf(total_loss), "CRITICAL: Total loss is Inf"

return total_loss, loss_impute, loss_forecast
```

**Change 3.3: Verify `scale_invariant_mse()` Exists**

**Location:** Check if function exists in `decoders.py`, add if missing

**Function to Add (if not present):**
```python
def scale_invariant_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Logarithmic Error (MSLE) for scale-invariant loss.
    
    MSLE = mean((log(1 + pred) - log(1 + target))^2)
    
    Args:
        pred: Predicted raw counts [B, N, T] or any shape
        target: Target raw counts [same shape as pred]
    
    Returns:
        Scalar loss value
    """
    # log1p is numerically stable for log(1 + x)
    log_pred = torch.log1p(pred)
    log_target = torch.log1p(target)
    
    # MSE in log space
    mse = F.mse_loss(log_pred, log_target)
    
    return mse
```

**Expected Impact:**
- Changes imputation loss computation from single MSE to split (MSLE + MSE)
- Adds inverse transform math for traffic channel
- Preserves existing ZINB forecasting loss
- Return signature unchanged (still 3 tensors)

**Risk Assessment:** 🟡 MEDIUM
- Risk: Inverse transform math error causing NaN loss
- Mitigation: Desk check script validates transform correctness
- Rollback: Revert to unified MSE on all channels

---

### FILE 4: `src/train.py`

#### Current State Analysis
- Lines 0-98: Argument parsing, config loading, mock data utilities
- Main training loop starts around line 100+
- Dataset initialization happens before pipeline creation

#### Required Changes

**Change 4.1: Extract Scaler Stats from Dataset**

**Location:** After dataset initialization, before pipeline creation (around line 200-250 area)

**Action:** 
1. Extract `traffic_mean` and `traffic_std` from train dataset
2. Validate against validation dataset statistics
3. Call `pipeline.set_scaler_stats()` after pipeline creation

**Code Pattern to Find:**
```python
# Current pattern (approximate)
train_dataset = RealIoTDataset(...)
val_dataset = RealIoTDataset(...)

pipeline = ResilientIoTPipeline(...)
pipeline = pipeline.to(device)
```

**Code to Add:**
```python
# After dataset initialization
train_dataset = RealIoTDataset(...)
val_dataset = RealIoTDataset(...)

# Extract and register scaler stats
traffic_mean = train_dataset.traffic_mean
traffic_std = train_dataset.traffic_std

logger.info(f"Extracted scaler stats from train_dataset: mean={traffic_mean:.4f}, std={traffic_std:.4f}")

# Create pipeline
pipeline = ResilientIoTPipeline(...)
pipeline = pipeline.to(device)

# Set scaler buffers AFTER moving to device
pipeline.set_scaler_stats(traffic_mean, traffic_std)

# DEFENSIVE CHECK: Validation set should have similar statistics
val_mean_diff = abs(val_dataset.traffic_mean - traffic_mean)
val_std_diff = abs(val_dataset.traffic_std - traffic_std)
relative_mean_diff = val_mean_diff / traffic_mean if traffic_mean != 0 else float('inf')
relative_std_diff = val_std_diff / traffic_std if traffic_std != 0 else float('inf')

if relative_mean_diff > 0.2 or relative_std_diff > 0.2:
    logger.warning(
        f"Train/val scaler mismatch detected: "
        f"Δmean={val_mean_diff:.2f} ({relative_mean_diff*100:.1f}%), "
        f"Δstd={val_std_diff:.2f} ({relative_std_diff*100:.1f}%)"
    )
else:
    logger.info(
        f"Train/val scaler stats are consistent: "
        f"Δmean={val_mean_diff:.4f}, Δstd={val_std_diff:.4f}"
    )
```

**Expected Impact:**
- Adds scaler stats extraction before pipeline creation
- Populates buffers with real statistics from data
- Logs warning if train/val distributions are very different

**Risk Assessment:** 🟢 LOW
- Risk: Dataset doesn't have attributes yet (if Change 1.1 not applied)
- Mitigation: Sequential implementation (dataset first, then train.py)
- Rollback: Remove stats extraction code

---

## VERIFICATION PLAN

### Desk Check Script: `scripts/desk_check_scale_invariant.py`

**Purpose:** Mathematically validate inverse transform and MSLE correctness

**Script Content:**
```python
#!/usr/bin/env python3
"""
Desk Check: Scale-Invariant Loss (Phase 1, Layer 1)
Validates:
1. Inverse transform correctly recovers original scale
2. MSLE differs significantly between raw and pseudo-scaled inputs
3. Scaler buffer device consistency
"""
import torch
import torch.nn.functional as F
import sys

def scale_invariant_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Copy of production function for testing"""
    log_pred = torch.log1p(pred)
    log_target = torch.log1p(target)
    return F.mse_loss(log_pred, log_target)

def test_inverse_transform():
    """Verify inverse transform recovers original scale."""
    print("\n=== TEST 1: Inverse Transform Correctness ===")
    
    # Real-world traffic counts (vehicles per 15min)
    traffic_raw = torch.tensor([0.0, 10.0, 100.0, 1000.0, 5000.0])
    mean, std = 500.0, 1000.0  # Realistic scaler stats
    
    print(f"Original raw counts: {traffic_raw.tolist()}")
    print(f"Scaler: mean={mean}, std={std}")
    
    # Forward transform (StandardScaler)
    traffic_scaled = (traffic_raw - mean) / std
    print(f"After StandardScaler: {traffic_scaled.tolist()}")
    
    # Inverse transform (our implementation)
    traffic_recovered = (traffic_scaled * std) + mean
    print(f"After inverse transform: {traffic_recovered.tolist()}")
    
    # Validation
    max_error = torch.abs(traffic_raw - traffic_recovered).max().item()
    assert torch.allclose(traffic_raw, traffic_recovered, atol=1e-5), \
        f"Inverse transform failed: max error {max_error}"
    
    print("✓ Inverse transform verified (max error < 1e-5)")
    return True

def test_msle_raw_vs_scaled():
    """Demonstrate MSLE sensitivity to raw vs scaled inputs."""
    print("\n=== TEST 2: Raw MSLE vs Pseudo-MSLE Difference ===")
    
    # Realistic prediction errors (50% off)
    pred_raw = torch.tensor([5.0, 50.0, 500.0, 5000.0])
    target_raw = torch.tensor([10.0, 100.0, 1000.0, 10000.0])
    
    mean, std = 500.0, 1000.0
    
    # Scale predictions
    pred_scaled = (pred_raw - mean) / std
    target_scaled = (target_raw - mean) / std
    
    print(f"Raw predictions: {pred_raw.tolist()}")
    print(f"Raw targets: {target_raw.tolist()}")
    print(f"Scaled predictions: {pred_scaled.tolist()}")
    print(f"Scaled targets: {target_scaled.tolist()}")
    
    # Compute MSLE on raw counts (TRUE MSLE)
    msle_raw = scale_invariant_mse(pred_raw, target_raw)
    
    # Compute pseudo-MSLE on scaled (with clamp for negative values)
    pred_scaled_clamped = torch.clamp(pred_scaled, min=0.0)
    target_scaled_clamped = torch.clamp(target_scaled, min=0.0)
    msle_pseudo = scale_invariant_mse(pred_scaled_clamped, target_scaled_clamped)
    
    print(f"\nTrue MSLE (on raw counts): {msle_raw:.6f}")
    print(f"Pseudo-MSLE (on scaled, clamped): {msle_pseudo:.6f}")
    print(f"Ratio (pseudo / true): {msle_pseudo / msle_raw:.2f}x")
    
    # Scaled version will be incorrect due to negative values being clamped
    # and log1p behavior changing
    assert abs(msle_raw.item() - msle_pseudo.item()) > 0.01, \
        "Expected significant difference between raw and pseudo MSLE"
    
    print("✓ Raw MSLE vs pseudo-MSLE difference confirmed (significant divergence)")
    return True

def test_device_consistency():
    """Verify buffers move to GPU with model"""
    print("\n=== TEST 3: Device Consistency ===")
    
    # Simulate buffer registration
    traffic_mean = torch.tensor(500.0)
    traffic_std = torch.tensor(1000.0)
    
    print(f"Initial device: {traffic_mean.device}")
    
    if torch.cuda.is_available():
        traffic_mean_gpu = traffic_mean.to('cuda')
        traffic_std_gpu = traffic_std.to('cuda')
        print(f"After .to('cuda'): {traffic_mean_gpu.device}")
        
        # Simulate inverse transform on GPU
        scaled = torch.tensor([-0.5, 0.0, 0.5]).to('cuda')
        raw = (scaled * traffic_std_gpu) + traffic_mean_gpu
        
        assert raw.device.type == 'cuda', "Device mismatch after transform"
        print(f"Transform output device: {raw.device}")
        print("✓ GPU device consistency verified")
    else:
        print("⚠ CUDA not available, skipping GPU test")
    
    return True

def test_extreme_values():
    """Test MSLE behavior with edge cases"""
    print("\n=== TEST 4: Extreme Values ===")
    
    # Edge cases
    cases = [
        ("Zero counts", torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])),
        ("Very large counts", torch.tensor([1e6, 1e7, 1e8]), torch.tensor([1e6, 1e7, 1e8])),
        ("Mixed zeros and large", torch.tensor([0.0, 100.0, 10000.0]), torch.tensor([1.0, 200.0, 20000.0]))
    ]
    
    for name, pred, target in cases:
        msle = scale_invariant_mse(pred, target)
        print(f"{name}: MSLE = {msle:.6f}")
        
        assert not torch.isnan(msle), f"NaN loss for {name}"
        assert not torch.isinf(msle), f"Inf loss for {name}"
    
    print("✓ All extreme value cases produced finite loss")
    return True

def main():
    print("=" * 60)
    print("PHASE 1 DESK CHECK: Scale-Invariant Loss")
    print("=" * 60)
    
    try:
        test_inverse_transform()
        test_msle_raw_vs_scaled()
        test_device_consistency()
        test_extreme_values()
        
        print("\n" + "=" * 60)
        print("✅ ALL DESK CHECKS PASSED")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ DESK CHECK FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

### Unit Test Updates

**File: `tests/test_decoders.py`**

Add test cases:
1. `test_compute_loss_channel_split()` - Verify traffic uses MSLE, weather uses MSE
2. `test_inverse_transform_in_loss()` - Check scaler buffers are used correctly
3. `test_loss_nan_tripwires()` - Verify assertions catch NaN

### Database Smoke Test

**Command:**
```bash
docker compose exec agent-workspace python src/train.py \
    --use-database \
    --config configs/train/train_test.yaml
```

**Success Criteria:**
1. 1 epoch completes without errors
2. No NaN losses logged
3. Loss values are finite and positive
4. Log shows scaler stats: `mean=XXX, std=XXX`
5. Train/val scaler consistency check passes

---

## IMPLEMENTATION SEQUENCE

### Step 1: Git Safety Commit
```bash
git add .
git commit -m "Auto-commit before Phase 1: Scale-Invariant Loss"
```

### Step 2: Implement Changes (Sequential Order)
1. **FILE 1:** `src/dataset.py` - Change 1.1 (scaler separation)
2. **FILE 2:** `src/pipeline.py` - Changes 2.1, 2.2, 2.3 (buffers + set method + forward update)
3. **FILE 3:** `src/decoders.py` - Changes 3.1, 3.2, 3.3 (loss signature + channel split + MSLE function)
4. **FILE 4:** `src/train.py` - Change 4.1 (stats extraction + set call)

### Step 3: Create Desk Check Script
```bash
# Create script
mkdir -p scripts
# Write desk_check_scale_invariant.py content above

# Run desk check
python scripts/desk_check_scale_invariant.py
```

### Step 4: Run Unit Tests
```bash
pytest tests/test_decoders.py -v
```

### Step 5: Database Smoke Test
```bash
docker compose exec agent-workspace python src/train.py \
    --use-database \
    --config configs/train/train_test.yaml
```

### Step 6: Verification Checklist
- [ ] Desk check script passes all 4 tests
- [ ] Unit tests pass
- [ ] Database smoke test completes 1 epoch without NaN
- [ ] Scaler stats logged correctly (mean ~XXX, std ~XXX based on data)
- [ ] Train/val scaler consistency check passes
- [ ] No Python syntax errors
- [ ] No import errors

---

## ROLLBACK PLAN

If Phase 1 fails verification:

1. **Immediate rollback:**
   ```bash
   git reset --hard HEAD~1
   ```

2. **Partial rollback (if specific file breaks):**
   - Restore individual file from git: `git checkout HEAD~1 -- src/dataset.py`
   - Fix issue
   - Re-run verification

3. **Debug mode:**
   - Add verbose logging to inverse transform
   - Print tensor shapes at each step
   - Verify scaler stats are reasonable for the dataset

---

## ESTIMATED EFFORT

- **File modifications:** 2-3 hours
- **Desk check script:** 1 hour  
- **Testing & debugging:** 1-2 hours
- **Total:** 4-6 hours

---

## APPROVAL CHECKPOINT

**STOP HERE AND WAIT FOR USER APPROVAL**

Before proceeding with code modifications, please confirm:

1. ✅ File-by-file plan is clear and complete
2. ✅ Offensive assertions are adequate
3. ✅ Desk check script covers critical math
4. ✅ Rollback plan is acceptable
5. ✅ Ready to proceed with implementation

**Reply with approval to begin Phase 1 implementation.**
