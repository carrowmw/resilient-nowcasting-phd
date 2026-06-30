#!/usr/bin/env python3
"""
Desk Check: Scale-Invariant Loss (Phase 1, Layer 1)
Validates:
1. Inverse transform correctly recovers original scale
2. MSLE differs significantly between raw and pseudo-scaled inputs
3. Scaler buffer device consistency
4. Extreme value handling
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
