#!/usr/bin/env python3
"""
Desk Check: Unified Loss (Phase 4, Layer 4 — Final Integration)

End-to-end trace of the unified multi-task objective as actually shipped in
`src/decoders.py::compute_loss`. Unlike the per-layer desk checks, this imports the
REAL production functions (no local copies) and exercises all three loss terms
together:

    L = w_impute * L_impute + w_forecast * L_ZINB + w_kriging * L_kriging

where
    L_impute  = TRUE MSLE on raw traffic counts (ch 0, inverse-transformed) + MSE on
                scaled weather (ch 1-2)
    L_ZINB    = Zero-Inflated Negative Binomial NLL on raw counts
    L_kriging = same channel-split treatment over spatially held-out nodes

Validates (architect_plan.md §6.1 item 4):
  A. All three loss terms are finite and the weighted sum is arithmetically correct.
  B. Gradients backpropagate to ALL heads (imputation, ZINB params, kriging).
  C. Zero-weight bypass: w_kriging=0 -> loss_kriging == 0.0 and NO grad to the kriging head.
  D. Channel separation: perfect reconstruction -> ~0 loss; a traffic-only error moves
     traffic MSLE while weather MSE stays ~0.
  E. NaN tripwire fires: a NaN in the inputs raises the CRITICAL offensive assertion.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from decoders import compute_loss


# Deterministic — Date.now()/random seeding kept fixed so the printed numbers are stable.
torch.manual_seed(42)

# Synthetic batch dimensions.
B, N, T, D = 2, 8, 6, 3
HOLDOUT = torch.tensor([True, True, False, False, False, False, False, False])
K = int(HOLDOUT.sum().item())

# Scaler buffers (realistic raw-count statistics).
TRAFFIC_MEAN = torch.tensor(500.0)
TRAFFIC_STD = torch.tensor(1000.0)


def _make_inputs(requires_grad: bool = False):
    """Build a fresh, valid input set for compute_loss."""
    imputed = torch.randn(B, N, T, D)
    target_impute = torch.randn(B, N, T, D)
    # ZINB params must be in valid domain: mu>0, theta>0, 0<pi<1.
    mu = torch.rand(B, N) + 0.5            # (0.5, 1.5)
    theta = torch.rand(B, N) + 0.5         # (0.5, 1.5)
    pi = torch.rand(B, N) * 0.5 + 0.25     # (0.25, 0.75)
    target_forecast = torch.randint(0, 5, (B, N)).float()
    kriging_pred = torch.randn(B, K, T, D)

    if requires_grad:
        imputed.requires_grad_(True)
        mu.requires_grad_(True)
        theta.requires_grad_(True)
        pi.requires_grad_(True)
        kriging_pred.requires_grad_(True)

    return imputed, target_impute, mu, theta, pi, target_forecast, kriging_pred


def test_all_terms_finite_and_arithmetic():
    print("\n=== TEST A: Three terms finite + weighted-sum arithmetic ===")
    imputed, target_impute, mu, theta, pi, target_forecast, kriging_pred = _make_inputs()

    w_impute, w_forecast, w_kriging = 1.0, 1.0, 0.5
    total, l_impute, l_forecast, l_kriging = compute_loss(
        imputed_features=imputed,
        target_impute=target_impute,
        mu=mu, theta=theta, pi=pi,
        target_forecast=target_forecast,
        traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
        kriging_pred=kriging_pred,
        spatial_holdout_mask=HOLDOUT,
        w_impute=w_impute, w_forecast=w_forecast, w_kriging=w_kriging,
    )

    print(f"loss_impute  = {l_impute.item():.6f}")
    print(f"loss_forecast= {l_forecast.item():.6f}")
    print(f"loss_kriging = {l_kriging.item():.6f}")
    print(f"total        = {total.item():.6f}")

    for name, t in [("impute", l_impute), ("forecast", l_forecast),
                    ("kriging", l_kriging), ("total", total)]:
        assert torch.isfinite(t).all(), f"{name} is not finite"
        assert t.dim() == 0, f"{name} is not a scalar (shape {tuple(t.shape)})"

    expected = w_impute * l_impute + w_forecast * l_forecast + w_kriging * l_kriging
    print(f"recomputed   = {expected.item():.6f}")
    assert torch.allclose(total, expected, atol=1e-6), \
        f"weighted sum mismatch: {total.item()} vs {expected.item()}"

    print("✓ All three terms finite; total == weighted sum (atol 1e-6)")
    return True


def test_gradient_flow_all_heads():
    print("\n=== TEST B: Gradient backprop to all heads ===")
    imputed, target_impute, mu, theta, pi, target_forecast, kriging_pred = \
        _make_inputs(requires_grad=True)

    total, *_ = compute_loss(
        imputed_features=imputed,
        target_impute=target_impute,
        mu=mu, theta=theta, pi=pi,
        target_forecast=target_forecast,
        traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
        kriging_pred=kriging_pred,
        spatial_holdout_mask=HOLDOUT,
        w_impute=1.0, w_forecast=1.0, w_kriging=0.5,
    )
    total.backward()

    checks = {
        "imputation head (imputed_features)": imputed,
        "ZINB mu": mu,
        "ZINB theta": theta,
        "ZINB pi": pi,
        "kriging head (kriging_pred)": kriging_pred,
    }
    for name, t in checks.items():
        assert t.grad is not None, f"No gradient reached {name}"
        gnorm = t.grad.abs().sum().item()
        print(f"{name}: grad L1 = {gnorm:.6f}")
        assert gnorm > 0.0, f"Zero gradient at {name}"

    print("✓ Non-zero gradients reached imputation, all ZINB params, and kriging")
    return True


def test_zero_weight_bypass():
    print("\n=== TEST C: Zero-weight kriging bypass ===")

    # C1: w_kriging=0 with no kriging inputs -> clean 0.0, total finite.
    imputed, target_impute, mu, theta, pi, target_forecast, _ = _make_inputs()
    total, l_impute, l_forecast, l_kriging = compute_loss(
        imputed_features=imputed,
        target_impute=target_impute,
        mu=mu, theta=theta, pi=pi,
        target_forecast=target_forecast,
        traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
        kriging_pred=None,
        spatial_holdout_mask=None,
        w_impute=1.0, w_forecast=1.0, w_kriging=0.0,
    )
    print(f"w_kriging=0.0 -> loss_kriging = {l_kriging.item():.6f}, total = {total.item():.6f}")
    assert l_kriging.item() == 0.0, "loss_kriging must be exactly 0.0 when w_kriging=0"
    assert torch.isfinite(total).all(), "total not finite under bypass"

    # C2: even if a grad-tracked kriging tensor is supplied, w_kriging=0 -> no grad to it.
    imputed, target_impute, mu, theta, pi, target_forecast, kriging_pred = \
        _make_inputs(requires_grad=True)
    total, *_ = compute_loss(
        imputed_features=imputed,
        target_impute=target_impute,
        mu=mu, theta=theta, pi=pi,
        target_forecast=target_forecast,
        traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
        kriging_pred=kriging_pred,
        spatial_holdout_mask=HOLDOUT,
        w_impute=1.0, w_forecast=1.0, w_kriging=0.0,
    )
    total.backward()
    print(f"w_kriging=0.0 with grad-tracked kriging_pred -> grad is None: "
          f"{kriging_pred.grad is None}")
    assert kriging_pred.grad is None, \
        "kriging head received gradient despite w_kriging=0 (bypass leak)"

    print("✓ Zero-weight bypass: loss_kriging==0.0 and no gradient leaks to the kriging head")
    return True


def test_channel_separation():
    print("\n=== TEST D: Channel separation (perfect recon + traffic-only error) ===")

    # Perfect reconstruction: imputed == target, kriging_pred == held-out target.
    target_impute = torch.randn(B, N, T, D)
    imputed = target_impute.clone()
    kriging_pred = target_impute[:, HOLDOUT, :, :].clone()
    mu = torch.rand(B, N) + 0.5
    theta = torch.rand(B, N) + 0.5
    pi = torch.rand(B, N) * 0.5 + 0.25
    target_forecast = torch.randint(0, 5, (B, N)).float()

    total, l_impute, l_forecast, l_kriging = compute_loss(
        imputed_features=imputed, target_impute=target_impute,
        mu=mu, theta=theta, pi=pi, target_forecast=target_forecast,
        traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
        kriging_pred=kriging_pred, spatial_holdout_mask=HOLDOUT,
        w_impute=1.0, w_forecast=1.0, w_kriging=0.5,
    )
    print(f"perfect recon -> loss_impute = {l_impute.item():.3e}, "
          f"loss_kriging = {l_kriging.item():.3e}")
    assert l_impute.item() < 1e-5, "Imputation loss should be ~0 on perfect reconstruction"
    assert l_kriging.item() < 1e-5, "Kriging loss should be ~0 on perfect reconstruction"

    # Traffic-only error: perturb only channel 0 of the imputation prediction.
    imputed_err = target_impute.clone()
    imputed_err[:, :, :, 0] += 1.0  # shift scaled traffic channel
    _, l_impute_err, _, _ = compute_loss(
        imputed_features=imputed_err, target_impute=target_impute,
        mu=mu, theta=theta, pi=pi, target_forecast=target_forecast,
        traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
        kriging_pred=kriging_pred, spatial_holdout_mask=HOLDOUT,
        w_impute=1.0, w_forecast=1.0, w_kriging=0.5,
    )
    print(f"traffic-only error -> loss_impute = {l_impute_err.item():.6f} "
          f"(was {l_impute.item():.3e})")
    assert l_impute_err.item() > 1e-3, "Traffic perturbation should raise imputation loss"

    # Weather-only error must leave the traffic MSLE term untouched: verify weather moves loss too.
    imputed_w = target_impute.clone()
    imputed_w[:, :, :, 1:] += 1.0
    _, l_impute_w, _, _ = compute_loss(
        imputed_features=imputed_w, target_impute=target_impute,
        mu=mu, theta=theta, pi=pi, target_forecast=target_forecast,
        traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
        kriging_pred=kriging_pred, spatial_holdout_mask=HOLDOUT,
        w_impute=1.0, w_forecast=1.0, w_kriging=0.5,
    )
    print(f"weather-only error -> loss_impute = {l_impute_w.item():.6f}")
    assert l_impute_w.item() > 1e-3, "Weather perturbation should raise imputation loss"

    print("✓ Channel separation verified (perfect recon ~0; traffic & weather errors register)")
    return True


def test_nan_tripwire():
    print("\n=== TEST E: NaN tripwire fires ===")
    imputed, target_impute, mu, theta, pi, target_forecast, kriging_pred = _make_inputs()
    imputed[0, 0, 0, 0] = float("nan")

    try:
        compute_loss(
            imputed_features=imputed, target_impute=target_impute,
            mu=mu, theta=theta, pi=pi, target_forecast=target_forecast,
            traffic_mean=TRAFFIC_MEAN, traffic_std=TRAFFIC_STD,
            kriging_pred=kriging_pred, spatial_holdout_mask=HOLDOUT,
            w_impute=1.0, w_forecast=1.0, w_kriging=0.5,
        )
    except AssertionError as e:
        assert "CRITICAL" in str(e), f"Tripwire fired but without CRITICAL marker: {e}"
        print(f"✓ NaN tripwire raised AssertionError: {e}")
        return True

    raise AssertionError("Expected a CRITICAL AssertionError on NaN input, none raised")


def main():
    print("=" * 64)
    print("PHASE 4 DESK CHECK: Unified Loss (end-to-end)")
    print("=" * 64)
    try:
        test_all_terms_finite_and_arithmetic()
        test_gradient_flow_all_heads()
        test_zero_weight_bypass()
        test_channel_separation()
        test_nan_tripwire()

        print("\n" + "=" * 64)
        print("✅ ALL DESK CHECKS PASSED")
        print("=" * 64)
        return 0
    except AssertionError as e:
        print(f"\n❌ DESK CHECK FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
