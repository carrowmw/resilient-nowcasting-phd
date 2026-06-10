"""
Phase 2 (feat/kriging-infrastructure) infrastructure tests.

Covers the dormant building blocks added for the Spatial Estimation Architecture:
  - decoders.scale_invariant_mse (MSLE)
  - decoders.SpatialKrigingDecoder
  - metrics.calculate_kriging_metrics / compute_msle

These exercise shapes and numerical edge cases only; none of this is wired into
the live training path yet (that happens on exp/spatial-kriging-head).
"""
import math

import pytest
import torch

from decoders import scale_invariant_mse, SpatialKrigingDecoder
from metrics import compute_msle, calculate_kriging_metrics


# --------------------------------------------------------------------------- #
# scale_invariant_mse (MSLE)                                                   #
# --------------------------------------------------------------------------- #
def test_msle_zero_when_identical():
    x = torch.tensor([0.0, 5.0, 200.0, 1e6])
    assert scale_invariant_mse(x, x).item() == pytest.approx(0.0, abs=1e-12)


def test_msle_zeros_no_division_by_zero():
    z = torch.zeros(4, 5)
    out = scale_invariant_mse(z, z)
    assert torch.isfinite(out).all()
    assert out.item() == pytest.approx(0.0, abs=1e-12)


def test_msle_large_integers_finite():
    pred = torch.full((4, 5), 1_000_000.0)
    target = torch.full((4, 5), 10.0)
    out = scale_invariant_mse(pred, target)
    assert torch.isfinite(out).all()
    assert out.item() > 0.0


def test_msle_clamps_negative_inputs():
    # Negatives are clamped to 0 before log1p; must stay finite.
    pred = torch.tensor([-5.0, -1.0, 0.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    out = scale_invariant_mse(pred, target)
    assert torch.isfinite(out).all()


def test_msle_nan_propagates():
    # By design NaN is NOT silently masked — it propagates for the downstream tripwire.
    pred = torch.tensor([float("nan"), 1.0, 2.0])
    target = torch.tensor([0.0, 1.0, 2.0])
    assert torch.isnan(scale_invariant_mse(pred, target)).item()


def test_msle_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        scale_invariant_mse(torch.zeros(4, 5), torch.zeros(4, 6))


def test_msle_scale_invariance_property():
    # Relative-error intuition: 5-vs-0 and 50-vs-0 differ far less under MSLE than
    # under raw MSE, where the 50-vs-0 error is 100x larger.
    p5 = scale_invariant_mse(torch.full((100,), 5.0), torch.zeros(100)).item()
    p50 = scale_invariant_mse(torch.full((100,), 50.0), torch.zeros(100)).item()
    mse5 = torch.mean((torch.full((100,), 5.0) - torch.zeros(100)) ** 2).item()
    mse50 = torch.mean((torch.full((100,), 50.0) - torch.zeros(100)) ** 2).item()
    msle_ratio = p50 / p5
    mse_ratio = mse50 / mse5
    assert msle_ratio < mse_ratio  # MSLE compresses the magnitude gap
    # ~ (log1p(51)/log1p(6))^2 ≈ 4.82; vastly smaller than the 100x raw-MSE gap.
    assert msle_ratio < 5.0
    assert mse_ratio == pytest.approx(100.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# SpatialKrigingDecoder                                                        #
# --------------------------------------------------------------------------- #
def test_kriging_decoder_output_shape():
    torch.manual_seed(42)
    b, n, t, h, d = 4, 20, 12, 32, 3
    dec = SpatialKrigingDecoder(hidden_dim=h, time_series_dim=d)
    emb = torch.randn(b, n, t, h)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 4, 9]] = True  # K = 3 held-out nodes
    out = dec(emb, mask)
    assert out.shape == (b, 3, t, d)
    assert torch.isfinite(out).all()


def test_kriging_decoder_gradients_flow():
    b, n, t, h, d = 2, 10, 6, 16, 3
    dec = SpatialKrigingDecoder(hidden_dim=h, time_series_dim=d)
    emb = torch.randn(b, n, t, h, requires_grad=True)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[0] = True
    out = dec(emb, mask)
    out.sum().backward()
    assert dec.kriging_head[0].weight.grad is not None


def test_kriging_decoder_rejects_bad_mask():
    dec = SpatialKrigingDecoder(hidden_dim=16, time_series_dim=3)
    emb = torch.randn(2, 10, 6, 16)
    with pytest.raises(AssertionError):
        dec(emb, torch.zeros(9, dtype=torch.bool))      # wrong length
    with pytest.raises(AssertionError):
        dec(emb, torch.zeros(10))                        # non-boolean


# --------------------------------------------------------------------------- #
# kriging metrics                                                              #
# --------------------------------------------------------------------------- #
def test_kriging_metrics_perfect_reconstruction():
    series = torch.rand(2, 3, 12, 3) * 50.0
    m = calculate_kriging_metrics(series.clone(), series.clone())
    assert m["kriging_msle"] == pytest.approx(0.0, abs=1e-10)
    assert m["kriging_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["kriging_mae"] == pytest.approx(0.0, abs=1e-6)


def test_kriging_metrics_empty_holdout():
    empty = torch.zeros(2, 0, 12, 3)
    m = calculate_kriging_metrics(empty, empty)
    assert m == {"kriging_msle": 0.0, "kriging_rmse": 0.0, "kriging_mae": 0.0}


def test_compute_msle_matches_loss_fn():
    pred = torch.rand(4, 7) * 100.0
    target = torch.rand(4, 7) * 100.0
    assert compute_msle(pred, target) == pytest.approx(
        scale_invariant_mse(pred, target).item(), rel=1e-6
    )
