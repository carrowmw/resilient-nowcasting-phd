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

from decoders import (
    scale_invariant_mse,
    SpatialKrigingDecoder,
    compute_kriging_loss,
    compute_loss,
)
from dataset import make_spatial_holdout_mask, apply_spatial_holdout
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


# =========================================================================== #
# PHASE 2 — LIVE-PATH WIRING (holdout masking + kriging loss + compute_loss)   #
# These exercise the now-active training path, not just the dormant building   #
# blocks above.                                                                #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# make_spatial_holdout_mask                                                    #
# --------------------------------------------------------------------------- #
def test_holdout_mask_shape_and_dtype():
    mask = make_spatial_holdout_mask(num_nodes=50, holdout_prob=0.2, seed=1)
    assert mask.shape == (50,)
    assert mask.dtype == torch.bool


def test_holdout_mask_deterministic_under_seed():
    a = make_spatial_holdout_mask(40, 0.3, seed=42)
    b = make_spatial_holdout_mask(40, 0.3, seed=42)
    assert torch.equal(a, b)


def test_holdout_mask_prob_extremes():
    none = make_spatial_holdout_mask(30, 0.0, seed=3)
    alln = make_spatial_holdout_mask(30, 1.0, seed=3)
    assert none.sum().item() == 0
    assert alln.sum().item() == 30


def test_holdout_mask_rejects_bad_prob():
    with pytest.raises(AssertionError):
        make_spatial_holdout_mask(10, 1.5)
    with pytest.raises(AssertionError):
        make_spatial_holdout_mask(0, 0.2)


# --------------------------------------------------------------------------- #
# apply_spatial_holdout                                                        #
# --------------------------------------------------------------------------- #
def test_apply_holdout_zeros_held_out_only():
    b, n, t, d = 4, 20, 12, 3
    feats = torch.randn(b, n, t, d)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 4, 9]] = True
    out = apply_spatial_holdout(feats, mask)
    assert (out[:, mask, :, :] == 0.0).all()
    assert torch.equal(out[:, ~mask, :, :], feats[:, ~mask, :, :])
    # original untouched (clone semantics)
    assert not torch.equal(out, feats)


def test_apply_holdout_rejects_shape_mismatch():
    feats = torch.randn(2, 10, 6, 3)
    with pytest.raises(AssertionError):
        apply_spatial_holdout(feats, torch.zeros(9, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# compute_kriging_loss                                                         #
# --------------------------------------------------------------------------- #
def _scaler():
    return torch.tensor(50.0), torch.tensor(200.0)  # mean, std


def test_kriging_loss_zero_on_perfect_reconstruction():
    b, n, t, d = 2, 10, 6, 3
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[2, 5]] = True
    target = torch.randn(b, n, t, d)
    pred = target[:, mask, :, :].clone()
    mean, std = _scaler()
    loss = compute_kriging_loss(pred, target, mask, mean, std)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kriging_loss_empty_holdout_returns_zero_no_nan():
    b, n, t, d = 2, 10, 6, 3
    mask = torch.zeros(n, dtype=torch.bool)  # nothing held out
    target = torch.randn(b, n, t, d)
    pred = target[:, mask, :, :].clone()     # [B, 0, T, D]
    mean, std = _scaler()
    loss = compute_kriging_loss(pred, target, mask, mean, std)
    assert torch.isfinite(loss).all()
    assert loss.item() == 0.0


def test_kriging_loss_positive_and_finite_on_error():
    b, n, t, d = 2, 8, 5, 3
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[0, 3]] = True
    target = torch.randn(b, n, t, d)
    pred = target[:, mask, :, :] + 1.0
    mean, std = _scaler()
    loss = compute_kriging_loss(pred, target, mask, mean, std)
    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0


def test_kriging_loss_shape_mismatch_raises():
    b, n, t, d = 2, 8, 5, 3
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[0, 3]] = True
    target = torch.randn(b, n, t, d)
    bad_pred = torch.randn(b, 3, t, d)  # K should be 2, not 3
    mean, std = _scaler()
    with pytest.raises(AssertionError):
        compute_kriging_loss(bad_pred, target, mask, mean, std)


# --------------------------------------------------------------------------- #
# compute_loss 4-output contract                                              #
# --------------------------------------------------------------------------- #
def _loss_inputs(b=2, n=10, t=6, d=3):
    imputed = torch.randn(b, n, t, d)
    target_impute = torch.randn(b, n, t, d)
    mu = torch.rand(b, n) + 0.1
    theta = torch.rand(b, n) + 0.1
    pi = torch.rand(b, n) * 0.5
    target_forecast = torch.randint(0, 30, (b, n)).float()
    return imputed, target_impute, mu, theta, pi, target_forecast


def test_compute_loss_returns_four_values_kriging_off():
    imputed, target_impute, mu, theta, pi, tf = _loss_inputs()
    out = compute_loss(
        imputed_features=imputed, target_impute=target_impute,
        mu=mu, theta=theta, pi=pi, target_forecast=tf,
        traffic_mean=torch.tensor(0.0), traffic_std=torch.tensor(1.0),
    )
    assert len(out) == 4
    total, l_imp, l_fc, l_kr = out
    assert l_kr.item() == 0.0
    assert torch.isfinite(total)


def test_compute_loss_kriging_on_contributes():
    b, n, t, d = 2, 10, 6, 3
    imputed, target_impute, mu, theta, pi, tf = _loss_inputs(b, n, t, d)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 7]] = True
    kriging_pred = torch.randn(b, int(mask.sum()), t, d)
    total, l_imp, l_fc, l_kr = compute_loss(
        imputed_features=imputed, target_impute=target_impute,
        mu=mu, theta=theta, pi=pi, target_forecast=tf,
        traffic_mean=torch.tensor(0.0), traffic_std=torch.tensor(1.0),
        kriging_pred=kriging_pred, spatial_holdout_mask=mask, w_kriging=0.5,
    )
    assert l_kr.item() > 0.0
    assert torch.isfinite(total)


def test_compute_loss_kriging_pred_none_w_kriging_positive_is_valid():
    """Under the decoupled contract, kriging_pred=None with w_kriging>0 is valid
    (loss_kriging stays 0.0, total is finite). This supports configurations where
    holdout masking is off but a weight is set."""
    imputed, target_impute, mu, theta, pi, tf = _loss_inputs()
    total, _, _, l_kr = compute_loss(
        imputed_features=imputed, target_impute=target_impute,
        mu=mu, theta=theta, pi=pi, target_forecast=tf,
        traffic_mean=torch.tensor(0.0), traffic_std=torch.tensor(1.0),
        kriging_pred=None, spatial_holdout_mask=None, w_kriging=0.5,
    )
    assert l_kr.item() == 0.0
    assert torch.isfinite(total)


def test_compute_loss_kriging_pred_without_mask_raises():
    """If kriging_pred is supplied but spatial_holdout_mask is None, the CRITICAL
    assertion must fire (fail fast)."""
    b, n, t, d = 2, 10, 6, 3
    imputed, target_impute, mu, theta, pi, tf = _loss_inputs(b, n, t, d)
    kriging_pred = torch.randn(b, 2, t, d)
    with pytest.raises(AssertionError, match="CRITICAL"):
        compute_loss(
            imputed_features=imputed, target_impute=target_impute,
            mu=mu, theta=theta, pi=pi, target_forecast=tf,
            traffic_mean=torch.tensor(0.0), traffic_std=torch.tensor(1.0),
            kriging_pred=kriging_pred, spatial_holdout_mask=None, w_kriging=0.0,
        )
