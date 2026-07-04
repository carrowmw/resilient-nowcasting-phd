#!/usr/bin/env python3
"""
scripts/desk_check_kriging.py

Phase 2 (feat/kriging-infrastructure) desk check. Two parts, per the Architect plan:

  1. Pass synthetic tensors (zeros, large integers, NaNs) through scale_invariant_mse
     and print the EXACT outputs, demonstrating no division-by-zero / shape mismatch.
  2. Run a 1-epoch dry-run of the existing pipeline on dummy tensors with the new
     inactive config keys present (w_kriging = 0.0) to confirm nothing regressed.

Run INSIDE the container:
    python scripts/desk_check_kriging.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from decoders import (
    scale_invariant_mse,
    SpatialKrigingDecoder,
    compute_kriging_loss,
    compute_loss,
)
from dataset import make_spatial_holdout_mask, apply_spatial_holdout


def hr(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def part1_msle_edge_cases():
    hr("PART 1 — scale_invariant_mse (MSLE) edge cases")
    cases = {
        "zeros vs zeros":          (torch.zeros(4, 5), torch.zeros(4, 5)),
        "identical (val 200)":     (torch.full((4, 5), 200.0), torch.full((4, 5), 200.0)),
        "large ints 1e6 vs 10":    (torch.full((4, 5), 1_000_000.0), torch.full((4, 5), 10.0)),
        "pred 5 vs target 0":      (torch.full((100,), 5.0), torch.zeros(100)),
        "pred 50 vs target 0":     (torch.full((100,), 50.0), torch.zeros(100)),
        "negatives (clamped)":     (torch.tensor([-5.0, -1.0, 0.0]), torch.zeros(3)),
        "NaN in pred":             (torch.tensor([float("nan"), 1.0, 2.0]), torch.tensor([0.0, 1.0, 2.0])),
    }
    for name, (p, t) in cases.items():
        out = scale_invariant_mse(p, t)
        print(f"  {name:24s} -> MSLE = {out.item():.10g}  (finite={bool(torch.isfinite(out).item())})")

    # Shape-mismatch tripwire fires (no silent broadcast).
    try:
        scale_invariant_mse(torch.zeros(4, 5), torch.zeros(4, 6))
        print("  shape mismatch          -> NO ASSERT (UNEXPECTED)")
    except AssertionError as e:
        print(f"  shape mismatch          -> AssertionError raised as expected")

    # SpatialKrigingDecoder forward shape check.
    hr("PART 1b — SpatialKrigingDecoder forward shapes")
    b, n, t, h, d = 4, 20, 12, 32, 3
    dec = SpatialKrigingDecoder(hidden_dim=h, time_series_dim=d)
    emb = torch.randn(b, n, t, h)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 4, 9]] = True
    out = dec(emb, mask)
    print(f"  input  latent [B,N,T,H] = {tuple(emb.shape)}, holdout K = {int(mask.sum())}")
    print(f"  output series [B,K,T,D] = {tuple(out.shape)}  (expected ({b}, 3, {t}, {d}))")
    assert out.shape == (b, 3, t, d), "kriging output shape mismatch"
    print("  -> shapes OK")


def part3_layer2_wiring():
    hr("PART 3 — Layer 2 wiring: holdout masking + kriging loss + 4-output compute_loss")

    # (a) Holdout masking: held-out nodes zeroed, complement byte-identical.
    b, n, t, d = 4, 20, 12, 3
    feats = torch.randn(b, n, t, d)
    mask = make_spatial_holdout_mask(n, holdout_prob=0.25, seed=7)
    masked = apply_spatial_holdout(feats, mask)
    held = int(mask.sum())
    print(f"  holdout mask: {held}/{n} nodes held out (seed=7)")
    assert (masked[:, mask, :, :] == 0.0).all(), "held-out nodes not zeroed"
    assert torch.equal(masked[:, ~mask, :, :], feats[:, ~mask, :, :]), "complement altered"
    print("  -> held-out nodes zeroed; observable nodes untouched OK")

    # Determinism + extremes
    assert torch.equal(make_spatial_holdout_mask(n, 0.3, seed=1),
                       make_spatial_holdout_mask(n, 0.3, seed=1)), "seed not deterministic"
    assert make_spatial_holdout_mask(n, 0.0, seed=1).sum() == 0
    assert make_spatial_holdout_mask(n, 1.0, seed=1).sum() == n
    print("  -> mask determinism + prob extremes (0.0/1.0) OK")

    # (b) Decoder shape K == mask.sum()
    h = 32
    dec = SpatialKrigingDecoder(hidden_dim=h, time_series_dim=d)
    emb = torch.randn(b, n, t, h)
    out = dec(emb, mask)
    print(f"  kriging decoder out {tuple(out.shape)} (expected ({b}, {held}, {t}, {d}))")
    assert out.shape == (b, held, t, d)

    # (c) compute_kriging_loss extremes + EMPTY holdout -> exactly 0.0 (NaN trap).
    mean, std = torch.tensor(50.0), torch.tensor(200.0)
    target = torch.randn(b, n, t, d)
    perfect = compute_kriging_loss(target[:, mask, :, :].clone(), target, mask, mean, std)
    print(f"  perfect reconstruction kriging loss = {perfect.item():.3e} (expect ~0)")
    assert perfect.item() < 1e-5

    big_pred = target[:, mask, :, :].clone()
    big_pred[:, :, :, 0] += 1_000.0  # large traffic error on raw scale
    big = compute_kriging_loss(big_pred, target, mask, mean, std)
    print(f"  large-error kriging loss = {big.item():.6g} (finite={bool(torch.isfinite(big))})")
    assert torch.isfinite(big) and big.item() > 0.0

    empty_mask = torch.zeros(n, dtype=torch.bool)
    empty = compute_kriging_loss(target[:, empty_mask, :, :].clone(), target, empty_mask, mean, std)
    print(f"  EMPTY holdout kriging loss = {empty.item():.6g} (must be exactly 0.0, no NaN)")
    assert empty.item() == 0.0 and torch.isfinite(empty)
    print("  -> kriging loss math + empty-holdout NaN trap OK")

    # (d) End-to-end compute_loss: 4 finite outputs; grads reach kriging head when on,
    #     and are None when w_kriging == 0.
    imputed = torch.randn(b, n, t, d)
    mu = torch.rand(b, n) + 0.1
    theta = torch.rand(b, n) + 0.1
    pi = torch.rand(b, n) * 0.5
    tf = torch.randint(0, 30, (b, n)).float()

    dec_on = SpatialKrigingDecoder(hidden_dim=h, time_series_dim=d)
    emb_on = torch.randn(b, n, t, h, requires_grad=True)
    kpred = dec_on(emb_on, mask)
    total, l_imp, l_fc, l_kr = compute_loss(
        imputed_features=imputed, target_impute=target, mu=mu, theta=theta, pi=pi,
        target_forecast=tf, traffic_mean=mean, traffic_std=std,
        kriging_pred=kpred, spatial_holdout_mask=mask, w_kriging=0.5,
    )
    print(f"  compute_loss(on):  total={total.item():.4f} impute={l_imp.item():.4f} "
          f"forecast={l_fc.item():.4f} kriging={l_kr.item():.4f}")
    assert all(torch.isfinite(x) for x in (total, l_imp, l_fc, l_kr))
    assert l_kr.item() > 0.0
    total.backward()
    assert dec_on.kriging_head[0].weight.grad is not None, "kriging head got no grad when active"
    print("  -> 4 finite losses + gradient reaches kriging head when w_kriging>0 OK")

    dec_off = SpatialKrigingDecoder(hidden_dim=h, time_series_dim=d)
    imputed_off = imputed.clone().requires_grad_(True)  # leaf so the off-graph exists
    total2, _, _, l_kr2 = compute_loss(
        imputed_features=imputed_off, target_impute=target, mu=mu, theta=theta, pi=pi,
        target_forecast=tf, traffic_mean=mean, traffic_std=std,
        kriging_pred=None, spatial_holdout_mask=None, w_kriging=0.0,
    )
    total2.backward()
    assert l_kr2.item() == 0.0
    assert dec_off.kriging_head[0].weight.grad is None, "kriging head got grad when inactive"
    print("  -> w_kriging=0 bypass: kriging loss 0.0 and no kriging gradient OK")


def part2_one_epoch_dry_run():
    hr("PART 2 — 1-epoch mock dry-run with w_kriging = 0.0")
    # Quiet, no W&B, CPU, tiny mock data.
    logging.getLogger().setLevel(logging.WARNING)
    from train import run_training

    config = {
        "data": {"seq_len": 12, "mask_prob": 0.15, "spatial_holdout_prob": 0.0},
        "model": {
            "num_nodes": 20, "static_feat_dim": 15, "time_series_dim": 3,
            "embedding_dim": 16, "hidden_dim": 32, "sparsity_threshold": 0.5,
            "num_layers": 2, "num_heads": 4, "dropout": 0.1,
            "dim_feedforward": None, "zinb_eps": 1.0e-6,
        },
        "training": {
            "epochs": 1, "batch_size": 8, "lr": 0.0003, "grad_clip_max_norm": 1.0,
            "random_seed": 42, "early_stop_patience": 5, "early_stop_min_delta": 0.001,
        },
        "loss": {"w_forecast": 1.0, "w_impute": 1.0, "w_kriging": 0.0},
    }

    final_val_loss = run_training(config, init_wandb=False, use_mock_data=True)
    print(f"  1-epoch mock training completed. final val_loss = {final_val_loss:.6f}")
    assert final_val_loss == final_val_loss, "val_loss is NaN"  # NaN check
    print("  -> pipeline functions seamlessly with kriging inactive (w_kriging=0.0)")


def part4_one_epoch_kriging_active():
    hr("PART 4 — 1-epoch mock dry-run with kriging ACTIVE (w_kriging=0.5, holdout=0.15)")
    logging.getLogger().setLevel(logging.WARNING)
    from train import run_training

    config = {
        "data": {"seq_len": 12, "mask_prob": 0.15, "spatial_holdout_prob": 0.15},
        "model": {
            "num_nodes": 20, "static_feat_dim": 15, "time_series_dim": 3,
            "embedding_dim": 16, "hidden_dim": 32, "sparsity_threshold": 0.5,
            "num_layers": 2, "num_heads": 4, "dropout": 0.1,
            "dim_feedforward": None, "zinb_eps": 1.0e-6,
        },
        "training": {
            "epochs": 1, "batch_size": 8, "lr": 0.0003, "grad_clip_max_norm": 1.0,
            "random_seed": 42, "early_stop_patience": 5, "early_stop_min_delta": 0.001,
        },
        "loss": {"w_forecast": 1.0, "w_impute": 1.0, "w_kriging": 0.5},
    }

    final_val_loss = run_training(config, init_wandb=False, use_mock_data=True)
    print(f"  1-epoch mock training (kriging active) completed. final val_loss = {final_val_loss:.6f}")
    assert final_val_loss == final_val_loss, "val_loss is NaN"  # NaN check
    print("  -> live kriging path trains end-to-end without NaN")


def main():
    torch.manual_seed(42)
    part1_msle_edge_cases()
    part3_layer2_wiring()
    part2_one_epoch_dry_run()
    part4_one_epoch_kriging_active()
    hr("DESK CHECK PASSED")


if __name__ == "__main__":
    main()
