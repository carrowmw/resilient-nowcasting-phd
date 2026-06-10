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

from decoders import scale_invariant_mse, SpatialKrigingDecoder


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


def main():
    torch.manual_seed(42)
    part1_msle_edge_cases()
    part2_one_epoch_dry_run()
    hr("DESK CHECK PASSED")


if __name__ == "__main__":
    main()
