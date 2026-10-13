#!/usr/bin/env python3
"""
scripts/desk_checks/desk_check_spatial_splits.py

Spatial-track split arithmetic desk check: prints the split boundaries, purge
window counts, and per-split usable-target coverage FOR THE REAL DATASET
(smoke config window) so the arithmetic is human-verifiable before any GPU
time. Also verifies the no-shared-grid-steps guarantee by brute force.

Run INSIDE the container:
    python scripts/desk_checks/desk_check_spatial_splits.py [--config configs/spatial/spatial_smoke.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from spatial.loader import GraphSeriesStore  # noqa: E402
from spatial.splits import assign_split_windows, build_graph_splits, count_usable_targets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spatial/spatial_smoke.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    t = config["temporal"]
    window, horizons = int(t["window"]), [int(h) for h in t["horizons"]]

    print("=" * 70)
    print("1. Pure arithmetic (synthetic T=1000)")
    print("=" * 70)
    idx = assign_split_windows(1000, window, max(horizons))
    purge_k = window + max(horizons) - 1
    for name in ("train", "val", "test"):
        s = idx[name]
        print(f"  {name:5s}: {len(s):4d} windows, starts [{s.min()}, {s.max()}]")
    print(f"  purge_k = window + max_h - 1 = {window} + {max(horizons)} - 1 = {purge_k}")

    # Brute-force the disjointness guarantee.
    def touched(starts):
        steps = set()
        for s in starts:
            steps.update(range(int(s), int(s) + window))
            steps.update(int(s) + window - 1 + h for h in horizons)
        return steps

    assert not (touched(idx["train"]) & touched(idx["val"])), "train/val overlap!"
    assert not (touched(idx["val"]) & touched(idx["test"])), "val/test overlap!"
    print("  brute-force check: train∩val = ∅, val∩test = ∅  ✓")

    print()
    print("=" * 70)
    print(f"2. Real dataset ({config['data']['start_date']} → {config['data']['end_date']})")
    print("=" * 70)
    store = GraphSeriesStore(config)
    ctx = store.build_context()
    splits = build_graph_splits(
        ctx, stride=int(t.get("stride", 1)), train_frac=float(t["train_frac"]),
        val_frac=float(t["val_frac"]), purge_boundaries=bool(t["purge_boundaries"]))
    print(f"  nodes: {ctx.n_nodes}, grid steps: {len(ctx.grid_index)}")
    for name, split in splits.items():
        span = f"{split.origin_dt[0]} → {split.origin_dt[-1]}"
        print(f"  {name:5s}: {len(split):6d} windows, origins {span}")
    print()
    print("  Per-split observed-target coverage:")
    for name, split in splits.items():
        df = count_usable_targets(split)
        frac = df.n_targets_observed.sum() / df.n_windows.sum()
        print(f"    {name:5s}: {frac:6.1%} of (window × node × horizon) targets observed")

    report = store.node_report()
    print(f"\n  node selection: {int(report.included.sum())}/{len(report)} included")
    for reason, grp in report[~report.included].groupby("exclusion_reason"):
        print(f"    excluded ({len(grp):2d}): {reason}")

    print("\nDESK CHECK PASSED")


if __name__ == "__main__":
    main()
