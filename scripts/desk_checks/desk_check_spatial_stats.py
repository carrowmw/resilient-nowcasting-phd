#!/usr/bin/env python3
"""
scripts/desk_checks/desk_check_spatial_stats.py

Statistical framework desk check with known-answer controls:

  1. POSITIVE control — model A has genuinely smaller errors (AR(1) error
     processes, sigma 1.0 vs 2.0): per-node DM must flag A better on ~all
     nodes and survive BH-FDR.
  2. NEGATIVE control — two models with the SAME error process: the
     FDR-significant fraction must be ≈ alpha (no fabricated significance).
  3. MASE known answer — a forecaster that always predicts yesterday's value
     must score MASE ≈ 1 by construction.

Run INSIDE the container:
    python scripts/desk_checks/desk_check_spatial_stats.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from spatial.evaluate import evaluate_split, mase_scale  # noqa: E402
from spatial.splits import build_graph_splits  # noqa: E402
from spatial.stats import dm_by_node, dm_summary  # noqa: E402
from spatial_helpers import make_context  # noqa: E402


def _ar1_preds(model: str, sigma: float, seed: int, n_nodes=8, n_windows=400):
    """Long-format predictions with AR(1) errors (phi=0.5) — the serial
    correlation the HAC variance is there to absorb."""
    truth_rng = np.random.default_rng(999)
    rng = np.random.default_rng(seed)
    dts = pd.date_range("2024-06-01", periods=n_windows, freq="15min")
    rows = []
    for j in range(n_nodes):
        truth = truth_rng.uniform(5, 50, n_windows)
        e = np.zeros(n_windows)
        innov = rng.normal(0, sigma, n_windows)
        for t in range(1, n_windows):
            e[t] = 0.5 * e[t - 1] + innov[t]
        for t in range(n_windows):
            rows.append({"model": model, "graph_variant": "g", "seed": 0,
                         "node_id": f"n{j}", "horizon": 1, "target_dt": dts[t],
                         "split": "test", "target_observed": True,
                         "y_true_raw": truth[t], "y_pred_raw": truth[t] + e[t]})
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("1. DM positive control (sigma 1.0 vs 2.0, AR(1) phi=0.5)")
    print("=" * 70)
    a = _ar1_preds("a", 1.0, seed=1)
    b = _ar1_preds("b", 2.0, seed=2)
    s = dm_summary(dm_by_node(a, b, horizon=1))
    print(f"  frac A better: {s['frac_a_better']:.2f}   "
          f"FDR-significant: {s['frac_significant_fdr']:.2f}   "
          f"median DM: {s['median_dm_stat']:.2f}")
    assert s["frac_a_better"] >= 0.9, "positive control failed direction"
    assert s["frac_significant_fdr"] >= 0.75, "positive control failed power"

    print()
    print("=" * 70)
    print("2. DM negative control (identical error process, 20 repetitions)")
    print("=" * 70)
    fracs = []
    for rep in range(20):
        a = _ar1_preds("a", 1.5, seed=100 + rep)
        b = _ar1_preds("b", 1.5, seed=200 + rep)
        fracs.append(dm_summary(dm_by_node(a, b, horizon=1))["frac_significant_fdr"])
    mean_frac = float(np.mean(fracs))
    print(f"  mean FDR-significant fraction over reps: {mean_frac:.3f} "
          f"(alpha = 0.05; should be near it)")
    assert mean_frac <= 0.10, "negative control: fabricated significance!"

    print()
    print("=" * 70)
    print("3. MASE known answer (predict-yesterday scores ~1)")
    print("=" * 70)
    ctx = make_context(n_nodes=5, T_total=96 * 10, window=8, horizons=(1,),
                       missing_frac=0.0, seed=3)
    splits = build_graph_splits(ctx)
    scale = mase_scale(ctx, splits["train"])
    split = splits["test"]
    season = ctx.steps_per_day()
    preds = ctx.values_scaled[split.target_idx - season].transpose(0, 2, 1)
    df = evaluate_split(split, preds.astype(np.float32), scale)
    med = df["mase"].median()
    print(f"  median MASE of predict-yesterday on test: {med:.3f}")
    assert 0.8 < med < 1.25, (
        "MASE of the seasonal-naive strategy should be ~1 (train/test regime "
        f"shift only), got {med:.3f}")

    print("\nDESK CHECK PASSED")


if __name__ == "__main__":
    main()
