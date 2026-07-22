#!/usr/bin/env python3
"""
Desk check for the statistics package (Phase G): every test exercised on
synthetic data with KNOWN ground truth, including the figure pipeline the
aggregator uses (so the ≥10-sensor code paths are verified before the full
matrix run, which the 3-sensor smoke test cannot reach).

Run INSIDE the container:
    python scripts/desk_checks/desk_check_quality_stats.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import numpy as np
import pandas as pd

from quality.plotting import plot_error_vs_quality, plot_importances
from quality.stats import (
    bootstrap_ci,
    diebold_mariano,
    partial_spearman,
    rf_quality_to_error,
    segmented_threshold_fit,
    spearman_with_fdr,
)


def hr(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    rng = np.random.default_rng(42)
    n_sensors = 80

    # Ground truth: error driven by avg_seq_len with a REAL threshold at 20 h;
    # completeness moderately correlated with seq_len (confounded, as in the
    # real data) but with no independent effect on error.
    seq_len = rng.uniform(4, 60, n_sensors)
    completeness = np.clip(0.7 + 0.002 * seq_len + rng.normal(0, 0.08, n_sensors), 0, 1)
    error = np.where(seq_len < 20, 4.0 - 0.12 * seq_len, 1.6) + rng.normal(0, 0.15, n_sensors)
    qdf = pd.DataFrame({
        "avg_seq_len_hours": seq_len,
        "completeness_active": completeness,
        "std_seq_len_hours": rng.uniform(1, 10, n_sensors),
        "seq_count": rng.integers(10, 200, n_sensors).astype(float),
    })
    err = pd.Series(error)

    hr("PART 1 — Spearman + FDR (seq_len real, completeness confounded)")
    corr = spearman_with_fdr(qdf, err)
    print(corr.to_string(index=False))
    assert bool(corr.set_index("metric").loc["avg_seq_len_hours", "significant_fdr"])

    hr("PART 2 — partial correlations disentangle the confound")
    r_seq, p_seq = partial_spearman(seq_len, error, completeness.reshape(-1, 1))
    r_comp, p_comp = partial_spearman(completeness, error, seq_len.reshape(-1, 1))
    print(f"  seq_len | completeness : rho={r_seq:+.3f}  p={p_seq:.2e}")
    print(f"  completeness | seq_len : rho={r_comp:+.3f}  p={p_comp:.2e}")
    assert p_seq < 0.001 and abs(r_seq) > abs(r_comp), \
        "partial correlation must keep seq_len and kill completeness"

    hr("PART 3 — segmented regression recovers the 20 h breakpoint")
    fit = segmented_threshold_fit(seq_len, error, n_bootstrap=500)
    print(f"  breakpoint = {fit.breakpoint:.1f} h  CI95 = "
          f"[{fit.breakpoint_ci[0]:.1f}, {fit.breakpoint_ci[1]:.1f}]")
    print(f"  slopes: below={fit.slope_below:+.3f}, above={fit.slope_above:+.3f}")
    print(f"  ΔAIC (linear − segmented) = {fit.delta_aic:.1f}  (>10 ⇒ strong)")
    assert abs(fit.breakpoint - 20) < 4 and fit.delta_aic > 10

    ci = bootstrap_ci(
        lambda a, b: float(np.corrcoef(a, b)[0, 1]), seq_len, error, n_bootstrap=500
    )
    print(f"  bootstrap CI on Pearson(seq_len, error): "
          f"{ci['point']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    hr("PART 4 — Diebold–Mariano ladder")
    n = 800
    e_good, e_mid, e_bad = (rng.normal(0, s, n) for s in (1.0, 1.15, 2.0))
    for label, ea, eb in (("good vs bad", e_good, e_bad),
                          ("good vs mid", e_good, e_mid),
                          ("good vs good-clone", e_good, e_good + rng.normal(0, 0.01, n))):
        out = diebold_mariano(ea, eb, horizon=8)
        print(f"  {label:20s} DM={out['dm_stat']:+7.2f}  p={out['p_value']:.4f}")

    hr("PART 5 — exploratory RF (flagged exploratory, honest R²)")
    rf = rf_quality_to_error(qdf, err, n_repeats=3)
    print(f"  exploratory={rf['exploratory']}  n={rf['n_sensors']}")
    print(f"  R² (repeated 5-fold) = {rf['r2_cv_mean']:.3f} ± {rf['r2_cv_std']:.3f}  "
          f"(OOB {rf['oob_r2']:.3f})")
    for k, v in rf["permutation_importance"].items():
        print(f"    {k:22s} {v['mean']:+.3f} ± {v['std']:.3f}")
    assert rf["permutation_importance"]["avg_seq_len_hours"]["mean"] == max(
        v["mean"] for v in rf["permutation_importance"].values()
    ), "seq_len must dominate the permutation importance"

    hr("PART 6 — figure pipeline (threshold scatter + importances)")
    with tempfile.TemporaryDirectory() as tmp:
        results = pd.DataFrame({"avg_seq_len_hours": seq_len, "rmse_raw": error})
        p1 = os.path.join(tmp, "threshold.png")
        p2 = os.path.join(tmp, "importance.png")
        plot_error_vs_quality(results, "avg_seq_len_hours", "rmse_raw", p1,
                              segmented_fit=fit,
                              xlabel="Average sequence length (hours)",
                              ylabel="Test RMSE (counts)")
        plot_importances(rf["permutation_importance"], p2)
        assert os.path.getsize(p1) > 10_000 and os.path.getsize(p2) > 10_000
        print(f"  figures rendered: {os.path.getsize(p1)//1024} KB, "
              f"{os.path.getsize(p2)//1024} KB")

    hr("DESK CHECK PASSED")


if __name__ == "__main__":
    main()
