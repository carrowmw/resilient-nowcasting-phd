# Tests for src/quality/stats.py — synthetic cases with known ground truth.
import numpy as np
import pandas as pd
import pytest

from quality.stats import (
    bootstrap_ci,
    diebold_mariano,
    dm_matrix,
    partial_spearman,
    rf_quality_to_error,
    segmented_threshold_fit,
    spearman_with_fdr,
)


def test_spearman_fdr_detects_real_and_null():
    rng = np.random.default_rng(11)
    n = 60
    signal = rng.normal(0, 1, n)
    error = pd.Series(-0.9 * signal + rng.normal(0, 0.3, n))
    qdf = pd.DataFrame({
        "real_metric": signal,
        "noise_1": rng.normal(0, 1, n),
        "noise_2": rng.normal(0, 1, n),
    })
    out = spearman_with_fdr(qdf, error)
    real_row = out[out["metric"] == "real_metric"].iloc[0]
    assert bool(real_row["significant_fdr"])
    assert real_row["rho"] < -0.6
    # FDR-adjusted p never smaller than raw p.
    assert (out["p_fdr"] >= out["p"] - 1e-12).all()


def test_partial_spearman_removes_confounder():
    rng = np.random.default_rng(5)
    n = 200
    confounder = rng.normal(0, 1, n)
    x = confounder + rng.normal(0, 0.1, n)   # x is mostly the confounder
    y = confounder + rng.normal(0, 0.1, n)   # y too
    r_raw, _ = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(x, y)
    r_partial, p = partial_spearman(x, y, confounder.reshape(-1, 1))
    assert abs(r_partial) < abs(r_raw) - 0.3  # correlation collapses given control


def test_bootstrap_ci_covers_point():
    rng = np.random.default_rng(2)
    x = rng.normal(5, 1, 100)
    out = bootstrap_ci(lambda a: float(np.mean(a)), x, n_bootstrap=500)
    assert out["ci_lower"] < out["point"] < out["ci_upper"]
    assert out["ci_lower"] < 5 < out["ci_upper"]


def test_segmented_fit_recovers_known_breakpoint():
    rng = np.random.default_rng(9)
    n = 120
    x = rng.uniform(5, 60, n)
    # Steep decline below 20, flat above — the paper's hypothesized shape.
    y = np.where(x < 20, 3.0 - 0.10 * x, 1.0) + rng.normal(0, 0.08, n)
    fit = segmented_threshold_fit(x, y, n_bootstrap=200, seed=1)
    assert fit.breakpoint == pytest.approx(20.0, abs=3.0)
    assert fit.delta_aic > 10            # threshold model strongly preferred
    assert fit.slope_below < fit.slope_above - 0.05
    assert fit.breakpoint_ci[0] <= fit.breakpoint <= fit.breakpoint_ci[1]


def test_segmented_fit_no_threshold_in_linear_data():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 50, 100)
    y = 2.0 - 0.03 * x + rng.normal(0, 0.1, 100)
    fit = segmented_threshold_fit(x, y, n_bootstrap=200, seed=1)
    assert fit.delta_aic < 10            # no strong evidence for a break


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(6)
    n = 500
    errors_good = rng.normal(0, 1.0, n)
    errors_bad = rng.normal(0, 2.0, n)
    out = diebold_mariano(errors_good, errors_bad, horizon=8)
    assert out["dm_stat"] < 0            # negative → first model better
    assert out["p_value"] < 0.01


def test_diebold_mariano_equal_models_not_significant():
    rng = np.random.default_rng(8)
    n = 500
    ea = rng.normal(0, 1.0, n)
    eb = rng.normal(0, 1.0, n)
    out = diebold_mariano(ea, eb, horizon=8)
    assert out["p_value"] > 0.05


def test_dm_matrix_pairs_on_shared_windows():
    rng = np.random.default_rng(12)
    n = 200
    target_dt = pd.date_range("2024-01-01", periods=n, freq="15min")
    y = rng.normal(50, 10, n)
    rows = []
    for model, sigma in (("lstm_uv", 1.0), ("persistence", 3.0)):
        rows.append(pd.DataFrame({
            "sensor_id": "S__a", "model": model, "horizon": 8,
            "interpolation": "none", "ablation": "", "split": "test",
            "origin_dt": target_dt, "target_dt": target_dt,
            "seq_id": 0,
            "y_true_raw": y, "y_pred_raw": y + rng.normal(0, sigma, n),
            "y_true_scaled": y, "y_pred_scaled": y,
            "target_interpolated": False,
        }))
    preds = pd.concat(rows, ignore_index=True)
    dm = dm_matrix(preds, ["persistence"], "lstm_uv", horizon=8)
    assert len(dm) == 1
    assert dm.iloc[0]["dm_stat"] < 0 and dm.iloc[0]["p_value"] < 0.05


def test_rf_exploratory_flags_and_importance():
    rng = np.random.default_rng(3)
    n = 80
    seq_len = rng.uniform(5, 60, n)
    qdf = pd.DataFrame({
        "avg_seq_len_hours": seq_len,
        "completeness_active": rng.uniform(0.5, 1.0, n),
        "noise": rng.normal(0, 1, n),
    })
    error = pd.Series(3.0 - 0.04 * seq_len + rng.normal(0, 0.1, n))
    out = rf_quality_to_error(qdf, error, n_estimators=100, n_repeats=3)
    assert out["exploratory"] is True
    assert out["r2_cv_mean"] > 0.3
    imp = out["permutation_importance"]
    assert imp["avg_seq_len_hours"]["mean"] > imp["noise"]["mean"]
