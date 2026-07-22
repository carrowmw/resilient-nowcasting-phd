"""
Statistical analysis package for the quality track (reviewer requirements:
formal backing for the sequence-length-over-completeness claim, the ~20 h
threshold, and model comparisons; exploratory framing for the RF model).

Contents:
  - Spearman correlations with Benjamini–Hochberg FDR correction
  - Partial Spearman correlations (rank-residualization)
  - Bootstrap confidence intervals (sensor-level resampling)
  - Segmented (broken-stick) regression with grid-searched breakpoint,
    AIC comparison vs a single-slope model, bootstrap CI on the breakpoint
  - Diebold–Mariano paired forecast-accuracy test (HAC variance,
    Harvey–Leybourne–Newbold small-sample correction)
  - Exploratory Random Forest quality→error model with repeated K-fold CV
    and permutation importance
"""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- correlations

def spearman_with_fdr(
    quality_df: pd.DataFrame,
    error: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Spearman rho of each quality column vs the per-sensor error, with
    Benjamini–Hochberg FDR-adjusted p-values across the family of metrics.
    """
    from statsmodels.stats.multitest import multipletests

    assert len(quality_df) == len(error), \
        f"CRITICAL: {len(quality_df)} quality rows vs {len(error)} error rows"
    assert len(quality_df) >= 5, \
        f"CRITICAL: only {len(quality_df)} sensors — too few for correlation tests"

    rows = []
    for col in quality_df.columns:
        x = quality_df[col].to_numpy(dtype=float)
        y = error.to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        rho, p = sps.spearmanr(x[ok], y[ok])
        rows.append({"metric": col, "n": int(ok.sum()), "rho": float(rho), "p": float(p)})
    out = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(out["p"], alpha=alpha, method="fdr_bh")
    out["p_fdr"] = p_adj
    out["significant_fdr"] = reject
    return out


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
) -> Tuple[float, float]:
    """
    Partial Spearman correlation of x and y given controls [n, k]:
    rank-transform everything, residualize x and y on the ranked controls by
    OLS, then Pearson-correlate the residuals (equivalent to partial Spearman).
    """
    assert x.ndim == 1 and y.ndim == 1, "CRITICAL: x, y must be 1-D"
    controls = np.atleast_2d(controls)
    if controls.shape[0] != len(x):
        controls = controls.T
    assert controls.shape[0] == len(x), \
        f"CRITICAL: controls shape {controls.shape} incompatible with n={len(x)}"

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    n = int(ok.sum())
    k = controls.shape[1]
    assert n > k + 3, f"CRITICAL: n={n} too small for {k} controls"

    rx = sps.rankdata(x[ok])
    ry = sps.rankdata(y[ok])
    rc = np.column_stack([sps.rankdata(controls[ok, j]) for j in range(k)])
    design = np.column_stack([np.ones(n), rc])

    def residual(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    r, _ = sps.pearsonr(residual(rx), residual(ry))
    # p-value with reduced degrees of freedom (n - 2 - k)
    dof = n - 2 - k
    t = r * np.sqrt(dof / max(1e-12, 1.0 - r ** 2))
    p = 2.0 * sps.t.sf(abs(t), dof)
    return float(r), float(p)


# ------------------------------------------------------------------- bootstrap

def bootstrap_ci(
    stat_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Percentile bootstrap CI for a scalar statistic of one or more aligned
    arrays (resampled jointly — sensor-level resampling when rows are sensors).
    """
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays), "CRITICAL: arrays must be aligned"
    assert n >= 5, f"CRITICAL: n={n} too small to bootstrap"
    rng = np.random.default_rng(seed)
    point = float(stat_fn(*arrays))
    draws = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(*(a[idx] for a in arrays))
    draws = draws[np.isfinite(draws)]
    assert len(draws) > 0.9 * n_bootstrap, \
        "CRITICAL: >10% of bootstrap draws non-finite — statistic is unstable"
    return {
        "point": point,
        "ci_lower": float(np.percentile(draws, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "n_bootstrap": int(len(draws)),
    }


# ---------------------------------------------------------- threshold analysis

@dataclass
class SegmentedFit:
    breakpoint: float
    slope_below: float
    slope_above: float
    intercept: float
    aic_segmented: float
    aic_linear: float
    delta_aic: float          # linear - segmented (positive favors a threshold)
    breakpoint_ci: Tuple[float, float]
    n: int


def segmented_threshold_fit(
    x: np.ndarray,
    y: np.ndarray,
    n_grid: int = 200,
    n_bootstrap: int = 1000,
    seed: int = 42,
    grid_quantiles: Tuple[float, float] = (0.10, 0.90),
) -> SegmentedFit:
    """
    Broken-stick regression y = b0 + b1·x + b2·max(0, x - c), with the
    breakpoint c grid-searched over the central quantile range of x.

    Evidence for a threshold is reported as ΔAIC (single-slope minus
    segmented) plus a bootstrap CI on c. A classical p-value for b2 at the
    *estimated* breakpoint is not valid (Davies' problem: the breakpoint is a
    nuisance parameter absent under H0), so none is fabricated here.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    assert n >= 10, f"CRITICAL: n={n} too small for segmented regression"

    def fit_at(c: float, xs: np.ndarray, ys: np.ndarray) -> Tuple[float, np.ndarray]:
        design = np.column_stack([np.ones(len(xs)), xs, np.maximum(0.0, xs - c)])
        beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
        rss = float(np.sum((ys - design @ beta) ** 2))
        return rss, beta

    def best_breakpoint(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, np.ndarray]:
        lo, hi = np.quantile(xs, grid_quantiles)
        grid = np.linspace(lo, hi, n_grid)
        rss_best, c_best, beta_best = np.inf, grid[0], None
        for c in grid:
            rss, beta = fit_at(c, xs, ys)
            if rss < rss_best:
                rss_best, c_best, beta_best = rss, c, beta
        return c_best, rss_best, beta_best

    c_hat, rss_seg, beta = best_breakpoint(x, y)

    # Single-slope comparison model.
    design_lin = np.column_stack([np.ones(n), x])
    beta_lin, *_ = np.linalg.lstsq(design_lin, y, rcond=None)
    rss_lin = float(np.sum((y - design_lin @ beta_lin) ** 2))

    def aic(rss: float, k: int) -> float:
        return n * np.log(max(rss, 1e-300) / n) + 2 * k

    aic_seg = aic(rss_seg, 4)   # b0, b1, b2 + breakpoint counts as a parameter
    aic_lin = aic(rss_lin, 2)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        cb, _, _ = best_breakpoint(x[idx], y[idx])
        boots.append(cb)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    return SegmentedFit(
        breakpoint=float(c_hat),
        slope_below=float(beta[1]),
        slope_above=float(beta[1] + beta[2]),
        intercept=float(beta[0]),
        aic_segmented=float(aic_seg),
        aic_linear=float(aic_lin),
        delta_aic=float(aic_lin - aic_seg),
        breakpoint_ci=ci,
        n=n,
    )


# ------------------------------------------------------------ Diebold–Mariano

def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
) -> Dict[str, float]:
    """
    DM test for equal forecast accuracy of two models over the SAME targets.
    errors_a/b are forecast errors (pred - true), aligned window-for-window.
    HAC (Bartlett) variance with lag h-1; Harvey–Leybourne–Newbold
    small-sample correction; two-sided p from Student-t(n-1).

    Negative DM statistic → model A more accurate.
    """
    assert errors_a.shape == errors_b.shape, "CRITICAL: DM inputs must be aligned"
    assert loss in ("squared", "absolute"), f"CRITICAL: unknown DM loss {loss!r}"
    ok = np.isfinite(errors_a) & np.isfinite(errors_b)
    ea, eb = errors_a[ok], errors_b[ok]
    n = len(ea)
    assert n >= 30, f"CRITICAL: only {n} paired forecasts — DM test unreliable"

    if loss == "squared":
        d = ea ** 2 - eb ** 2
    else:
        d = np.abs(ea) - np.abs(eb)

    d_bar = float(d.mean())
    h = max(1, int(horizon))
    dc = d - d_bar
    gamma0 = float(np.mean(dc * dc))
    var_d = gamma0
    for lag in range(1, h):
        cov = float(np.mean(dc[lag:] * dc[:-lag]))
        var_d += 2.0 * cov
    if var_d <= 0:
        # Long-run variance can go negative with truncated kernels; fall back
        # to gamma0 and say so rather than fabricating a huge statistic.
        logger.warning("DM: non-positive HAC variance, falling back to lag-0 variance")
        var_d = gamma0
    assert var_d > 0, "CRITICAL: zero variance of loss differential"

    dm = d_bar / np.sqrt(var_d / n)
    # Harvey–Leybourne–Newbold correction.
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2.0 * sps.t.sf(abs(dm_hln), df=n - 1)
    return {
        "dm_stat": float(dm_hln),
        "p_value": float(p),
        "mean_loss_diff": d_bar,
        "n": int(n),
        "horizon": h,
        "loss": loss,
    }


def dm_matrix(
    predictions: pd.DataFrame,
    baseline_models: Sequence[str],
    focal_model: str,
    horizon: int,
    loss: str = "squared",
) -> pd.DataFrame:
    """
    Per-sensor DM tests of focal_model vs each baseline on the shared test
    windows, from the tidy predictions frame (see evaluate.predictions_frame).
    """
    test = predictions[
        (predictions["split"] == "test") & (predictions["horizon"] == horizon)
    ]
    rows = []
    for sensor_id, group in test.groupby("sensor_id"):
        pivots = {}
        for model in [focal_model, *baseline_models]:
            sub = group[group["model"] == model].sort_values("target_dt")
            pivots[model] = sub
        focal = pivots[focal_model]
        if focal.empty:
            continue
        for baseline in baseline_models:
            base = pivots[baseline]
            if base.empty:
                continue
            merged = focal.merge(
                base, on=["target_dt", "seq_id"], suffixes=("_f", "_b"), how="inner"
            )
            assert len(merged) == len(focal) == len(base), (
                f"CRITICAL: unpaired forecasts for {sensor_id} "
                f"({focal_model}:{len(focal)} vs {baseline}:{len(base)} vs merged:{len(merged)}) "
                "— models did not predict the identical window set"
            )
            ea = (merged["y_pred_raw_f"] - merged["y_true_raw_f"]).to_numpy()
            eb = (merged["y_pred_raw_b"] - merged["y_true_raw_b"]).to_numpy()
            res = diebold_mariano(ea, eb, horizon=horizon, loss=loss)
            rows.append({
                "sensor_id": sensor_id,
                "focal": focal_model,
                "baseline": baseline,
                **res,
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------- exploratory RF model

def rf_quality_to_error(
    quality_df: pd.DataFrame,
    error: pd.Series,
    n_estimators: int = 300,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict:
    """
    EXPLORATORY Random Forest predicting per-sensor error from quality
    indicators. Repeated K-fold R² with dispersion and permutation importance
    — reported with uncertainty, per the reviewers' insistence that this
    analysis not be oversold.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import RepeatedKFold, cross_val_score

    X = quality_df.to_numpy(dtype=float)
    y = error.to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    assert len(y) >= n_splits * 2, \
        f"CRITICAL: {len(y)} sensors too few for {n_splits}-fold CV"

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, oob_score=True)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")

    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=seed, scoring="r2")

    return {
        "exploratory": True,
        "n_sensors": int(len(y)),
        "r2_cv_mean": float(scores.mean()),
        "r2_cv_std": float(scores.std(ddof=0)),
        "r2_cv_quantiles": {
            "q05": float(np.quantile(scores, 0.05)),
            "q50": float(np.quantile(scores, 0.50)),
            "q95": float(np.quantile(scores, 0.95)),
        },
        "oob_r2": float(rf.oob_score_),
        "permutation_importance": {
            col: {"mean": float(m), "std": float(s)}
            for col, m, s in zip(
                quality_df.columns, perm.importances_mean, perm.importances_std
            )
        },
    }
