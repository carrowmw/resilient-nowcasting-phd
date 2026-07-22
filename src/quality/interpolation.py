"""
Gap-filling variants for the interpolation experiment (paper §4.2, extended
per review: linear / spline / KNN in addition to the original night zero-fill).

Every method returns (filled_series, interpolated_mask). The mask travels with
the series so evaluation can be computed both on all targets and on
real-observations-only targets — interpolation must never silently launder
synthetic values into the metrics.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = ("none", "night_zero", "linear", "spline", "knn")


def apply_interpolation(
    series: pd.Series,
    method: str,
    night_hours: Tuple[int, int] = (0, 6),
    spline_order: int = 3,
    knn_neighbors: int = 5,
    max_gap_steps: int = 0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Args:
        series: NaN-gapped continuous-grid series.
        method: one of INTERPOLATION_METHODS.
        night_hours: [start, end) hour range for night_zero.
        max_gap_steps: for linear/spline — only fill gaps of at most this many
            steps (0 = no limit). Prevents interpolating across multi-day
            outages, which the original paper identified as the failure mode
            of naive filling.

    Returns:
        (filled, was_interpolated) — was_interpolated is a boolean Series
        aligned to the grid, True exactly where a value was synthesized.
    """
    assert method in INTERPOLATION_METHODS, \
        f"CRITICAL: unknown interpolation method {method!r}, expected one of {INTERPOLATION_METHODS}"
    assert isinstance(series.index, pd.DatetimeIndex), \
        "CRITICAL: series must have a DatetimeIndex"

    missing_before = series.isna()

    if method == "none":
        filled = series.copy()

    elif method == "night_zero":
        start_h, end_h = night_hours
        assert 0 <= start_h < end_h <= 24, \
            f"CRITICAL: bad night_hours {night_hours}"
        is_night = (series.index.hour >= start_h) & (series.index.hour < end_h)
        filled = series.copy()
        filled[missing_before & is_night] = 0.0

    elif method in ("linear", "spline"):
        limit = max_gap_steps if max_gap_steps > 0 else None
        if method == "linear":
            filled = series.interpolate(method="linear", limit=limit, limit_area="inside")
        else:
            filled = series.interpolate(
                method="spline", order=spline_order, limit=limit, limit_area="inside"
            )
        # Spline can overshoot into negative counts — clamp to the count domain
        # and record how much was clamped (no silent behavior).
        n_neg = int((filled < 0).sum())
        if n_neg > 0:
            logger.info(f"{method}: clamped {n_neg} negative interpolated values to 0")
            filled = filled.clip(lower=0.0)

    elif method == "knn":
        filled = _knn_impute(series, knn_neighbors)

    was_interpolated = missing_before & filled.notna()
    # Tripwire: interpolation must never alter observed values.
    observed = ~missing_before
    altered = int((filled[observed] != series[observed]).sum())
    assert altered == 0, \
        f"CRITICAL: {method} altered {altered} observed values"
    n_filled = int(was_interpolated.sum())
    logger.info(
        f"{series.name} [{method}]: filled {n_filled} of {int(missing_before.sum())} missing steps"
    )
    return filled, was_interpolated


def _knn_impute(series: pd.Series, n_neighbors: int) -> pd.Series:
    """
    KNN interpolation in a temporal feature space: a KNeighborsRegressor is
    fit on the OBSERVED points, described by (sin/cos hour-of-day, sin/cos
    day-of-week, normalized time), and missing counts are predicted from the
    k nearest temporal neighbors. (sklearn's KNNImputer is O(n^2) over ~1e5
    grid points and computationally infeasible here; a KD-tree regressor has
    identical semantics for a single target column.)
    """
    from sklearn.neighbors import KNeighborsRegressor

    assert n_neighbors >= 1, f"CRITICAL: knn_neighbors must be >= 1, got {n_neighbors}"
    idx = series.index
    t_norm = ((idx - idx.min()) / max((idx.max() - idx.min()), pd.Timedelta("1s"))).astype(float)
    hour = idx.hour + idx.minute / 60.0
    features = np.column_stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * idx.dayofweek / 7.0),
        np.cos(2 * np.pi * idx.dayofweek / 7.0),
        np.asarray(t_norm),
    ])
    observed = series.notna().to_numpy()
    missing = ~observed
    if not missing.any():
        return series.copy()
    assert observed.sum() >= n_neighbors, \
        f"CRITICAL: only {int(observed.sum())} observed points < k={n_neighbors}"

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    knn.fit(features[observed], series.to_numpy(dtype=float)[observed])
    predicted = knn.predict(features[missing])

    filled = series.copy()
    filled.iloc[np.flatnonzero(missing)] = predicted
    filled = filled.clip(lower=0.0)
    assert filled.notna().all(), "CRITICAL: KNN interpolation left unexpected NaNs"
    return filled
