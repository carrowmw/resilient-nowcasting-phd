# Tests for src/quality/per_sensor_dataset.py — splits, scaling, features.
import numpy as np
import pandas as pd
import pytest
import torch

from quality.contract import FEATURE_GROUP_DIMS, ALL_FEATURE_GROUPS
from quality.per_sensor_dataset import (
    WindowDataset,
    assign_split_windows,
    build_feature_matrix,
    build_sensor_splits,
    train_segments,
)
from quality.quality_metrics import count_usable_windows, find_consecutive_sequences


def make_series(n=2000, gap_at=(800, 850), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    hours = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    values = 50 + 40 * np.sin(2 * np.pi * hours / 24.0) + rng.normal(0, 3, n)
    values = np.clip(values, 0, None)
    values[gap_at[0]: gap_at[1]] = np.nan
    s = pd.Series(values, index=idx, name="SYN__a_to_b")
    mask = pd.Series(False, index=idx)
    return s, mask


def build(series=None, mask=None, window=24, horizon=8, purge=True):
    if series is None:
        series, mask = make_series()
    return build_sensor_splits(
        series, mask, window=window, horizon=horizon,
        train_frac=0.7, val_frac=0.15, purge_boundaries=purge,
    )


def test_split_counts_match_quality_metrics():
    """The viability arithmetic and the actual datasets must agree exactly."""
    series, mask = make_series()
    spans = find_consecutive_sequences(series, min_length=32)
    expected = count_usable_windows(spans, 24, 8, 1, 0.7, 0.15, True)
    _, splits = build(series, mask)
    assert len(splits["train"]) == expected["train_windows"]
    assert len(splits["val"]) == expected["val_windows"]
    assert len(splits["test"]) == expected["test_windows"]


def test_splits_chronological_and_disjoint_within_sequence():
    _, splits = build()
    for seq in np.unique(splits["train"].seq_id):
        tr = splits["train"].window_start_idx[splits["train"].seq_id == seq]
        va = splits["val"].window_start_idx[splits["val"].seq_id == seq]
        te = splits["test"].window_start_idx[splits["test"].seq_id == seq]
        if len(va):
            assert tr.max() < va.min()
        if len(te) and len(va):
            assert va.max() < te.min()


def test_purge_removes_boundary_overlap():
    _, splits = build(purge=True)
    span = 24 + 8
    for a, b in (("train", "val"), ("val", "test")):
        for seq in np.unique(splits[a].seq_id):
            ia = splits[a].window_start_idx[splits[a].seq_id == seq]
            ib = splits[b].window_start_idx[splits[b].seq_id == seq]
            if len(ia) and len(ib):
                # No step covered by both splits' (input+target) extents.
                assert ib.min() - ia.max() >= span


def test_scaler_fit_on_train_only():
    """Leakage regression test: scaler stats must ignore val/test values."""
    series, mask = make_series()
    # Poison the tail (test region) with huge values; scaler must not move.
    poisoned = series.copy()
    ctx_clean, _ = build(series, mask)
    poisoned.iloc[-200:] = poisoned.iloc[-200:] + 10000.0
    ctx_poisoned, _ = build(poisoned, mask)
    assert ctx_clean.scaler_mean == pytest.approx(ctx_poisoned.scaler_mean, rel=1e-9)
    assert ctx_clean.scaler_std == pytest.approx(ctx_poisoned.scaler_std, rel=1e-9)


def test_no_nan_in_targets_or_windows():
    _, splits = build()
    for split in splits.values():
        assert not np.isnan(split.y_raw).any()
        assert not np.isnan(split.y_scaled).any()


def test_target_arithmetic():
    ctx, splits = build()
    split = splits["test"]
    assert (split.target_idx == split.window_start_idx + 24 + 8 - 1).all()
    np.testing.assert_allclose(
        split.y_raw, ctx.values_raw[split.window_start_idx + 31]
    )


def test_feature_matrix_shapes_and_ranges():
    ctx, _ = build()
    full = list(ALL_FEATURE_GROUPS)
    m = build_feature_matrix(ctx, full)
    assert m.shape == (len(ctx.grid_index), sum(FEATURE_GROUP_DIMS[g] for g in full))
    # All non-count features bounded in [-1, 1].
    assert np.nanmax(np.abs(m[:, 1:])) <= 1.0 + 1e-6
    uv = build_feature_matrix(ctx, ["count"])
    assert uv.shape[1] == 1


def test_feature_matrix_day_period():
    ctx, _ = build()
    m = build_feature_matrix(ctx, ["count", "day"])
    sin_day = m[:, 1]
    # 96 steps apart (24 h) must give identical encodings.
    np.testing.assert_allclose(sin_day[:500], sin_day[96:596], atol=1e-5)


def test_window_dataset_item():
    ctx, splits = build()
    ds = WindowDataset(splits["train"], ["count", "day"])
    x, y = ds[0]
    assert x.shape == (24, 3)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all() and torch.isfinite(y)
    assert len(ds) == len(splits["train"])


def test_train_segments_contiguous_and_nan_free():
    ctx, splits = build()
    segs = train_segments(splits["train"])
    assert segs
    for start, end in segs:
        assert not np.isnan(ctx.values_raw[start: end + 1]).any()


def test_constant_series_does_not_divide_by_zero():
    idx = pd.date_range("2024-01-01", periods=500, freq="15min")
    series = pd.Series(5.0, index=idx, name="CONST__x")
    mask = pd.Series(False, index=idx)
    ctx, _ = build(series, mask)
    assert ctx.scaler_std == 1.0
    assert np.isfinite(ctx.values_scaled).all()
