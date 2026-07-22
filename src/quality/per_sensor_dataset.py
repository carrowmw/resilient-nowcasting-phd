"""
Per-sensor windowing, splitting, feature construction and scaling.

Split design (documented deviation matrix vs the original paper):
  - Per-sequence chronological split: each consecutive run's windows are
    split 70/15/15 in time order, so every split draws from the sensor's whole
    active period rather than test being only the final months.
  - Boundary purging (default on): the first (window + horizon - 1) windows of
    val and test are dropped so no input/target step is shared across a split
    boundary. The original stride-1 sequential split leaked overlapping
    windows across boundaries.
  - StandardScaler (count and timestamp features) is fit on TRAIN data only.
    The original code fit on the full series — leakage, fixed here.
"""
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from quality.contract import (
    ALL_FEATURE_GROUPS,
    FEATURE_GROUP_DIMS,
    PerSensorSplit,
    SensorContext,
)
from quality.quality_metrics import (
    SequenceSpan,
    assign_split_windows,
    find_consecutive_sequences,
    windows_in_span,
)

logger = logging.getLogger(__name__)


def build_sensor_splits(
    series: pd.Series,
    interpolated_mask: pd.Series,
    window: int,
    horizon: int,
    stride: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_boundaries: bool = True,
) -> Tuple[SensorContext, Dict[str, PerSensorSplit]]:
    """
    Build the SensorContext (train-only scaler) and the three splits for one
    sensor at one horizon. `series` may already be interpolated; the
    interpolated_mask keeps synthetic values traceable through evaluation.
    """
    assert series.index.equals(interpolated_mask.index), \
        "CRITICAL: series and interpolated_mask must share the grid index"
    spans = find_consecutive_sequences(series, min_length=window + horizon)
    split_idx = assign_split_windows(
        spans, window, horizon, stride, train_frac, val_frac, purge_boundaries
    )

    values_raw = series.to_numpy(dtype=float)

    # Train-only scaler: fit over every step touched by a train window
    # (inputs and target), never over val/test steps.
    train_starts = split_idx["train"][0]
    assert len(train_starts) > 0, \
        f"CRITICAL: sensor {series.name} has no train windows at horizon {horizon}"
    span_len = window + horizon
    touched = np.unique(
        (train_starts[:, None] + np.arange(span_len)[None, :]).ravel()
    )
    train_values = values_raw[touched]
    train_values = train_values[~np.isnan(train_values)]
    assert len(train_values) > 0, "CRITICAL: no observed values under train windows"
    mean = float(train_values.mean())
    std = float(train_values.std(ddof=0))
    if std < 1e-8:
        # A constant train series is degenerate but must not silently divide
        # by ~0; scale by 1 and record the fact via the context.
        logger.warning(f"{series.name}: constant train values (std={std:.3g}); using std=1")
        std = 1.0

    context = SensorContext(
        sensor_id=str(series.name),
        grid_index=series.index,
        values_raw=values_raw,
        values_scaled=(values_raw - mean) / std,
        interpolated_mask=np.array(interpolated_mask.to_numpy(dtype=bool)),
        scaler_mean=mean,
        scaler_std=std,
        window=window,
        horizon=horizon,
        freq=pd.infer_freq(series.index) or "unknown",
    )

    splits = {
        name: PerSensorSplit(
            context=context,
            split=name,
            window_start_idx=idx,
            seq_id=seq,
        )
        for name, (idx, seq) in split_idx.items()
    }
    # Tripwire: no window may straddle a NaN (spans guarantee this; verify on
    # a sample rather than materializing everything).
    for name, split in splits.items():
        if len(split) == 0:
            continue
        probe = split.window_start_idx[:: max(1, len(split) // 32)]
        for s in probe:
            chunk = values_raw[s: s + span_len]
            assert not np.isnan(chunk).any(), \
                f"CRITICAL: NaN inside a {name} window of {series.name} at idx {s}"
    return context, splits


def train_segments(train_split: PerSensorSplit) -> List[Tuple[int, int]]:
    """
    Contiguous grid-index ranges [start, end] (inclusive) covered by train
    windows, one per consecutive run — the fitting series for ARIMA-style
    models that consume history rather than windows.
    """
    segs: List[Tuple[int, int]] = []
    ctx = train_split.context
    span_len = ctx.window + ctx.horizon
    for seq in np.unique(train_split.seq_id):
        starts = train_split.window_start_idx[train_split.seq_id == seq]
        segs.append((int(starts.min()), int(starts.max()) + span_len - 1))
    return segs


def build_feature_matrix(
    context: SensorContext,
    feature_groups: Sequence[str],
    fractional_time: bool = True,
) -> np.ndarray:
    """
    [T, D] feature matrix over the sensor grid for the selected groups.
    Cached on the context per group tuple.

    Cyclic encodings port the legacy create_periodicity_features semantics;
    with fractional_time=True, hour-of-day is fractional (15-min resolution)
    instead of the legacy integer hour (which made the encodings step-wise
    constant within each hour).
    """
    key = (tuple(feature_groups), fractional_time)
    if key in context._feature_cache:
        return context._feature_cache[key]

    assert feature_groups[0] == "count", \
        f"CRITICAL: 'count' must be the first feature group, got {feature_groups}"
    unknown = set(feature_groups) - set(ALL_FEATURE_GROUPS)
    assert not unknown, f"CRITICAL: unknown feature groups {unknown}"

    idx = context.grid_index
    hour = idx.hour + (idx.minute / 60.0 if fractional_time else 0.0)
    week = idx.isocalendar().week.to_numpy(dtype=float)
    doy = idx.dayofyear.to_numpy(dtype=float)

    cols: List[np.ndarray] = []
    for group in feature_groups:
        if group == "count":
            cols.append(context.values_scaled)
        elif group == "half_day":
            cols.append(np.sin(2 * np.pi * hour / 12.0))
            cols.append(np.cos(2 * np.pi * hour / 12.0))
        elif group == "day":
            cols.append(np.sin(2 * np.pi * hour / 24.0))
            cols.append(np.cos(2 * np.pi * hour / 24.0))
        elif group == "week":
            cols.append(np.sin(2 * np.pi * week / 52.0))
            cols.append(np.cos(2 * np.pi * week / 52.0))
        elif group == "quarter":
            cols.append(np.sin(2 * np.pi * doy / 91.25))
            cols.append(np.cos(2 * np.pi * doy / 91.25))
        elif group == "year":
            cols.append(np.sin(2 * np.pi * doy / 365.0))
            cols.append(np.cos(2 * np.pi * doy / 365.0))
        elif group == "timestamp":
            # Normalized against the grid span; affine, so train-only fitting
            # is irrelevant here (unlike the count scaler).
            t = (idx - idx[0]) / (idx[-1] - idx[0])
            cols.append(t.to_numpy(dtype=float) * 2.0 - 1.0)

    matrix = np.column_stack(cols).astype(np.float32)
    expected_d = sum(FEATURE_GROUP_DIMS[g] for g in feature_groups)
    assert matrix.shape == (len(idx), expected_d), \
        f"CRITICAL: feature matrix shape {matrix.shape} != ({len(idx)}, {expected_d})"
    context._feature_cache[key] = matrix
    return matrix


class WindowDataset(Dataset):
    """
    Lazy torch dataset over one PerSensorSplit: gathers [window, D] inputs
    from the shared feature matrix per item; targets are scaled scalars.
    """

    def __init__(self, split: PerSensorSplit, feature_groups: Sequence[str],
                 fractional_time: bool = True):
        assert len(split) > 0, \
            f"CRITICAL: empty {split.split} split for {split.context.sensor_id}"
        self._features = build_feature_matrix(split.context, feature_groups, fractional_time)
        self._starts = split.window_start_idx
        self._window = split.context.window
        self._targets = split.y_scaled.astype(np.float32)
        nan_count = int(np.isnan(self._features[:, 0][self._starts]).sum())
        assert nan_count == 0, "CRITICAL: NaN at window starts in feature matrix"

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, i: int):
        s = self._starts[i]
        x = self._features[s: s + self._window]
        return torch.from_numpy(x.copy()), torch.tensor(self._targets[i])
