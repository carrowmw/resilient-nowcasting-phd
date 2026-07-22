"""
Per-sensor data quality metrics and the sensor viability rule.

Quality indicators (per directional series):
  - completeness_active: observed / expected over the series' own active span
  - completeness_window: observed / expected over the configured analysis window
  - avg/std/count/max of maximal consecutive (gap-free) run lengths
  - usable sliding-window counts per split and horizon

The consecutive-run semantics port the legacy find_consecutive_sequences onto
the current NaN-gapped continuous grid: a "gap" is a NaN run, not a missing
timestamp. Runs are maximal non-NaN stretches; window counting applies the
window/horizon arithmetic per run.
"""
import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceSpan:
    """A maximal consecutive (gap-free) run within a gridded series."""
    start: pd.Timestamp
    end: pd.Timestamp        # inclusive
    n_steps: int
    start_idx: int           # positional index into the gridded series
    end_idx: int             # inclusive


def find_consecutive_sequences(
    series: pd.Series,
    min_length: int = 1,
) -> List[SequenceSpan]:
    """
    Extract maximal non-NaN runs of at least min_length steps.

    The series must be on a continuous grid (constant frequency index);
    missingness is expressed as NaN values.
    """
    assert isinstance(series.index, pd.DatetimeIndex), \
        "CRITICAL: series must have a DatetimeIndex"
    assert min_length >= 1, f"CRITICAL: min_length must be >= 1, got {min_length}"
    if len(series) > 1:
        deltas = series.index.to_series().diff().dropna().unique()
        assert len(deltas) == 1, (
            f"CRITICAL: series index is not a continuous grid — "
            f"{len(deltas)} distinct step sizes found"
        )

    observed = series.notna().to_numpy()
    if not observed.any():
        return []

    # Run-length encode the observed mask.
    change = np.flatnonzero(np.diff(observed.astype(np.int8)))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [len(observed) - 1]))

    spans: List[SequenceSpan] = []
    for s, e in zip(starts, ends):
        if not observed[s]:
            continue
        n = int(e - s + 1)
        if n < min_length:
            continue
        spans.append(SequenceSpan(
            start=series.index[s],
            end=series.index[e],
            n_steps=n,
            start_idx=int(s),
            end_idx=int(e),
        ))

    total_observed = int(observed.sum())
    counted = sum(sp.n_steps for sp in spans)
    assert counted <= total_observed, \
        f"CRITICAL: run-length accounting error ({counted} > {total_observed})"
    return spans


def compute_completeness(series: pd.Series, expected_steps: int = 0) -> float:
    """
    observed / expected. With expected_steps=0, expected = len(series)
    (the series' own active-span grid).
    """
    observed = int(series.notna().sum())
    expected = expected_steps if expected_steps > 0 else len(series)
    assert expected > 0, "CRITICAL: expected_steps must be positive"
    completeness = observed / expected
    assert 0.0 <= completeness <= 1.0, \
        f"CRITICAL: completeness {completeness:.4f} outside [0, 1] — expected_steps too small?"
    return completeness


def compute_sequence_stats(spans: List[SequenceSpan], steps_per_hour: float) -> Dict[str, float]:
    """Summary statistics of consecutive-run lengths, in steps and hours."""
    assert steps_per_hour > 0, f"CRITICAL: steps_per_hour must be positive, got {steps_per_hour}"
    if not spans:
        return {
            "seq_count": 0,
            "avg_seq_len_steps": 0.0, "std_seq_len_steps": 0.0, "max_seq_len_steps": 0,
            "avg_seq_len_hours": 0.0, "std_seq_len_hours": 0.0, "max_seq_len_hours": 0.0,
        }
    lengths = np.array([sp.n_steps for sp in spans], dtype=float)
    return {
        "seq_count": int(len(lengths)),
        "avg_seq_len_steps": float(lengths.mean()),
        "std_seq_len_steps": float(lengths.std(ddof=0)),
        "max_seq_len_steps": int(lengths.max()),
        "avg_seq_len_hours": float(lengths.mean() / steps_per_hour),
        "std_seq_len_hours": float(lengths.std(ddof=0) / steps_per_hour),
        "max_seq_len_hours": float(lengths.max() / steps_per_hour),
    }


def windows_in_span(n_steps: int, window: int, horizon: int, stride: int = 1) -> int:
    """
    Number of sliding windows in a run of n_steps: input steps [i, i+window),
    scalar target at step i + window + horizon - 1.
    """
    assert window >= 1 and horizon >= 1 and stride >= 1, \
        f"CRITICAL: window/horizon/stride must be >= 1 (got {window}/{horizon}/{stride})"
    usable = n_steps - (window + horizon) + 1
    return max(0, (usable + stride - 1) // stride) if usable > 0 else 0


def assign_split_windows(
    spans: List[SequenceSpan],
    window: int,
    horizon: int,
    stride: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_boundaries: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Chronological split of window start indices, shared by the viability rule
    (count_usable_windows) and the actual datasets (build_sensor_splits) so
    the two can never disagree.

    Long runs are split train/val/test = train_frac/val_frac/rest internally,
    in time order. With purge_boundaries, the first (window + horizon - 1)
    windows of val and test are dropped so no input/target step is shared
    across a split boundary (the original paper's stride-1 sequential split
    leaked overlapping windows across boundaries).

    Runs too short to give every split at least one window after purging are
    assigned WHOLLY to a single split, chosen to keep the global window
    counts closest to the target fractions (deterministic, chronological).
    Whole-run assignment has zero cross-split leakage by construction, and it
    keeps short-run (low-quality) sensors in the study — the very cases the
    quality analysis is about; purging alone would systematically select
    them out.

    Returns {split: (window_start_idx [n], seq_id [n])}.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1, \
        f"CRITICAL: invalid split fractions train={train_frac}, val={val_frac}"
    purge = (window + horizon - 1) if purge_boundaries else 0
    targets = {"train": train_frac, "val": val_frac, "test": 1.0 - train_frac - val_frac}
    out = {s: ([], []) for s in ("train", "val", "test")}
    totals = {s: 0 for s in ("train", "val", "test")}

    def push(split: str, starts: np.ndarray, seq_id: int):
        if len(starts):
            out[split][0].append(starts)
            out[split][1].append(np.full(len(starts), seq_id, dtype=np.int64))
            totals[split] += len(starts)

    for seq_id, sp in enumerate(spans):
        n = windows_in_span(sp.n_steps, window, horizon, stride)
        if n == 0:
            continue
        starts = sp.start_idx + np.arange(n, dtype=np.int64) * stride
        n_train = int(np.floor(n * train_frac))
        n_val = int(np.floor(n * val_frac))
        n_test = n - n_train - n_val
        internal_ok = n_train >= 1 and (n_val - purge) >= 1 and (n_test - purge) >= 1
        if internal_ok:
            push("train", starts[:n_train], seq_id)
            push("val", starts[n_train + purge: n_train + n_val], seq_id)
            push("test", starts[n_train + n_val + purge:], seq_id)
        else:
            grand = sum(totals.values()) + n
            deficits = {s: targets[s] * grand - totals[s] for s in totals}
            # Deterministic tie-break: train > val > test.
            order = {"train": 0, "val": 1, "test": 2}
            chosen = max(deficits, key=lambda s: (deficits[s], -order[s]))
            push(chosen, starts, seq_id)

    return {
        split: (
            np.concatenate(starts) if starts else np.empty(0, dtype=np.int64),
            np.concatenate(seqs) if seqs else np.empty(0, dtype=np.int64),
        )
        for split, (starts, seqs) in out.items()
    }


def count_usable_windows(
    spans: List[SequenceSpan],
    window: int,
    horizon: int,
    stride: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_boundaries: bool = True,
) -> Dict[str, int]:
    """Windows per split — thin wrapper over assign_split_windows."""
    assigned = assign_split_windows(
        spans, window, horizon, stride, train_frac, val_frac, purge_boundaries
    )
    return {f"{split}_windows": len(idx) for split, (idx, _) in assigned.items()}


def _steps_per_hour(freq: str) -> float:
    step = pd.Timedelta(freq)
    assert step > pd.Timedelta(0), f"CRITICAL: bad frequency {freq!r}"
    return pd.Timedelta(hours=1) / step


def build_quality_table(store, config: dict) -> pd.DataFrame:
    """
    One row per directional series with all quality indicators and per-horizon
    usable-window counts, plus the viability verdict.

    `store` is a SensorSeriesStore (or any object with series_ids()/get_series()/
    expected_steps_in_window()). Interpolation variants pass a store whose
    get_series() returns the interpolated grid.
    """
    q_cfg = config["quality"]
    window = int(q_cfg["window"])
    horizons = [int(h) for h in q_cfg["horizons"]]
    stride = int(q_cfg.get("stride", 1))
    train_frac = float(q_cfg["train_frac"])
    val_frac = float(q_cfg["val_frac"])
    purge = bool(q_cfg.get("purge_boundaries", True))
    min_windows = q_cfg["viability"]["min_windows"]  # {'train': int, 'val': int, 'test': int}
    sph = _steps_per_hour(config["data"]["freq"])

    rows = []
    for series_id in store.series_ids():
        series = store.get_series(series_id)
        spans = find_consecutive_sequences(series, min_length=1)
        row: Dict = {"sensor_id": series_id}
        row["first_obs"] = series.index.min()
        row["last_obs"] = series.index.max()
        row["completeness_active"] = compute_completeness(series)
        row["completeness_window"] = compute_completeness(
            series, expected_steps=store.expected_steps_in_window()
        )
        row.update(compute_sequence_stats(spans, sph))

        viable = True
        for h in horizons:
            counts = count_usable_windows(
                spans, window, h, stride, train_frac, val_frac, purge
            )
            for split_name, n in counts.items():
                row[f"{split_name}_h{h}"] = n
            for split in ("train", "val", "test"):
                if counts[f"{split}_windows"] < int(min_windows[split]):
                    viable = False
        row["viable"] = viable
        rows.append(row)
        store.drop_from_cache(series_id)

    table = pd.DataFrame(rows).set_index("sensor_id").sort_index()
    assert not table.empty, "CRITICAL: quality table is empty — no series found"
    n_viable = int(table["viable"].sum())
    logger.info(f"Quality table: {len(table)} series, {n_viable} viable")
    return table


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def build_sensor_manifest(quality_table: pd.DataFrame, config: dict) -> dict:
    """
    Reproducible record of the sensor-selection process (reviewer requirement:
    'specify the pre-selection process of sensors'). The rule, its parameters,
    and every inclusion/exclusion decision are persisted.
    """
    q_cfg = config["quality"]
    viable = quality_table[quality_table["viable"]]
    excluded = quality_table[~quality_table["viable"]]
    exclusion_reasons = {}
    min_windows = q_cfg["viability"]["min_windows"]
    horizons = [int(h) for h in q_cfg["horizons"]]
    for sensor_id, row in excluded.iterrows():
        reasons = []
        for h in horizons:
            for split in ("train", "val", "test"):
                col = f"{split}_windows_h{h}"
                if row[col] < int(min_windows[split]):
                    reasons.append(f"{col}={int(row[col])} < {int(min_windows[split])}")
        exclusion_reasons[sensor_id] = reasons

    manifest = {
        "git_hash": _git_hash(),
        "date_range": [str(config["data"]["start_date"]), str(config["data"]["end_date"])],
        "veh_class": config["data"]["veh_class"],
        "freq": config["data"]["freq"],
        "viability_rule": {
            "window": int(q_cfg["window"]),
            "horizons": horizons,
            "stride": int(q_cfg.get("stride", 1)),
            "train_frac": float(q_cfg["train_frac"]),
            "val_frac": float(q_cfg["val_frac"]),
            "purge_boundaries": bool(q_cfg.get("purge_boundaries", True)),
            "min_windows": {k: int(v) for k, v in min_windows.items()},
        },
        "n_candidate_series": int(len(quality_table)),
        "n_viable_series": int(len(viable)),
        "viable_sensors": viable.index.tolist(),
        "excluded_sensors": [
            {"sensor_id": sid, "reasons": reasons}
            for sid, reasons in exclusion_reasons.items()
        ],
    }
    return manifest


def save_sensor_manifest(manifest: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Sensor manifest saved to {path}")
