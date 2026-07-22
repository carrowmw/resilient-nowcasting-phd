#!/usr/bin/env python3
"""
Desk check for the quality metrics module (Phase A of the quality track).

Three synthetic sensors with known ground truth:
  A — perfect:     100% complete, one long run
  B — fragmented:  ~50% complete, many short runs (the paper's problem case)
  C — night gaps:  ~75% complete, gaps only 00:00–06:00 (structural MNAR)

plus interpolation effects on the quality metrics themselves (the paper's
"sequence-length inflation" phenomenon, reproduced for every method).

Run INSIDE the container:
    python scripts/desk_checks/desk_check_quality_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import numpy as np
import pandas as pd

from quality.interpolation import INTERPOLATION_METHODS, apply_interpolation
from quality.quality_metrics import (
    compute_completeness,
    compute_sequence_stats,
    count_usable_windows,
    find_consecutive_sequences,
)


def hr(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def make_sensors(days=14):
    n = 96 * days
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    hours = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    base = np.clip(50 + 40 * np.sin(2 * np.pi * (hours - 6) / 24.0) + rng.normal(0, 3, n), 0, None)

    a = pd.Series(base.copy(), index=idx, name="A__perfect")

    b = pd.Series(base.copy(), index=idx, name="B__fragmented")
    # Kill every other 6-hour block → runs of 24 steps (6 h), ~50% complete.
    block = 24
    for start in range(block, n, 2 * block):
        b.iloc[start: start + block] = np.nan

    c = pd.Series(base.copy(), index=idx, name="C__night_gaps")
    c[np.asarray((idx.hour >= 0) & (idx.hour < 6))] = np.nan

    return [a, b, c]


def main():
    sensors = make_sensors()

    hr("PART 1 — quality metrics per synthetic sensor (window=24, horizon=8)")
    print(f"{'sensor':16s} {'complete':>9s} {'runs':>5s} {'avg_h':>7s} {'std_h':>7s} "
          f"{'train_w':>8s} {'val_w':>6s} {'test_w':>7s}")
    for s in sensors:
        spans = find_consecutive_sequences(s)
        comp = compute_completeness(s)
        stats = compute_sequence_stats(spans, steps_per_hour=4)
        wins = count_usable_windows(spans, window=24, horizon=8)
        print(f"{s.name:16s} {comp:9.3f} {stats['seq_count']:5d} "
              f"{stats['avg_seq_len_hours']:7.2f} {stats['std_seq_len_hours']:7.2f} "
              f"{wins['train_windows']:8d} {wins['val_windows']:6d} {wins['test_windows']:7d}")

    a, b, c = sensors
    assert compute_completeness(a) == 1.0, "sensor A must be 100% complete"
    b_stats = compute_sequence_stats(find_consecutive_sequences(b), 4)
    assert b_stats["avg_seq_len_hours"] == 6.0, "sensor B runs must average exactly 6 h"
    # B is the paper's pathology: decent completeness, runs too short for any
    # window+horizon=32 usable window.
    b_wins = count_usable_windows(find_consecutive_sequences(b), 24, 8)
    assert sum(b_wins.values()) == 0, "sensor B must yield zero usable windows"
    print("\n  → B shows the paper's core pathology: ~50% complete but ZERO usable "
          "windows\n    (every run shorter than window+horizon).")

    hr("PART 2 — interpolation inflates sequence length (per method, sensor C)")
    print(f"{'method':12s} {'complete':>9s} {'runs':>5s} {'avg_seq_h':>10s} {'filled':>7s}")
    for method in INTERPOLATION_METHODS:
        filled, mask = apply_interpolation(c, method, max_gap_steps=0)
        spans = find_consecutive_sequences(filled)
        stats = compute_sequence_stats(spans, 4)
        print(f"{method:12s} {compute_completeness(filled):9.3f} {stats['seq_count']:5d} "
              f"{stats['avg_seq_len_hours']:10.2f} {int(mask.sum()):7d}")
    zero_filled, _ = apply_interpolation(c, "night_zero")
    inflated = compute_sequence_stats(find_consecutive_sequences(zero_filled), 4)
    original = compute_sequence_stats(find_consecutive_sequences(c), 4)
    assert inflated["avg_seq_len_hours"] > original["avg_seq_len_hours"], \
        "night_zero must inflate average sequence length"
    print("\n  → night gaps filled ⇒ avg sequence length inflates from "
          f"{original['avg_seq_len_hours']:.1f} h to {inflated['avg_seq_len_hours']:.1f} h "
          "(paper Fig. 4 effect).")

    hr("DESK CHECK PASSED")


if __name__ == "__main__":
    main()
