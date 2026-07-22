# Tests for src/quality/quality_metrics.py — synthetic series with known gaps.
import numpy as np
import pandas as pd
import pytest

from quality.quality_metrics import (
    compute_completeness,
    compute_sequence_stats,
    count_usable_windows,
    find_consecutive_sequences,
    windows_in_span,
)


def make_series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="15min")
    return pd.Series(values, index=idx, name="TEST__a_to_b")


def test_find_sequences_known_gaps():
    # 10 obs | 3 NaN | 5 obs | 1 NaN | 20 obs → runs of 10, 5, 20
    values = [1.0] * 10 + [np.nan] * 3 + [2.0] * 5 + [np.nan] + [3.0] * 20
    spans = find_consecutive_sequences(make_series(values))
    assert [sp.n_steps for sp in spans] == [10, 5, 20]
    assert spans[0].start_idx == 0 and spans[0].end_idx == 9
    assert spans[2].start_idx == 19 and spans[2].end_idx == 38


def test_find_sequences_min_length_filter():
    values = [1.0] * 10 + [np.nan] * 3 + [2.0] * 5 + [np.nan] + [3.0] * 20
    spans = find_consecutive_sequences(make_series(values), min_length=8)
    assert [sp.n_steps for sp in spans] == [10, 20]


def test_find_sequences_all_nan():
    assert find_consecutive_sequences(make_series([np.nan] * 12)) == []


def test_find_sequences_rejects_irregular_grid():
    s = make_series([1.0] * 10)
    s = s.drop(s.index[4])  # punch a hole in the index itself
    with pytest.raises(AssertionError, match="continuous grid"):
        find_consecutive_sequences(s)


def test_completeness_half_missing():
    values = [1.0, np.nan] * 50
    assert compute_completeness(make_series(values)) == pytest.approx(0.5)


def test_completeness_against_window():
    s = make_series([1.0] * 96)  # one full day observed
    # Expected over 2 days → 0.5
    assert compute_completeness(s, expected_steps=192) == pytest.approx(0.5)


def test_sequence_stats_hours_conversion():
    values = [1.0] * 96 + [np.nan] * 4 + [1.0] * 48  # 24 h and 12 h runs
    spans = find_consecutive_sequences(make_series(values))
    stats = compute_sequence_stats(spans, steps_per_hour=4)
    assert stats["seq_count"] == 2
    assert stats["avg_seq_len_hours"] == pytest.approx(18.0)
    assert stats["max_seq_len_hours"] == pytest.approx(24.0)


def test_windows_in_span_arithmetic():
    # L=100, window=24, horizon=8 → 100 - 32 + 1 = 69
    assert windows_in_span(100, 24, 8) == 69
    assert windows_in_span(32, 24, 8) == 1
    assert windows_in_span(31, 24, 8) == 0
    assert windows_in_span(100, 24, 8, stride=2) == 35


def test_count_usable_windows_no_purge_internal_split():
    values = [1.0] * 132  # single run: n = 132 - 32 + 1 = 101 windows
    spans = find_consecutive_sequences(make_series(values))
    no_purge = count_usable_windows(spans, 24, 8, purge_boundaries=False)
    assert no_purge == {"train_windows": 70, "val_windows": 15, "test_windows": 16}


def test_short_run_whole_assignment_keeps_all_windows():
    # One run of 101 windows cannot survive internal purging (purge=31 > 15
    # val windows) → the whole run is assigned to one split, nothing is lost.
    values = [1.0] * 132
    spans = find_consecutive_sequences(make_series(values))
    purged = count_usable_windows(spans, 24, 8, purge_boundaries=True)
    assert sum(purged.values()) == 101
    assert purged["train_windows"] == 101  # first allocation goes to train


def test_short_run_allocation_approximates_fractions():
    # Many short runs (each 41 windows, too short for internal purged split)
    # must be allocated whole-run at ~70/15/15 window proportions.
    values = ([1.0] * 72 + [np.nan]) * 20
    spans = find_consecutive_sequences(make_series(values))
    counts = count_usable_windows(spans, 24, 8, purge_boundaries=True)
    total = sum(counts.values())
    assert total == 20 * 41
    assert counts["train_windows"] / total == pytest.approx(0.7, abs=0.06)
    assert counts["val_windows"] / total == pytest.approx(0.15, abs=0.06)
    assert counts["test_windows"] / total == pytest.approx(0.15, abs=0.06)


def test_long_runs_still_split_internally_with_purge():
    values = [1.0] * 1000  # n = 969 windows; val floor = 145 > purge 31
    spans = find_consecutive_sequences(make_series(values))
    counts = count_usable_windows(spans, 24, 8, purge_boundaries=True)
    assert counts["train_windows"] == 678          # floor(0.7 * 969)
    assert counts["val_windows"] == 145 - 31       # floor(0.15 * 969) - purge
    assert counts["test_windows"] == (969 - 678 - 145) - 31


def test_count_usable_windows_invalid_fracs():
    with pytest.raises(AssertionError, match="split fractions"):
        count_usable_windows([], 24, 8, train_frac=0.9, val_frac=0.2)
