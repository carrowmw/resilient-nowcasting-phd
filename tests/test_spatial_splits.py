# Tests for src/spatial/splits.py — chronology, purge, leakage, target counts.
import numpy as np
import pytest

from spatial.splits import assign_split_windows, build_graph_splits, count_usable_targets
from spatial_helpers import make_context


def test_chronological_and_disjoint():
    idx = assign_split_windows(T_total=1000, window=12, max_horizon=4)
    train, val, test = idx["train"], idx["val"], idx["test"]
    assert train.max() < val.min() < test.min()
    all_idx = np.concatenate([train, val, test])
    assert len(np.unique(all_idx)) == len(all_idx)


def test_no_shared_grid_steps_after_purge():
    window, h = 12, 4
    idx = assign_split_windows(T_total=1000, window=window, max_horizon=h,
                               purge_boundaries=True)

    def touched(starts):
        steps = set()
        for s in starts:
            steps.update(range(s, s + window))                  # inputs
            steps.update(s + window - 1 + hh for hh in (1, h))  # targets
        return steps

    assert not (touched(idx["train"]) & touched(idx["val"]))
    assert not (touched(idx["val"]) & touched(idx["test"]))


def test_without_purge_boundaries_overlap():
    idx = assign_split_windows(T_total=1000, window=12, max_horizon=4,
                               purge_boundaries=False)
    # Sanity: purge=False keeps the head windows (documented contamination).
    assert idx["val"].min() == idx["train"].max() + 1


def test_purge_count_arithmetic():
    a = assign_split_windows(T_total=1000, window=12, max_horizon=4,
                             purge_boundaries=False)
    b = assign_split_windows(T_total=1000, window=12, max_horizon=4,
                             purge_boundaries=True)
    purge_k = 12 + 4 - 1
    assert len(a["val"]) - len(b["val"]) == purge_k
    assert len(a["test"]) - len(b["test"]) == purge_k
    assert (a["train"] == b["train"]).all()


def test_too_short_grid_asserts():
    with pytest.raises(AssertionError, match="too short"):
        assign_split_windows(T_total=10, window=12, max_horizon=4)
    with pytest.raises(AssertionError, match="empty after purge"):
        assign_split_windows(T_total=80, window=12, max_horizon=4)


def test_stride_respected():
    idx = assign_split_windows(T_total=1000, window=12, max_horizon=4, stride=4)
    diffs = np.diff(idx["train"])
    assert (diffs == 4).all()


def test_build_graph_splits_and_target_counts():
    ctx = make_context(T_total=600, missing_frac=0.2, seed=3)
    splits = build_graph_splits(ctx)
    assert set(splits) == {"train", "val", "test"}
    df = count_usable_targets(splits["test"])
    assert len(df) == ctx.n_nodes * len(ctx.horizons)
    # Counts must agree with the split's own target_observed sums.
    obs = splits["test"].target_observed
    for k, h in enumerate(ctx.horizons):
        row = df[(df.node_id == ctx.node_ids[0]) & (df.horizon == h)]
        assert int(row.n_targets_observed.iloc[0]) == int(obs[:, 0, k].sum())
    # ~20% missing → observed fraction near 80%.
    frac = df.n_targets_observed.sum() / (df.n_windows.sum())
    assert 0.7 < frac < 0.9
