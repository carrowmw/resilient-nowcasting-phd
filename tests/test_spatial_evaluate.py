# Tests for src/spatial/evaluate.py — metrics, MASE, observed-only masking.
import numpy as np
import pandas as pd
import pytest

from spatial.contract import GraphSplit
from spatial.evaluate import (
    METRIC_KEYS,
    aggregate_nodes,
    evaluate_split,
    mase_scale,
    predictions_frame,
)
from spatial.splits import build_graph_splits
from spatial_helpers import make_context


@pytest.fixture(scope="module")
def world():
    ctx = make_context(n_nodes=4, T_total=96 * 6, window=8, horizons=(1, 2),
                       missing_frac=0.15, seed=11)
    splits = build_graph_splits(ctx)
    scale = mase_scale(ctx, splits["train"])
    return ctx, splits, scale


def test_mase_scale_positive(world):
    ctx, splits, scale = world
    assert scale.shape == (4,)
    assert np.isfinite(scale).all() and (scale > 0).all()


def test_perfect_predictions_zero_error(world):
    ctx, splits, scale = world
    split = splits["test"]
    preds = np.nan_to_num(split.y_scaled, nan=0.0)
    df = evaluate_split(split, preds, scale)
    assert len(df) == 4 * 2
    assert (df["rmse_raw"].dropna() < 1e-3).all()
    assert (df["mase"].dropna() < 1e-3).all()
    assert set(METRIC_KEYS) <= set(df.columns)


def test_known_answer_single_cell():
    ctx = make_context(n_nodes=4, T_total=96 * 6, window=8, horizons=(1,),
                       missing_frac=0.0, seed=2)
    splits = build_graph_splits(ctx)
    split = splits["test"]
    scale = mase_scale(ctx, splits["train"])
    # Constant +1 raw-count error on node 0 via scaled offset.
    preds = split.y_scaled.copy()
    preds[:, 0, 0] += 1.0 / ctx.scaler_std[0]
    df = evaluate_split(split, preds, scale)
    row = df[(df.node_id == ctx.node_ids[0]) & (df.horizon == 1)].iloc[0]
    assert row["mae_raw"] == pytest.approx(1.0, abs=1e-3)
    assert row["rmse_raw"] == pytest.approx(1.0, abs=1e-3)
    assert row["mase"] == pytest.approx(1.0 / scale[0], abs=1e-3)
    # Other nodes untouched.
    other = df[(df.node_id != ctx.node_ids[0])]
    assert (other["mae_raw"] < 1e-3).all()


def test_masked_targets_ignored(world):
    ctx, splits, scale = world
    split = splits["test"]
    preds = np.nan_to_num(split.y_scaled, nan=0.0)
    # Garbage where unobserved must not move any metric.
    garbage = preds.copy()
    garbage[~split.target_observed] = 1e6
    a = evaluate_split(split, preds, scale)
    b = evaluate_split(split, garbage, scale)
    pd.testing.assert_frame_equal(a, b)


def test_evaluate_rejects_bad_shapes(world):
    ctx, splits, scale = world
    with pytest.raises(AssertionError, match="predictions shape"):
        evaluate_split(splits["test"], np.zeros((3, 4, 2)), scale)


def test_aggregate_nodes(world):
    ctx, splits, scale = world
    split = splits["test"]
    df = evaluate_split(split, np.nan_to_num(split.y_scaled, nan=0.0), scale)
    agg = aggregate_nodes(df)
    assert len(agg) == 2
    assert {"mae_raw_median", "mae_raw_iqr", "mase_median", "n_nodes"} <= set(agg.columns)
    assert (agg["n_nodes"] == 4).all()


def test_predictions_frame_layout(world):
    ctx, splits, scale = world
    split = splits["test"]
    preds = np.nan_to_num(split.y_scaled, nan=0.0)
    pf = predictions_frame(split, preds, model="m", graph_variant="g", seed=1)
    assert len(pf) == len(split) * 4 * 2
    assert pf["target_observed"].sum() == split.target_observed.sum()
    # Row alignment: (window, node, horizon) reshape order.
    i, j, k = 5, 2, 1
    flat = i * (4 * 2) + j * 2 + k
    row = pf.iloc[flat]
    assert row["node_id"] == ctx.node_ids[j]
    assert row["horizon"] == ctx.horizons[k]
    if split.target_observed[i, j, k]:
        assert row["y_true_raw"] == pytest.approx(split.y_raw[i, j, k])
    # target_dt matches the grid.
    expected_dt = ctx.grid_index[split.target_idx[i, k]]
    assert pd.Timestamp(row["target_dt"]) == expected_dt
