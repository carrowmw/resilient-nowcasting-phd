# Tests for src/spatial/contract.py — shapes, invariants, inverse transform.
import numpy as np
import pytest

from spatial.contract import GraphSplit, GraphSpec, assert_predictions_valid
from spatial_helpers import make_context


def test_context_invariants():
    ctx = make_context()
    assert ctx.n_nodes == 6
    assert ctx.max_horizon == 2
    assert ctx.steps_per_day() == 96
    # Unobserved cells are NaN in raw, False in mask.
    assert not np.isfinite(ctx.values_raw[~ctx.observed_mask]).any()


def test_context_rejects_unsorted_nodes():
    ctx = make_context()
    with pytest.raises(AssertionError, match="sorted"):
        make_bad = ctx.__class__(
            node_ids=list(reversed(ctx.node_ids)),
            grid_index=ctx.grid_index,
            values_raw=ctx.values_raw,
            values_scaled=ctx.values_scaled,
            observed_mask=ctx.observed_mask,
            static_features=ctx.static_features,
            scaler_mean=ctx.scaler_mean,
            scaler_std=ctx.scaler_std,
            window=ctx.window,
            horizons=ctx.horizons,
            freq=ctx.freq,
        )
        del make_bad


def test_context_rejects_unordered_horizons():
    ctx = make_context()
    with pytest.raises(AssertionError, match="ascending"):
        ctx.__class__(
            node_ids=ctx.node_ids, grid_index=ctx.grid_index,
            values_raw=ctx.values_raw, values_scaled=ctx.values_scaled,
            observed_mask=ctx.observed_mask, static_features=ctx.static_features,
            scaler_mean=ctx.scaler_mean, scaler_std=ctx.scaler_std,
            window=ctx.window, horizons=(2, 1), freq=ctx.freq,
        )


def test_inverse_transform_round_trip_and_clip():
    ctx = make_context(missing_frac=0.0)
    y_scaled = ctx.values_scaled[50:60].T[None]        # [1, N, 10] node axis at -2
    back = ctx.inverse_transform(y_scaled)
    truth = ctx.values_raw[50:60].T[None]
    np.testing.assert_allclose(back, truth, rtol=1e-4, atol=1e-3)
    # Strongly negative scaled values clip to 0 (count domain).
    assert (ctx.inverse_transform(np.full((1, ctx.n_nodes, 3), -100.0)) == 0).all()


def test_split_lazy_properties():
    ctx = make_context()
    split = GraphSplit(ctx, "train", np.arange(0, 50, dtype=np.int64))
    assert len(split) == 50
    assert split.origin_idx[0] == ctx.window - 1
    assert split.target_idx.shape == (50, 2)
    assert (split.target_idx[:, 0] == split.origin_idx + 1).all()
    assert split.y_scaled.shape == (50, ctx.n_nodes, 2)
    assert split.y_raw.shape == (50, ctx.n_nodes, 2)
    assert split.target_observed.shape == (50, ctx.n_nodes, 2)
    # NaN exactly where unobserved.
    nan_mask = np.isnan(split.y_scaled)
    assert (nan_mask == ~split.target_observed).all()
    assert split.target_dt.shape == (50, 2)


def test_split_rejects_bad_name_and_overrun():
    ctx = make_context()
    with pytest.raises(AssertionError, match="bad split name"):
        GraphSplit(ctx, "training", np.arange(5, dtype=np.int64))
    with pytest.raises(AssertionError, match="overruns"):
        GraphSplit(ctx, "train", np.array([10_000], dtype=np.int64))


def test_graph_spec_invariants():
    ei = np.array([[0, 1], [1, 0]], dtype=np.int64)
    spec = GraphSpec(method="knn", edge_index=ei, edge_attr=None, edge_dim=0)
    assert spec.n_edges == 2
    with pytest.raises(AssertionError, match="edge_dim must be 0"):
        GraphSpec(method="knn", edge_index=ei, edge_attr=None, edge_dim=3)
    with pytest.raises(AssertionError, match="edge_attr shape"):
        GraphSpec(method="knn", edge_index=ei,
                  edge_attr=np.zeros((3, 2), dtype=np.float32), edge_dim=2)


def test_assert_predictions_valid():
    ctx = make_context()
    split = GraphSplit(ctx, "test", np.arange(10, dtype=np.int64))
    good = np.zeros((10, ctx.n_nodes, 2))
    assert_predictions_valid(split, good, "m")
    with pytest.raises(AssertionError, match="shape"):
        assert_predictions_valid(split, good[:, :, :1], "m")
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(AssertionError, match="non-finite"):
        assert_predictions_valid(split, bad, "m")
