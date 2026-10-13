# Tests for src/spatial/dataset.py — channels, masks, collate, masked loss.
import numpy as np
import pytest
import torch

from spatial.contract import GraphSplit
from spatial.dataset import (
    GraphWindowDataset,
    build_time_features,
    collate_graph,
    input_dim,
    masked_mse,
)
from spatial_helpers import make_context


def _split(ctx, n=40):
    return GraphSplit(ctx, "train", np.arange(n, dtype=np.int64))


def test_input_dim_arithmetic():
    assert input_dim(True, ("day", "week")) == 1 + 1 + 4
    assert input_dim(False, ()) == 1
    with pytest.raises(AssertionError, match="unknown time features"):
        input_dim(True, ("month",))


def test_time_features_periodicity():
    ctx = make_context(T_total=96 * 8)
    tf = build_time_features(ctx.grid_index, ("day",))
    assert tf.shape == (96 * 8, 2)
    # 24 h period at 15-min steps: exact repetition after 96 steps.
    np.testing.assert_allclose(tf[:96], tf[96:192], atol=1e-6)
    np.testing.assert_allclose(np.square(tf).sum(axis=1), 1.0, atol=1e-6)


def test_item_shapes_and_mask_semantics():
    ctx = make_context(missing_frac=0.3, seed=7)
    ds = GraphWindowDataset(_split(ctx), mask_channel=True, time_features=("day", "week"))
    item = ds[0]
    N, T, D = ctx.n_nodes, ctx.window, ds.n_channels
    assert item["x"].shape == (N, T, D)
    assert item["y"].shape == (N, len(ctx.horizons))
    assert item["y_mask"].shape == (N, len(ctx.horizons))
    assert D == 6
    # No NaN anywhere in tensors.
    for v in item.values():
        assert torch.isfinite(v.float()).all()
    # Mask channel equals observedness of the window slice.
    mask_chan = item["x"][:, :, 1].numpy().astype(bool)
    expected = ctx.observed_mask[0:ctx.window].T
    assert (mask_chan == expected).all()
    # Count channel is zero exactly where unobserved.
    count_chan = item["x"][:, :, 0].numpy()
    assert (count_chan[~expected] == 0).all()
    # y is zero where y_mask is False (dead values).
    y, ym = item["y"].numpy(), item["y_mask"].numpy()
    assert (y[~ym] == 0).all()


def test_collate_stacks():
    ctx = make_context()
    ds = GraphWindowDataset(_split(ctx))
    batch = collate_graph([ds[i] for i in range(4)])
    assert batch["x"].shape == (4, ctx.n_nodes, ctx.window, ds.n_channels)
    assert batch["y"].shape == (4, ctx.n_nodes, len(ctx.horizons))


def test_masked_mse_known_answer():
    pred = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    y = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
    mask = torch.tensor([[True, True], [False, True]])
    # Errors on observed entries: 1², 0², 1² → mean 2/3.
    loss = masked_mse(pred, y, mask)
    assert loss.item() == pytest.approx(2.0 / 3.0)


def test_masked_mse_ignores_masked_garbage():
    pred = torch.tensor([[1.0, 1e6]])
    y = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    assert masked_mse(pred, y, mask).item() == pytest.approx(0.0)


def test_masked_mse_all_masked_asserts():
    with pytest.raises(AssertionError, match="zero observed targets"):
        masked_mse(torch.ones(2, 2), torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.bool))
