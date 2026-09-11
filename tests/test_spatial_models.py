# Tests for src/spatial/models + registry — shapes, isolation proofs, determinism.
import copy

import numpy as np
import pytest
import torch

from spatial.contract import GraphSplit, assert_predictions_valid
from spatial.graphs import build_gabriel_graph, build_identity_graph
from spatial.registry import KNOWN_TYPES, build_forecaster
from spatial.splits import build_graph_splits
from spatial_helpers import make_context

TRAIN_BLOCK = dict(lr=1e-2, batch_size=8, epochs=2, patience=2,
                   scheduler_step_size=1, scheduler_gamma=0.9, grad_clip=1.0)
INPUTS = {"mask_channel": True, "time_features": ["day"]}

SPECS = {
    "persistence": {"type": "persistence"},
    "snaive_daily": {"type": "seasonal_naive", "seasonality_days": 1},
    "hist_avg": {"type": "historical_average"},
    "lstm": {"type": "lstm", "hidden_dim": 16, "num_layers": 1, **TRAIN_BLOCK},
    "transformer": {"type": "transformer", "d_model": 16, "nhead": 2,
                    "num_encoder_layers": 1, "dim_feedforward": 32, **TRAIN_BLOCK},
    "gat_plain": {"type": "gat", "temporal_encoder": "none", "gat_layers": 2,
                  "heads": 2, "hidden_dim": 16, "use_static_features": False,
                  **TRAIN_BLOCK},
    "gat_lstm": {"type": "gat", "temporal_encoder": "lstm", "num_layers": 1,
                 "gat_layers": 1, "heads": 2, "hidden_dim": 16,
                 "use_static_features": True, **TRAIN_BLOCK},
    "gat_learned": {"type": "gat_learned", "temporal_encoder": "none",
                    "gat_layers": 1, "heads": 2, "hidden_dim": 16,
                    "use_static_features": False, "embedding_dim": 8,
                    "sparsity_threshold": 0.0, "edge_attr_mode": "learned_only",
                    **TRAIN_BLOCK},
}


@pytest.fixture(scope="module")
def world():
    ctx = make_context(n_nodes=6, T_total=400, window=8, horizons=(1, 2), seed=5)
    splits = build_graph_splits(ctx)
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1000, (6, 2))
    diff = points[:, None] - points[None, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return {
        "ctx": ctx, "splits": splits,
        "identity": build_identity_graph(6),
        "gabriel": build_gabriel_graph(dist, ctx.node_ids),
    }


def _fit_predict(key, world, graph_key="identity", seed=0):
    spec = copy.deepcopy(SPECS[key])
    f = build_forecaster(key, spec, seed=seed, device="cpu", inputs_cfg=INPUTS)
    graph = world["identity"] if key == "gat_learned" else world[graph_key]
    report = f.fit(world["splits"]["train"], world["splits"]["val"], graph)
    assert report.fitted
    preds = f.predict(world["splits"]["test"], graph)
    assert_predictions_valid(world["splits"]["test"], preds, key)
    return f, preds


@pytest.mark.parametrize("key", list(SPECS))
def test_fit_predict_shapes(key, world):
    _fit_predict(key, world)


def test_registry_asserts():
    with pytest.raises(AssertionError, match="unknown model type"):
        build_forecaster("x", {"type": "delaunay_net"})
    with pytest.raises(AssertionError, match="missing 'type'"):
        build_forecaster("x", {})
    with pytest.raises(AssertionError, match="missing training key"):
        build_forecaster("lstm", {"type": "lstm", "hidden_dim": 8, "num_layers": 1},
                         inputs_cfg=INPUTS)
    f = build_forecaster("my_label", {"type": "persistence"})
    assert f.name == "my_label"


def test_predict_before_fit_asserts(world):
    f = build_forecaster("lstm", copy.deepcopy(SPECS["lstm"]), seed=0,
                         device="cpu", inputs_cfg=INPUTS)
    with pytest.raises(AssertionError, match="before fit"):
        f.predict(world["splits"]["test"], world["identity"])


def test_persistence_known_answer(world):
    ctx, splits = world["ctx"], world["splits"]
    _, preds = _fit_predict("persistence", world)
    split = splits["test"]
    # Node 0, first test window: last observed scaled value in the window.
    start = int(split.window_start_idx[0])
    win_obs = ctx.observed_mask[start:start + ctx.window, 0]
    if win_obs.any():
        last = np.where(win_obs)[0].max() + start
        expected = ctx.values_scaled[last, 0]
        assert preds[0, 0, 0] == pytest.approx(expected, abs=1e-6)
        assert preds[0, 0, 1] == pytest.approx(expected, abs=1e-6)  # flat across horizons


def test_snaive_future_leakage_assert(world):
    f = build_forecaster("bad_snaive",
                         {"type": "seasonal_naive", "seasonality_days": 1 / 96},
                         inputs_cfg=INPUTS)
    f.fit(world["splits"]["train"], world["splits"]["val"], world["identity"])
    with pytest.raises(AssertionError, match="future leakage"):
        f.predict(world["splits"]["test"], world["identity"])


def _perturbed_context(ctx, node_j=3, seed=99):
    """Copy of ctx with node j's series replaced by unrelated noise."""
    rng = np.random.default_rng(seed)
    new = copy.deepcopy(ctx)
    T = len(ctx.grid_index)
    new.values_raw[:, node_j] = np.clip(rng.normal(50, 20, T), 0, None).astype(np.float32)
    new.values_raw[~new.observed_mask] = np.nan
    new.values_scaled[:, node_j] = (
        (new.values_raw[:, node_j] - ctx.scaler_mean[node_j]) / ctx.scaler_std[node_j]
    ).astype(np.float32)
    return new


@pytest.mark.parametrize("key", ["lstm", "transformer", "gat_plain"])
def test_cross_node_isolation_on_identity(key, world):
    """Perturbing node j must leave node i's predictions bit-identical when
    there is no message passing (per-node models; GAT on self-loops only)."""
    f, preds_a = _fit_predict(key, world, graph_key="identity")
    ctx_b = _perturbed_context(world["ctx"], node_j=3)
    split_b = GraphSplit(ctx_b, "test", world["splits"]["test"].window_start_idx)
    preds_b = f.predict(split_b, world["identity"])
    others = [i for i in range(6) if i != 3]
    np.testing.assert_array_equal(preds_a[:, others], preds_b[:, others])
    # And node j itself must change (the perturbation reached the model).
    assert not np.array_equal(preds_a[:, 3], preds_b[:, 3])


def test_cross_node_flow_on_gabriel(world):
    """On a real graph the same perturbation MUST reach other nodes."""
    f, preds_a = _fit_predict("gat_plain", world, graph_key="gabriel")
    ctx_b = _perturbed_context(world["ctx"], node_j=3)
    split_b = GraphSplit(ctx_b, "test", world["splits"]["test"].window_start_idx)
    preds_b = f.predict(split_b, world["gabriel"])
    others = [i for i in range(6) if i != 3]
    assert not np.array_equal(preds_a[:, others], preds_b[:, others])


def test_seed_determinism(world):
    _, a = _fit_predict("gat_plain", world, seed=7)
    _, b = _fit_predict("gat_plain", world, seed=7)
    np.testing.assert_array_equal(a, b)
    _, c = _fit_predict("gat_plain", world, seed=8)
    assert not np.array_equal(a, c)


def test_gat_learned_requires_identity_spec(world):
    f = build_forecaster("gat_learned", copy.deepcopy(SPECS["gat_learned"]),
                         seed=0, device="cpu", inputs_cfg=INPUTS)
    with pytest.raises(AssertionError, match="identity"):
        f.fit(world["splits"]["train"], world["splits"]["val"], world["gabriel"])


def test_gat_attention_record(world):
    f, _ = _fit_predict("gat_plain", world, graph_key="gabriel")
    rec = f.predict_attention(world["splits"]["test"], world["gabriel"], max_windows=16)
    E = world["gabriel"].n_edges
    assert rec.edge_index.shape == (2, E)
    assert rec.alpha_mean.shape == (E, 2)
    assert rec.n_windows == 16
    # Softmax over inbound edges: per-destination attention sums to ~1.
    alpha = rec.alpha_flat()
    dst = rec.edge_index[1]
    for j in range(6):
        assert alpha[dst == j].sum() == pytest.approx(1.0, abs=1e-4)


def test_baseline_capability_flags():
    for key in ("persistence", "snaive_daily", "hist_avg"):
        f = build_forecaster(key, copy.deepcopy(SPECS[key]))
        assert not f.capabilities.uses_graph
        assert not f.capabilities.emits_attention
    assert set(KNOWN_TYPES) == {s["type"] for s in SPECS.values()}
