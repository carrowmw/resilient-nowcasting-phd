import json
import os
import tempfile

import numpy as np
import pytest
import torch

from graph_learner import DynamicGraphLearner, _ROAD_KEYS


def test_dynamic_graph_learner():
    torch.manual_seed(42)
    np.random.seed(42)

    num_nodes = 50
    static_feat_dim = 10
    embedding_dim = 16
    sparsity_threshold = 0.5

    model = DynamicGraphLearner(
        static_feat_dim=static_feat_dim,
        embedding_dim=embedding_dim,
        sparsity_threshold=sparsity_threshold
    )

    static_features = torch.randn(num_nodes, static_feat_dim)
    # PHASE 3: forward now returns a 3-tuple (edge_index, edge_weight, edge_attr).
    edge_index, edge_weight, edge_attr = model(static_features)

    E = edge_weight.shape[0]
    assert edge_index.shape == (2, E), f"edge_index shape mismatch: {edge_index.shape}"
    assert edge_weight.shape == (E,), f"edge_weight shape mismatch: {edge_weight.shape}"

    # No cache -> learned-only fallback edge_attr [E, 1] == edge_weight.
    assert edge_attr.shape == (E, 1), f"edge_attr fallback shape mismatch: {edge_attr.shape}"
    assert torch.allclose(edge_attr.squeeze(1), edge_weight)

    assert E > 0, "Graph is completely disconnected (E=0)."
    assert E < (num_nodes * num_nodes), f"Graph is fully connected (E={E}), sparsity threshold not working."

    loss = edge_weight.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient."


def _pair(sp, eucl, directness, roads):
    d = {"shortest_path_distance": sp, "euclidean_distance": eucl, "directness": directness}
    for k, v in zip(_ROAD_KEYS, roads):
        d[k] = v
    return d


def _write_cache(cache: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cache, f)
    f.close()
    return f.name


def test_edge_attr_with_cache():
    """PHASE 3: with a cache, edge_attr is [E, 10] and the road-sum invariant holds for
    connected (~1.0) and disconnected (~0.0) pairs alike."""
    torch.manual_seed(0)
    connected_a = (0.5, 0.3, 0.1, 0.1, 0.0, 0.0)   # sums to 1.0
    connected_b = (0.0, 0.0, 0.0, 0.4, 0.6, 0.0)   # sums to 1.0
    disconnected = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # sums to 0.0 (no OSM path)
    cache = {
        "S1": {"S2": _pair(100.0, 90.0, 0.9, connected_a),
               "S3": _pair(8000.0, 7000.0, 0.875, disconnected)},
        "S2": {"S1": _pair(100.0, 90.0, 0.9, connected_a),
               "S3": _pair(200.0, 180.0, 0.9, connected_b)},
        "S3": {"S1": _pair(8000.0, 7000.0, 0.875, disconnected),
               "S2": _pair(200.0, 180.0, 0.9, connected_b)},
    }
    path = _write_cache(cache)
    try:
        learner = DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0,
            edge_features_cache_path=path,
        )
        assert learner.max_distance == 8000.0
        static = torch.ones(3, 4)
        edge_index, edge_weight, edge_attr = learner(static, location_ids=["S1", "S2", "S3"])

        E = edge_index.shape[1]
        assert edge_attr.shape == (E, 10)
        # Distance channels normalized into [0, 1].
        assert (edge_attr[:, 0] >= 0).all() and (edge_attr[:, 0] <= 1.0 + 1e-6).all()
        assert (edge_attr[:, 1] >= 0).all() and (edge_attr[:, 1] <= 1.0 + 1e-6).all()
        # Road-sum invariant: ~1.0 (connected) OR ~0.0 (disconnected).
        road_sums = edge_attr[:, 3:9].sum(dim=1)
        connected = (road_sums - 1.0).abs() < 1e-3
        disconnected_m = road_sums.abs() < 1e-3
        assert torch.all(connected | disconnected_m)
        # 10th column is the learned similarity.
        assert torch.allclose(edge_attr[:, 9], edge_weight)
    finally:
        os.unlink(path)


def test_missing_cache_fail_fast():
    """PHASE 3: a location_id absent from the cache raises a CRITICAL ValueError."""
    cache = {
        "S1": {"S2": _pair(100.0, 90.0, 0.9, (0.5, 0.3, 0.1, 0.1, 0.0, 0.0))},
        "S2": {"S1": _pair(100.0, 90.0, 0.9, (0.5, 0.3, 0.1, 0.1, 0.0, 0.0))},
    }
    path = _write_cache(cache)
    try:
        learner = DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0,
            edge_features_cache_path=path,
        )
        static = torch.ones(2, 4)
        with pytest.raises(ValueError, match="CRITICAL: Missing cache"):
            learner(static, location_ids=["S1", "S99"])
    finally:
        os.unlink(path)


def test_missing_cache_file_fail_fast():
    """PHASE 3: a non-existent cache path raises FileNotFoundError at construction."""
    with pytest.raises(FileNotFoundError, match="CRITICAL: Edge features cache not found"):
        DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8,
            edge_features_cache_path="/nonexistent/edge_cache.json",
        )
