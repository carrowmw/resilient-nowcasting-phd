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


# ---------------------------------------------------------------------------
# Dual-cache (real + simulated) tests
# ---------------------------------------------------------------------------

def _make_pair(sp, eucl=None, directness=0.9, roads=(0.5, 0.3, 0.1, 0.1, 0.0, 0.0)):
    """Build a feature dict with road fractions that sum to 1.0 (connected pair)."""
    if eucl is None:
        eucl = sp * 0.9
    d = {
        "shortest_path_distance": sp,
        "euclidean_distance": eucl,
        "directness": directness,
    }
    for k, v in zip(_ROAD_KEYS, roads):
        d[k] = v
    return d


def _build_dual_caches(real_max_sp=5000.0, sim_max_sp=8000.0):
    """
    Build and write two cache JSON files for 3 real nodes (R1,R2,R3) and 2 simulated
    nodes (Simulated_Node_0, Simulated_Node_1). Returns (real_path, sim_path).
    """
    # Real cache: only R↔R pairs.
    real_cache = {
        "R1": {
            "R2": _make_pair(real_max_sp),
            "R3": _make_pair(1000.0),
        },
        "R2": {
            "R1": _make_pair(real_max_sp),
            "R3": _make_pair(2000.0),
        },
        "R3": {
            "R1": _make_pair(1000.0),
            "R2": _make_pair(2000.0),
        },
    }

    # Simulated cache: all pairs touching Simulated_Node_0 or Simulated_Node_1.
    sim_nodes = ["Simulated_Node_0", "Simulated_Node_1"]
    real_nodes = ["R1", "R2", "R3"]
    sim_cache = {}
    # real → sim
    for r in real_nodes:
        sim_cache[r] = {}
        for s in sim_nodes:
            sim_cache[r][s] = _make_pair(sim_max_sp)
    # sim → real
    for s in sim_nodes:
        sim_cache[s] = {}
        for r in real_nodes:
            sim_cache[s][r] = _make_pair(sim_max_sp)
    # sim → sim
    sim_cache["Simulated_Node_0"]["Simulated_Node_1"] = _make_pair(sim_max_sp)
    sim_cache["Simulated_Node_1"]["Simulated_Node_0"] = _make_pair(sim_max_sp)

    real_path = _write_cache(real_cache)
    sim_path = _write_cache(sim_cache)
    return real_path, sim_path


def test_dual_cache_merge_shape():
    """Dual cache: forward with 5 nodes (3 real + 2 sim) produces edge_attr [E, 10]."""
    torch.manual_seed(0)
    real_path, sim_path = _build_dual_caches()
    try:
        learner = DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0,
            edge_features_cache_path=real_path,
            simulated_edge_features_cache_path=sim_path,
        )
        location_ids = ["R1", "R2", "R3", "Simulated_Node_0", "Simulated_Node_1"]
        static = torch.ones(5, 4)
        edge_index, edge_weight, edge_attr = learner(static, location_ids=location_ids)
        E = edge_index.shape[1]
        assert edge_attr.shape == (E, 10), f"Expected (E, 10), got {edge_attr.shape}"
    finally:
        os.unlink(real_path)
        os.unlink(sim_path)


def test_dual_cache_max_distance():
    """Dual cache: max_distance is recomputed from the merged cache (sim pair wins)."""
    real_path, sim_path = _build_dual_caches(real_max_sp=5000.0, sim_max_sp=15000.0)
    try:
        learner = DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0,
            edge_features_cache_path=real_path,
            simulated_edge_features_cache_path=sim_path,
        )
        assert learner.max_distance == 15000.0, (
            f"Expected max_distance=15000.0 (from sim cache), got {learner.max_distance}"
        )
    finally:
        os.unlink(real_path)
        os.unlink(sim_path)


def test_dual_cache_road_sum_invariant():
    """Dual cache: road fractions (cols 3:9) sum to ~1.0 for every edge in the 5-node graph."""
    torch.manual_seed(1)
    real_path, sim_path = _build_dual_caches()
    try:
        learner = DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0,
            edge_features_cache_path=real_path,
            simulated_edge_features_cache_path=sim_path,
        )
        location_ids = ["R1", "R2", "R3", "Simulated_Node_0", "Simulated_Node_1"]
        static = torch.ones(5, 4)
        _, _, edge_attr = learner(static, location_ids=location_ids)
        road_sums = edge_attr[:, 3:9].sum(dim=1)
        connected = (road_sums - 1.0).abs() < 1e-3
        disconnected = road_sums.abs() < 1e-3
        assert torch.all(connected | disconnected), (
            "Road fractions invariant violated; offending sums="
            f"{road_sums[~(connected | disconnected)]}"
        )
    finally:
        os.unlink(real_path)
        os.unlink(sim_path)


def test_simulated_cache_required_guard():
    """Guard: setting simulated_edge_features_cache_path without edge_features_cache_path raises ValueError."""
    with pytest.raises(ValueError, match="CRITICAL: simulated_edge_features_cache_path requires"):
        DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8,
            edge_features_cache_path=None,
            simulated_edge_features_cache_path="/some/sim_cache.json",
        )


def test_simulated_cache_missing_file_fail_fast():
    """Guard: a nonexistent simulated cache path raises FileNotFoundError."""
    real_cache = {
        "R1": {"R2": _make_pair(100.0)},
        "R2": {"R1": _make_pair(100.0)},
    }
    real_path = _write_cache(real_cache)
    try:
        with pytest.raises(FileNotFoundError, match="CRITICAL: Simulated edge features cache not found"):
            DynamicGraphLearner(
                static_feat_dim=4, embedding_dim=8,
                edge_features_cache_path=real_path,
                simulated_edge_features_cache_path="/nonexistent/sim_cache.json",
            )
    finally:
        os.unlink(real_path)
