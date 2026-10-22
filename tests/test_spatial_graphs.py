# Tests for src/spatial/graphs.py — Gabriel vs brute force, kNN, edge attrs.
import json

import numpy as np
import pytest

from spatial.graphs import (
    build_fc_graph,
    build_gabriel_graph,
    build_graph,
    build_identity_graph,
    build_knn_graph,
    graph_manifest,
    load_distance_matrix,
    load_physical_edge_attr,
)


def _pairwise(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _gabriel_brute_force(points):
    """O(N³) reference: edge iff no w strictly inside the circle with diameter uv."""
    n = len(points)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            centre = (points[i] + points[j]) / 2
            r2 = ((points[i] - points[j]) ** 2).sum() / 4
            inside = False
            for w in range(n):
                if w in (i, j):
                    continue
                if ((points[w] - centre) ** 2).sum() < r2:
                    inside = True
                    break
            if not inside:
                edges.add((i, j))
                edges.add((j, i))
    return edges


def _non_loop_edges(spec):
    return {(int(s), int(d)) for s, d in spec.edge_index.T if s != d}


@pytest.mark.parametrize("seed", range(10))
def test_gabriel_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, (20, 2))
    node_ids = [f"n{i:02d}" for i in range(20)]
    spec = build_gabriel_graph(_pairwise(points), node_ids)
    assert _non_loop_edges(spec) == _gabriel_brute_force(points)


def test_gabriel_scale_invariance():
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 10, (25, 2))
    ids = [f"n{i:02d}" for i in range(25)]
    dist = _pairwise(points)
    a = build_gabriel_graph(dist, ids)
    b = build_gabriel_graph(dist * 3.7, ids)
    assert _non_loop_edges(a) == _non_loop_edges(b)


def test_gabriel_connected_on_general_position():
    rng = np.random.default_rng(2)
    points = rng.uniform(0, 50, (30, 2))
    spec = build_gabriel_graph(_pairwise(points), [f"n{i:02d}" for i in range(30)])
    assert spec.params["n_components"] == 1


def test_identity_graph_self_loops_only():
    spec = build_identity_graph(7)
    assert spec.n_edges == 7
    assert (spec.edge_index[0] == spec.edge_index[1]).all()


def test_fc_graph_edge_count():
    spec = build_fc_graph(6, [f"n{i}" for i in range(6)])
    assert spec.n_edges == 6 * 5 + 6  # ordered pairs + self-loops


def test_knn_symmetrised_and_directed():
    rng = np.random.default_rng(3)
    points = rng.uniform(0, 10, (12, 2))
    dist = _pairwise(points)
    ids = [f"n{i:02d}" for i in range(12)]
    sym = build_knn_graph(dist, ids, k=3, symmetrise=True)
    edges = _non_loop_edges(sym)
    assert all((d, s) in edges for s, d in edges)
    asym = build_knn_graph(dist, ids, k=3, symmetrise=False)
    # Each node receives exactly k inbound non-loop messages.
    dst = [d for s, d in _non_loop_edges(asym)]
    assert all(dst.count(i) == 3 for i in range(12))
    with pytest.raises(AssertionError, match="invalid"):
        build_knn_graph(dist, ids, k=12)


# ---------------------------------------------------------------- cache-backed

@pytest.fixture
def cache_file(tmp_path):
    rng = np.random.default_rng(4)
    ids = [f"s{i}" for i in range(5)]
    points = rng.uniform(0, 1000, (5, 2))
    dist = _pairwise(points)
    cache = {}
    for i, src in enumerate(ids):
        cache[src] = {}
        for j, dst in enumerate(ids):
            if i == j:
                continue
            d = float(dist[i, j])
            cache[src][dst] = {
                "shortest_path_distance": d * 1.3,
                "euclidean_distance": d,
                "directness": 1 / 1.3,
                "pedestrian": 0.5, "residential": 0.5, "tertiary": 0.0,
                "secondary": 0.0, "primary": 0.0, "service": 0.0,
            }
    path = tmp_path / "edge_cache.json"
    path.write_text(json.dumps(cache))
    return str(path), ids, dist


def test_load_distance_matrix(cache_file):
    path, ids, dist = cache_file
    loaded = load_distance_matrix(ids, path)
    np.testing.assert_allclose(loaded, dist, rtol=1e-6)
    with pytest.raises(AssertionError, match="edge cache missing"):
        load_distance_matrix(ids + ["ghost"], path)


def test_physical_edge_attr(cache_file):
    path, ids, dist = cache_file
    spec = build_fc_graph(len(ids), ids)
    attr = load_physical_edge_attr(spec.edge_index, ids, path)
    assert attr.shape == (spec.n_edges, 9)
    assert np.isfinite(attr).all()
    # Distances normalised to (0, 1]; max shortest-path row hits exactly 1.
    assert attr[:, 0].max() == pytest.approx(1.0)
    assert attr[:, 0].min() >= 0.0
    # Self-loop rows: all zero except directness=1.
    loops = spec.edge_index[0] == spec.edge_index[1]
    assert (attr[loops, 2] == 1.0).all()
    assert (attr[loops][:, [0, 1, 3, 4, 5, 6, 7, 8]] == 0).all()
    # Road fractions sum to ~1 on real edges.
    assert np.allclose(attr[~loops, 3:9].sum(1), 1.0, atol=0.05)


def test_build_graph_dispatcher_and_manifest(cache_file):
    path, ids, dist = cache_file
    spec = build_graph("gabriel", {"method": "gabriel", "edge_attr": "physical"}, ids, path)
    assert spec.method == "gabriel" and spec.edge_dim == 9
    man = graph_manifest(spec, ids)
    assert man["n_nodes"] == 5
    assert man["n_self_loops"] == 5
    assert len(man["sha16"]) == 16
    # Deterministic sha.
    assert graph_manifest(spec, ids)["sha16"] == man["sha16"]
    with pytest.raises(AssertionError, match="unknown graph method"):
        build_graph("bad", {"method": "delaunay"}, ids, path)
    with pytest.raises(AssertionError, match="meaningless"):
        build_graph("identity", {"method": "identity", "edge_attr": "physical"}, ids, path)
