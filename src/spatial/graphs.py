"""
Static graph construction for the spatial track.

All topologies are built from the cached pairwise euclidean distances in
data/external/edge_features_cache.json — the same file that supplies the
physical edge features — so topology and edge attributes share one source of
truth and no coordinate pipeline is needed:

  - identity        : self-loops only — the matched-capacity control.
  - gabriel         : edge (u,v) iff no third node w satisfies
                      d(u,w)² + d(w,v)² < d(u,v)². Pure distance form of the
                      Gabriel test (w inside the circle with diameter uv);
                      scale-invariant, so the metres-vs-normalised question
                      is moot.
  - knn             : k nearest neighbours, symmetrised by default.
  - fully_connected : every ordered pair.

Every GraphSpec includes self-loops (models set add_self_loops=False).
Physical edge attributes (edge_dim=9) follow the column order of
graph_learner.py / features.EDGE_FEATURE_KEYS; self-loop rows are zero except
directness=1 (a node is perfectly direct to itself).
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from spatial.contract import GraphSpec

# Column order contract for edge_dim=9 physical attributes (matches the
# [E, 10] layout in src/graph_learner.py minus the learned-similarity col 9).
EDGE_FEATURE_KEYS_PHYSICAL = (
    "shortest_path_distance",   # 0 — normalised by global max_distance
    "euclidean_distance",       # 1 — normalised by global max_distance
    "directness",               # 2 — already in (0, 1]
    "pedestrian",               # 3 ┐
    "residential",              # 4 │
    "tertiary",                 # 5 │ road-category fractions of path length
    "secondary",                # 6 │ (sum to ~1.0, or ~0.0 for degenerate paths)
    "primary",                  # 7 │
    "service",                  # 8 ┘
)
_DIRECTNESS_COL = 2

KNOWN_METHODS = ("identity", "gabriel", "knn", "fully_connected")


def _load_cache(cache_path: str) -> Dict:
    path = Path(cache_path)
    assert path.exists(), f"CRITICAL: edge features cache not found: {path}"
    with open(path) as f:
        return json.load(f)


def load_distance_matrix(node_ids: Sequence[str], cache_path: str) -> np.ndarray:
    """
    [N, N] euclidean metres between nodes from the pairwise cache.
    Diagonal is 0. Asserts every directed pair present, positivity, and
    symmetry within tolerance (haversine is symmetric; drift means cache
    corruption).
    """
    cache = _load_cache(cache_path)
    n = len(node_ids)
    dist = np.zeros((n, n), dtype=np.float64)
    for i, src in enumerate(node_ids):
        assert src in cache, f"CRITICAL: edge cache missing source node {src}"
        for j, dst in enumerate(node_ids):
            if i == j:
                continue
            assert dst in cache[src], \
                f"CRITICAL: edge cache missing directed pair ({src} → {dst})"
            d = float(cache[src][dst]["euclidean_distance"])
            assert d > 0, f"CRITICAL: non-positive distance for ({src}, {dst})"
            dist[i, j] = d
    asym = np.abs(dist - dist.T).max()
    assert asym <= 1.0, \
        f"CRITICAL: distance matrix asymmetric by {asym:.2f} m — cache corruption"
    return (dist + dist.T) / 2.0


def _edge_index(pairs: List, n: int) -> np.ndarray:
    """Directed pairs (sorted, deduplicated) + self-loops → [2, E] int64."""
    uniq = sorted(set(pairs) | {(i, i) for i in range(n)})
    return np.array(uniq, dtype=np.int64).T.reshape(2, -1)


def build_identity_graph(n: int) -> GraphSpec:
    assert n >= 1
    return GraphSpec(method="identity", edge_index=_edge_index([], n),
                     edge_attr=None, edge_dim=0, params={"n_nodes": n})


def build_gabriel_graph(dist: np.ndarray, node_ids: Sequence[str]) -> GraphSpec:
    n = len(node_ids)
    assert dist.shape == (n, n), f"CRITICAL: dist shape {dist.shape} != ({n}, {n})"
    d2 = dist ** 2
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Gabriel: no w with d(i,w)² + d(w,j)² < d(i,j)².
            others = d2[i] + d2[j]          # [n]
            others[i] = others[j] = np.inf
            if (others >= d2[i, j]).all():
                pairs.append((i, j))
                pairs.append((j, i))
    spec = GraphSpec(method="gabriel", edge_index=_edge_index(pairs, n),
                     edge_attr=None, edge_dim=0, params={"n_nodes": n})
    # Gabriel graphs on points in general position are connected; a violation
    # flags cache corruption. Warn-and-record rather than die (recorded in the
    # manifest via params).
    n_components = _count_components(spec.edge_index, n)
    object.__setattr__(spec, "params", {**spec.params, "n_components": n_components})
    return spec


def build_knn_graph(dist: np.ndarray, node_ids: Sequence[str], k: int,
                    symmetrise: bool = True) -> GraphSpec:
    n = len(node_ids)
    assert 1 <= k < n, f"CRITICAL: knn k={k} invalid for {n} nodes"
    pairs = []
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            pairs.append((int(j), i))       # neighbour → target (message direction)
            if symmetrise:
                pairs.append((i, int(j)))
    return GraphSpec(method="knn", edge_index=_edge_index(pairs, n),
                     edge_attr=None, edge_dim=0,
                     params={"n_nodes": n, "k": k, "symmetrise": symmetrise})


def build_fc_graph(n: int, node_ids: Sequence[str]) -> GraphSpec:
    assert n == len(node_ids)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return GraphSpec(method="fully_connected", edge_index=_edge_index(pairs, n),
                     edge_attr=None, edge_dim=0, params={"n_nodes": n})


def load_physical_edge_attr(edge_index: np.ndarray, node_ids: Sequence[str],
                            cache_path: str) -> np.ndarray:
    """
    [E, 9] float32 physical features per directed edge, normalised as in
    DynamicGraphLearner (both distance channels / global max shortest-path
    distance over the cache). Self-loop rows are zero except directness=1.
    """
    cache = _load_cache(cache_path)
    max_distance = max(
        pair["shortest_path_distance"]
        for pairs in cache.values() for pair in pairs.values()
    )
    assert max_distance > 0, "CRITICAL: max_distance must be positive"

    E = edge_index.shape[1]
    attr = np.zeros((E, len(EDGE_FEATURE_KEYS_PHYSICAL)), dtype=np.float32)
    for e in range(E):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i == j:
            attr[e, _DIRECTNESS_COL] = 1.0
            continue
        src, dst = node_ids[i], node_ids[j]
        assert src in cache and dst in cache[src], \
            f"CRITICAL: edge cache missing pair ({src} → {dst})"
        feats = cache[src][dst]
        row = []
        for key in EDGE_FEATURE_KEYS_PHYSICAL:
            assert key in feats, f"CRITICAL: cache pair ({src},{dst}) missing {key!r}"
            v = float(feats[key])
            if key in ("shortest_path_distance", "euclidean_distance"):
                v = v / max_distance
            row.append(v)
        attr[e] = row
        road_sum = attr[e, 3:9].sum()
        assert abs(road_sum - 1.0) < 0.05 or road_sum < 0.05, (
            f"CRITICAL: road fractions for ({src},{dst}) sum to {road_sum:.3f} "
            f"(expected ~1.0 or ~0.0)"
        )
    assert np.isfinite(attr).all(), "CRITICAL: non-finite physical edge_attr"
    return attr


def build_graph(variant_key: str, variant_cfg: dict, node_ids: Sequence[str],
                cache_path: Optional[str]) -> GraphSpec:
    """Config-driven dispatcher; attaches physical edge_attr when requested."""
    method = variant_cfg["method"]
    assert method in KNOWN_METHODS, \
        f"CRITICAL: unknown graph method {method!r} for variant {variant_key!r}"
    n = len(node_ids)

    if method == "identity":
        spec = build_identity_graph(n)
    else:
        assert cache_path, \
            f"CRITICAL: graph variant {variant_key!r} needs edge_features_cache"
        dist = load_distance_matrix(node_ids, cache_path)
        if method == "gabriel":
            spec = build_gabriel_graph(dist, node_ids)
        elif method == "knn":
            spec = build_knn_graph(dist, node_ids, k=int(variant_cfg["k"]),
                                   symmetrise=bool(variant_cfg.get("symmetrise", True)))
        else:
            spec = build_fc_graph(n, node_ids)

    edge_attr_mode = variant_cfg.get("edge_attr", "none")
    assert edge_attr_mode in ("none", "physical"), \
        f"CRITICAL: unknown edge_attr mode {edge_attr_mode!r}"
    if edge_attr_mode == "physical":
        assert method != "identity", \
            "CRITICAL: identity graph with physical edge_attr is meaningless"
        attr = load_physical_edge_attr(spec.edge_index, node_ids, cache_path)
        spec = GraphSpec(method=spec.method, edge_index=spec.edge_index,
                         edge_attr=attr, edge_dim=attr.shape[1],
                         params={**spec.params, "edge_attr": "physical"})
    return spec


def graph_manifest(spec: GraphSpec, node_ids: Sequence[str]) -> dict:
    """Persistable description incl. sha16 over the resolved edge list."""
    n = len(node_ids)
    src, dst = spec.edge_index
    non_loop = src != dst
    in_degree = np.bincount(dst[non_loop], minlength=n)
    edge_pairs = [
        [node_ids[int(s)], node_ids[int(d)]]
        for s, d in zip(src, dst) if s != d
    ]
    sha = hashlib.sha256(
        json.dumps({"nodes": list(node_ids), "edges": edge_pairs},
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    return {
        "method": spec.method,
        "params": {k: v for k, v in spec.params.items()},
        "n_nodes": n,
        "n_edges_directed": int(non_loop.sum()),
        "n_self_loops": int((~non_loop).sum()),
        "edge_dim": spec.edge_dim,
        "in_degree": {"min": int(in_degree.min()), "median": float(np.median(in_degree)),
                      "max": int(in_degree.max())},
        "edges": edge_pairs,
        "sha16": sha,
    }


def _count_components(edge_index: np.ndarray, n: int) -> int:
    """Undirected connected components via union-find (self-loops ignored)."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in edge_index.T:
        if s == d:
            continue
        ra, rb = find(int(s)), find(int(d))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})
