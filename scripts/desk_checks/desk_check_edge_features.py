#!/usr/bin/env python3
"""
scripts/desk_check_edge_features.py

Phase 3 (Layer 3: Edge Features) desk check. Proves the mathematical / structural
correctness of DynamicGraphLearner's physical edge feature construction BEFORE any
training run, using a small synthetic cache so the numbers are hand-verifiable.

Covers:
  1. edge_attr shape [E, 10]; road fractions (cols 3:9) sum to ~1.0 on connected edges.
  2. Disconnected-pair tolerance: an all-zero road-fraction pair passes the corrected
     assertion (road-sum ~0.0 OR ~1.0), unlike the architect's strict sum==1.0.
  3. Distance normalization: shortest_path / euclidean both land in [0, 1].
  4. Fail-fast: a missing cache entry raises ValueError("CRITICAL: Missing cache ...").
  5. Fallback: no cache -> edge_attr is [E, 1] and equals edge_weight.
  6. Vectorized gather matches a reference per-edge dict lookup (no silent reindexing).

Run INSIDE the container:
    python scripts/desk_check_edge_features.py
"""
import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from graph_learner import DynamicGraphLearner, _ROAD_KEYS


def hr(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def _pair(sp, eucl, directness, roads):
    """Build one cache pair dict. `roads` is a 6-tuple matching _ROAD_KEYS order."""
    d = {
        "shortest_path_distance": sp,
        "euclidean_distance": eucl,
        "directness": directness,
    }
    for k, v in zip(_ROAD_KEYS, roads):
        d[k] = v
    return d


def make_mock_cache():
    """
    3 nodes S1, S2, S3 (fully connected, directed). S1<->S2 and S2<->S3 are connected
    (road fractions sum to 1.0). S1->S3 / S3->S1 are DISCONNECTED (all-zero road
    fractions) to exercise the disconnected-pair tolerance.
    """
    connected_a = (0.5, 0.3, 0.1, 0.1, 0.0, 0.0)   # sums to 1.0
    connected_b = (0.0, 0.0, 0.0, 0.4, 0.6, 0.0)   # sums to 1.0
    disconnected = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # sums to 0.0
    return {
        "S1": {
            "S2": _pair(100.0, 90.0, 0.9, connected_a),
            "S3": _pair(8000.0, 7000.0, 0.875, disconnected),   # disconnected, max dist
        },
        "S2": {
            "S1": _pair(100.0, 90.0, 0.9, connected_a),
            "S3": _pair(200.0, 180.0, 0.9, connected_b),
        },
        "S3": {
            "S1": _pair(8000.0, 7000.0, 0.875, disconnected),
            "S2": _pair(200.0, 180.0, 0.9, connected_b),
        },
    }


def _make_learner(cache: dict):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cache, f)
    f.close()
    learner = DynamicGraphLearner(
        static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0,
        edge_features_cache_path=f.name,
    )
    return learner, f.name


def part1_shape_and_road_sums():
    hr("PART 1 — edge_attr shape [E, 10] + road-sum invariant (incl. disconnected)")
    cache = make_mock_cache()
    learner, path = _make_learner(cache)
    location_ids = ["S1", "S2", "S3"]

    # Force a dense (near-complete) graph: large positive static features so the
    # dot-product similarity clears the (zero) sparsity threshold for all off-diagonal pairs.
    torch.manual_seed(0)
    static = torch.ones(3, 4)
    edge_index, edge_weight, edge_attr = learner(static, location_ids=location_ids)

    E = edge_index.shape[1]
    print(f"E = {E}, edge_attr.shape = {tuple(edge_attr.shape)}")
    assert edge_attr.shape == (E, 10), f"Expected ({E}, 10), got {tuple(edge_attr.shape)}"

    road_sums = edge_attr[:, 3:9].sum(dim=1)
    print(f"road_sums = {road_sums.tolist()}")
    connected = (road_sums - 1.0).abs() < 1e-3
    disconnected = road_sums.abs() < 1e-3
    assert torch.all(connected | disconnected), "Road-sum invariant violated"
    print(f"connected edges: {int(connected.sum())}, disconnected edges: {int(disconnected.sum())}")
    assert int(disconnected.sum()) >= 1, "Expected at least one disconnected pair in this graph"
    print("OK  shape [E,10] and road-sum (~1.0 connected | ~0.0 disconnected) verified")
    os.unlink(path)


def part2_distance_normalization():
    hr("PART 2 — distance normalization in [0, 1] (cols 0 and 1)")
    cache = make_mock_cache()
    learner, path = _make_learner(cache)
    location_ids = ["S1", "S2", "S3"]
    static = torch.ones(3, 4)
    _, _, edge_attr = learner(static, location_ids=location_ids)

    sp_norm = edge_attr[:, 0]
    eucl_norm = edge_attr[:, 1]
    print(f"max_distance (global) = {learner.max_distance:.2f}")
    print(f"sp_norm   range = [{sp_norm.min():.4f}, {sp_norm.max():.4f}]")
    print(f"eucl_norm range = [{eucl_norm.min():.4f}, {eucl_norm.max():.4f}]")
    assert learner.max_distance == 8000.0, f"Expected global max 8000.0, got {learner.max_distance}"
    assert (sp_norm >= 0).all() and (sp_norm <= 1.0 + 1e-6).all(), "sp_norm out of [0,1]"
    assert (eucl_norm >= 0).all() and (eucl_norm <= 1.0 + 1e-6).all(), "eucl_norm out of [0,1]"
    # The S1->S3 pair has shortest_path == global max -> normalized to exactly 1.0.
    assert torch.isclose(sp_norm.max(), torch.tensor(1.0)), "Max sp should normalize to 1.0"
    print("OK  both distance channels normalized to [0, 1]")
    os.unlink(path)


def part3_fail_fast_missing_cache():
    hr("PART 3 — fail-fast ValueError on missing cache entry")
    # Cache only knows S1, S2; request a node S99 absent from the cache.
    cache = {
        "S1": {"S2": _pair(100.0, 90.0, 0.9, (0.5, 0.3, 0.1, 0.1, 0.0, 0.0))},
        "S2": {"S1": _pair(100.0, 90.0, 0.9, (0.5, 0.3, 0.1, 0.1, 0.0, 0.0))},
    }
    learner, path = _make_learner(cache)
    static = torch.ones(2, 4)
    raised = False
    try:
        learner(static, location_ids=["S1", "S99"])
    except ValueError as e:
        raised = True
        print(f"Raised as expected: {e}")
        assert "CRITICAL: Missing cache" in str(e), f"Unexpected message: {e}"
    assert raised, "Expected ValueError for missing cache entry"
    print("OK  fail-fast on missing cache verified")
    os.unlink(path)


def part4_fallback_no_cache():
    hr("PART 4 — fallback: no cache -> edge_attr [E, 1] == edge_weight")
    learner = DynamicGraphLearner(static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0)
    static = torch.ones(3, 4)
    edge_index, edge_weight, edge_attr = learner(static)  # no location_ids
    E = edge_index.shape[1]
    print(f"E = {E}, edge_attr.shape = {tuple(edge_attr.shape)}")
    assert edge_attr.shape == (E, 1), f"Expected ({E}, 1), got {tuple(edge_attr.shape)}"
    assert torch.allclose(edge_attr.squeeze(1), edge_weight), "Fallback edge_attr != edge_weight"
    print("OK  learned-only fallback verified")


def part5_gather_matches_reference():
    hr("PART 5 — vectorized gather matches per-edge dict lookup")
    cache = make_mock_cache()
    learner, path = _make_learner(cache)
    location_ids = ["S1", "S2", "S3"]
    static = torch.ones(3, 4)
    edge_index, edge_weight, edge_attr = learner(static, location_ids=location_ids)

    mismatches = 0
    for e in range(edge_index.shape[1]):
        i, j = edge_index[0, e].item(), edge_index[1, e].item()
        if i == j:
            # Self-loops are absent from the cache; the dense diagonal is zero by design.
            ref = [0.0] * 9 + [edge_weight[e].item()]
        else:
            src, dst = location_ids[i], location_ids[j]
            pair = cache[src][dst]
            ref = [
                pair["shortest_path_distance"] / learner.max_distance,
                pair["euclidean_distance"] / learner.max_distance,
                pair["directness"],
                *[pair[k] for k in _ROAD_KEYS],
                edge_weight[e].item(),
            ]
        got = edge_attr[e].tolist()
        if not torch.allclose(torch.tensor(got), torch.tensor(ref), atol=1e-5):
            mismatches += 1
            print(f"  MISMATCH edge {e} ({src}->{dst}): got={got} ref={ref}")
    assert mismatches == 0, f"{mismatches} edge(s) mismatched the reference lookup"
    print(f"OK  all {edge_index.shape[1]} edges match the reference dict lookup")
    os.unlink(path)


def main():
    logging.basicConfig(level=logging.WARNING)
    part1_shape_and_road_sums()
    part2_distance_normalization()
    part3_fail_fast_missing_cache()
    part4_fallback_no_cache()
    part5_gather_matches_reference()
    hr("ALL EDGE-FEATURE DESK CHECKS PASSED")


if __name__ == "__main__":
    main()
