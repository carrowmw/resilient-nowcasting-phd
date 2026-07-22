import json
import os
import sys
import tempfile

import pytest

# Add scripts dir so we can import cache_simulated_edge_features as a module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from cache_simulated_edge_features import (
    assert_coords_in_graph_bbox,
    filter_simulated_pairs,
    run_self_test,
)


def test_self_test_runs():
    """run_self_test() completes without exception on the synthetic mock graph."""
    run_self_test()


def test_output_excludes_real_real_pairs():
    """filter_simulated_pairs removes all real→real directed pairs from the output."""
    from cache_edge_features import build_mock_graph
    from src.features import compute_pairwise_edge_features

    G, all_coords, node_ids = build_mock_graph()
    all_pairs = compute_pairwise_edge_features(
        G, all_coords, cache_path=None, nearest_nodes=node_ids
    )

    is_simulated = lambda name: name in {"S4", "S5"}
    filtered = filter_simulated_pairs(all_pairs, is_simulated)

    real_nodes = {"S1", "S2", "S3"}
    for src in real_nodes:
        for dst in real_nodes:
            if src != dst:
                if src in filtered:
                    assert dst not in filtered[src], (
                        f"Real→real pair {src}→{dst} should not appear in the simulated cache"
                    )


def test_bounding_box_fail_fast():
    """assert_coords_in_graph_bbox raises ValueError for a coord outside the graph bbox."""
    from cache_edge_features import build_mock_graph

    G, _, _ = build_mock_graph()
    # A coordinate wildly outside the mock graph's node extent (lat ~54.978, lon ~-1.610).
    bad_coords = [("FakeSimulated_0", 90.0, 0.0)]
    with pytest.raises(ValueError, match="outside OSM graph bbox"):
        assert_coords_in_graph_bbox(G, bad_coords)
