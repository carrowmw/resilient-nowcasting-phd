#!/usr/bin/env python3
"""
scripts/desk_checks/desk_check_spatial_graphs.py

Builds every configured graph variant on the REAL node set and prints edge
counts, degree distributions, Gabriel connectivity, and edge-attribute
summaries; renders an MDS layout figure per graph to /tmp. Human-verifiable
structure before any training run.

Run INSIDE the container:
    python scripts/desk_checks/desk_check_spatial_graphs.py [--config configs/spatial/spatial_smoke.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from spatial.graphs import build_graph, graph_manifest, load_distance_matrix  # noqa: E402
from spatial.loader import GraphSeriesStore  # noqa: E402
from spatial.plotting import mds_layout, plot_attention_graph  # noqa: E402
from spatial.attention import AttentionRecord  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spatial/spatial_smoke.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    store = GraphSeriesStore(config)
    ctx = store.build_context()
    cache = config["nodes"]["edge_features_cache"]
    dist = load_distance_matrix(ctx.node_ids, cache)
    print(f"Node set: {ctx.n_nodes} nodes")
    print(f"Distance matrix: min {dist[dist > 0].min():.0f} m, "
          f"median {np.median(dist[dist > 0]):.0f} m, max {dist.max():.0f} m")

    for variant, vcfg in config["graph"]["variants"].items():
        spec = build_graph(variant, vcfg, ctx.node_ids, cache)
        man = graph_manifest(spec, ctx.node_ids)
        print(f"\n{variant} ({spec.method}):")
        print(f"  directed edges: {man['n_edges_directed']}, "
              f"self-loops: {man['n_self_loops']}, edge_dim: {spec.edge_dim}")
        print(f"  in-degree: min {man['in_degree']['min']}, "
              f"median {man['in_degree']['median']}, max {man['in_degree']['max']}")
        if spec.method == "gabriel":
            print(f"  connected components: {spec.params['n_components']} "
                  f"(1 expected for points in general position)")
        if spec.edge_attr is not None:
            attr = spec.edge_attr
            loops = spec.edge_index[0] == spec.edge_index[1]
            print(f"  edge_attr: dist_norm [{attr[~loops, 0].min():.3f}, "
                  f"{attr[~loops, 0].max():.3f}], "
                  f"directness median {np.median(attr[~loops, 2]):.3f}, "
                  f"road-fraction sums ~1: "
                  f"{np.isclose(attr[~loops, 3:9].sum(1), 1.0, atol=0.05).mean():.1%}")
        # MDS layout figure (uniform pseudo-attention → pure structure).
        if man["n_edges_directed"] > 0:
            E = spec.edge_index.shape[1]
            rec = AttentionRecord(
                edge_index=spec.edge_index,
                alpha_mean=np.full((E, 1), 0.5, dtype=np.float32),
                alpha_by_hour=np.full((24, E), np.nan, dtype=np.float32),
                n_windows=0, seed=0)
            out = f"/tmp/desk_check_graph_{variant}.png"
            plot_attention_graph(rec, dist, ctx.node_ids, out,
                                 title=f"{variant} topology (MDS layout)")
            print(f"  layout figure: {out}")

    print("\nDESK CHECK PASSED")


if __name__ == "__main__":
    main()
