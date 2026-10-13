"""
Thesis figures for the spatial track. Mirrors the quality track's paper
styling (same PAPER_RC values) so Chapter 5 and Chapter 6 figures read as
one thesis. Node layouts are classical MDS on the cached distance matrix —
no coordinate pipeline needed.
"""
import logging
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PAPER_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


def mds_layout(dist: np.ndarray) -> np.ndarray:
    """[N, 2] classical MDS (Torgerson) embedding of a distance matrix."""
    n = dist.shape[0]
    assert dist.shape == (n, n)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dist ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:2]
    coords = vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))
    return coords


def plot_model_comparison(results: pd.DataFrame, metric: str, out_path: str,
                          title: Optional[str] = None) -> None:
    """
    Box plot of the per-node metric across models, one panel per horizon.
    `results` = concatenated per-job evaluate_split frames with a 'label'
    column (model or model|graph).
    """
    apply_paper_style()
    horizons = sorted(results["horizon"].unique())
    labels = list(dict.fromkeys(results["label"]))
    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(4.2 * len(horizons) + 2, 5.5),
                             sharey=True, squeeze=False)
    for ax, h in zip(axes[0], horizons):
        grp = results[results["horizon"] == h]
        data = [grp[grp["label"] == lb][metric].dropna() for lb in labels]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"h = {h} step{'s' if h > 1 else ''}")
        ax.tick_params(axis="x", rotation=60)
    axes[0][0].set_ylabel(metric)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_spatial_delta(dm_df: pd.DataFrame, node_metric_delta: pd.Series,
                       out_path: str, title: Optional[str] = None,
                       alpha: float = 0.05) -> None:
    """
    Per-node paired metric deltas (focal - baseline), DM-significant nodes
    marked. `dm_df` from stats.dm_by_node, `node_metric_delta` indexed by
    node_id.
    """
    apply_paper_style()
    df = dm_df.set_index("node_id").join(node_metric_delta.rename("delta")).dropna(
        subset=["delta"])
    df = df.sort_values("delta")
    sig = (df["excluded_reason"] == "") & (df["p_value"] < alpha)
    fig, ax = plt.subplots(figsize=(max(7, 0.16 * len(df)), 5))
    colors = np.where(sig, "#1f77b4", "#bbbbbb")
    ax.bar(range(len(df)), df["delta"], color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks([])
    ax.set_xlabel(f"nodes (sorted; blue = DM p < {alpha})")
    ax.set_ylabel("Δ metric (focal − baseline)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_graph_ablation(agg: pd.DataFrame, metric: str, out_path: str,
                        order: Optional[Sequence[str]] = None) -> None:
    """Metric (± seed sd) across graph variants, one line per horizon.
    `agg` needs columns: graph_variant, horizon, mean, sd (from seed_stability)."""
    apply_paper_style()
    variants = list(order) if order else list(dict.fromkeys(agg["graph_variant"]))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for h, grp in agg.groupby("horizon"):
        grp = grp.set_index("graph_variant").reindex(variants)
        ax.errorbar(variants, grp["mean"], yerr=grp["sd"], marker="o",
                    capsize=4, label=f"h = {h}")
    ax.set_ylabel(metric)
    ax.set_xlabel("graph variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_attention_graph(record, dist: np.ndarray, node_ids: Sequence[str],
                         out_path: str, title: Optional[str] = None) -> None:
    """Network drawn at the MDS layout, edge width/alpha ∝ mean attention."""
    apply_paper_style()
    coords = mds_layout(dist)
    alpha = record.alpha_flat()
    src, dst = record.edge_index
    keep = src != dst
    a = alpha[keep]
    a_norm = a / a.max() if a.max() > 0 else a
    fig, ax = plt.subplots(figsize=(8, 8))
    for (s, d, w) in zip(src[keep], dst[keep], a_norm):
        ax.plot(coords[[s, d], 0], coords[[s, d], 1],
                color="#1f77b4", alpha=float(0.15 + 0.85 * w),
                linewidth=float(0.5 + 3.0 * w), zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=60, c="#d62728", zorder=2,
               edgecolor="k", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_attention_distance_decay(df: pd.DataFrame, out_path: str,
                                  title: Optional[str] = None) -> None:
    """Scatter + binned median of edge attention vs distance (attention.attention_vs_distance)."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["distance_m"], df["alpha"], s=24, alpha=0.4, edgecolor="none")
    bins = np.quantile(df["distance_m"], np.linspace(0, 1, 9))
    mids, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df["distance_m"] >= lo) & (df["distance_m"] <= hi)
        if m.any():
            mids.append((lo + hi) / 2)
            meds.append(df.loc[m, "alpha"].median())
    ax.plot(mids, meds, color="#d62728", marker="o", label="binned median")
    ax.set_xlabel("edge distance (m)")
    ax.set_ylabel("mean attention α")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_seed_stability(stab: pd.DataFrame, metric_label: str, out_path: str) -> None:
    """Across-seed mean ± range per (model|graph, horizon) — stats.seed_stability."""
    apply_paper_style()
    stab = stab.assign(label=stab["model"] + "|" + stab["graph_variant"])
    horizons = sorted(stab["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons) + 2, 5),
                             sharey=True, squeeze=False)
    for ax, h in zip(axes[0], horizons):
        grp = stab[stab["horizon"] == h]
        x = np.arange(len(grp))
        yerr = np.vstack([grp["mean"] - grp["min"], grp["max"] - grp["mean"]])
        ax.errorbar(x, grp["mean"], yerr=yerr, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(grp["label"], rotation=60, ha="right")
        ax.set_title(f"h = {h}")
    axes[0][0].set_ylabel(metric_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
