"""
plot_attention_distance_decay.py
Generates publication-quality distance decay plots for spatial attention weights.
Degree-normalizes attention: \\alpha_{ij} * |N(i)| to eliminate degree confounding.
Includes uniform prior reference line (y = 1.0) and binned median curve.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Consistent style settings
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e5e7eb",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
})

CONFIGS = [
    {
        "key": "gabriel",
        "csv": "artifacts/spatial_full/aggregated/attention_distance_gat_transformer_gabriel.csv",
        "out": "attention_distance_gat_transformer_gabriel.png",
        "title": "Gabriel Planar Graph",
        "subtitle": "156 directed edges (median in-degree: 3.0)",
        "r_table": -0.52,
        "r_log_table": -0.55,
        "color": "#1f77b4",
    },
    {
        "key": "knn8",
        "csv": "artifacts/spatial_full/aggregated/attention_distance_gat_transformer_knn8.csv",
        "out": "attention_distance_gat_transformer_knn8.png",
        "title": r"$k$-NN-8 Metric Proximity Graph",
        "subtitle": "544 directed edges (median in-degree: 9.5)",
        "r_table": -0.03,
        "r_log_table": -0.05,
        "color": "#6366f1",
    },
    {
        "key": "fc",
        "csv": "artifacts/spatial_full/aggregated/attention_distance_gat_transformer_fc.csv",
        "out": "attention_distance_gat_transformer_fc.png",
        "title": "Fully-Connected Topology",
        "subtitle": "2,862 directed edges (in-degree: 53.0)",
        "r_table": -0.42,
        "r_log_table": -0.55,
        "color": "#0ea5e9",
    },
    {
        "key": "learned",
        "csv": "artifacts/spatial_full/aggregated/attention_distance_gat_learned_identity.csv",
        "out": "attention_distance_gat_learned_identity.png",
        "title": "Learned Adaptive Adjacency",
        "subtitle": r"$\sim$1,756 active links (median in-degree: 33.5)",
        "r_table": 0.08,
        "r_log_table": 0.05,
        "color": "#8b5cf6",
    },
    {
        "key": "gabriel_plain",
        "csv": "artifacts/spatial_full/aggregated/attention_distance_gat_transformer_gabriel_plain.csv",
        "out": "attention_distance_gat_transformer_gabriel_plain.png",
        "title": "Gabriel Graph (Unconditioned / No Edge Features)",
        "subtitle": "156 directed edges (median in-degree: 3.0)",
        "r_table": -0.21,
        "r_log_table": -0.31,
        "color": "#64748b",
    },
]

OUT_DIRS = [
    "artifacts/spatial_full/aggregated/figures",
    "/Users/administrator/Code/latex-documents/thesis/figures/experiment2",
]

def plot_single(cfg):
    csv_path = cfg["csv"]
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_path}, not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Destination node in-degree (number of candidate incoming edges)
    k_in = df.groupby("dst")["alpha"].transform("count")
    # Degree-normalized attention
    df["deg_norm"] = df["alpha"] * k_in

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

    # 1. Background grid
    ax.grid(True, linestyle="--", alpha=0.6, color="#d1d5db")

    # 2. Reference line for uniform attention prior (y = 1.0)
    ax.axhline(
        1.0,
        color="#6b7280",
        linestyle=":",
        linewidth=1.3,
        alpha=0.9,
        label=r"Uniform Prior ($\tilde{\alpha} = 1.0$)",
        zorder=2,
    )

    # 3. Scatter points
    scatter = ax.scatter(
        df["distance_m"],
        df["deg_norm"],
        c=cfg["color"],
        alpha=0.38,
        s=24,
        edgecolors="none",
        label=r"Edge Attention ($\alpha_{ij} \cdot |\mathcal{N}(i)|$)",
        zorder=3,
    )

    # 4. Binned median curve (6 quantile bins)
    df["bin"] = pd.qcut(df["distance_m"], q=6, duplicates="drop")
    binned = df.groupby("bin", observed=True).agg(
        dist_med=("distance_m", "median"),
        norm_med=("deg_norm", "median"),
    ).reset_index()

    ax.plot(
        binned["dist_med"],
        binned["norm_med"],
        color="#dc2626",
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="Binned Median",
        zorder=4,
    )

    # 5. Titles & labels
    r_stat = cfg["r_table"]
    r_log_stat = cfg["r_log_table"]
    sign = "+" if r_stat > 0 else ""
    sign_log = "+" if r_log_stat > 0 else ""

    ax.set_title(cfg["title"], fontsize=12.5, fontweight="bold", pad=18)
    subtitle = f"{cfg['subtitle']}  |  $r = {sign}{r_stat:.2f}$ ($r_{{\\log}} = {sign_log}{r_log_stat:.2f}$)"
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, fontsize=10, ha="center", va="bottom", color="#444444")

    ax.set_xlabel("Inter-Sensor Distance (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Degree-Normalized Attention ($\alpha_{ij} \cdot |\mathcal{N}(i)|$)", fontsize=11, fontweight="bold")

    # Limits and aesthetics
    ax.set_xlim(left=-50, right=max(df["distance_m"]) * 1.03)
    y_max = max(df["deg_norm"]) * 1.08
    ax.set_ylim(bottom=-0.05, top=y_max)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()

    # Save to local artifacts and latex directory if available
    local_out = os.path.join("artifacts/spatial_full/aggregated/figures", cfg["out"])
    plt.savefig(local_out, dpi=300)
    print(f"Saved: {local_out}")

    latex_out = os.path.join("/Users/administrator/Code/latex-documents/thesis/figures/experiment2", cfg["out"])
    if os.path.exists(os.path.dirname(latex_out)):
        plt.savefig(latex_out, dpi=300)
        print(f"Synced to: {latex_out}")

    plt.close()

def main():
    os.makedirs("artifacts/spatial_full/aggregated/figures", exist_ok=True)
    for cfg in CONFIGS:
        plot_single(cfg)

if __name__ == "__main__":
    main()
