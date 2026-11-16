"""
plot_seed_stability.py
Generates publication-quality seed stability figure for Chapter 6 (Figure 6.6).
- Panel A: Cross-horizon forecast error (MAE +- 1 SD) across h=1, h=2, h=4 for neural configurations.
- Panel B: High-resolution zoom on h=1 lead time, showing individual seeds (42, 43, 44) and error bars.
- Uses clean display names, publication styling, and eliminates raw baselines.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14.5,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e5e7eb",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
})

CSV_PATH = "artifacts/spatial_full/aggregated/seed_stability.csv"

# Clean display names and category mapping
MODEL_MAP = {
    # Topologies
    ("gat_lstm", "gabriel"): ("GAT-LSTM (Gabriel) [Champ]", "Topology", "#1e3a8a"),
    ("gat_transformer", "gabriel"): ("GAT-Trans (Gabriel)", "Topology", "#2563eb"),
    ("gat_transformer", "knn8"): ("GAT-Trans (kNN-8)", "Topology", "#0284c7"),
    ("gat_transformer", "fc"): ("GAT-Trans (FC)", "Topology", "#0d9488"),
    ("gat_learned", "identity"): ("GAT-Learned (Adaptive)", "Topology", "#059669"),
    # Conditioning
    ("gat_transformer", "gabriel_plain"): ("GAT-Trans (Gab Plain)", "Ablation", "#ca8a04"),
    ("gat_plain", "gabriel"): ("GATv2 (Plain)", "Ablation", "#d97706"),
    # Capacity
    ("gat_l1", "gabriel"): ("GAT Depth L=1", "Capacity", "#9333ea"),
    ("gat_l3", "gabriel"): ("GAT Depth L=3", "Capacity", "#7e22ce"),
    ("gat_h1", "gabriel"): ("GAT Heads H=1", "Capacity", "#c026d3"),
    ("gat_h8", "gabriel"): ("GAT Heads H=8", "Capacity", "#db2777"),
    # Temporal Controls
    ("gat_transformer", "identity"): ("GAT-Trans (Identity)", "Temporal", "#64748b"),
    ("lstm", "identity"): ("LSTM (Temporal)", "Temporal", "#475569"),
    ("transformer", "identity"): ("Transformer (Temporal)", "Temporal", "#334155"),
}

ORDER = [
    ("gat_lstm", "gabriel"),
    ("gat_transformer", "gabriel"),
    ("gat_transformer", "knn8"),
    ("gat_transformer", "fc"),
    ("gat_learned", "identity"),
    ("gat_transformer", "gabriel_plain"),
    ("gat_plain", "gabriel"),
    ("gat_l1", "gabriel"),
    ("gat_l3", "gabriel"),
    ("gat_h1", "gabriel"),
    ("gat_h8", "gabriel"),
    ("gat_transformer", "identity"),
    ("lstm", "identity"),
    ("transformer", "identity"),
]

def main():
    df = pd.read_csv(CSV_PATH)
    # Filter to neural models with SD
    df = df.dropna(subset=["sd"]).copy()
    
    # Map display names
    df["key"] = list(zip(df["model"], df["graph_variant"]))
    df = df[df["key"].isin(MODEL_MAP.keys())].copy()
    df["display_name"] = df["key"].apply(lambda k: MODEL_MAP[k][0])
    df["category"] = df["key"].apply(lambda k: MODEL_MAP[k][1])
    df["color"] = df["key"].apply(lambda k: MODEL_MAP[k][2])

    fig = plt.figure(figsize=(13, 9.6), dpi=300)
    
    # Grid: 2 rows. Row 1 has 3 subplots for h=1, h=2, h=4.
    # Row 2 has a wide plot focusing on h=1 ranking and seed error bar overlap.
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.78, wspace=0.22)

    # -------------------------------------------------------------
    # Row 1: h = 1, h = 2, h = 4 comparisons across all configurations
    # -------------------------------------------------------------
    horizons = [(1, "h = 1 (15 min)"), (2, "h = 2 (30 min)"), (4, "h = 4 (60 min)")]
    
    for col_idx, (h_val, h_label) in enumerate(horizons):
        ax = fig.add_subplot(gs[0, col_idx])
        sub_df = df[df["horizon"] == h_val].set_index("key").reindex(ORDER).reset_index()
        
        x_positions = np.arange(len(sub_df))
        means = sub_df["mean"].values
        sds = sub_df["sd"].values
        colors = sub_df["color"].values

        ax.errorbar(
            x_positions,
            means,
            yerr=sds,
            fmt="o",
            color="#1e293b",
            ecolor="#dc2626",
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.4,
            markersize=5.5,
            zorder=4,
        )
        
        # Color markers individually
        for xp, m, c in zip(x_positions, means, colors):
            ax.scatter([xp], [m], color=c, s=40, zorder=5, edgecolors="#1e293b", linewidth=0.6)

        ax.set_title(h_label, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sub_df["display_name"], rotation=52, ha="right", rotation_mode="anchor", fontsize=10.2)
        ax.set_ylim(4.2, 5.8)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        if col_idx == 0:
            ax.set_ylabel("Node-Median MAE (pedestrian counts)", fontsize=11.5, fontweight="bold")
        else:
            ax.set_ylabel("")

    # -------------------------------------------------------------
    # Row 2: High-Resolution Focal View of Top Configurations at h = 1
    # -------------------------------------------------------------
    ax_zoom = fig.add_subplot(gs[1, :])
    
    # Focus on top 10 models at h=1
    h1_df = df[df["horizon"] == 1].copy()
    h1_top = h1_df[h1_df["mean"] < 4.47].sort_values("mean").reset_index(drop=True)
    
    y_positions = np.arange(len(h1_top))[::-1]
    
    for yp, (_, row) in zip(y_positions, h1_top.iterrows()):
        m_val = row["mean"]
        sd_val = row["sd"]
        mn_val = row["min"]
        mx_val = row["max"]
        c = row["color"]
        name = row["display_name"]
        
        # Horizontal error bar +- 1 SD
        ax_zoom.errorbar(
            m_val,
            yp,
            xerr=sd_val,
            fmt="o",
            color=c,
            ecolor="#2563eb",
            elinewidth=2.2,
            capsize=5,
            capthick=1.8,
            markersize=7.5,
            zorder=4,
            label="Mean ± 1 SD" if yp == y_positions[0] else "",
        )
        
        # Cross-seed min / max range markers
        ax_zoom.scatter(
            [mn_val, mx_val],
            [yp, yp],
            color="#64748b",
            marker="|",
            s=80,
            linewidths=1.5,
            zorder=5,
            label="Min / Max seed" if yp == y_positions[0] else "",
        )
        
        # Numerical label text
        stat_txt = f"{m_val:.3f} ± {sd_val:.3f}"
        ax_zoom.text(
            m_val + sd_val + 0.003,
            yp,
            stat_txt,
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="semibold",
            color="#1e293b",
        )

    # Shaded band highlighting mutual seed overlap of top 5 models
    champ_mean = h1_top.loc[0, "mean"]
    champ_sd = h1_top.loc[0, "sd"]
    ax_zoom.axvspan(
        champ_mean - champ_sd,
        champ_mean + champ_sd,
        color="#2563eb",
        alpha=0.08,
        label=r"Champion ± 1 SD Band",
        zorder=1,
    )

    ax_zoom.set_yticks(y_positions)
    ax_zoom.set_yticklabels(h1_top["display_name"], fontsize=11, fontweight="semibold")
    ax_zoom.set_xlabel("Node-Median MAE (pedestrian counts) at $h = 1$ (15 min)", fontsize=12, fontweight="bold")
    ax_zoom.set_xlim(4.32, 4.53)
    ax_zoom.grid(True, linestyle="--", alpha=0.6)
    ax_zoom.set_title("Focus on Lead Time $h = 1$: Top Architectures Fall Within Cross-Seed Variability ($\pm 0.008$–$0.023$ Counts)",
                      fontsize=13, fontweight="bold", pad=8)
    ax_zoom.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    plt.suptitle("Cross-Seed Forecast Error Variability Across Model Architectures and Lead Times ($n = 3$ Seeds)",
                 fontsize=14.5, fontweight="bold", y=0.98)

    local_out = "artifacts/spatial_full/aggregated/figures/seed_stability.png"
    plt.savefig(local_out, dpi=300, bbox_inches="tight")
    print(f"Saved: {local_out}")

    latex_out = "/Users/administrator/Code/latex-documents/thesis/figures/experiment2/seed_stability.png"
    if os.path.exists(os.path.dirname(latex_out)):
        plt.savefig(latex_out, dpi=300, bbox_inches="tight")
        print(f"Synced to: {latex_out}")

    plt.close()

if __name__ == "__main__":
    main()
