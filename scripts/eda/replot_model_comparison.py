#!/usr/bin/env python3
"""
scripts/eda/replot_model_comparison.py

Regenerates model_comparison_mae.png and model_comparison_mase.png
with publication-quality styling:
- "pedestrian counts" on MAE y-axis (fixing "raw vehicle counts")
- Clean unclipped MASE y-axis label
- Full un-truncated display names (e.g. "GAT-Transformer (Plain Gabriel)")
- Syncs outputs to workspace artifacts and thesis figure directories.
"""
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER_RC = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

LABEL_MAP = {
    "gat_lstm|gabriel": "GAT-LSTM (Gabriel)",
    "gat_learned|identity": "GAT-Learned",
    "gat_transformer|knn8": "GAT-Transformer (kNN-8)",
    "gat_transformer|fc": "GAT-Transformer (FC)",
    "gat_transformer|gabriel": "GAT-Transformer (Gabriel)",
    "gat_transformer|gabriel_plain": "GAT-Transformer (Plain Gabriel)",
    "gat_plain|gabriel": "GAT-Plain (Gabriel)",
    "lstm|identity": "LSTM (Temporal)",
    "transformer|identity": "Transformer (Temporal)",
    "snaive_weekly|identity": "S-Naive (Weekly)",
    "persistence|identity": "Persistence",
    "hist_avg|identity": "Historical Average",
    "snaive_daily|identity": "S-Naive (Daily)",
}

def load_data(spatial_dir: Path):
    runs = sorted((spatial_dir / "runs").glob("*/manifest.json"))
    frames = []
    for man_path in runs:
        with open(man_path) as f:
            man = json.load(f)
        res_file = man_path.parent / "results.csv"
        if res_file.exists():
            df = pd.read_csv(res_file)
            df["stage"] = man.get("stage", "")
            frames.append(df)
    results = pd.concat(frames, ignore_index=True)
    test = results[results["split"] == "test"].copy()
    test["raw_label"] = test["model"] + "|" + test["graph_variant"]
    return test

def plot_metric(test_df: pd.DataFrame, metric: str, out_paths: list, ylabel: str, title: str):
    plt.rcParams.update(PAPER_RC)
    horizons = sorted(test_df["horizon"].unique())
    raw_labels = list(LABEL_MAP.keys())
    
    # Filter only labels present in test set
    present_labels = [lb for lb in raw_labels if lb in test_df["raw_label"].unique()]
    display_labels = [LABEL_MAP[lb] for lb in present_labels]

    fig, axes = plt.subplots(
        1, len(horizons),
        figsize=(16, 5.8),
        sharey=True,
        squeeze=False
    )
    
    for idx, (ax, h) in enumerate(zip(axes[0], horizons)):
        grp = test_df[test_df["horizon"] == h]
        data = [grp[grp["raw_label"] == lb][metric].dropna() for lb in present_labels]
        
        # Color coding: GAT deep blue, GAT plain lighter blue, temporal orange, heuristic purple
        box = ax.boxplot(
            data,
            tick_labels=display_labels,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.6),
            widths=0.62
        )
        
        for patch, lb in zip(box['boxes'], present_labels):
            if "gat" in lb and lb != "gat_plain|gabriel":
                patch.set_facecolor("#2b5c8f")  # deep blue
                patch.set_alpha(0.72)
            elif lb == "gat_plain|gabriel":
                patch.set_facecolor("#5c8bb8")  # lighter blue
                patch.set_alpha(0.65)
            elif "lstm" in lb or "transformer" in lb:
                patch.set_facecolor("#d95f02")  # orange
                patch.set_alpha(0.72)
            else:
                patch.set_facecolor("#7570b3")  # muted purple/heuristic
                patch.set_alpha(0.55)

        ax.set_title(f"Horizon $h = {h}$ ({h * 15} min)", pad=12, weight="bold", fontsize=12.5)
        ax.set_xticks(range(1, len(display_labels) + 1))
        ax.set_xticklabels(display_labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0][0].set_ylabel(ylabel, weight="bold", fontsize=12, labelpad=8)
    fig.suptitle(title, y=0.98, weight="bold", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32, top=0.88, left=0.07, right=0.98)
    
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)

def main():
    spatial_dir = Path("artifacts/spatial_full")
    dest_dir = Path("latex-documents/thesis/figures/experiment2")
    src_fig_dir = spatial_dir / "aggregated" / "figures"
    art_fig_dir = Path("artifacts/figures")
    art_fig_dir.mkdir(parents=True, exist_ok=True)
    src_fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading run results...")
    test_df = load_data(spatial_dir)
    print(f"Loaded {len(test_df)} test rows across {test_df['raw_label'].nunique()} models.")
    
    mae_targets = [src_fig_dir / "model_comparison_mae.png", art_fig_dir / "model_comparison_mae.png"]
    if dest_dir.exists():
        mae_targets.append(dest_dir / "model_comparison_mae.png")
    
    mase_targets = [src_fig_dir / "model_comparison_mase.png", art_fig_dir / "model_comparison_mase.png"]
    if dest_dir.exists():
        mase_targets.append(dest_dir / "model_comparison_mase.png")

    # 1. Plot MAE
    plot_metric(
        test_df, "mae_raw",
        mae_targets,
        ylabel="Node-Median MAE (pedestrian counts)",
        title="Per-Node Forecast Error Distribution (MAE) Across Horizons"
    )
    
    # 2. Plot MASE
    plot_metric(
        test_df, "mase",
        mase_targets,
        ylabel="Node-Median MASE",
        title="Per-Node Scaled Forecast Error (MASE) Across Horizons"
    )
    
    # Copy script for thesis reproducibility if thesis dir exists
    code_dir = Path("latex-documents/thesis/code/notebooks")
    if code_dir.exists():
        shutil.copy2(__file__, code_dir / "replot_model_comparison.py")
    print("All model comparison figures regenerated successfully!")

if __name__ == "__main__":
    main()
