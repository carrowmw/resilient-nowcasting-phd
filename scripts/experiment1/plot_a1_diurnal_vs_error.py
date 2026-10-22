#!/usr/bin/env python3
"""
scripts/experiment1/plot_a1_diurnal_vs_error.py

Regenerates Figure 5.4: Predictability vs Diurnal Strength
using artifacts/quality_full_v2/quality_table_none.parquet
with cleaned publication title (replacing internal code name 'h8').
Saves to artifacts and copies to thesis figures directory.
"""
import os
import json
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    with open("artifacts/quality_full_v2/sensor_manifest.json") as f:
        qm = json.load(f)
    viable = qm["viable_sensors"]

    results_path = "artifacts/quality_full/analysis/results.parquet"
    quality_path = "artifacts/quality_full_v2/quality_table_none.parquet"

    results = pd.read_parquet(results_path)
    quality = pd.read_parquet(quality_path)

    target_model = "lstm_mv"
    target_horizon = 8

    base_mask = (results["interpolation"] == "none") & (results["ablation"] == "")
    sub_results = results[
        base_mask & 
        (results["horizon"] == target_horizon) & 
        (results["model"] == target_model) &
        (results["sensor_id"].isin(viable))
    ].copy()

    sub_results = sub_results.drop(columns=[c for c in quality.columns if c in sub_results.columns and c != "sensor_id"])
    sub_results = sub_results.merge(quality.reset_index(), on="sensor_id", how="left")

    df = sub_results.dropna(subset=["diurnal_capture_ratio", "cv_rmse_raw", "avg_seq_len_hours"])
    print(f"Loaded {len(df)} sensors for diurnal predictability plot.")

    # Match original thesis font & styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    scatter = ax.scatter(
        df["diurnal_capture_ratio"],
        df["cv_rmse_raw"],
        c=df["avg_seq_len_hours"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="k",
        s=48,
        linewidth=0.8
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Avg Seq Length (h)", fontsize=13, labelpad=8)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel("Diurnal Capture Ratio (24h power fraction)", fontsize=13, labelpad=8)
    ax.set_ylabel("CV-RMSE (%)", fontsize=13, labelpad=8)
    # Clean publication title removing internal code name 'h8'
    ax.set_title("Predictability vs Diurnal Strength (LSTM-MV, 2-Hour Horizon)", fontsize=13.5, fontweight="bold", pad=12)

    ax.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(1.0)

    fig.tight_layout()

    out_dir = Path("artifacts/quality_full/analysis/deep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "a1_diurnal_vs_error.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    pdf_path = out_dir / "a1_diurnal_vs_error.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    # Also save to thesis figures directory
    thesis_fig = Path("/Users/administrator/Code/latex-documents/thesis/figures/experiment1/a1_diurnal_vs_error.png")
    if thesis_fig.parent.exists():
        shutil.copyfile(out_path, thesis_fig)
        print(f"Copied to thesis: {thesis_fig}")

if __name__ == "__main__":
    main()
