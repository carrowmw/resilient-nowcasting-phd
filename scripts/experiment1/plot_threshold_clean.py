#!/usr/bin/env python3
"""
scripts/experiment1/plot_threshold_clean.py

Regenerates Figure 5.6: threshold_cvrmse_lstm_mv_h8.png
with a clean title (LSTM-MV, h = 8 (2 h)), proper labels,
and exports breakpoint CI and slopes.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, "src")
from quality.stats import segmented_threshold_fit

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
    sub = results[
        base_mask & 
        (results["horizon"] == target_horizon) & 
        (results["model"] == target_model) &
        (results["sensor_id"].isin(viable))
    ].copy()

    sub = sub.drop(columns=[c for c in quality.columns if c in sub.columns and c != "sensor_id"])
    sub = sub.merge(quality.reset_index(), on="sensor_id", how="left")

    x = sub["avg_seq_len_hours"].to_numpy(dtype=float)
    y = sub["cv_rmse_raw"].to_numpy(dtype=float)

    fit = segmented_threshold_fit(x, y, n_bootstrap=1000, seed=42)
    print(f"n: {len(x)}")
    print(f"Breakpoint: {fit.breakpoint:.2f} h")
    print(f"Breakpoint 95% CI: [{fit.breakpoint_ci[0]:.2f}, {fit.breakpoint_ci[1]:.2f}] h")
    print(f"Slope below: {fit.slope_below:.2f} %/h")
    print(f"Slope above: {fit.slope_above:.2f} %/h")
    print(f"Intercept: {fit.intercept:.2f}")
    print(f"Delta AIC: {fit.delta_aic:.2f}")

    # Plot
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=300)
    ax.scatter(x, y, s=48, alpha=0.75, color="#1f77b4", edgecolor="#111111", linewidth=0.6)

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ys = (
        fit.intercept
        + fit.slope_below * xs
        + (fit.slope_above - fit.slope_below) * np.maximum(0.0, xs - fit.breakpoint)
    )
    ax.plot(xs, ys, color="crimson", lw=2.2, label="Segmented fit")
    ax.axvline(
        fit.breakpoint, color="crimson", ls="--", lw=1.6,
        label=f"Breakpoint = {fit.breakpoint:.2f} h"
    )
    ax.axvspan(
        *fit.breakpoint_ci, color="crimson", alpha=0.12,
        label=f"95% CI [{fit.breakpoint_ci[0]:.2f}, {fit.breakpoint_ci[1]:.2f}] h"
    )

    ax.set_xlabel("Average sequence length (hours)", fontsize=12.5, labelpad=8)
    ax.set_ylabel("CV-RMSE (%)", fontsize=12.5, labelpad=8)
    # Clean publication title requested in audit
    ax.set_title("Predictability Threshold (LSTM-MV, h = 8 [2 h])", fontsize=13.5, fontweight="bold", pad=12)

    ax.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(1.0)

    ax.legend(frameon=True, fontsize=10.5, loc="upper right")
    fig.tight_layout()

    out_dir = Path("artifacts/quality_full/analysis/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "threshold_cvrmse_lstm_mv_h8.png"
    out_pdf = out_dir / "threshold_cvrmse_lstm_mv_h8.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

    # Also save to thesis figures
    thesis_dir = Path("/Users/administrator/Code/latex-documents/thesis/figures/experiment1")
    if thesis_dir.exists():
        shutil.copyfile(out_png, thesis_dir / out_png.name)
        shutil.copyfile(out_pdf, thesis_dir / out_pdf.name)
        print(f"Copied to thesis: {out_png.name}")

if __name__ == "__main__":
    main()
