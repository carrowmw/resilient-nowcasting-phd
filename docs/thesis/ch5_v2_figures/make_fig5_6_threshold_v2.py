#!/usr/bin/env python3
"""Regenerate Figure 5.6 (CV-RMSE vs average sequence length, LSTM-MV, h=8) from the
canonical v2 run set only (artifacts/quality_full_v2: v2 test predictions + v2 quality table).

The original figure (scripts/experiment1/plot_threshold_clean.py) combined v1 errors
(artifacts/quality_full) with v2 quality features. Run from the repository root:

    python docs/thesis/ch5_v2_figures/make_fig5_6_threshold_v2.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from quality.stats import segmented_threshold_fit  # noqa: E402

ART = Path("artifacts/quality_full_v2")
OUT = Path(__file__).resolve().parent


def v2_errors(model="lstm_mv", h=8, interp="none"):
    p = pd.read_parquet(ART / f"runs/{model}_h{h}_{interp}/predictions.parquet")
    p = p[p.split == "test"]
    g = p.groupby("sensor_id").apply(
        lambda d: pd.Series({
            "rmse_raw": np.sqrt(((d.y_true_raw - d.y_pred_raw) ** 2).mean()),
            "mean_count_raw": d.y_true_raw.mean(),
        })
    )
    g["cv_rmse_raw"] = 100 * g.rmse_raw / g.mean_count_raw
    return g


def main():
    viable = json.load(open(ART / "sensor_manifest.json"))["viable_sensors"]
    q = pd.read_parquet(ART / "quality_table_none.parquet")
    d = v2_errors().join(q[["avg_seq_len_hours"]], how="inner")
    d = d.loc[d.index.isin(viable)].dropna(subset=["avg_seq_len_hours", "cv_rmse_raw"])

    x = d.avg_seq_len_hours.to_numpy(float)
    y = d.cv_rmse_raw.to_numpy(float)
    fit = segmented_threshold_fit(x, y, n_bootstrap=1000, seed=42)

    # The three severely fragmented high-error streams named in the thesis (line 294).
    worst3 = pd.Index([
        "NclQueenVictoriaRdBedsonBuilding__from_south_to_north",
        "NclUrbanScienceBuildingSquareMid__west_to_east",
        "NclQueenVictoriaRdPercyBuilding__from_south_to_north",
    ])
    d_ex = d.drop(worst3)
    fit_ex = segmented_threshold_fit(d_ex.avg_seq_len_hours.to_numpy(float),
                                     d_ex.cv_rmse_raw.to_numpy(float), n_bootstrap=1000, seed=42)

    lines = [
        f"n = {len(x)}",
        f"breakpoint = {fit.breakpoint:.2f} h, 95% CI [{fit.breakpoint_ci[0]:.2f}, {fit.breakpoint_ci[1]:.2f}]",
        f"slope below = {fit.slope_below:.2f} pp/h, slope above = {fit.slope_above:.2f} pp/h, dAIC = {fit.delta_aic:.2f}",
        f"excluding fragmented outliers ({', '.join(worst3)}): n = {len(d_ex)}",
        f"  breakpoint = {fit_ex.breakpoint:.2f} h, 95% CI [{fit_ex.breakpoint_ci[0]:.2f}, {fit_ex.breakpoint_ci[1]:.2f}]",
        f"  slope below = {fit_ex.slope_below:.2f}, slope above = {fit_ex.slope_above:.2f}, dAIC = {fit_ex.delta_aic:.2f}",
        f"streams > 6 h: {(x > 6).sum()}",
        "worst 5: " + "; ".join(f"{s} {v:.2f}% ({d.loc[s,'avg_seq_len_hours']:.2f} h)"
                                 for s, v in d.cv_rmse_raw.nlargest(5).items()),
    ]
    (OUT / "fig5_6_threshold_v2_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=300)
    ax.scatter(x, y, s=48, alpha=0.75, color="#1f77b4", edgecolor="#111111", linewidth=0.6)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = fit.intercept + fit.slope_below * xs + (fit.slope_above - fit.slope_below) * np.maximum(0, xs - fit.breakpoint)
    ax.plot(xs, ys, color="crimson", lw=2.2, label="Segmented fit")
    ax.axvline(fit.breakpoint, color="crimson", ls="--", lw=1.6, label=f"Breakpoint = {fit.breakpoint:.2f} h")
    ax.axvspan(*fit.breakpoint_ci, color="crimson", alpha=0.12,
               label=f"95% CI [{fit.breakpoint_ci[0]:.2f}, {fit.breakpoint_ci[1]:.2f}] h")
    ax.set_xlabel("Average sequence length (hours)", fontsize=12.5, labelpad=8)
    ax.set_ylabel("CV-RMSE (%)", fontsize=12.5, labelpad=8)
    ax.grid(True, alpha=0.3, color="#d1d5db")
    ax.legend(frameon=True, fontsize=10.5, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"threshold_cvrmse_lstm_mv_h8.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
