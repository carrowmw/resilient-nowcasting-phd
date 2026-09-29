#!/usr/bin/env python3
"""
scripts/plot_experiment1_split_comparison.py

Generates a publication-quality comparative figure specifically for Experiment 1
illustrating the profound impact of interpolation on sequence splitting:
  - Panel A: Experiment 1 with Single-Node Imputation (Continuous Macro-Splits: 70/15/15%)
  - Panel B: Experiment 1 without Imputation (Fragmented Micro-Splits: ~5,088 interleaved runs/sensor)
  - Panel C: 90-Day Zoomed Callout (Daily assignment to Train, Val, Test based on running deficit)

Saves both 300 DPI PNG and vector PDF to artifacts and thesis figure directories.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates

import sys
sys.path.insert(0, "src")

from quality.quality_metrics import find_consecutive_sequences, assign_split_windows

# ---------------------------------------------------------
# 1. Palette & Styles (GOV.UK design tokens)
# ---------------------------------------------------------
C_INACTIVE = "#ffffff"       # White for unobserved / offline / gaps
C_TRAIN_SOLID = "#d4351c"    # GOV.UK Red (Observed)
C_TRAIN_LIGHT = "#f8b8ae"    # Light Red (Interpolated Gaps)
C_VAL_SOLID = "#1d70b8"      # GOV.UK Blue (Observed)
C_VAL_LIGHT = "#b9d5ee"      # Light Blue (Interpolated Gaps)
C_TEST_SOLID = "#00703c"     # GOV.UK Green (Observed)
C_TEST_LIGHT = "#b7e3cc"     # Light Green (Interpolated Gaps)
C_BORDER = "#b1b4b6"         # Border Grey
C_TEXT = "#0b0c0c"           # Text Black
C_SEC_TEXT = "#505a5f"       # Secondary Text Grey

def main():
    print("Loading sensor manifests...")
    with open("artifacts/quality_full_v2/sensor_manifest.json") as f:
        qm = json.load(f)
    viable_110 = qm["viable_sensors"]

    df_qual = pd.read_parquet("artifacts/quality_full_v2/quality_table_none.parquet")
    first_last = df_qual.loc[viable_110, ["first_obs", "last_obs"]]

    print("Loading master dataset...")
    table = pq.read_table(
        "data/processed/master_dataset_15T.parquet",
        columns=["dt", "location", "dir", "value"],
        filters=[("veh_class", "=", "person"), ("category", "=", "flow")]
    )
    df = table.to_pandas()
    df = df[df["value"].notna() & (df["value"] > 0)]
    df["sensor"] = df["location"] + "__" + df["dir"]
    df = df[df["sensor"].isin(viable_110)]

    # Time bounds
    start_mon = pd.Timestamp("2023-01-02")
    end_mon = pd.Timestamp("2026-06-08")
    all_mondays = pd.date_range(start_mon, end_mon, freq="7D")
    n_weeks = len(all_mondays)

    # Sort sensors by active weeks for clean visual hierarchy
    df["week"] = df["dt"].dt.normalize() - pd.to_timedelta(df["dt"].dt.weekday, unit="D")
    grp_w = df.groupby(["sensor", "week"]).size().unstack(fill_value=0)
    mat_w = (grp_w > 0).astype(int).reindex(index=viable_110, columns=all_mondays, fill_value=0)
    sorted_110_idx = mat_w.sum(axis=1).sort_values(ascending=False).index
    mat_110_sorted = mat_w.loc[sorted_110_idx]

    print("Building Panel A (Interpolated Macro-Splits)...")
    grid_interp = np.zeros((110, n_weeks), dtype=int)
    for r_idx, s in enumerate(sorted_110_idx):
        f_obs = pd.Timestamp(first_last.loc[s, "first_obs"])
        l_obs = pd.Timestamp(first_last.loc[s, "last_obs"])
        dur = l_obs - f_obs
        t_val = f_obs + 0.70 * dur
        t_test = f_obs + 0.85 * dur

        for w_idx, monday in enumerate(all_mondays):
            if (monday + pd.Timedelta(days=7) < f_obs) or (monday > l_obs):
                grid_interp[r_idx, w_idx] = 0
                continue
            if monday < t_val:
                solid_c, light_c = 1, 2
            elif monday < t_test:
                solid_c, light_c = 3, 4
            else:
                solid_c, light_c = 5, 6

            is_active = (mat_110_sorted.iloc[r_idx, w_idx] == 1)
            grid_interp[r_idx, w_idx] = solid_c if is_active else light_c

    print("Building Panel B (Uninterpolated Micro-Splits at Daily Resolution)...")
    all_days = pd.date_range(start_mon, end_mon, freq="1D")
    n_days = len(all_days)
    grid_uninterp = np.zeros((110, n_days), dtype=int)

    zoom_sensor = sorted_110_idx[0]
    zoom_days_data = []

    for r_idx, s in enumerate(sorted_110_idx):
        s_df = df[df["sensor"] == s].set_index("dt")["value"].sort_index()
        if s_df.empty:
            continue
        s_series = s_df.reindex(pd.date_range(s_df.index.min(), s_df.index.max(), freq="15min"))
        spans = find_consecutive_sequences(s_series, min_length=20)
        split_idx = assign_split_windows(spans, window=12, horizon=8)

        day_counts = {"train": {}, "val": {}, "test": {}}
        for sp_name in ["train", "val", "test"]:
            starts = split_idx[sp_name][0]
            if len(starts) > 0:
                times = s_series.index[starts]
                days = times.normalize()
                day_counts[sp_name] = days.value_counts().to_dict()

        for d_idx, day in enumerate(all_days):
            c_tr = day_counts["train"].get(day, 0)
            c_va = day_counts["val"].get(day, 0)
            c_te = day_counts["test"].get(day, 0)
            tot = c_tr + c_va + c_te
            if tot == 0:
                grid_uninterp[r_idx, d_idx] = 0
            else:
                best = max([("train", c_tr), ("val", c_va), ("test", c_te)], key=lambda x: x[1])[0]
                if best == "train":
                    grid_uninterp[r_idx, d_idx] = 1
                elif best == "val":
                    grid_uninterp[r_idx, d_idx] = 3
                else:
                    grid_uninterp[r_idx, d_idx] = 5

        if s == zoom_sensor:
            zoom_days_data = [
                (day, day_counts["train"].get(day, 0), day_counts["val"].get(day, 0), day_counts["test"].get(day, 0))
                for day in all_days
            ]

    # Colormap: 0=Inactive, 1=Train Solid, 2=Train Light, 3=Val Solid, 4=Val Light, 5=Test Solid, 6=Test Light
    cmap = ListedColormap([
        C_INACTIVE,     # 0
        C_TRAIN_SOLID,  # 1
        C_TRAIN_LIGHT,  # 2
        C_VAL_SOLID,    # 3
        C_VAL_LIGHT,    # 4
        C_TEST_SOLID,   # 5
        C_TEST_LIGHT,   # 6
    ])

    print("Rendering figure...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.8, 1.0], hspace=0.32, wspace=0.10,
        left=0.07, right=0.98, top=0.92, bottom=0.12
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[1, :])

    x_num_w = mdates.date2num([m.to_pydatetime() for m in all_mondays])
    x_num_d = mdates.date2num([d.to_pydatetime() for d in all_days])

    # Panel A: Interpolated Macro-Splits
    ax0.imshow(
        grid_interp, cmap=cmap, vmin=0, vmax=6, aspect="auto", interpolation="nearest",
        extent=[x_num_w[0], x_num_w[-1], 110, 0]
    )
    ax0.set_title("Panel A: Single-Node Baselines with Imputation (KNN / Linear / Spline)", fontsize=11, fontweight="bold", pad=18, color=C_TEXT)
    ax0.text(0.5, 1.02, "Continuous Macro-Splits: Contiguous Train (70%) → Val (15%) → Test (15%) per Sensor", transform=ax0.transAxes, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=C_SEC_TEXT)

    # Panel B: Uninterpolated Micro-Splits
    ax1.imshow(
        grid_uninterp, cmap=cmap, vmin=0, vmax=6, aspect="auto", interpolation="nearest",
        extent=[x_num_d[0], x_num_d[-1], 110, 0]
    )
    ax1.set_title("Panel B: Single-Node Baselines without Imputation (Raw Uninterpolated Observations)", fontsize=11, fontweight="bold", pad=18, color=C_TEXT)
    ax1.text(0.5, 1.02, "Fragmented Micro-Splits: ~5,088 Interleaved Spans per Sensor with Boundary Purging", transform=ax1.transAxes, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=C_SEC_TEXT)

    for ax in [ax0, ax1]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.set_xlabel("Time (2023 – 2026)", fontsize=9.5, color=C_TEXT, labelpad=6)
        ax.set_xlim(x_num_w[0], x_num_w[-1])
        ax.set_yticks([1, 25, 54, 75, 110])
        ax.set_yticklabels(["1", "25", "54", "75", "110"], fontsize=9)
        for s in ax.spines.values():
            s.set_edgecolor(C_BORDER)

    ax0.set_ylabel("Sensor Node Index (Sorted by Completeness)", fontsize=10, color=C_TEXT, labelpad=6)

    # Panel C: 90-day zoom for top sensor (Feb 15 to May 15, 2024)
    zoom_start = pd.Timestamp("2024-02-15")
    zoom_end = pd.Timestamp("2024-05-15")
    z_days = [d for d, tr, va, te in zoom_days_data if zoom_start <= d <= zoom_end]
    z_tr = [tr for d, tr, va, te in zoom_days_data if zoom_start <= d <= zoom_end]
    z_va = [va for d, tr, va, te in zoom_days_data if zoom_start <= d <= zoom_end]
    z_te = [te for d, tr, va, te in zoom_days_data if zoom_start <= d <= zoom_end]

    ax2.bar(z_days, z_tr, width=0.85, color=C_TRAIN_SOLID, label="Train Windows")
    ax2.bar(z_days, z_va, bottom=z_tr, width=0.85, color=C_VAL_SOLID, label="Val Windows")
    bottom_te = np.array(z_tr) + np.array(z_va)
    ax2.bar(z_days, z_te, bottom=bottom_te, width=0.85, color=C_TEST_SOLID, label="Test Windows")

    ax2.set_title("Panel C: Micro-Level Split Assignment (90-Day Zoomed Detail for Sensor AdelaideTerBeechSt, Feb – May 2024)", fontsize=10.5, fontweight="bold", pad=14, color=C_TEXT)
    ax2.text(0.5, 1.02, "Shows individual continuous daytime observation runs assigned deterministically to Train, Val, or Test based on running deficit", transform=ax2.transAxes, ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_SEC_TEXT)
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    ax2.xaxis.set_minor_locator(mdates.DayLocator())
    ax2.set_xlim(zoom_start - pd.Timedelta(days=1), zoom_end + pd.Timedelta(days=1))
    ax2.set_ylabel("Number of Sliding Windows (h=8)", fontsize=9.5, color=C_TEXT, labelpad=6)
    ax2.set_xlabel("Calendar Date (Daily Resolution)", fontsize=9.5, color=C_TEXT, labelpad=6)
    for s in ax2.spines.values():
        s.set_edgecolor(C_BORDER)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Clean grouped legend
    legend_patches = [
        mpatches.Patch(facecolor=C_TRAIN_SOLID, edgecolor="none", label="Train (Observed)"),
        mpatches.Patch(facecolor=C_TRAIN_LIGHT, edgecolor="none", label="Train (Interpolated Gaps)"),
        mpatches.Patch(facecolor=C_VAL_SOLID, edgecolor="none", label="Val (Observed)"),
        mpatches.Patch(facecolor=C_VAL_LIGHT, edgecolor="none", label="Val (Interpolated Gaps)"),
        mpatches.Patch(facecolor=C_TEST_SOLID, edgecolor="none", label="Test (Observed)"),
        mpatches.Patch(facecolor=C_TEST_LIGHT, edgecolor="none", label="Test (Interpolated Gaps)"),
        mpatches.Patch(facecolor=C_INACTIVE, edgecolor=C_BORDER, label="Offline / Missing Gaps (Uninterpolated)"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=4, columnspacing=2.0,
        frameon=True, facecolor="#ffffff", edgecolor=C_BORDER, fontsize=8.8, bbox_to_anchor=(0.5, 0.01)
    )

    out_dir_ws = "artifacts/figures"
    os.makedirs(out_dir_ws, exist_ok=True)
    out_png_ws = os.path.join(out_dir_ws, "experiment1_split_comparison.png")
    out_pdf_ws = os.path.join(out_dir_ws, "experiment1_split_comparison.pdf")

    print(f"Saving to workspace artifacts: {out_png_ws} and {out_pdf_ws}...")
    plt.savefig(out_png_ws, dpi=300)
    plt.savefig(out_pdf_ws)
    print("Saved successfully.")

if __name__ == "__main__":
    main()
