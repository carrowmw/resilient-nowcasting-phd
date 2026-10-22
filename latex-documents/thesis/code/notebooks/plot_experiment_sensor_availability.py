#!/usr/bin/env python3
"""
scripts/plot_experiment_sensor_availability.py

Generates a three-panel publication-quality figure comparing sensor availability
across the three PhD thesis experiments:
  - Panel 1: Experiment 1 (Single-Node Baselines, N=110, with single-node interpolation)
  - Panel 2: Experiment 2 (Multi-Node Spatio-Temporal Graph, N=54)
  - Panel 3: Experiment 3 (Resilient Architecture / RIoT, N=110)

Uses weekly aggregation bins and GOV.UK color palette to highlight the data
advantage of the resilient architecture.
"""
import os
import json
import shutil
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates

sys.path.insert(0, "src")
from quality.quality_metrics import find_consecutive_sequences, assign_split_windows

# ---------------------------------------------------------
# 1. Colors & Style (GOV.UK palette)
# ---------------------------------------------------------
C_INACTIVE = "#ffffff"       # White for unobserved / offline
C_TRAIN_SOLID = "#d4351c"    # GOV.UK Red (Observed)
C_TRAIN_LIGHT = "#f8b8ae"    # Light Red (Interpolated)
C_VAL_SOLID = "#1d70b8"      # GOV.UK Blue (Observed)
C_VAL_LIGHT = "#b9d5ee"      # Light Blue (Interpolated)
C_TEST_SOLID = "#00703c"     # GOV.UK Green (Observed)
C_TEST_LIGHT = "#b7e3cc"     # Light Green (Interpolated)
C_EXCLUDED = "#d2d2d6"       # Medium Grey for excluded nodes in Exp 2
C_BORDER = "#b1b4b6"         # Border Grey
C_TEXT = "#0b0c0c"           # Text Black
C_SEC_TEXT = "#505a5f"       # Secondary Text Grey

# Split dates
TRAIN_END = pd.Timestamp("2025-05-29")
VAL_END = pd.Timestamp("2025-12-03")
TEST_END = pd.Timestamp("2026-06-08")

def main():
    print("Loading sensor manifests...")
    with open("artifacts/quality_full_v2/sensor_manifest.json") as f:
        qm = json.load(f)
    viable_110 = qm["viable_sensors"]

    print("Loading 81-location spatial audit report...")
    node_rep = pd.read_csv("artifacts/spatial_full/node_report.csv")
    inc_54 = list(node_rep[node_rep["included"] == True].sort_values("completeness", ascending=False)["node_id"])
    exc_27 = list(node_rep[node_rep["included"] == False].sort_values("completeness", ascending=False)["node_id"])
    all_81_sorted = inc_54 + exc_27
    print(f"Physical locations: {len(inc_54)} included + {len(exc_27)} excluded = {len(all_81_sorted)} total")

    print("Loading 54-node spatial context manifest...")
    with open("artifacts/spatial_full/context_manifest.json") as f:
        sm = json.load(f)
    spatial_54 = sm["node_ids"]

    # Load sensor lifespan bounds for interpolation masking (Exp 1)
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
    df["week"] = df["dt"].dt.normalize() - pd.to_timedelta(df["dt"].dt.weekday, unit="D")

    # Full weekly calendar grid (Mondays)
    start_mon = pd.Timestamp("2023-01-02")
    end_mon = pd.Timestamp("2026-06-08")
    all_mondays = pd.date_range(start_mon, end_mon, freq="7D")
    n_weeks = len(all_mondays)
    print(f"Calendar span: {start_mon.date()} to {end_mon.date()} ({n_weeks} weeks)")

    # ---------------------------------------------------------
    # 2. Build presence matrices
    # ---------------------------------------------------------
    print("Computing 110-stream weekly availability matrix (Exp 1)...")
    df_110 = df[df["sensor"].isin(viable_110)]
    grp_110 = df_110.groupby(["sensor", "week"]).size().unstack(fill_value=0)
    mat_110 = (grp_110 > 0).astype(int).reindex(index=viable_110, columns=all_mondays, fill_value=0)

    # Sort 110 sensors by total active weeks (descending) for a clean visual gradient
    sensor_active_counts = mat_110.sum(axis=1)
    sorted_110_idx = sensor_active_counts.sort_values(ascending=False).index
    mat_110_sorted = mat_110.loc[sorted_110_idx]

    print("Computing 81-location weekly availability matrix (Exp 2 & 3)...")
    df_81 = df[df["location"].isin(all_81_sorted)]
    grp_81 = df_81.groupby(["location", "week"]).size().unstack(fill_value=0)
    mat_81_sorted = (grp_81 > 0).astype(int).reindex(index=all_81_sorted, columns=all_mondays, fill_value=0)

    # ---------------------------------------------------------
    # 3. Colormap mapping:
    # 0 = Inactive / Unobserved (White)
    # 1 = Train (Observed) (Dark Red)
    # 2 = Train (Interpolated) (Light Red)
    # 3 = Val (Observed) (Dark Blue)
    # 4 = Val (Interpolated) (Light Blue)
    # 5 = Test (Observed) (Dark Green)
    # 6 = Test (Interpolated) (Light Green)
    # 7 = Excluded from Architecture (Medium Grey)
    # ---------------------------------------------------------
    cmap = ListedColormap([
        C_INACTIVE,      # 0
        C_TRAIN_SOLID,   # 1
        C_TRAIN_LIGHT,   # 2
        C_VAL_SOLID,     # 3
        C_VAL_LIGHT,     # 4
        C_TEST_SOLID,    # 5
        C_TEST_LIGHT,    # 6
        C_EXCLUDED,      # 7
    ])

    # ---------------------------------------------------------
    # 3. Build presence matrices
    # ---------------------------------------------------------
    print("Building Experiment 1: Un-imputed alternating-day micro-splits (N=110)...")
    all_days = pd.date_range(start_mon, end_mon, freq="1D")
    n_days = len(all_days)
    x_num_d = mdates.date2num([d.to_pydatetime() for d in all_days])
    x_start_d = x_num_d[0]
    x_end_d = x_num_d[-1]

    grid_exp1_uninterp = np.zeros((110, n_days), dtype=int)
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
                grid_exp1_uninterp[r_idx, d_idx] = 0
            else:
                best = max([("train", c_tr), ("val", c_va), ("test", c_te)], key=lambda x: x[1])[0]
                grid_exp1_uninterp[r_idx, d_idx] = 1 if best == "train" else (3 if best == "val" else 5)

    # Also build interpolated macro-split variant for comparison if needed
    grid_exp1_interp = np.zeros((110, n_weeks), dtype=int)
    for r_idx, s in enumerate(sorted_110_idx):
        f_obs = pd.Timestamp(first_last.loc[s, "first_obs"])
        l_obs = pd.Timestamp(first_last.loc[s, "last_obs"])
        dur = l_obs - f_obs
        t_val = f_obs + 0.70 * dur
        t_test = f_obs + 0.85 * dur
        for w_idx, monday in enumerate(all_mondays):
            if (monday + pd.Timedelta(days=7) < f_obs) or (monday > l_obs):
                grid_exp1_interp[r_idx, w_idx] = 0
                continue
            if monday < t_val:
                solid_c, light_c = 1, 2
            elif monday < t_test:
                solid_c, light_c = 3, 4
            else:
                solid_c, light_c = 5, 6
            is_active = (mat_110_sorted.iloc[r_idx, w_idx] == 1)
            grid_exp1_interp[r_idx, w_idx] = solid_c if is_active else light_c

    # Build Exp 2: 81 total locations (54 active under global calendar split, 27 excluded)
    grid_exp2 = np.zeros((81, n_weeks), dtype=int)
    for w_idx, monday in enumerate(all_mondays):
        if monday <= TRAIN_END:
            solid_c = 1
        elif monday <= VAL_END:
            solid_c = 3
        else:
            solid_c = 5

        for r_idx in range(54):
            if mat_81_sorted.iloc[r_idx, w_idx] == 1:
                grid_exp2[r_idx, w_idx] = solid_c
            else:
                grid_exp2[r_idx, w_idx] = 0
    grid_exp2[54:, :] = 7  # 27 excluded sparse locations

    # Build Exp 3: All 81 physical locations active under global calendar split (RIoT reclaims the 27 sparse locations)
    grid_exp3 = np.zeros((81, n_weeks), dtype=int)
    for w_idx, monday in enumerate(all_mondays):
        if monday <= TRAIN_END:
            solid_c = 1
        elif monday <= VAL_END:
            solid_c = 3
        else:
            solid_c = 5

        for r_idx in range(81):
            if mat_81_sorted.iloc[r_idx, w_idx] == 1:
                grid_exp3[r_idx, w_idx] = solid_c
            else:
                grid_exp3[r_idx, w_idx] = 0

    # ---------------------------------------------------------
    # 4. Helper to configure an axis
    # ---------------------------------------------------------
    x_num = mdates.date2num([m.to_pydatetime() for m in all_mondays])
    x_start = x_num[0]
    x_end = x_num[-1]

    train_x = mdates.date2num(TRAIN_END.to_pydatetime())
    val_x = mdates.date2num(VAL_END.to_pydatetime())

    def draw_panel(ax, grid, title, subtitle, show_global_splits, y_max, y_ticks, y_label, p_x_start=None, p_x_end=None):
        cur_x_start = p_x_start if p_x_start is not None else x_start
        cur_x_end = p_x_end if p_x_end is not None else x_end
        ax.imshow(
            grid,
            cmap=cmap,
            vmin=0,
            vmax=7,
            aspect="auto",
            interpolation="nearest",
            extent=[cur_x_start, cur_x_end, y_max, 0]  # row 0 at top, row y_max at bottom
        )

        if show_global_splits:
            ax.axvline(train_x, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.axvline(val_x, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=20, color=C_TEXT)
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=C_SEC_TEXT
        )

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.set_xlabel("Time (2023 – 2026)", fontsize=9.5, color=C_TEXT, labelpad=8)
        ax.set_xlim(x_start, x_end)
        ax.set_ylim(y_max, 0)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t) for t in y_ticks], fontsize=9)
        if y_label:
            ax.set_ylabel(y_label, fontsize=10, color=C_TEXT, labelpad=6)

        for spine in ax.spines.values():
            spine.set_edgecolor(C_BORDER)
            spine.set_linewidth(1.0)

    def add_excluded_annotation(ax):
        ax.axhline(54, color=C_TEXT, linestyle="-", linewidth=1.2, alpha=0.85)
        ax.text(
            x_start + (x_end - x_start) * 0.5, 67.5,
            "110 streams vs 54 station nodes;\n27 of 81 candidate stations excluded at < 30% completeness",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#2c2d30",
            bbox=dict(boxstyle="round,pad=0.40", facecolor="#ffffff", edgecolor=C_BORDER, alpha=0.95)
        )

    def add_reclaimed_annotation(ax):
        ax.axhline(54, color=C_TEST_SOLID, linestyle="--", linewidth=1.1, alpha=0.75)
        ax.text(
            x_start + (x_end - x_start) * 0.5, 67.5,
            "27 Sparse Locations Reclaimed by RIoT\n(100% of 81 physical locations active)",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_TEST_SOLID,
            bbox=dict(boxstyle="round,pad=0.40", facecolor="#ffffff", edgecolor=C_TEST_SOLID, alpha=0.95)
        )

    legend_patches_8 = [
        mpatches.Patch(facecolor=C_TRAIN_SOLID, edgecolor="none", label="Train (Observed)"),
        mpatches.Patch(facecolor=C_TRAIN_LIGHT, edgecolor="none", label="Train (Interpolated)"),
        mpatches.Patch(facecolor=C_VAL_SOLID, edgecolor="none", label="Val (Observed)"),
        mpatches.Patch(facecolor=C_VAL_LIGHT, edgecolor="none", label="Val (Interpolated)"),
        mpatches.Patch(facecolor=C_TEST_SOLID, edgecolor="none", label="Test (Observed)"),
        mpatches.Patch(facecolor=C_TEST_LIGHT, edgecolor="none", label="Test (Interpolated)"),
        mpatches.Patch(facecolor=C_INACTIVE, edgecolor=C_BORDER, label="Offline / Unobserved"),
        mpatches.Patch(facecolor=C_EXCLUDED, edgecolor=C_BORDER, label="Excluded from Graph"),
    ]

    legend_patches_5 = [
        mpatches.Patch(facecolor=C_TRAIN_SOLID, edgecolor="none", label="Train (Observed)"),
        mpatches.Patch(facecolor=C_VAL_SOLID, edgecolor="none", label="Val (Observed)"),
        mpatches.Patch(facecolor=C_TEST_SOLID, edgecolor="none", label="Test (Observed)"),
        mpatches.Patch(facecolor=C_INACTIVE, edgecolor=C_BORDER, label="Offline / Unobserved"),
        mpatches.Patch(facecolor=C_EXCLUDED, edgecolor=C_BORDER, label="Excluded from Graph"),
    ]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # =========================================================================
    # FIGURE 1: 3-Panel Comparison for Chapter 4 (Methodology)
    # =========================================================================
    print("Rendering 3-Panel Figure (Methodology: Exp 1, 2, 3)...")
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 6.8))
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.17, top=0.88, wspace=0.16)

    draw_panel(axes1[0], grid_exp1_uninterp, "Experiment 1: Single-Node Baselines", "N = 110 Directional Streams (Un-imputed Alternating-Day Splits)", False, 110, [1, 25, 55, 80, 110], "Directional Stream Index (N = 110)", p_x_start=x_start_d, p_x_end=x_end_d)
    draw_panel(axes1[1], grid_exp2, "Experiment 2: Multi-Node Spatio-Temporal Graph", "N = 54 Connected Graph Nodes (Global Calendar Split)", True, 81, [1, 20, 40, 54, 81], "Physical Station Index (N = 81)")
    draw_panel(axes1[2], grid_exp3, "Experiment 3: Resilient Architecture (RIoT)", "N = 81 Resilient Graph Nodes (Global Calendar Split)", True, 81, [1, 20, 40, 54, 81], "")

    add_excluded_annotation(axes1[1])
    add_reclaimed_annotation(axes1[2])

    fig1.legend(
        handles=legend_patches_5,
        loc="lower center",
        ncol=5,
        columnspacing=2.0,
        frameon=True,
        facecolor="#ffffff",
        edgecolor=C_BORDER,
        fontsize=8.8,
        bbox_to_anchor=(0.5, 0.01)
    )

    out_dir_ws = "artifacts/figures"
    os.makedirs(out_dir_ws, exist_ok=True)
    f1_png = os.path.join(out_dir_ws, "experiment_sensor_availability_comparison.png")
    f1_pdf = os.path.join(out_dir_ws, "experiment_sensor_availability_comparison.pdf")
    fig1.savefig(f1_png, dpi=300)
    fig1.savefig(f1_pdf)
    plt.close(fig1)
    print(f"Saved: {f1_png}")

    # =========================================================================
    # FIGURE 2: 2-Panel Comparison for Chapter 6 (Exp 1 vs Exp 2)
    # =========================================================================
    print("Rendering 2-Panel Figure (Chapter 6: Exp 1 vs Exp 2)...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.0, 6.8))
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.88, wspace=0.18)

    draw_panel(axes2[0], grid_exp1_uninterp, "Experiment 1: Single-Node Baselines", "N = 110 Directional Streams (Un-imputed Alternating-Day Splits)", False, 110, [1, 25, 55, 80, 110], "Directional Stream Index (N = 110)", p_x_start=x_start_d, p_x_end=x_end_d)
    draw_panel(axes2[1], grid_exp2, "Experiment 2: Multi-Node Spatio-Temporal Graph", "N = 54 Connected Graph Nodes (Global Calendar Split)", True, 81, [1, 20, 40, 54, 81], "Physical Station Index (N = 81)")

    add_excluded_annotation(axes2[1])

    fig2.legend(
        handles=legend_patches_5,
        loc="lower center",
        ncol=5,
        columnspacing=2.0,
        frameon=True,
        facecolor="#ffffff",
        edgecolor=C_BORDER,
        fontsize=8.8,
        bbox_to_anchor=(0.5, 0.01)
    )

    f2_png = os.path.join(out_dir_ws, "sensor_availability_exp1_vs_exp2.png")
    f2_pdf = os.path.join(out_dir_ws, "sensor_availability_exp1_vs_exp2.pdf")
    fig2.savefig(f2_png, dpi=300)
    fig2.savefig(f2_pdf)
    plt.close(fig2)
    print(f"Saved: {f2_png}")


    # =========================================================================
    # FIGURE 3: 2-Panel Comparison for Chapter 7 / Experiment 3 (Exp 2 vs Exp 3)
    # =========================================================================
    print("Rendering 2-Panel Figure (Experiment 3: Exp 2 vs Exp 3)...")
    fig3, axes3 = plt.subplots(1, 2, figsize=(12.0, 6.8), sharey=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.88, wspace=0.14)

    draw_panel(axes3[0], grid_exp2, "Experiment 2: Multi-Node Spatio-Temporal Graph", "N = 54 Connected Graph Nodes (Global Calendar Split)", True, 81, [1, 20, 40, 54, 81], "Physical Station Index (N = 81)")
    draw_panel(axes3[1], grid_exp3, "Experiment 3: Resilient Architecture (RIoT)", "N = 81 Resilient Graph Nodes (Global Calendar Split)", True, 81, [1, 20, 40, 54, 81], "")

    add_excluded_annotation(axes3[0])
    add_reclaimed_annotation(axes3[1])

    fig3.legend(
        handles=legend_patches_5,
        loc="lower center",
        ncol=5,
        columnspacing=2.0,
        frameon=True,
        facecolor="#ffffff",
        edgecolor=C_BORDER,
        fontsize=9.0,
        bbox_to_anchor=(0.5, 0.02)
    )

    f3_png = os.path.join(out_dir_ws, "sensor_availability_exp2_vs_exp3.png")
    f3_pdf = os.path.join(out_dir_ws, "sensor_availability_exp2_vs_exp3.pdf")
    fig3.savefig(f3_png, dpi=300)
    fig3.savefig(f3_pdf)
    plt.close(fig3)
    print(f"Saved: {f3_png}")

    # =========================================================================
    # 5. Sync all figures to Thesis directories
    # =========================================================================
    thesis_sync_targets = {
        # Methodology gets the 3-panel overview as well as sub-comparisons
        "latex-documents/thesis/figures/methodology": [
            ("experiment_sensor_availability_comparison.png", f1_png),
            ("experiment_sensor_availability_comparison.pdf", f1_pdf),
            ("sensor_availability_exp1_vs_exp2.png", f2_png),
            ("sensor_availability_exp1_vs_exp2.pdf", f2_pdf),
            ("sensor_availability_exp2_vs_exp3.png", f3_png),
            ("sensor_availability_exp2_vs_exp3.pdf", f3_pdf),
        ],
        # Chapter 6 gets Exp 1 vs Exp 2 side-by-side
        "latex-documents/thesis/figures/experiment2": [
            ("sensor_availability_exp1_vs_exp2.png", f2_png),
            ("sensor_availability_exp1_vs_exp2.pdf", f2_pdf),
            ("experiment_sensor_availability_comparison.png", f1_png),
            ("experiment_sensor_availability_comparison.pdf", f1_pdf),
        ],
        # Chapter 7 gets Exp 2 vs Exp 3 side-by-side
        "latex-documents/thesis/figures/experiment3": [
            ("sensor_availability_exp2_vs_exp3.png", f3_png),
            ("sensor_availability_exp2_vs_exp3.pdf", f3_pdf),
            ("experiment_sensor_availability_comparison.png", f1_png),
            ("experiment_sensor_availability_comparison.pdf", f1_pdf),
        ],
    }

    print("\nSyncing figures to LaTeX thesis directories...")
    for target_dir, file_pairs in thesis_sync_targets.items():
        os.makedirs(target_dir, exist_ok=True)
        for dest_name, src_path in file_pairs:
            dest_path = os.path.join(target_dir, dest_name)
            shutil.copy2(src_path, dest_path)
            print(f"  Copied -> {dest_path}")

    # Copy script for thesis reproducibility
    code_dir = "latex-documents/thesis/code/notebooks"
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy2(__file__, os.path.join(code_dir, "plot_experiment_sensor_availability.py"))
    print(f"  Copied script -> {code_dir}/plot_experiment_sensor_availability.py")

    print("\nAll figures generated and synced successfully!")

if __name__ == "__main__":
    main()
