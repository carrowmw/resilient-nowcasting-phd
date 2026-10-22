#!/usr/bin/env python3
"""
scripts/experiment1/plot_ch5_audit_temporal_scale.py

Regenerates Chapter 5 figures:
- 5.10: exp_h_seasonal_error_side_by_side.png (clean legend title)
- 5.11: exp_i_case_study_NclShieldsRdRobinsonStNorth__west_to_east.png (clean title, matched dates)
- 5.12: exp_i_case_study_NclHeberSt__east_to_west.png (clean title, matched dates, clean y-axis)
- 5.14: exp_k_extreme_events_side_by_side.png (2x2 with RMSE and CV-RMSE)
- 5.15: exp_l_matched_pairs_comparison.png (clean units [pp], 3.7x coverage, clean Panel D subplots)

Saves to artifacts/quality_full_v2/analysis/temporal_scale/ and copies to thesis directory.
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def get_season_uk(dt_series):
    m = dt_series.dt.month
    conditions = [
        m.isin([12, 1, 2]),
        m.isin([3, 4, 5]),
        m.isin([6, 7, 8]),
        m.isin([9, 10, 11])
    ]
    choices = ["Winter", "Spring", "Summer", "Autumn"]
    return np.select(conditions, choices, default="Unknown")

def main():
    out_dir = Path("artifacts/quality_full_v2/analysis/temporal_scale")
    out_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir = Path("/Users/administrator/Code/latex-documents/thesis/figures/experiment1")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12

    print("Loading predictions...")
    p_none = pd.read_parquet("artifacts/quality_full_v2/runs/lstm_mv_h8_none/predictions.parquet")
    p_knn = pd.read_parquet("artifacts/quality_full_v2/runs/lstm_mv_h8_knn/predictions.parquet")

    p_none["target_dt"] = pd.to_datetime(p_none["target_dt"])
    p_knn["target_dt"] = pd.to_datetime(p_knn["target_dt"])

    # =========================================================================
    # 1. Figure 5.10: exp_h_seasonal_error_side_by_side
    # =========================================================================
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    rows = []
    for name, preds in [("None (Un-imputed)", p_none), ("KNN Imputation", p_knn)]:
        p_copy = preds.copy()
        p_copy["season"] = get_season_uk(p_copy["target_dt"])
        for season, g in p_copy.groupby("season"):
            y = g["y_true_raw"].to_numpy(dtype=float)
            p = g["y_pred_raw"].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((p - y) ** 2)))
            mean_cnt = float(np.mean(y))
            cv_rmse = float(rmse / max(mean_cnt, 1e-8) * 100.0)
            rows.append({
                "Imputation Strategy": name,
                "season": season,
                "rmse_raw": rmse,
                "cv_rmse_raw": cv_rmse
            })

    agg = pd.DataFrame(rows)
    agg["season"] = pd.Categorical(agg["season"], categories=season_order, ordered=True)
    agg = agg.sort_values("season")

    palette = {"None (Un-imputed)": "#66c2a5", "KNN Imputation": "#fc8d62"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), dpi=300)
    sns.barplot(data=agg, x="season", y="rmse_raw", hue="Imputation Strategy", ax=axes[0], palette=palette)
    axes[0].set_title("RMSE by Season (None vs KNN)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_ylabel("RMSE (Counts)", fontsize=12)
    axes[0].set_xlabel("Season (UK)", fontsize=12)
    axes[0].legend(title="Imputation Strategy", frameon=True, fontsize=10.5)
    axes[0].grid(True, linestyle="-", alpha=0.3, color="#d1d5db", axis="y")

    sns.barplot(data=agg, x="season", y="cv_rmse_raw", hue="Imputation Strategy", ax=axes[1], palette=palette)
    axes[1].set_title("CV-RMSE by Season (None vs KNN)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_ylabel("CV-RMSE (%)", fontsize=12)
    axes[1].set_xlabel("Season (UK)", fontsize=12)
    axes[1].legend(title="Imputation Strategy", frameon=True, fontsize=10.5)
    axes[1].grid(True, linestyle="-", alpha=0.3, color="#d1d5db", axis="y")

    fig.tight_layout()
    fig5_10_png = out_dir / "exp_h_seasonal_error_side_by_side.png"
    fig5_10_pdf = out_dir / "exp_h_seasonal_error_side_by_side.pdf"
    fig.savefig(fig5_10_png, bbox_inches="tight")
    fig.savefig(fig5_10_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.10 (exp_h_seasonal_error_side_by_side)")

    # =========================================================================
    # 2. Figure 5.11 & 5.12: Case Studies (Matched Dates)
    # =========================================================================
    # Build matched pairs set
    merged = pd.merge(
        p_none[["sensor_id", "target_dt", "y_true_raw", "y_pred_raw"]],
        p_knn[["sensor_id", "target_dt", "y_true_raw", "y_pred_raw"]],
        on=["sensor_id", "target_dt"],
        suffixes=("_none", "_knn")
    )

    # 5.11: Shields Road (Benchmark) - 3 matched test days showing daily 16:00 commute surge
    shields_id = "NclShieldsRdRobinsonStNorth__west_to_east"
    s_days = ["2026-04-17", "2026-04-24", "2026-05-01"]
    s_day_labels = ["Fri 17 Apr 2026", "Fri 24 Apr 2026", "Fri 01 May 2026"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.8), sharey=True, dpi=300)

    for col_idx, (d_str, d_lbl) in enumerate(zip(s_days, s_day_labels)):
        sub = merged[(merged["sensor_id"] == shields_id) & (merged["target_dt"].dt.date == pd.to_datetime(d_str).date())].sort_values("target_dt")
        times = [t.strftime("%H:%M") for t in sub["target_dt"]]

        # Row 0: None
        ax0 = axes[0, col_idx]
        ax0.plot(times, sub["y_true_raw_none"], "o-", color="#222222", markersize=3.5, label="Actuals", linewidth=1.2, alpha=0.75)
        ax0.plot(times, sub["y_pred_raw_none"], ".--", color="#d62728", markersize=3.5, label="Predicted (LSTM-MV)", linewidth=1.4)
        ax0.set_title(f"{d_lbl}\n[None: Un-imputed Baseline]", fontsize=11, fontweight="bold", pad=6)
        ax0.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")
        ax0.set_xticks(range(0, len(times), 8))
        ax0.tick_params(axis="x", rotation=30)
        if col_idx == 0:
            ax0.set_ylabel("Pedestrian Count", fontsize=11.5)
            ax0.legend(loc="upper right", frameon=True, fontsize=9.5)

        # Row 1: KNN
        ax1 = axes[1, col_idx]
        ax1.plot(times, sub["y_true_raw_knn"], "o-", color="#222222", markersize=3.5, label="Actuals", linewidth=1.2, alpha=0.75)
        ax1.plot(times, sub["y_pred_raw_knn"], ".--", color="#1f77b4", markersize=3.5, label="Predicted (KNN Imputed)", linewidth=1.4)
        ax1.set_title(f"{d_lbl}\n[KNN Imputation]", fontsize=11, fontweight="bold", pad=6)
        ax1.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")
        ax1.set_xticks(range(0, len(times), 8))
        ax1.tick_params(axis="x", rotation=30)
        ax1.set_xlabel("Time of Day (UTC)", fontsize=11)
        if col_idx == 0:
            ax1.set_ylabel("Pedestrian Count", fontsize=11.5)
            ax1.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.suptitle("Benchmark Case Study: Shields Road (North, West-to-East) — Matched Test Days", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig5_11_png = out_dir / "exp_i_case_study_NclShieldsRdRobinsonStNorth__west_to_east.png"
    fig5_11_pdf = out_dir / "exp_i_case_study_NclShieldsRdRobinsonStNorth__west_to_east.pdf"
    fig.savefig(fig5_11_png, bbox_inches="tight")
    fig.savefig(fig5_11_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.11 (Shields Road Case Study: 3 Matched Days)")

    # 5.12: Heber Street (Stadium Crowd Surge) - 3 matched test days (Routine vs Surge)
    heber_id = "NclHeberSt__east_to_west"
    h_days = ["2026-04-14", "2026-04-21", "2026-04-28"]
    h_day_labels = [
        "Tue 14 Apr 2026\n[Baseline Routine Day]",
        "Tue 21 Apr 2026\n[Matchday Surge: Peak 150]",
        "Tue 28 Apr 2026\n[Matchday Surge: Peak 206]"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.8), sharey=True, dpi=300)

    for col_idx, (d_str, d_lbl) in enumerate(zip(h_days, h_day_labels)):
        sub = merged[(merged["sensor_id"] == heber_id) & (merged["target_dt"].dt.date == pd.to_datetime(d_str).date())].sort_values("target_dt")
        times = [t.strftime("%H:%M") for t in sub["target_dt"]]

        # Row 0: None
        ax0 = axes[0, col_idx]
        ax0.plot(times, sub["y_true_raw_none"], "o-", color="#222222", markersize=3.5, label="Actuals", linewidth=1.2, alpha=0.75)
        ax0.plot(times, sub["y_pred_raw_none"], ".--", color="#d62728", markersize=3.5, label="Predicted (LSTM-MV)", linewidth=1.4)
        ax0.set_title(f"{d_lbl}\n[None: Un-imputed Baseline]", fontsize=10.5, fontweight="bold", pad=6)
        ax0.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")
        ax0.set_xticks(range(0, len(times), 8))
        ax0.tick_params(axis="x", rotation=30)
        ax0.set_ylim(-5, 230)
        if col_idx == 0:
            ax0.set_ylabel("Pedestrian Count", fontsize=11.5)
            ax0.legend(loc="upper right", frameon=True, fontsize=9.5)

        # Row 1: KNN
        ax1 = axes[1, col_idx]
        ax1.plot(times, sub["y_true_raw_knn"], "o-", color="#222222", markersize=3.5, label="Actuals", linewidth=1.2, alpha=0.75)
        ax1.plot(times, sub["y_pred_raw_knn"], ".--", color="#1f77b4", markersize=3.5, label="Predicted (KNN Imputed)", linewidth=1.4)
        ax1.set_title(f"{d_lbl}\n[KNN Imputation]", fontsize=10.5, fontweight="bold", pad=6)
        ax1.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")
        ax1.set_xticks(range(0, len(times), 8))
        ax1.tick_params(axis="x", rotation=30)
        ax1.set_xlabel("Time of Day (UTC)", fontsize=11)
        ax1.set_ylim(-5, 230)
        if col_idx == 0:
            ax1.set_ylabel("Pedestrian Count", fontsize=11.5)
            ax1.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.suptitle("Boundary Case Study: Heber Street (East-to-West) — Stadium Matchday Surges", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig5_12_png = out_dir / "exp_i_case_study_NclHeberSt__east_to_west.png"
    fig5_12_pdf = out_dir / "exp_i_case_study_NclHeberSt__east_to_west.pdf"
    fig.savefig(fig5_12_png, bbox_inches="tight")
    fig.savefig(fig5_12_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.12 (Heber Street Case Study: 3 Matched Days)")

    # =========================================================================
    # 3. Figure 5.14: exp_k_extreme_events_side_by_side (2x2 Grid with CV-RMSE)
    # =========================================================================
    extreme_records = []
    for interp_name, preds_df in [("None (Un-imputed)", p_none), ("KNN Imputation", p_knn)]:
        for sensor_id, g in preds_df.groupby("sensor_id"):
            mu = g["y_true_raw"].mean()
            sigma = g["y_true_raw"].std()
            if pd.isna(sigma) or sigma < 1.0:
                continue
            thresh = mu + 3 * sigma
            is_ext = g["y_true_raw"] > thresh
            is_norm = (g["y_true_raw"] <= mu + sigma) & (g["y_true_raw"] >= mu - sigma)
            if not is_ext.any() or not is_norm.any():
                continue

            y_ext = g.loc[is_ext, "y_true_raw"].to_numpy(dtype=float)
            p_ext = g.loc[is_ext, "y_pred_raw"].to_numpy(dtype=float)
            rmse_ext = float(np.sqrt(np.mean((p_ext - y_ext)**2)))
            cv_ext = float(rmse_ext / max(np.mean(y_ext), 1e-8) * 100.0)

            y_norm = g.loc[is_norm, "y_true_raw"].to_numpy(dtype=float)
            p_norm = g.loc[is_norm, "y_pred_raw"].to_numpy(dtype=float)
            rmse_norm = float(np.sqrt(np.mean((p_norm - y_norm)**2)))
            cv_norm = float(rmse_norm / max(np.mean(y_norm), 1e-8) * 100.0)

            extreme_records.append({
                "sensor_id": sensor_id,
                "imputation": interp_name,
                "Period": "Normal (±1σ)",
                "RMSE": rmse_norm,
                "CV-RMSE": cv_norm
            })
            extreme_records.append({
                "sensor_id": sensor_id,
                "imputation": interp_name,
                "Period": "Extreme (>3σ)",
                "RMSE": rmse_ext,
                "CV-RMSE": cv_ext
            })

    df_ext = pd.DataFrame(extreme_records)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # Row 1: RMSE (Absolute Error)
    for col_idx, (interp_name, title_suffix) in enumerate([("None (Un-imputed)", "None (Un-imputed)"), ("KNN Imputation", "KNN Imputation")]):
        ax = axes[0, col_idx]
        sub_d = df_ext[df_ext["imputation"] == interp_name]
        sns.boxplot(data=sub_d, x="Period", y="RMSE", ax=ax, palette=["#66c2a5", "#fc8d62"], width=0.45)
        ax.set_title(f"Absolute RMSE: Normal vs Extreme\n[{title_suffix}]", fontsize=12.5, fontweight="bold", pad=8)
        ax.set_ylabel("RMSE (Counts)", fontsize=11.5)
        ax.set_xlabel("")
        ax.grid(True, linestyle="-", alpha=0.3, color="#d1d5db", axis="y")

    # Row 2: CV-RMSE (Relative Error)
    for col_idx, (interp_name, title_suffix) in enumerate([("None (Un-imputed)", "None (Un-imputed)"), ("KNN Imputation", "KNN Imputation")]):
        ax = axes[1, col_idx]
        sub_d = df_ext[df_ext["imputation"] == interp_name]
        sns.boxplot(data=sub_d, x="Period", y="CV-RMSE", ax=ax, palette=["#66c2a5", "#fc8d62"], width=0.45)
        ax.set_title(f"Relative CV-RMSE: Normal vs Extreme\n[{title_suffix}]", fontsize=12.5, fontweight="bold", pad=8)
        ax.set_ylabel("CV-RMSE (%)", fontsize=11.5)
        ax.set_xlabel("Operating Period", fontsize=11.5)
        ax.grid(True, linestyle="-", alpha=0.3, color="#d1d5db", axis="y")

    fig.tight_layout()
    fig5_14_png = out_dir / "exp_k_extreme_events_side_by_side.png"
    fig5_14_pdf = out_dir / "exp_k_extreme_events_side_by_side.pdf"
    fig.savefig(fig5_14_png, bbox_inches="tight")
    fig.savefig(fig5_14_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.14 (Extreme Events 2x2 with CV-RMSE)")

    # =========================================================================
    # 4. Figure 5.15: exp_l_matched_pairs_comparison (Clean Units & Split Panel D)
    # =========================================================================
    def sensor_metrics(g):
        r_n = np.sqrt(np.mean((g["y_pred_raw_none"] - g["y_true_raw_none"])**2))
        r_k = np.sqrt(np.mean((g["y_pred_raw_knn"] - g["y_true_raw_knn"])**2))
        mu = g["y_true_raw_none"].mean()
        cv_n = (r_n / max(mu, 1e-8)) * 100.0
        cv_k = (r_k / max(mu, 1e-8)) * 100.0
        return pd.Series({
            "n_matched": len(g),
            "mean_count": mu,
            "rmse_none": r_n,
            "rmse_knn": r_k,
            "cv_none": cv_n,
            "cv_knn": cv_k,
            "diff_cv": cv_k - cv_n,
            "diff_rmse": r_k - r_n
        })

    sensor_df = merged.groupby("sensor_id").apply(sensor_metrics).reset_index()

    merged["hour"] = merged["target_dt"].dt.hour
    def hourly_metrics(g):
        r_n = np.sqrt(np.mean((g["y_pred_raw_none"] - g["y_true_raw_none"])**2))
        r_k = np.sqrt(np.mean((g["y_pred_raw_knn"] - g["y_true_raw_knn"])**2))
        return pd.Series({"rmse_none": r_n, "rmse_knn": r_k, "n_obs": len(g)})

    hourly_df = merged.groupby("hour").apply(hourly_metrics).reset_index()

    fig = plt.figure(figsize=(14.5, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # Panel A: Sensor-level CV-RMSE Scatter
    ax_a = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=sensor_df, x="cv_none", y="cv_knn", ax=ax_a, alpha=0.75, color="#1f77b4", s=50)
    max_val = max(sensor_df["cv_none"].quantile(0.98), sensor_df["cv_knn"].quantile(0.98)) * 1.1
    ax_a.plot([0, max_val], [0, max_val], "r--", label="Identity Line (y = x)")
    ax_a.set_xlim(20, max_val)
    ax_a.set_ylim(20, max_val)
    ax_a.set_xlabel("None (Un-imputed) CV-RMSE (%)", fontsize=11.5)
    ax_a.set_ylabel("KNN Imputation CV-RMSE (%)", fontsize=11.5)
    ax_a.set_title("Panel A: Paired Sensor Relative Error (CV-RMSE)\n[Points above line = None outperforms KNN]", fontsize=12.5, fontweight="bold", pad=8)
    ax_a.legend(frameon=True)
    ax_a.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")

    # Panel B: Hourly RMSE on Matched Timestamps
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(hourly_df["hour"], hourly_df["rmse_none"], marker="o", label="None (Un-imputed)", color="#1f77b4", linewidth=2.2)
    ax_b.plot(hourly_df["hour"], hourly_df["rmse_knn"], marker="s", label="KNN Imputation", color="#ff7f0e", linewidth=2.2)
    ax_b.set_xticks(range(0, 24, 2))
    ax_b.set_xlabel("Hour of Day (UTC)", fontsize=11.5)
    ax_b.set_ylabel("RMSE (Counts)", fontsize=11.5)
    ax_b.set_title(f"Panel B: Hourly Forecast Error on Matched Timestamps\n[Exact same N={len(merged):,} time points across 108 sensors]", fontsize=12.5, fontweight="bold", pad=8)
    ax_b.legend(frameon=True)
    ax_b.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")

    # Panel C: Distribution of Paired CV-RMSE Difference (pp units)
    ax_c = fig.add_subplot(gs[1, 0])
    sns.histplot(sensor_df["diff_cv"], kde=True, ax=ax_c, color="#6a3d9a", bins=20)
    ax_c.axvline(0, color="red", linestyle="--", label="Zero Difference")
    mean_diff = sensor_df["diff_cv"].mean()
    ax_c.axvline(mean_diff, color="black", linestyle="-", label=f"Mean Diff: +{mean_diff:.2f} pp")
    ax_c.set_xlabel(r"$\Delta$ CV-RMSE (KNN - None) [pp]", fontsize=11.5)
    ax_c.set_ylabel("Sensor Count", fontsize=11.5)
    ax_c.set_title("Panel C: Distribution of Paired Performance Shift\n[Positive values indicate Un-imputed Baseline has lower error]", fontsize=12.5, fontweight="bold", pad=8)
    ax_c.legend(frameon=True)
    ax_c.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")

    # Panel D: Subdivided cleanly into D1 (Coverage) and D2 (Matched CV-RMSE)
    gs_d = gs[1, 1].subgridspec(1, 2, wspace=0.35)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    # D1: Test Windows Coverage
    cov_none = len(p_none) / 1000.0
    cov_knn = len(p_knn) / 1000.0
    ratio = len(p_knn) / len(p_none)
    bars1 = ax_d1.bar(["None", "KNN"], [cov_none, cov_knn], color=["#2ca02c", "#1b7837"], width=0.55, edgecolor="#111111")
    ax_d1.set_ylabel("Test Predictions (x1,000)", fontsize=11)
    ax_d1.set_title(f"Operational Coverage\n[{ratio:.1f}x Expansion]", fontsize=11.5, fontweight="bold", pad=8)
    ax_d1.grid(True, linestyle="-", alpha=0.3, color="#d1d5db", axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax_d1.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval:.0f}k", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax_d1.set_ylim(0, cov_knn * 1.18)

    # D2: Matched Pairs CV-RMSE
    cv_none_mean = sensor_df["cv_none"].mean()
    cv_knn_mean = sensor_df["cv_knn"].mean()
    bars2 = ax_d2.bar(["None", "KNN"], [cv_none_mean, cv_knn_mean], color=["#3182bd", "#e6550d"], width=0.55, edgecolor="#111111")
    ax_d2.set_ylabel("Matched CV-RMSE (%)", fontsize=11)
    ax_d2.set_title(f"Matched Relative Error\n[+{mean_diff:.2f} pp Penalty]", fontsize=11.5, fontweight="bold", pad=8)
    ax_d2.grid(True, linestyle="-", alpha=0.3, color="#d1d5db", axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax_d2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax_d2.set_ylim(0, cv_knn_mean * 1.2)

    fig5_15_png = out_dir / "exp_l_matched_pairs_comparison.png"
    fig5_15_pdf = out_dir / "exp_l_matched_pairs_comparison.pdf"
    fig.savefig(fig5_15_png, bbox_inches="tight")
    fig.savefig(fig5_15_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.15 (exp_l_matched_pairs_comparison)")

    # Copy all to thesis directory
    if thesis_dir.exists():
        for f in [
            fig5_10_png, fig5_10_pdf,
            fig5_11_png, fig5_11_pdf,
            fig5_12_png, fig5_12_pdf,
            fig5_14_png, fig5_14_pdf,
            fig5_15_png, fig5_15_pdf
        ]:
            shutil.copyfile(f, thesis_dir / f.name)
            print(f"Copied to thesis: {f.name}")

if __name__ == "__main__":
    main()
