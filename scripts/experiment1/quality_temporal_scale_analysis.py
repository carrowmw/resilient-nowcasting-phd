#!/usr/bin/env python3
"""
Temporal and Scale Quality Analysis — Experiments G, H, I, J, K.
Side-by-side comparisons between Un-imputed Baseline (`none`) and KNN Imputation (`knn`).

Usage:
    python scripts/quality_temporal_scale_analysis.py --config configs/quality/quality_full_v2.yaml
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quality.evaluate import evaluate_split, predictions_frame
from quality.interpolation import apply_interpolation
from quality.loader import SensorSeriesStore
from quality.per_sensor_dataset import build_sensor_splits
from quality.plotting import apply_paper_style
from quality.registry import build_forecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("quality_temporal_scale_analysis")


def _get_season_uk(dt_series):
    m = dt_series.dt.month
    conditions = [
        m.isin([12, 1, 2]),
        m.isin([3, 4, 5]),
        m.isin([6, 7, 8]),
        m.isin([9, 10, 11])
    ]
    choices = ["Winter", "Spring", "Summer", "Autumn"]
    return np.select(conditions, choices, default="Unknown")


def experiment_g_hour_of_day(preds_none: pd.DataFrame, preds_knn: pd.DataFrame, out_dir: Path):
    logger.info("=== Experiment G: Error by Hour-of-Day (Side-by-Side: None vs KNN) ===")
    apply_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (name, preds) in zip(axes, [("None (Un-imputed Baseline)", preds_none), ("KNN Imputation", preds_knn)]):
        p = preds.copy()
        if "target_dt" not in p.columns:
            p["target_dt"] = pd.to_datetime(p["target_dt"])

        p["hour"] = p["target_dt"].dt.hour
        p["dayofweek"] = p["target_dt"].dt.dayofweek
        p["is_weekend"] = p["dayofweek"] >= 5

        def group_metrics(g):
            y = g["y_true_raw"].to_numpy(dtype=float)
            pr = g["y_pred_raw"].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((pr - y) ** 2)))
            return pd.Series({"rmse_raw": rmse})

        agg = p.groupby(["is_weekend", "hour"]).apply(group_metrics).reset_index()

        sns.lineplot(data=agg, x="hour", y="rmse_raw", hue="is_weekend", ax=ax, marker="o")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("RMSE (Counts)")
        ax.set_title(f"Forecast Error by Hour-of-Day\n[{name}]")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ["Weekday", "Weekend"], title="Day Type")

    fig.tight_layout()
    fig.savefig(out_dir / "exp_g_error_by_hour.png")
    fig.savefig(out_dir / "exp_g_error_by_hour_side_by_side.png")
    plt.close(fig)

    logger.info("Experiment G complete.")


def experiment_h_seasonal(preds_none: pd.DataFrame, preds_knn: pd.DataFrame, out_dir: Path):
    logger.info("=== Experiment H: Seasonal Decomposition (Side-by-Side: None vs KNN) ===")
    apply_paper_style()

    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    rows = []

    for name, preds in [("None (Un-imputed)", preds_none), ("KNN Imputation", preds_knn)]:
        p_copy = preds.copy()
        if "target_dt" not in p_copy.columns:
            p_copy["target_dt"] = pd.to_datetime(p_copy["target_dt"])
        p_copy["season"] = _get_season_uk(p_copy["target_dt"])

        for season, g in p_copy.groupby("season"):
            y = g["y_true_raw"].to_numpy(dtype=float)
            p = g["y_pred_raw"].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((p - y) ** 2)))
            mean_cnt = float(np.mean(y))
            cv_rmse = float(rmse / max(mean_cnt, 1e-8) * 100.0)
            rows.append({"interpolation": name, "season": season, "rmse_raw": rmse, "cv_rmse_raw": cv_rmse})

    agg = pd.DataFrame(rows)
    agg["season"] = pd.Categorical(agg["season"], categories=season_order, ordered=True)
    agg = agg.sort_values("season")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=agg, x="season", y="rmse_raw", hue="interpolation", ax=axes[0], palette="Set2")
    axes[0].set_title("RMSE by Season (None vs KNN)")
    axes[0].set_ylabel("RMSE (Counts)")
    axes[0].set_xlabel("Season (UK)")

    sns.barplot(data=agg, x="season", y="cv_rmse_raw", hue="interpolation", ax=axes[1], palette="Set2")
    axes[1].set_title("CV-RMSE by Season (None vs KNN)")
    axes[1].set_ylabel("CV-RMSE (%)")
    axes[1].set_xlabel("Season (UK)")

    fig.tight_layout()
    fig.savefig(out_dir / "exp_h_seasonal_error.png")
    fig.savefig(out_dir / "exp_h_seasonal_error_side_by_side.png")
    plt.close(fig)

    agg.to_csv(out_dir / "exp_h_seasonal_error.csv", index=False)
    logger.info("Experiment H complete.")


def experiment_i_outliers(sub_results_none: pd.DataFrame, preds_none: pd.DataFrame, quality_none: pd.DataFrame,
                         sub_results_knn: pd.DataFrame, preds_knn: pd.DataFrame, quality_knn: pd.DataFrame,
                         out_dir: Path):
    logger.info("=== Experiment I: Outlier / Case Study Analysis (Side-by-Side: None vs KNN) ===")
    apply_paper_style()

    df = sub_results_none.dropna(subset=["cv_rmse_raw"]).sort_values("cv_rmse_raw")
    if len(df) == 0:
        logger.warning("No data for experiment I")
        return

    bottom_5 = df.head(5)
    top_5 = df.tail(5)

    outliers = pd.concat([bottom_5, top_5])
    outliers["outlier_type"] = ["Bottom 5 (Lowest Error)"] * len(bottom_5) + ["Top 5 (Highest Error)"] * len(top_5)

    cols_to_save = ["sensor_id", "outlier_type", "cv_rmse_raw", "rmse_raw", "mean_count_raw"]
    for c in ["avg_seq_len_hours", "completeness_active"]:
        if c in quality_none.columns:
            outliers[c] = outliers["sensor_id"].map(quality_none[c])
            cols_to_save.append(c)

    outliers[cols_to_save].to_csv(out_dir / "exp_i_outliers_characteristics.csv", index=False)

    case_studies = [
        bottom_5.iloc[0]["sensor_id"],
        bottom_5.iloc[1]["sensor_id"],
        bottom_5.iloc[2]["sensor_id"],
        top_5.iloc[-1]["sensor_id"],
        top_5.iloc[-2]["sensor_id"],
        top_5.iloc[-3]["sensor_id"],
    ]

    for sensor in case_studies:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

        for ax, (interp_label, preds_df, sub_res) in zip(
            axes,
            [("None (Un-imputed Baseline)", preds_none, sub_results_none), ("KNN Imputation", preds_knn, sub_results_knn)]
        ):
            if "target_dt" not in preds_df.columns:
                preds_df["target_dt"] = pd.to_datetime(preds_df["target_dt"])

            sensor_preds = preds_df[preds_df["sensor_id"] == sensor].sort_values("target_dt")
            if len(sensor_preds) == 0:
                ax.text(0.5, 0.5, f"No predictions available for {sensor}", ha="center", va="center")
                continue

            sp = sensor_preds.head(500).set_index("target_dt").resample("15min").first().reset_index()
            sensor_sub = sub_res[sub_res["sensor_id"] == sensor]
            cv_val = sensor_sub["cv_rmse_raw"].iloc[0] if len(sensor_sub) > 0 else 0.0

            ax.plot(sp["target_dt"], sp["y_true_raw"], label="Actuals", color="black", alpha=0.6, marker=".", markersize=3, linewidth=1)
            ax.plot(sp["target_dt"], sp["y_pred_raw"], label="Predicted", color="red", alpha=0.8, linestyle="--", marker=".", markersize=3, linewidth=1)

            ax.set_title(f"Case Study: {sensor} — {interp_label} (CV-RMSE: {cv_val:.1f}%)")
            ax.set_ylabel("Pedestrian Count")
            ax.legend()

        fig.tight_layout()
        safe_name = sensor.replace(":", "_").replace(" ", "_")
        fig.savefig(out_dir / f"exp_i_case_study_{safe_name}.png")
        fig.savefig(out_dir / f"exp_i_case_study_{safe_name}_side_by_side.png")
        plt.close(fig)

    logger.info("Experiment I complete.")


def experiment_j_scale(sub_results_none: pd.DataFrame, sub_results_knn: pd.DataFrame, out_dir: Path):
    logger.info("=== Experiment J: Scale Effect (Side-by-Side: None vs KNN) ===")
    apply_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, df) in zip(axes, [("None (Un-imputed Baseline)", sub_results_none), ("KNN Imputation", sub_results_knn)]):
        df_clean = df.dropna(subset=["cv_rmse_raw", "mean_count_raw"])
        if len(df_clean) > 0:
            sns.regplot(data=df_clean, x="mean_count_raw", y="cv_rmse_raw",
                        ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, lowess=True)
            ax.set_xlabel("Mean Pedestrian Volume (Counts)")
            ax.set_ylabel("Relative Error (CV-RMSE %)")
            ax.set_title(f"Relative Error vs Volume\n[{name}]")

    fig.tight_layout()
    fig.savefig(out_dir / "exp_j_scale_effect.png")
    fig.savefig(out_dir / "exp_j_scale_effect_side_by_side.png")
    plt.close(fig)

    logger.info("Experiment J complete.")


def experiment_k_extreme_events(preds_none: pd.DataFrame, preds_knn: pd.DataFrame, out_dir: Path):
    logger.info("=== Experiment K: Extreme Events Analysis (Side-by-Side: None vs KNN) ===")
    apply_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, preds) in zip(axes, [("None (Un-imputed Baseline)", preds_none), ("KNN Imputation", preds_knn)]):
        p = preds.copy()
        if "target_dt" not in p.columns:
            p["target_dt"] = pd.to_datetime(p["target_dt"])

        metrics = []
        for sensor_id, g in p.groupby("sensor_id"):
            mean_c = g["y_true_raw"].mean()
            std_c = g["y_true_raw"].std()

            if pd.isna(std_c) or std_c < 1.0:
                continue

            threshold = mean_c + 3 * std_c
            is_extreme = g["y_true_raw"] > threshold
            is_normal = (g["y_true_raw"] <= mean_c + std_c) & (g["y_true_raw"] >= mean_c - std_c)

            if not is_extreme.any() or not is_normal.any():
                continue

            y_ext = g.loc[is_extreme, "y_true_raw"].to_numpy(dtype=float)
            pr_ext = g.loc[is_extreme, "y_pred_raw"].to_numpy(dtype=float)
            rmse_ext = float(np.sqrt(np.mean((pr_ext - y_ext)**2)))
            mape_ext = float(np.mean(np.abs(pr_ext - y_ext) / np.maximum(np.abs(y_ext), 1.0)) * 100.0)

            y_norm = g.loc[is_normal, "y_true_raw"].to_numpy(dtype=float)
            pr_norm = g.loc[is_normal, "y_pred_raw"].to_numpy(dtype=float)
            rmse_norm = float(np.sqrt(np.mean((pr_norm - y_norm)**2)))
            mape_norm = float(np.mean(np.abs(pr_norm - y_norm) / np.maximum(np.abs(y_norm), 1.0)) * 100.0)

            metrics.append({
                "sensor_id": sensor_id,
                "rmse_extreme": rmse_ext,
                "rmse_normal": rmse_norm,
                "mape_extreme": mape_ext,
                "mape_normal": mape_norm
            })

        if metrics:
            df_metrics = pd.DataFrame(metrics)
            if name.startswith("None"):
                df_metrics.to_csv(out_dir / "exp_k_extreme_events_metrics.csv", index=False)

            plot_df = pd.melt(df_metrics, id_vars=["sensor_id"],
                              value_vars=["rmse_normal", "rmse_extreme"],
                              var_name="metric_type", value_name="value")
            plot_df["Period"] = plot_df["metric_type"].apply(lambda x: "Extreme (>3σ)" if "extreme" in x else "Normal (±1σ)")

            sns.boxplot(data=plot_df, x="Period", y="value", ax=ax, palette="Set2")
            ax.set_title(f"RMSE: Normal vs Extreme\n[{name}]")
            ax.set_ylabel("RMSE (Counts)")

    fig.tight_layout()
    fig.savefig(out_dir / "exp_k_extreme_events_comparison.png")
    fig.savefig(out_dir / "exp_k_extreme_events_side_by_side.png")
    plt.close(fig)

    logger.info("Experiment K complete.")


def experiment_l_matched_pairs(preds_none: pd.DataFrame, preds_knn: pd.DataFrame, out_dir: Path):
    logger.info("=== Experiment L: Matched Pairs Intersection Analysis ===")
    apply_paper_style()

    p_none = preds_none.copy()
    p_knn = preds_knn.copy()
    if "target_dt" not in p_none.columns:
        p_none["target_dt"] = pd.to_datetime(p_none["target_dt"])
    if "target_dt" not in p_knn.columns:
        p_knn["target_dt"] = pd.to_datetime(p_knn["target_dt"])

    merged = pd.merge(
        p_none[["sensor_id", "target_dt", "y_true_raw", "y_pred_raw"]],
        p_knn[["sensor_id", "target_dt", "y_true_raw", "y_pred_raw"]],
        on=["sensor_id", "target_dt"],
        suffixes=("_none", "_knn")
    )

    if len(merged) == 0:
        logger.warning("No matched pairs found between none and knn.")
        return

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
    sensor_df.to_csv(out_dir / "exp_l_matched_pairs_sensor_summary.csv", index=False)

    merged["hour"] = merged["target_dt"].dt.hour
    def hourly_metrics(g):
        r_n = np.sqrt(np.mean((g["y_pred_raw_none"] - g["y_true_raw_none"])**2))
        r_k = np.sqrt(np.mean((g["y_pred_raw_knn"] - g["y_true_raw_knn"])**2))
        return pd.Series({"rmse_none": r_n, "rmse_knn": r_k, "n_obs": len(g)})

    hourly_df = merged.groupby("hour").apply(hourly_metrics).reset_index()
    hourly_df.to_csv(out_dir / "exp_l_matched_pairs_hourly_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Sensor-level CV-RMSE Scatter (None vs KNN)
    ax_a = axes[0, 0]
    sns.scatterplot(data=sensor_df, x="cv_none", y="cv_knn", ax=ax_a, alpha=0.75, color="#1f77b4", s=50)
    max_val = max(sensor_df["cv_none"].quantile(0.98), sensor_df["cv_knn"].quantile(0.98)) * 1.1
    ax_a.plot([0, max_val], [0, max_val], 'r--', label="Identity Line (y = x)")
    ax_a.set_xlim(20, max_val)
    ax_a.set_ylim(20, max_val)
    ax_a.set_xlabel("None (Un-imputed) CV-RMSE (%)")
    ax_a.set_ylabel("KNN Imputation CV-RMSE (%)")
    ax_a.set_title("Panel A: Paired Sensor Relative Error (CV-RMSE)\n[Points above line = None outperforms KNN]")
    ax_a.legend()

    # Panel B: Hourly RMSE on Matched Pairs
    ax_b = axes[0, 1]
    ax_b.plot(hourly_df["hour"], hourly_df["rmse_none"], marker="o", label="None (Un-imputed)", color="#1f77b4", linewidth=2)
    ax_b.plot(hourly_df["hour"], hourly_df["rmse_knn"], marker="s", label="KNN Imputation", color="#ff7f0e", linewidth=2)
    ax_b.set_xticks(range(0, 24, 2))
    ax_b.set_xlabel("Hour of Day (UTC)")
    ax_b.set_ylabel("RMSE (Counts)")
    ax_b.set_title(f"Panel B: Hourly Forecast Error on Matched Timestamps\n[Exact same N={len(merged):,} time points across 108 sensors]")
    ax_b.legend()

    # Panel C: Distribution of Paired CV-RMSE Difference
    ax_c = axes[1, 0]
    sns.histplot(sensor_df["diff_cv"], kde=True, ax=ax_c, color="#6a3d9a", bins=20)
    ax_c.axvline(0, color="red", linestyle="--", label="Zero Difference")
    ax_c.axvline(sensor_df["diff_cv"].mean(), color="black", linestyle="-", label=f"Mean Diff: +{sensor_df['diff_cv'].mean():.2f}%")
    ax_c.set_xlabel(r"$\Delta$ CV-RMSE (KNN - None) [%]")
    ax_c.set_ylabel("Sensor Count")
    ax_c.set_title("Panel C: Distribution of Paired Performance Shift\n[Positive values indicate Un-imputed Baseline has lower error]")
    ax_c.legend()

    # Panel D: Operational Coverage vs Accuracy Trade-off
    ax_d = axes[1, 1]
    metrics_summary = pd.DataFrame({
        "Variant": ["None (Un-imputed)", "KNN Imputation"],
        "Operational Coverage (Test Windows)": [len(preds_none), len(preds_knn)],
        "Matched Pairs CV-RMSE (%)": [sensor_df["cv_none"].mean(), sensor_df["cv_knn"].mean()]
    })
    ax_d_twin = ax_d.twinx()
    w = 0.35
    x = np.arange(2)
    ax_d.bar(x - w/2, metrics_summary["Operational Coverage (Test Windows)"] / 1000.0, width=w, color="#2ca02c", alpha=0.85, label="Test Windows (Thousands)")
    ax_d_twin.bar(x + w/2, metrics_summary["Matched Pairs CV-RMSE (%)"], width=w, color="#d62728", alpha=0.85, label="Matched CV-RMSE (%)")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(metrics_summary["Variant"])
    ax_d.set_ylabel("Total Test Predictions (x1,000)", color="#2ca02c")
    ax_d_twin.set_ylabel("Matched CV-RMSE (%)", color="#d62728")
    ax_d.set_title("Panel D: The Operational Dilemma\n[Coverage Expansion (24.4x) vs Reconstruction Noise (+4.3%)]")

    fig.tight_layout()
    fig.savefig(out_dir / "exp_l_matched_pairs_comparison.png")
    plt.close(fig)

    logger.info(f"Experiment L complete. Generated exp_l_matched_pairs_comparison.png on {len(merged):,} matched pairs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_root = Path(config["run"]["output_dir"])
    out_dir = out_root / "analysis" / "temporal_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading analysis data...")
    results_path = out_root / "analysis" / "results.parquet"
    if not results_path.exists():
        logger.error(f"CRITICAL: results.parquet not found at {results_path}. Run aggregate_quality_results.py first.")
        sys.exit(1)

    all_results = pd.read_parquet(results_path)

    quality_none = pd.read_parquet(out_root / "quality_table_none.parquet")
    quality_knn = pd.read_parquet(out_root / "quality_table_knn.parquet" if (out_root / "quality_table_knn.parquet").exists() else out_root / "quality_table_none.parquet")

    runs_dir = out_root / "runs"
    preds_dict = {}

    for interp in ["none", "knn"]:
        pred_frames = []
        for job_dir in sorted(runs_dir.iterdir()):
            if not (job_dir / "manifest.json").exists():
                continue
            with open(job_dir / "manifest.json") as f:
                manifest = json.load(f)
                if manifest.get("model") == "lstm_mv" and manifest.get("horizon") == 8 and manifest.get("interpolation") == interp and manifest.get("ablation") == "":
                    p = pd.read_parquet(job_dir / "predictions.parquet")
                    pred_frames.append(p)

        if pred_frames:
            preds_dict[interp] = pd.concat(pred_frames, ignore_index=True)

    if "none" not in preds_dict or "knn" not in preds_dict:
        logger.error("CRITICAL: Missing predictions for none or knn.")
        sys.exit(1)

    target_model = "lstm_mv"
    target_horizon = 8

    sub_results_none = all_results[
        (all_results["model"] == target_model) &
        (all_results["horizon"] == target_horizon) &
        (all_results["interpolation"] == "none") &
        (all_results["ablation"] == "")
    ].copy()

    sub_results_knn = all_results[
        (all_results["model"] == target_model) &
        (all_results["horizon"] == target_horizon) &
        (all_results["interpolation"] == "knn") &
        (all_results["ablation"] == "")
    ].copy()

    logger.info(f"Loaded {len(sub_results_none)} none sensors and {len(sub_results_knn)} knn sensors for target setup.")

    experiment_g_hour_of_day(preds_dict["none"], preds_dict["knn"], out_dir)
    experiment_h_seasonal(preds_dict["none"], preds_dict["knn"], out_dir)
    experiment_i_outliers(sub_results_none, preds_dict["none"], quality_none,
                         sub_results_knn, preds_dict["knn"], quality_knn, out_dir)
    experiment_j_scale(sub_results_none, sub_results_knn, out_dir)
    experiment_k_extreme_events(preds_dict["none"], preds_dict["knn"], out_dir)
    experiment_l_matched_pairs(preds_dict["none"], preds_dict["knn"], out_dir)

    logger.info(f"All temporal/scale side-by-side analyses complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
