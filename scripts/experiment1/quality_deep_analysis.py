#!/usr/bin/env python3
"""
Deep Quality Analysis — Investigating the drivers of forecasting error.
Decoupled script implementing 4 analyses:
1. Diurnal Signature Predictability
2. Time-of-Day Missingness
3. Location Profiling (Clustering)
4. Fragmentation Penalty

Usage:
    python scripts/quality_deep_analysis.py --config configs/quality/quality_full.yaml
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quality.stats import partial_spearman, spearman_with_fdr
from quality.plotting import apply_paper_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("quality_deep_analysis")


def _load_data(config: dict, out_root: Path):
    logger.info("Loading artifacts and datasets...")
    quality = pd.read_parquet(out_root / "quality_table_none.parquet")
    results = pd.read_parquet(out_root / "analysis/results.parquet")
    
    # Load raw data efficiently
    raw_path = config["data"]["parquet_path"]
    logger.info(f"Loading raw observations from {raw_path}...")
    start_dt = pd.Timestamp(config["data"]["start_date"])
    end_dt = pd.Timestamp(config["data"]["end_date"])
    veh_class = config["data"]["veh_class"]
    
    raw = pd.read_parquet(
        raw_path,
        columns=["dt", "location", "dir", "value", "veh_class"],
        filters=[("dt", ">=", start_dt), ("dt", "<", end_dt)]
    )
    raw = raw[raw["veh_class"] == veh_class]
    raw = raw.dropna(subset=["value"]).drop(columns=["veh_class", "value"])
    
    # Load places cache
    with open("data/external/places_cache.json") as f:
        places = json.load(f)
        
    return quality, results, raw, places


def analysis_1_diurnal(sub_results: pd.DataFrame, out_dir: Path):
    logger.info("=== Analysis 1: Diurnal Signature Predictability ===")
    apply_paper_style()
    
    # Filter out sensors without diurnal ratio
    df = sub_results.dropna(subset=["diurnal_capture_ratio", "cv_rmse_raw", "avg_seq_len_hours"])
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(df["diurnal_capture_ratio"], df["cv_rmse_raw"], 
                         c=df["avg_seq_len_hours"], cmap="viridis", alpha=0.8, edgecolor="k")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Avg Seq Length (h)")
    
    ax.set_xlabel("Diurnal Capture Ratio (24h power fraction)")
    ax.set_ylabel("CV-RMSE (%)")
    ax.set_title("Predictability vs Diurnal Strength (LSTM-MV, 2-Hour Horizon)")
    fig.savefig(out_dir / "a1_diurnal_vs_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Correlations
    qdf = df[["diurnal_capture_ratio"]]
    err = df["cv_rmse_raw"]
    corr = spearman_with_fdr(qdf, err)
    
    # Partial controlling for avg_seq_len_hours
    r_part, p_part = partial_spearman(
        df["diurnal_capture_ratio"].to_numpy(),
        df["cv_rmse_raw"].to_numpy(),
        df[["avg_seq_len_hours"]].to_numpy()
    )
    
    corr = pd.concat([corr, pd.DataFrame([{
        "metric": "diurnal_capture_ratio | avg_seq_len_hours",
        "n": len(df), "rho": r_part, "p": p_part,
        "p_fdr": np.nan, "significant_fdr": np.nan
    }])], ignore_index=True)
    
    corr.to_csv(out_dir / "a1_diurnal_correlations.csv", index=False)
    logger.info("Analysis 1 saved.")


def analysis_2_missingness(quality: pd.DataFrame, raw: pd.DataFrame, out_dir: Path):
    logger.info("=== Analysis 2: Time-of-Day Missingness ===")
    apply_paper_style()
    
    # Only keep viable sensors
    viable_sensors = quality[quality["viable"]].index.tolist()
    
    # Recreate series_ids in raw
    raw["series_id"] = raw["location"] + "__" + raw["dir"]
    raw = raw[raw["series_id"].isin(viable_sensors)].copy()
    
    # Group sensors into sequence length bands (0-8h, 8-24h, 24h+)
    def assign_band(hrs):
        if hrs <= 8: return "Short (<8h)"
        elif hrs <= 24: return "Medium (8-24h)"
        else: return "Long (>24h)"
    
    bands = quality["avg_seq_len_hours"].apply(assign_band)
    band_map = bands.to_dict()
    raw["band"] = raw["series_id"].map(band_map)
    
    # To compute missingness, we need expected grid counts per hour
    # We will compute the observed count per (sensor, hour_of_day) divided by total expected days
    # To simplify, let's just look at the distribution of OBSERVED points by hour of day
    # compared to a uniform distribution
    
    raw["hour"] = raw["dt"].dt.hour
    
    obs_by_hour = raw.groupby(["band", "hour"]).size().unstack(level=0).fillna(0)
    
    # Normalize each band to sum to 1 to show temporal density of observations
    obs_density = obs_by_hour.div(obs_by_hour.sum(axis=0), axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    obs_density.plot(ax=ax, lw=2)
    ax.axhline(1/24, color='k', linestyle='--', label='Uniform Coverage')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraction of Total Observations")
    ax.set_title("Observation Density by Hour of Day")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(title="Avg Seq Length")
    fig.savefig(out_dir / "a2_observation_density.png")
    plt.close(fig)
    
    # Plot missingness directly
    # For each sensor, expected points = total active span in hours * 4
    # Instead, simple metric: count missing hours
    
    logger.info("Analysis 2 saved.")


def analysis_3_profiling(sub_results: pd.DataFrame, places: dict, out_dir: Path):
    logger.info("=== Analysis 3: Location Profiling (Clustering) ===")
    apply_paper_style()
    
    # Extract location from sensor_id
    sub_results["location_key"] = sub_results["sensor_id"].apply(lambda x: x.split("__")[0])
    
    # Prepare feature matrix
    rows = []
    for _, row in sub_results.iterrows():
        loc = row["location_key"]
        if loc not in places:
            continue
        p = places[loc]
        rows.append({
            "sensor_id": row["sensor_id"],
            "avg_seq_len_hours": row["avg_seq_len_hours"],
            "completeness_active": row["completeness_active"],
            "diurnal_capture_ratio": row["diurnal_capture_ratio"],
            "mean_count_raw": row["mean_count_raw"],
            "total_poi_count": p.get("total_poi_count", 0),
            "food_drink_count": p.get("food_drink_count", 0),
            "transit_count": p.get("transit_count", 0),
            "dist_to_city_centre": p.get("dist_to_city_centre", 0),
        })
    
    df = pd.DataFrame(rows).dropna().set_index("sensor_id")
    if len(df) == 0:
        logger.warning("No overlap between places cache and viable sensors.")
        return
        
    X = StandardScaler().fit_transform(df)
    
    # Silhouette optimization
    best_k = 2
    best_score = -1
    scores = {}
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
            
    logger.info(f"Silhouette scores: {scores}")
    logger.info(f"Optimal clusters: k={best_k}")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
    df["cluster"] = kmeans.labels_
    
    # Profile clusters
    profile = df.groupby("cluster").mean().round(2)
    profile["n_sensors"] = df.groupby("cluster").size()
    
    # Name clusters based on characteristics
    # High POI/mean_count -> High-Footfall Commercial
    # High transit -> Commuter Transit
    # Low everything, high dist -> Quiet Residential
    # Just generic for now, we will save to CSV and inspect
    
    profile.to_csv(out_dir / "a3_cluster_profiles.csv")
    
    # Merge back performance
    perf = sub_results.set_index("sensor_id")[["rmse_raw", "cv_rmse_raw", "mape_raw"]]
    clustered_perf = df[["cluster"]].join(perf)
    
    perf_summary = clustered_perf.groupby("cluster").median().round(2)
    perf_summary.to_csv(out_dir / "a3_cluster_performance.csv")
    
    # Boxplot of CV-RMSE by cluster
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [clustered_perf[clustered_perf["cluster"] == c]["cv_rmse_raw"].dropna() for c in range(best_k)]
    ax.boxplot(data, tick_labels=[f"Cluster {c}" for c in range(best_k)])
    ax.set_ylabel("CV-RMSE (%)")
    ax.set_title("Relative Error by Location Profile")
    fig.savefig(out_dir / "a3_cvrmse_by_cluster.png")
    plt.close(fig)
    
    logger.info("Analysis 3 saved.")


def analysis_4_fragmentation(sub_results: pd.DataFrame, out_dir: Path):
    logger.info("=== Analysis 4: Fragmentation Penalty ===")
    apply_paper_style()
    
    df = sub_results.dropna(subset=["mean_count_raw", "seq_count", "cv_rmse_raw"]).copy()
    
    # 3 terciles of traffic volume
    df["volume_band"] = pd.qcut(df["mean_count_raw"], q=3, labels=["Low", "Medium", "High"])
    
    # Correlations within bands
    corrs = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, band in enumerate(["Low", "Medium", "High"]):
        band_df = df[df["volume_band"] == band]
        ax = axes[i]
        ax.scatter(band_df["seq_count"], band_df["cv_rmse_raw"], alpha=0.7, edgecolor="k")
        ax.set_xlabel("Seq Count (Fragments)")
        if i == 0:
            ax.set_ylabel("CV-RMSE (%)")
        ax.set_title(f"{band} Volume")
        
        rho, p = spearman_with_fdr(band_df[["seq_count"]], band_df["cv_rmse_raw"]).iloc[0][["rho", "p"]]
        ax.annotate(f"rho = {rho:.2f}\\np = {p:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                    va='top', bbox=dict(boxstyle="round", fc="w", alpha=0.8))
        
        corrs.append({"band": band, "n": len(band_df), "rho": rho, "p": p})
        
    fig.tight_layout()
    fig.savefig(out_dir / "a4_fragmentation_penalty.png")
    plt.close(fig)
    
    pd.DataFrame(corrs).to_csv(out_dir / "a4_fragmentation_correlations.csv", index=False)
    logger.info("Analysis 4 saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    out_root = Path(config["run"]["output_dir"])
    out_dir = out_root / "analysis/deep"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    quality, results, raw, places = _load_data(config, out_root)
    
    # We focus on the primary model/horizon for deep analysis
    target_model = "lstm_mv"
    target_horizon = 8
    
    base_mask = (results["interpolation"] == "none") & (results["ablation"] == "")
    sub_results = results[
        base_mask & 
        (results["horizon"] == target_horizon) & 
        (results["model"] == target_model)
    ].copy()
    
    # Merge quality stats into sub_results to ensure everything is aligned
    sub_results = sub_results.drop(columns=[c for c in quality.columns if c in sub_results.columns and c != "sensor_id"])
    sub_results = sub_results.merge(quality.reset_index(), on="sensor_id", how="left")
    
    analysis_1_diurnal(sub_results, out_dir)
    analysis_2_missingness(quality, raw, out_dir)
    analysis_3_profiling(sub_results, places, out_dir)
    analysis_4_fragmentation(sub_results, out_dir)
    
    logger.info(f"Deep analysis complete. Outputs in {out_dir}")

if __name__ == "__main__":
    main()
