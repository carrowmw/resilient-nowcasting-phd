#!/usr/bin/env python3
"""
Extended Quality Analysis — Further figures and statistical tests based on V2 data.
Generates radar charts, heatmaps, distance plots, and statistical validation of clusters.
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal, mannwhitneyu
import statsmodels.formula.api as smf
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from quality.plotting import apply_paper_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("quality_extended_analysis")

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
    
    with open("data/external/places_cache.json") as f:
        places = json.load(f)
        
    return quality, results, raw, places


def prepare_cluster_data(sub_results: pd.DataFrame, places: dict):
    sub_results["location_key"] = sub_results["sensor_id"].apply(lambda x: x.split("__")[0])
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
            "transit_count": p.get("transit_count", 0),
            "food_drink_count": p.get("food_drink_count", 0),
            "dist_to_city_centre": p.get("dist_to_city_centre", 0),
            "cv_rmse_raw": row["cv_rmse_raw"]
        })
    df = pd.DataFrame(rows).dropna().set_index("sensor_id")
    features = ["avg_seq_len_hours", "completeness_active", "diurnal_capture_ratio", 
                "mean_count_raw", "total_poi_count", "transit_count", "food_drink_count", "dist_to_city_centre"]
    X = StandardScaler().fit_transform(df[features])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    df["cluster"] = kmeans.labels_
    
    # Rename clusters to make sense based on distance
    c0_dist = df[df["cluster"]==0]["dist_to_city_centre"].mean()
    c1_dist = df[df["cluster"]==1]["dist_to_city_centre"].mean()
    if c0_dist < c1_dist:
        cluster_names = {0: "Urban Core", 1: "Suburban"}
    else:
        cluster_names = {1: "Urban Core", 0: "Suburban"}
    df["cluster_name"] = df["cluster"].map(cluster_names)
    return df


def analysis_radar(df: pd.DataFrame, out_dir: Path):
    logger.info("Generating Cluster Radar Chart...")
    apply_paper_style()
    features = ["dist_to_city_centre", "total_poi_count", "transit_count", 
                "mean_count_raw", "diurnal_capture_ratio", "avg_seq_len_hours"]
    
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)
    scaled_df["cluster_name"] = df["cluster_name"]
    
    means = scaled_df.groupby("cluster_name").mean()
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for idx, row in means.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=idx)
        ax.fill(angles, values, alpha=0.25)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Standardised Cluster Profiles", va='bottom')
    fig.tight_layout()
    fig.savefig(out_dir / "ext_cluster_radar.png")
    plt.close(fig)


def analysis_corr_heatmap(df: pd.DataFrame, out_dir: Path):
    logger.info("Generating Full Correlation Heatmap...")
    apply_paper_style()
    features = ["cv_rmse_raw", "avg_seq_len_hours", "completeness_active", 
                "diurnal_capture_ratio", "mean_count_raw", "total_poi_count", 
                "transit_count", "dist_to_city_centre"]
    corr = df[features].corr(method='spearman')
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman Rank Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "ext_correlation_heatmap.png")
    plt.close(fig)


def analysis_2d_missingness(quality: pd.DataFrame, raw: pd.DataFrame, out_dir: Path):
    logger.info("Generating 2D Missingness Heatmap...")
    apply_paper_style()
    viable_sensors = quality[quality["viable"]].index.tolist()
    raw["series_id"] = raw["location"] + "__" + raw["dir"]
    raw = raw[raw["series_id"].isin(viable_sensors)].copy()
    
    raw["hour"] = raw["dt"].dt.hour
    raw["dayofweek"] = raw["dt"].dt.dayofweek
    
    # Heatmap for short sequence sensors (<8h) to see if missingness is purely diurnal or weekend-driven
    short_sensors = quality[quality["avg_seq_len_hours"] <= 8].index
    raw_short = raw[raw["series_id"].isin(short_sensors)]
    
    counts = raw_short.groupby(["dayofweek", "hour"]).size().unstack(level=1).fillna(0)
    density = counts.div(counts.values.sum()) * 100 # % of total observations
    
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(density, cmap="viridis", ax=ax, cbar_kws={'label': '% of Total Observations'})
    ax.set_yticklabels(days, rotation=0)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Observation Density for Short-Sequence Sensors (<8h)")
    fig.tight_layout()
    fig.savefig(out_dir / "ext_2d_observation_density.png")
    plt.close(fig)


def analysis_dist_vs_error(df: pd.DataFrame, out_dir: Path):
    logger.info("Generating Error vs Distance Plot...")
    apply_paper_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=df, x="dist_to_city_centre", y="cv_rmse_raw", 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax, lowess=True)
    ax.set_xlabel("Distance to City Centre (m)")
    ax.set_ylabel("CV-RMSE (%)")
    ax.set_title("Forecast Error vs Distance to Urban Core")
    fig.tight_layout()
    fig.savefig(out_dir / "ext_dist_vs_error.png")
    plt.close(fig)


def analysis_stats(df: pd.DataFrame, out_dir: Path):
    logger.info("Running statistical tests...")
    results = {}
    
    # 1. Kruskal-Wallis on Clusters
    c1 = df[df["cluster_name"] == "Urban Core"]["cv_rmse_raw"]
    c2 = df[df["cluster_name"] == "Suburban"]["cv_rmse_raw"]
    stat, p = kruskal(c1, c2)
    results["kruskal_clusters"] = {"statistic": float(stat), "p_value": float(p)}
    
    # 2. Mann-Whitney U for Transit
    df["high_transit"] = df["transit_count"] >= df["transit_count"].median()
    t_high = df[df["high_transit"]]["diurnal_capture_ratio"]
    t_low = df[~df["high_transit"]]["diurnal_capture_ratio"]
    stat_t, p_t = mannwhitneyu(t_high, t_low, alternative="greater")
    results["mwu_transit_diurnal"] = {"statistic": float(stat_t), "p_value": float(p_t)}
    
    # 3. OLS Regression with Interactions
    model = smf.ols("cv_rmse_raw ~ avg_seq_len_hours * mean_count_raw", data=df).fit()
    with open(out_dir / "ext_ols_interaction.txt", "w") as f:
        f.write(model.summary().as_text())
    
    with open(out_dir / "ext_statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2)


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
    
    target_model = "lstm_mv"
    target_horizon = 8
    
    base_mask = (results["interpolation"] == "none") & (results["ablation"] == "")
    sub_results = results[
        base_mask & 
        (results["horizon"] == target_horizon) & 
        (results["model"] == target_model)
    ].copy()
    
    sub_results = sub_results.drop(columns=[c for c in quality.columns if c in sub_results.columns and c != "sensor_id"])
    sub_results = sub_results.merge(quality.reset_index(), on="sensor_id", how="left")
    
    df = prepare_cluster_data(sub_results, places)
    
    analysis_radar(df, out_dir)
    analysis_corr_heatmap(df, out_dir)
    analysis_2d_missingness(quality, raw, out_dir)
    analysis_dist_vs_error(df, out_dir)
    analysis_stats(df, out_dir)
    
    logger.info(f"Extended deep analysis complete. Outputs in {out_dir}")

if __name__ == "__main__":
    main()
