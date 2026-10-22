#!/usr/bin/env python3
"""
scripts/experiment1/plot_ch5_audit_cluster_and_corr.py

Regenerates Chapter 5 figures:
- 5.5: ext_correlation_heatmap.png (masked upper triangle, clean labels, horizon stated)
- 5.7: ext_cluster_radar.png (clean feature labels, fix -1.0 radial tick overlap)
- 5.8: a3_cvrmse_by_cluster.png (relabel Cluster 0 -> Suburban, Cluster 1 -> Urban Core)

Saves to artifacts/quality_full/analysis/deep/ and copies to thesis figures directory.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    with open("artifacts/quality_full_v2/sensor_manifest.json") as f:
        qm = json.load(f)
    viable = qm["viable_sensors"]

    with open("data/external/places_cache.json") as f:
        places = json.load(f)

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
    sub["location_key"] = sub["sensor_id"].apply(lambda x: x.split("__")[0])

    rows = []
    for _, row in sub.iterrows():
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
            "cv_rmse_raw": row["cv_rmse_raw"],
            "rmse_raw": row["rmse_raw"]
        })
    df = pd.DataFrame(rows).dropna().set_index("sensor_id")

    # K-means clustering (k=2) matching the 82 Suburban vs 28 Urban Core structure
    features = [
        "avg_seq_len_hours", "completeness_active", "diurnal_capture_ratio", 
        "mean_count_raw", "total_poi_count", "transit_count", "dist_to_city_centre"
    ]
    X = StandardScaler().fit_transform(df[features])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    df["cluster"] = kmeans.labels_

    c0_dist = df[df["cluster"] == 0]["dist_to_city_centre"].mean()
    c1_dist = df[df["cluster"] == 1]["dist_to_city_centre"].mean()
    if c0_dist < c1_dist:
        cluster_names = {0: "Urban Core", 1: "Suburban"}
    else:
        cluster_names = {1: "Urban Core", 0: "Suburban"}
    df["cluster_name"] = df["cluster"].map(cluster_names)

    out_dir = Path("artifacts/quality_full/analysis/deep")
    out_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir = Path("/Users/administrator/Code/latex-documents/thesis/figures/experiment1")

    # Common styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12

    # -------------------------------------------------------------
    # 1. Figure 5.5: ext_correlation_heatmap
    # -------------------------------------------------------------
    corr_cols = [
        "cv_rmse_raw", "avg_seq_len_hours", "completeness_active", 
        "diurnal_capture_ratio", "mean_count_raw", "total_poi_count", 
        "transit_count", "dist_to_city_centre"
    ]
    col_labels = [
        "CV-RMSE (%)", "Avg Seq Length (h)", "Active Completeness",
        "Diurnal Ratio", "Mean Flow Count", "Total POI Count",
        "Transit Count", "Dist to Centre (m)"
    ]
    corr_df = df[corr_cols].copy()
    corr_df.columns = col_labels
    corr = corr_df.corr(method="spearman")

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
        linewidths=0.5
    )
    ax.set_title("Spearman Rank Correlation Matrix (LSTM-MV, 2-Hour Horizon)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=10.5)
    plt.yticks(rotation=0, fontsize=10.5)
    fig.tight_layout()

    fig5_5_png = out_dir / "ext_correlation_heatmap.png"
    fig5_5_pdf = out_dir / "ext_correlation_heatmap.pdf"
    fig.savefig(fig5_5_png, bbox_inches="tight")
    fig.savefig(fig5_5_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.5 (ext_correlation_heatmap)")

    # -------------------------------------------------------------
    # 2. Figure 5.7: ext_cluster_radar
    # -------------------------------------------------------------
    radar_features = [
        "dist_to_city_centre", "total_poi_count", "transit_count", 
        "mean_count_raw", "diurnal_capture_ratio", "avg_seq_len_hours"
    ]
    radar_labels = [
        "Distance to Centre", "Total POI Count", "Transit Access",
        "Mean Flow Count", "Diurnal Strength", "Avg Seq Length"
    ]
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[radar_features]), columns=radar_labels, index=df.index)
    scaled_df["cluster_name"] = df["cluster_name"]

    means = scaled_df.groupby("cluster_name").mean()
    # Ensure Suburban first, Urban Core second for consistent colors
    ordered_clusters = [c for c in ["Suburban", "Urban Core"] if c in means.index]
    means = means.loc[ordered_clusters]

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), dpi=300)
    colors = {"Suburban": "#0072B2", "Urban Core": "#E69F00"}

    for idx, row in means.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, label=idx, color=colors.get(idx, "#333333"))
        ax.fill(angles, values, alpha=0.2, color=colors.get(idx, "#333333"))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11)
    
    # Clean radial ticks: avoid -1.0 overlap at origin by setting bounds and radial angle
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["-0.5", "0.0", "0.5", "1.0"], fontsize=9.5, color="#555555")
    ax.set_rlabel_position(22.5)  # Offset radial labels to clear lines and vertices

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=11, frameon=True)
    ax.set_title("Standardised Cluster Profiles", fontsize=13.5, fontweight="bold", pad=20)
    fig.tight_layout()

    fig5_7_png = out_dir / "ext_cluster_radar.png"
    fig5_7_pdf = out_dir / "ext_cluster_radar.pdf"
    fig.savefig(fig5_7_png, bbox_inches="tight")
    fig.savefig(fig5_7_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.7 (ext_cluster_radar)")

    # -------------------------------------------------------------
    # 3. Figure 5.8: a3_cvrmse_by_cluster
    # -------------------------------------------------------------
    suburban_data = df[df["cluster_name"] == "Suburban"]["cv_rmse_raw"].dropna()
    urbancore_data = df[df["cluster_name"] == "Urban Core"]["cv_rmse_raw"].dropna()

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    bp = ax.boxplot(
        [suburban_data, urbancore_data],
        tick_labels=[f"Suburban\n(n={len(suburban_data)})", f"Urban Core\n(n={len(urbancore_data)})"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="#d95f02", linewidth=2.0),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", color="#333333", alpha=0.7, markersize=5)
    )
    # Color boxes cleanly
    bp["boxes"][0].set_facecolor("#cfe2f3")
    bp["boxes"][1].set_facecolor("#fce5cd")

    ax.set_ylabel("CV-RMSE (%)", fontsize=12.5, labelpad=8)
    ax.set_title("Relative Forecast Error by Location Profile (LSTM-MV)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="-", alpha=0.3, color="#d1d5db")
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(1.0)
    fig.tight_layout()

    fig5_8_png = out_dir / "a3_cvrmse_by_cluster.png"
    fig5_8_pdf = out_dir / "a3_cvrmse_by_cluster.pdf"
    fig.savefig(fig5_8_png, bbox_inches="tight")
    fig.savefig(fig5_8_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.8 (a3_cvrmse_by_cluster)")

    # Copy files to thesis if thesis directory exists
    if thesis_dir.exists():
        for f in [fig5_5_png, fig5_5_pdf, fig5_7_png, fig5_7_pdf, fig5_8_png, fig5_8_pdf]:
            shutil.copyfile(f, thesis_dir / f.name)
            print(f"Copied to thesis: {f.name}")

if __name__ == "__main__":
    main()
