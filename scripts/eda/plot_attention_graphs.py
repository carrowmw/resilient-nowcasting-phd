#!/usr/bin/env python3
"""
scripts/eda/plot_attention_graphs.py

Generates publication-quality attention network graphs for Chapter 6:
- Desaturated OpenStreetMap basemap in background (roads, rivers, bridges)
- Physical 1 km scale bar with clear label
- Sequential edge-weight color bar (plasma)
- Top-k edge pruning for dense FC and Learned topologies (top 3 edges/node)
- Clean titles with model and seed attribution
- Syncs to workspace artifacts and thesis figure directories.
"""
import os
import json
import math
import shutil
import urllib.request
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PAPER_RC = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_osm_basemap(lon_min, lon_max, lat_min, lat_max, zoom=13, cache_dir="artifacts/cache/tiles_osm"):
    os.makedirs(cache_dir, exist_ok=True)
    x_min, y_min = deg2num(lat_max, lon_min, zoom)
    x_max, y_max = deg2num(lat_min, lon_max, zoom)
    
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1
    stitched = Image.new("RGB", (n_x * 256, n_y * 256))
    
    for i, x in enumerate(range(x_min, x_max + 1)):
        for j, y in enumerate(range(y_min, y_max + 1)):
            cache_file = os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")
            if not os.path.exists(cache_file):
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                req = urllib.request.Request(url, headers={"User-Agent": "ThesisNewcastlePedestrianAnalysis/1.0 (academic research)"})
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        with open(cache_file, "wb") as f_out:
                            f_out.write(resp.read())
                except Exception as e:
                    print(f"Failed to fetch tile {x},{y}: {e}")
                    tile = Image.new("RGB", (256, 256), (250, 250, 250))
                    stitched.paste(tile, (i * 256, j * 256))
                    continue
            raw_tile = Image.open(cache_file).convert("L").convert("RGB")
            # Blend with 65% white for a subtle, desaturated, publication-clean basemap
            white_bg = Image.new("RGB", raw_tile.size, (255, 255, 255))
            light_tile = Image.blend(raw_tile, white_bg, 0.65)
            stitched.paste(light_tile, (i * 256, j * 256))
            
    top_lat, left_lon = num2deg(x_min, y_min, zoom)
    bot_lat, right_lon = num2deg(x_max + 1, y_max + 1, zoom)
    return np.array(stitched), [left_lon, right_lon, bot_lat, top_lat]

def add_scale_bar(ax, lon_start, lat_start, length_km=1.0):
    """Draws an accurate scale bar in kilometers with a clear label."""
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_start))
    d_lon = length_km / km_per_deg_lon
    tick_h = 0.0016
    
    # Scale bar horizontal line
    ax.plot([lon_start, lon_start + d_lon], [lat_start, lat_start], color="#111111", linewidth=2.4, zorder=10)
    # End vertical ticks
    ax.plot([lon_start, lon_start], [lat_start - tick_h, lat_start + tick_h], color="#111111", linewidth=1.8, zorder=10)
    ax.plot([lon_start + d_lon, lon_start + d_lon], [lat_start - tick_h, lat_start + tick_h], color="#111111", linewidth=1.8, zorder=10)
    # Label positioned cleanly above the bar
    ax.text(lon_start + d_lon / 2.0, lat_start + tick_h * 1.5, f"{int(length_km) if length_km == int(length_km) else length_km} km",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#111111", zorder=10,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ffffff", edgecolor="#b1b4b6", alpha=0.90, linewidth=0.6))

def plot_graph(edges_df, coords_map, out_paths, title, subtitle, basemap_img, basemap_extent, top_k=None):
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(8.2, 8.0), dpi=300)
    
    # Filter self-loops
    df = edges_df[edges_df["src"] != edges_df["dst"]].copy()
    
    # Filter to top-k edges per destination node if requested
    if top_k is not None:
        df = df.sort_values(["dst", "alpha"], ascending=[True, False])
        df = df.groupby("dst").head(top_k).copy()
    
    # Render basemap
    ax.imshow(basemap_img, extent=basemap_extent, aspect="auto", alpha=0.90, zorder=1)
    
    # Color map for edges
    cmap = cm.plasma
    v_min = df["alpha"].min() if len(df) > 0 else 0.0
    v_max = df["alpha"].max() if len(df) > 0 else 1.0
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    
    # Sort edges so highest attention lines are drawn on top
    df = df.sort_values("alpha")
    
    for _, row in df.iterrows():
        s, d, a = row["src"], row["dst"], row["alpha"]
        if s not in coords_map or d not in coords_map:
            continue
        lon1, lat1 = coords_map[s]
        lon2, lat2 = coords_map[d]
        
        ratio = (a - v_min) / (v_max - v_min + 1e-8)
        lw = 1.0 + 3.2 * ratio
        line_alpha = 0.40 + 0.58 * ratio
        c = cmap(norm(a))
        
        ax.plot([lon1, lon2], [lat1, lat2], color=c, linewidth=lw, alpha=line_alpha, zorder=3)
    
    # Draw sensor nodes
    node_lons = [coords_map[n][0] for n in coords_map]
    node_lats = [coords_map[n][1] for n in coords_map]
    ax.scatter(node_lons, node_lats, s=38, facecolor="#0b0c0c", edgecolor="white", linewidth=0.9, zorder=5)
    
    # Set view bounds with padding
    lon_min, lon_max = min(node_lons), max(node_lons)
    lat_min, lat_max = min(node_lats), max(node_lats)
    pad_lon = (lon_max - lon_min) * 0.08
    pad_lat = (lat_max - lat_min) * 0.08
    
    x_lim = (lon_min - pad_lon, lon_max + pad_lon)
    y_lim = (lat_min - pad_lat, lat_max + pad_lat)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    # Scale bar in bottom-left
    scale_lon = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.05
    scale_lat = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05
    add_scale_bar(ax, scale_lon, scale_lat, length_km=1.0)
    
    # Add Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.035, pad=0.03, aspect=24)
    cbar.set_label("Learned Attention Weight ($\\alpha_{ij}$)", fontsize=10.5, weight="bold")
    cbar.ax.tick_params(labelsize=9)
    
    # Title & Subtitle with proper vertical separation
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=20, color="#0b0c0c")
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#505a5f")
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#b1b4b6")
        spine.set_linewidth(1.0)
        
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)

def main():
    spatial_dir = Path("artifacts/spatial_full")
    dest_dir = Path("latex-documents/thesis/figures/experiment2")
    src_fig_dir = spatial_dir / "aggregated" / "figures"
    art_fig_dir = Path("artifacts/figures")
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_fig_dir.mkdir(parents=True, exist_ok=True)
    art_fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading sensor locations...")
    with open("artifacts/spatial_full/traffic_sensors.json") as f:
        sensors = json.load(f)
    coords_map = {s["location"]: (s["lon"], s["lat"]) for s in sensors}
    
    with open("artifacts/spatial_full/context_manifest.json") as f:
        cm_data = json.load(f)
    node_ids = cm_data["node_ids"]
    coords_54 = {n: coords_map[n] for n in node_ids if n in coords_map}
    print(f"54 active nodes mapped: {len(coords_54)} coordinates.")
    
    # Compute bounds and fetch clean OSM basemap
    lons = [coords_54[n][0] for n in coords_54]
    lats = [coords_54[n][1] for n in coords_54]
    b_img, b_ext = fetch_osm_basemap(min(lons) - 0.02, max(lons) + 0.02, min(lats) - 0.02, max(lats) + 0.02, zoom=13)
    print("OSM basemap ready:", b_img.shape)
    
    configs = [
        {
            "name": "gabriel",
            "file": "attention_distance_gat_transformer_gabriel.csv",
            "out_name": "attention_graph_gat_transformer_gabriel.png",
            "title": "Gabriel Planar Graph (156 Edges)",
            "subtitle": "GAT-Transformer · Seed 42",
            "top_k": None,
        },
        {
            "name": "knn8",
            "file": "attention_distance_gat_transformer_knn8.csv",
            "out_name": "attention_graph_gat_transformer_knn8.png",
            "title": "kNN-8 Metric Proximity Graph (544 Edges)",
            "subtitle": "GAT-Transformer · Seed 42",
            "top_k": None,
        },
        {
            "name": "fc",
            "file": "attention_distance_gat_transformer_fc.csv",
            "out_name": "attention_graph_gat_transformer_fc.png",
            "title": "Fully-Connected Topology (Top-3 In-Edges / Node)",
            "subtitle": "GAT-Transformer · Seed 42 (Pruned from 2,862 Edges)",
            "top_k": 3,
        },
        {
            "name": "learned",
            "file": "attention_distance_gat_learned_identity.csv",
            "out_name": "attention_graph_gat_learned_identity.png",
            "title": "Learned Adjacency (Top-3 In-Edges / Node)",
            "subtitle": "GAT-Learned · Seed 42 (Pruned from ~1,756 Edges)",
            "top_k": 3,
        },
        {
            "name": "gabriel_plain",
            "file": "attention_distance_gat_transformer_gabriel_plain.csv",
            "out_name": "attention_graph_gat_transformer_gabriel_plain.png",
            "title": "Gabriel Planar Graph (Unconditioned, 156 Edges)",
            "subtitle": "GAT-Transformer (Plain) · Seed 42",
            "top_k": None,
        },
    ]
    
    for cfg in configs:
        csv_p = spatial_dir / "aggregated" / cfg["file"]
        if not csv_p.exists():
            print(f"Skipping {cfg['name']}, file not found: {csv_p}")
            continue
        df = pd.read_csv(csv_p)
        targets = [
            src_fig_dir / cfg["out_name"],
            art_fig_dir / cfg["out_name"],
            dest_dir / cfg["out_name"]
        ]
        plot_graph(
            df, coords_54, targets,
            cfg["title"], cfg["subtitle"],
            b_img, b_ext,
            top_k=cfg["top_k"]
        )
        
    print("All attention network graphs generated and synced successfully!")

if __name__ == "__main__":
    main()
