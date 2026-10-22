#!/usr/bin/env python3
"""
make_ch6_new_figures.py

Generates the four additional Chapter 6 figures recommended in the
13 October 2026 audit (audit_report_ch5_ch6_2026-10-13.md):

  1. fig_ch6_node_map_h{h}       Per-node MASE and spatial gain on an OSM basemap
  2. fig_ch6_completeness        Completeness vs MASE / spatial gain (+ neighbour completeness)
  3. fig_ch6_distance_decay      Four-panel degree-normalised attention vs distance
  4. fig_ch6_paired_gain         Paired per-node spatial gain by horizon and backbone

All metrics are test-split values averaged over seeds per node, read from
artifacts/spatial_full. Run from the repository root:

    python docs/thesis/ch6_new_figures/make_ch6_new_figures.py
"""
import glob
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "spatial_full"
OUT = Path(__file__).resolve().parent
TILE_CACHE = ROOT / "artifacts" / "cache" / "tiles_osm"

CHAMPION = "gat_lstm|gabriel"
BACKBONES = [
    # (spatial model, temporal-only twin, display label, colour)
    ("gat_lstm|gabriel", "lstm|identity", "LSTM backbone", "#1f5f8b"),
    ("gat_transformer|gabriel", "transformer|identity", "Transformer backbone", "#d9822b"),
]
HORIZONS = [1, 2, 4]
H_MIN = {1: 15, 2: 30, 4: 60}
ANNOTATE = {
    "NclFalconarStLoversLn": "Falconar St",
    "NclBarrasBridgeCivicCentre": "Barras Bridge",
}
# Map label offsets (points) chosen to keep labels clear of the city-centre cluster
MAP_OFFSETS = {"NclFalconarStLoversLn": (18, -34), "NclBarrasBridgeCivicCentre": (-95, 30)}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_node_metrics():
    """Seed-mean test MAE/MASE per (config, node, horizon)."""
    frames = [pd.read_csv(f) for f in glob.glob(str(ART / "runs" / "*" / "results.csv"))]
    res = pd.concat(frames, ignore_index=True)
    res = res[res["split"] == "test"].copy()
    res["key"] = res["model"] + "|" + res["graph_variant"]
    agg = (res.groupby(["key", "node_id", "horizon"])
              .agg(mae=("mae_raw", "mean"), mase=("mase", "mean"),
                   n_seeds=("seed", "nunique"), n_targets=("n_targets", "max"))
              .reset_index())
    return agg


def metric(pm, key, h, col="mae"):
    d = pm[(pm["key"] == key) & (pm["horizon"] == h)].set_index("node_id")[col]
    return d.dropna()


def spatial_gain(pm, spatial_key, temporal_key, h):
    """Percentage MAE reduction of the spatial model relative to its temporal twin."""
    a = metric(pm, spatial_key, h)
    b = metric(pm, temporal_key, h)
    idx = a.index.intersection(b.index)
    return (100.0 * (b[idx] - a[idx]) / b[idx]).rename("gain")


def load_coords():
    sensors = pd.DataFrame(json.load(open(ART / "traffic_sensors.json")))
    return sensors.drop_duplicates("location").set_index("location")[["lon", "lat"]]


def load_gabriel_edges():
    man = json.load(open(ART / "graphs" / "gabriel_manifest.json"))
    return [tuple(e) for e in man["edges"]]


# ----------------------------------------------------------------------------
# Basemap helpers (same tile cache and styling as scripts/eda/plot_attention_graphs.py)
# ----------------------------------------------------------------------------
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    return (int((lon_deg + 180.0) / 360.0 * n),
            int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    return lat, lon


def fetch_basemap(lon_min, lon_max, lat_min, lat_max, zoom=13):
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    x0, y0 = deg2num(lat_max, lon_min, zoom)
    x1, y1 = deg2num(lat_min, lon_max, zoom)
    img = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256), (250, 250, 250))
    for i, x in enumerate(range(x0, x1 + 1)):
        for j, y in enumerate(range(y0, y1 + 1)):
            f = TILE_CACHE / f"{zoom}_{x}_{y}.png"
            if not f.exists():
                req = urllib.request.Request(
                    f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png",
                    headers={"User-Agent": "ThesisNewcastlePedestrianAnalysis/1.0 (academic research)"})
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        f.write_bytes(r.read())
                except Exception as exc:
                    print(f"  tile {x},{y} unavailable ({exc}); leaving blank")
                    continue
            tile = Image.open(f).convert("L").convert("RGB")
            tile = Image.blend(tile, Image.new("RGB", tile.size, (255, 255, 255)), 0.65)
            img.paste(tile, (i * 256, j * 256))
    top, left = num2deg(x0, y0, zoom)
    bottom, right = num2deg(x1 + 1, y1 + 1, zoom)
    return np.array(img), [left, right, bottom, top]


def add_scale_bar(ax, lon, lat, km=1.0):
    d_lon = km / (111.32 * math.cos(math.radians(lat)))
    ax.plot([lon, lon + d_lon], [lat, lat], color="#111111", lw=2.2, zorder=10)
    for x in (lon, lon + d_lon):
        ax.plot([x, x], [lat - 0.0012, lat + 0.0012], color="#111111", lw=1.6, zorder=10)
    ax.text(lon + d_lon / 2, lat + 0.0018, f"{km:g} km", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b1b4b6", alpha=0.9, lw=0.6))


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


# ----------------------------------------------------------------------------
# Figure 1: node map
# ----------------------------------------------------------------------------
def fig_node_map(pm, coords, edges, report, h, stats_out):
    included = report[report["included"]].index
    excluded = report[~report["included"]].index
    mase = metric(pm, CHAMPION, h, "mase")
    gain = spatial_gain(pm, CHAMPION, "lstm|identity", h)
    no_test = [n for n in included if n not in mase.index]

    lons, lats = coords.loc[included, "lon"], coords.loc[included, "lat"]
    pad_lon, pad_lat = (lons.max() - lons.min()) * 0.08, (lats.max() - lats.min()) * 0.08
    xlim = (lons.min() - pad_lon, lons.max() + pad_lon)
    ylim = (lats.min() - pad_lat, lats.max() + pad_lat)
    basemap, extent = fetch_basemap(xlim[0], xlim[1], ylim[0], ylim[1])
    aspect = 1.0 / math.cos(math.radians(np.mean(ylim)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    gmax = np.ceil(np.abs(gain).max())
    panels = [
        (axes[0], mase, "viridis_r", mcolors.Normalize(mase.min(), mase.max()),
         f"(a) Test MASE, GAT-LSTM (Gabriel), $h={h}$ ({H_MIN[h]} min)", "MASE (lower is better)"),
        (axes[1], gain, "RdBu", mcolors.TwoSlopeNorm(0.0, -gmax, gmax),
         f"(b) Spatial gain over temporal-only LSTM, $h={h}$", "MAE reduction (%)  [positive = GAT better]"),
    ]
    for ax, values, cmap, norm, title, clabel in panels:
        ax.imshow(basemap, extent=extent, aspect="auto", alpha=0.9, zorder=1)
        for s, d in edges:
            if s < d and s in coords.index and d in coords.index:
                ax.plot(coords.loc[[s, d], "lon"], coords.loc[[s, d], "lat"],
                        color="#6b7280", lw=0.8, alpha=0.6, zorder=2)
        ax.scatter(coords.loc[excluded, "lon"], coords.loc[excluded, "lat"], marker="x",
                   s=22, color="#9ca3af", lw=1.0, zorder=3, label="Excluded (<30% complete)")
        ax.scatter(coords.loc[no_test, "lon"], coords.loc[no_test, "lat"], s=46,
                   facecolor="white", edgecolor="#4b5563", lw=1.1, zorder=4,
                   label="Included, no test targets")
        sc = ax.scatter(coords.loc[values.index, "lon"], coords.loc[values.index, "lat"],
                        c=values.values, cmap=cmap, norm=norm, s=62, edgecolor="#111111",
                        lw=0.6, zorder=5)
        for node, label in ANNOTATE.items():
            if node in values.index:
                ax.annotate(f"{label} ({values[node]:+.1f}%)" if values is gain else label,
                            coords.loc[node, ["lon", "lat"]].values, xytext=MAP_OFFSETS[node],
                            textcoords="offset points", fontsize=8.5, fontweight="bold", zorder=6,
                            arrowprops=dict(arrowstyle="-", color="#111111", lw=0.8),
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9ca3af", lw=0.6, alpha=0.95))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect(aspect)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontweight="bold")
        add_scale_bar(ax, xlim[0] + 0.05 * (xlim[1] - xlim[0]), ylim[0] + 0.05 * (ylim[1] - ylim[0]))
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03)
        cb.set_label(clabel)
    axes[0].legend(loc="upper left", framealpha=0.95)
    fig.text(0.5, -0.01, "Basemap © OpenStreetMap contributors. Grey lines: Gabriel graph edges.",
             ha="center", fontsize=8, color="#4b5563")
    save(fig, f"fig_ch6_node_map_h{h}")

    stats_out.append(f"### Node map (h={h})\n"
                     f"- Nodes with test targets: {len(mase)} of {len(included)} included; "
                     f"no test targets: {', '.join(sorted(no_test))}\n"
                     f"- MASE range {mase.min():.3f}–{mase.max():.3f} (median {mase.median():.3f})\n"
                     f"- Spatial gain range {gain.min():.1f}% to {gain.max():.1f}% "
                     f"(median {gain.median():.1f}%); nodes worse with GAT: "
                     f"{', '.join(f'{n} ({gain[n]:.1f}%)' for n in gain[gain < 0].sort_values().index)}\n")


# ----------------------------------------------------------------------------
# Figure 2: completeness scatter
# ----------------------------------------------------------------------------
def corr_text(x, y):
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return r, p, rho, p_rho, f"Pearson $r={r:+.2f}$ ($p={p:.3f}$)\nSpearman $\\rho={rho:+.2f}$ ($p={p_rho:.3f}$)\n$n={len(x)}$"


def partial_corr(x, y, z):
    """Pearson partial correlation of x and y controlling for z (residual method)."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r, _ = stats.pearsonr(rx, ry)
    n = len(x)
    t = r * math.sqrt((n - 3) / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), n - 3)
    return r, p


def scatter_panel(ax, x, y, colour, xlabel, ylabel, title, txt, box="upper right"):
    ax.scatter(x, y, s=34, color=colour, edgecolor="white", lw=0.5, zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color="#111111", lw=1.2, ls="--", zorder=2)
    for node, label in ANNOTATE.items():
        if node in x.index:
            ax.annotate(label, (x[node], y[node]), xytext=(6, -10), textcoords="offset points", fontsize=8)
    bx, by, ha, va = {"upper right": (0.97, 0.97, "right", "top"),
                      "lower right": (0.97, 0.03, "right", "bottom")}[box]
    ax.text(bx, by, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d1d5db", alpha=0.95))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")


def fig_completeness(pm, edges, report, stats_out, h=1):
    comp = report["completeness"] * 100
    mase = metric(pm, CHAMPION, h, "mase")
    gain = spatial_gain(pm, CHAMPION, "lstm|identity", h)
    nodes = mase.index.intersection(gain.index)

    # Mean completeness of each node's Gabriel in-neighbours
    nbrs = {}
    for s, d in edges:
        nbrs.setdefault(d, []).append(s)
    nbr_comp = pd.Series({n: comp[nbrs[n]].mean() for n in nodes if n in nbrs})
    nodes = nodes.intersection(nbr_comp.index)

    x, m, g, nc = comp[nodes], mase[nodes], gain[nodes], nbr_comp[nodes]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    r1 = corr_text(x, m)
    scatter_panel(axes[0], x, m, "#1f5f8b", "Node completeness (%)", "Test MASE",
                  f"(a) Completeness vs MASE (GAT-LSTM, $h={h}$)", r1[4])
    r2 = corr_text(x, g)
    scatter_panel(axes[1], x, g, "#2a9d8f", "Node completeness (%)", "Spatial gain (% MAE reduction)",
                  f"(b) Completeness vs spatial gain ($h={h}$)", r2[4])
    pr, pp = partial_corr(nc.values, g.values, x.values)
    r3 = corr_text(nc, g)
    scatter_panel(axes[2], nc, g, "#9b5de5", "Mean completeness of Gabriel neighbours (%)",
                  "Spatial gain (% MAE reduction)", f"(c) Neighbour completeness vs gain ($h={h}$)",
                  r3[4] + f"\npartial $r$ | own compl. $={pr:+.2f}$ ($p={pp:.3f}$)", )
    axes[2].set_ylim(top=g.max() + 6)  # headroom so the stats box clears the data
    for ax in axes[1:]:
        ax.axhline(0, color="#6b7280", lw=0.8)
    fig.tight_layout()
    save(fig, "fig_ch6_completeness")

    stats_out.append(f"### Completeness scatter (h={h}, n={len(nodes)})\n"
                     f"- (a) completeness vs MASE: Pearson r={r1[0]:+.3f} (p={r1[1]:.3f}); "
                     f"Spearman rho={r1[2]:+.3f} (p={r1[3]:.3f})\n"
                     f"- (b) completeness vs spatial gain: Pearson r={r2[0]:+.3f} (p={r2[1]:.3f}); "
                     f"Spearman rho={r2[2]:+.3f} (p={r2[3]:.3f})\n"
                     f"- (c) neighbour completeness vs gain: Pearson r={r3[0]:+.3f} (p={r3[1]:.3f}); "
                     f"partial r controlling own completeness={pr:+.3f} (p={pp:.3f})\n")


# ----------------------------------------------------------------------------
# Figure 3: four-panel distance decay
# ----------------------------------------------------------------------------
DECAY_PANELS = [
    ("gat_transformer_gabriel", "(a) Gabriel", "#1f77b4"),
    ("gat_transformer_knn8", "(b) $k$NN-8", "#6366f1"),
    ("gat_transformer_fc", "(c) Fully connected", "#0ea5e9"),
    ("gat_learned_identity", "(d) Learned adjacency", "#8b5cf6"),
]


def fmt_p(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.2f}$"


def fig_distance_decay(stats_out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    lines = ["### Distance decay (attention exports in artifacts/spatial_full/aggregated)"]
    for ax, (key, title, colour) in zip(axes.flat, DECAY_PANELS):
        df = pd.read_csv(ART / "aggregated" / f"attention_distance_{key}.csv")
        df = df[df["src"] != df["dst"]].copy()
        # Attention relative to a uniform split over the node's neighbours.
        # Dividing by the per-node mean removes both the in-degree effect
        # and the share of attention placed on the self-loop.
        df["alpha_rel"] = df["alpha"] / df.groupby("dst")["alpha"].transform("mean")
        k = df.groupby("dst").size()

        r_raw, p_raw = stats.pearsonr(df["distance_m"], df["alpha"])
        r_rel, p_rel = stats.pearsonr(df["distance_m"], df["alpha_rel"])
        rho_rel, p_rho = stats.spearmanr(df["distance_m"], df["alpha_rel"])

        ax.axhline(1.0, color="#6b7280", ls=":", lw=1.3, label="Uniform attention", zorder=2)
        ax.scatter(df["distance_m"], df["alpha_rel"], s=14 if len(df) > 1000 else 22,
                   color=colour, alpha=0.35 if len(df) > 1000 else 0.55, edgecolor="none",
                   label="Edge", zorder=3)
        df["bin"] = pd.qcut(df["distance_m"], q=8, duplicates="drop")
        b = df.groupby("bin", observed=True).agg(d=("distance_m", "median"), a=("alpha_rel", "median"))
        ax.plot(b["d"], b["a"], color="#dc2626", marker="o", ms=5, lw=2, label="Binned median", zorder=4)

        ax.set_title(title, fontweight="bold", loc="left")
        ax.text(0.98, 0.97,
                f"{len(df):,} edges, median in-degree {k.median():.1f}\n"
                f"Pearson $r={r_rel:+.2f}$ ({fmt_p(p_rel)})\nSpearman $\\rho={rho_rel:+.2f}$ ({fmt_p(p_rho)})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d1d5db", alpha=0.95))
        ax.set_xlabel("Inter-sensor distance (m)")
        ax.set_ylabel(r"Relative attention $\alpha_{ij} / \bar{\alpha}_{i}$")
        ax.set_ylim(bottom=0)
        lines.append(f"- {key}: {len(df)} edges, median in-degree {k.median():.1f}; "
                     f"raw alpha Pearson r={r_raw:+.3f} (p={p_raw:.2g}); "
                     f"relative alpha Pearson r={r_rel:+.3f} (p={p_rel:.2g}), Spearman rho={rho_rel:+.3f} (p={p_rho:.2g})")
    axes[0, 0].legend(loc="center right", framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig_ch6_distance_decay")
    stats_out.append("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Figure 4: paired per-node spatial gain
# ----------------------------------------------------------------------------
def fig_paired_gain(pm, stats_out):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rng = np.random.default_rng(0)
    width = 0.36
    lines = ["### Paired per-node spatial gain (seed-mean test MAE)"]
    for bi, (sp, tp, label, colour) in enumerate(BACKBONES):
        for hi, h in enumerate(HORIZONS):
            g = spatial_gain(pm, sp, tp, h)
            pos = hi + (bi - 0.5) * width
            ax.boxplot(g, positions=[pos], widths=width * 0.8, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=mcolors.to_rgba(colour, 0.25), edgecolor=colour),
                       medianprops=dict(color=colour, lw=2), whiskerprops=dict(color=colour),
                       capprops=dict(color=colour))
            ax.scatter(pos + rng.uniform(-width * 0.28, width * 0.28, len(g)), g, s=12,
                       color=colour, alpha=0.75, edgecolor="none", zorder=3,
                       label=label if hi == 0 else None)
            ax.text(pos, g.max() + 0.8,
                    f"{(g > 0).sum()}/{len(g)}", ha="center", va="bottom", fontsize=8, color=colour)
            lines.append(f"- {label}, h={h}: median {g.median():.1f}%, mean {g.mean():.1f}%, "
                         f"IQR {g.quantile(.25):.1f}–{g.quantile(.75):.1f}%, improved {(g > 0).sum()}/{len(g)} nodes, "
                         f"min {g.min():.1f}% ({g.idxmin()})")
    ax.axhline(0, color="#111111", lw=0.9)
    ax.set_ylim(top=ax.get_ylim()[1] + 1.5)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"$h={h}$ ({H_MIN[h]} min)" for h in HORIZONS])
    ax.set_ylabel("Spatial gain (% MAE reduction vs temporal-only twin)")
    ax.set_title("Per-node spatial gain of GAT (Gabriel) over the matched temporal-only model",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.text(0.99, 0.02, "Numbers above boxes: nodes improved / nodes evaluated",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#4b5563")
    fig.tight_layout()
    save(fig, "fig_ch6_paired_gain")
    stats_out.append("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pm = load_node_metrics()
    coords = load_coords()
    edges = load_gabriel_edges()
    report = pd.read_csv(ART / "node_report.csv").set_index("node_id")
    stats_out = ["# Chapter 6 new figures — summary statistics",
                 "",
                 "Test split; per-node metrics averaged over available seeds "
                 "(3 for trainable models, 1 for heuristics). Spatial gain = "
                 "100 × (MAE_temporal − MAE_spatial) / MAE_temporal, per node.",
                 ""]
    print("Figure 1: node maps")
    for h in (1, 4):
        fig_node_map(pm, coords, edges, report, h, stats_out)
    print("Figure 2: completeness scatter")
    fig_completeness(pm, edges, report, stats_out)
    print("Figure 3: distance decay")
    fig_distance_decay(stats_out)
    print("Figure 4: paired gain")
    fig_paired_gain(pm, stats_out)
    (OUT / "figure_stats.md").write_text("\n".join(stats_out))
    print("  wrote figure_stats.md")


if __name__ == "__main__":
    main()
