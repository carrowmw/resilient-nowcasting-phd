#!/usr/bin/env python3
"""
compare_runs.py — Cross-run spatial interpolation quality analysis.

Fetches R2-hosted inference payloads for multiple model runs and computes
spatial interpolation quality metrics, focusing on simulated node predictions
as a proxy for spatial generalization capability.

Uses only stdlib (urllib, json, math) — no pip install required.

Usage:
    python scripts/compare_runs.py dark-brook-37 radiant-voice-41 crisp-puddle-47
    python scripts/compare_runs.py --all               # compare all runs in registry
    python scripts/compare_runs.py --date 2026-05-15 dark-brook-37 crisp-puddle-47
"""

import argparse
import gzip
import json
import logging
import math
import sys
import urllib.request
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("compare_runs")

R2_BASE = "https://pub-b102fa91750348f78a6385248654688a.r2.dev/runs"


# ---------------------------------------------------------------------------
# Fetching helpers (stdlib only, handles gzip Content-Encoding from R2)
# ---------------------------------------------------------------------------

def fetch_json(url: str) -> Optional[Dict]:
    """Fetch and decode a JSON resource from R2. Returns None on 404/error.

    R2 objects are stored with ContentEncoding=gzip, so the raw response bytes
    may be gzip-compressed. We try JSON first, then gzip-decompress on failure.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "compare-runs/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            # Try plain JSON first
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            # Likely gzip-compressed
            try:
                decompressed = gzip.decompress(raw)
                return json.loads(decompressed)
            except Exception:
                logger.error(f"  Cannot decode response from {url}")
                return None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.warning(f"  404: {url}")
            return None
        raise
    except Exception as e:
        logger.error(f"  Failed to fetch {url}: {e}")
        return None


def fetch_registry() -> Dict:
    url = f"{R2_BASE}/run_registry.json"
    logger.info(f"Fetching registry: {url}")
    data = fetch_json(url)
    assert data is not None, f"CRITICAL: Cannot fetch run registry from {url}"
    return data


def fetch_spatial(run_id: str) -> Optional[Dict]:
    url = f"{R2_BASE}/{run_id}/spatial.json"
    logger.info(f"  Fetching spatial: {run_id}")
    return fetch_json(url)


def fetch_page(run_id: str, date: str) -> Optional[Dict]:
    url = f"{R2_BASE}/{run_id}/{date}/page.json"
    logger.info(f"  Fetching page: {run_id}/{date}")
    return fetch_json(url)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def extract_simulated_node_ids(spatial: Dict) -> List[int]:
    """Return node IDs marked as simulated in spatial.json."""
    return [n["id"] for n in spatial["nodes"] if n.get("is_simulated", False)]


def extract_real_node_ids(spatial: Dict) -> List[int]:
    """Return node IDs for real (non-simulated) sensors."""
    return [n["id"] for n in spatial["nodes"] if not n.get("is_simulated", False)]


def node_coords(spatial: Dict) -> Dict[int, Tuple[float, float]]:
    """Map node_id → (lat, lon) for all nodes."""
    return {n["id"]: (n["lat"], n["lon"]) for n in spatial["nodes"]}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_daily_mu(
    page: Dict, node_ids: List[int]
) -> Dict[int, List[float]]:
    """Extract per-node μ timeseries for specified node IDs from a page.json."""
    ts = page.get("node_timeseries", {})
    out: Dict[int, List[float]] = {}
    for nid in node_ids:
        key = str(nid)
        if key in ts:
            out[nid] = [step["prediction"] for step in ts[key]]
    return out


def daily_mean_mu(mu_series: Dict[int, List[float]]) -> Dict[int, float]:
    """Mean μ across all timesteps for each node."""
    return {nid: sum(vals) / len(vals) for nid, vals in mu_series.items() if vals}


def compute_spatial_metrics(
    spatial: Dict,
    page: Dict,
) -> Dict[str, Any]:
    """
    Compute spatial interpolation quality metrics for a single run/date.

    Returns a dict with:
      - sim_mu_variance: variance of daily-mean μ across simulated nodes
      - sim_mu_std: std of daily-mean μ across simulated nodes
      - sim_mu_mean: mean of daily-mean μ across simulated nodes
      - sim_dynamic_range: max/min ratio of daily-mean μ (clamped to avoid div-by-zero)
      - sim_cv: coefficient of variation (std/mean) — normalised diversity
      - real_mu_mean: mean of daily-mean μ across real nodes (for comparison)
      - real_sim_corr: Pearson correlation between spatial proximity and μ similarity
      - spatial_autocorr: simplified spatial autocorrelation (Moran's I proxy)
      - num_sim_nodes: count of simulated nodes
      - num_near_zero_sim: count of simulated nodes with mean μ < 1.0
      - zero_collapse_ratio: fraction of simulated nodes collapsed to near-zero
    """
    sim_ids = extract_simulated_node_ids(spatial)
    real_ids = extract_real_node_ids(spatial)
    coords = node_coords(spatial)

    sim_mu = compute_daily_mu(page, sim_ids)
    real_mu = compute_daily_mu(page, real_ids)

    sim_means = daily_mean_mu(sim_mu)
    real_means = daily_mean_mu(real_mu)

    sim_vals = list(sim_means.values())
    real_vals = list(real_means.values())

    if not sim_vals:
        return {"error": "No simulated nodes found in page data"}

    n = len(sim_vals)
    mean_sim = sum(sim_vals) / n
    var_sim = sum((v - mean_sim) ** 2 for v in sim_vals) / n
    std_sim = math.sqrt(var_sim)

    mean_real = sum(real_vals) / len(real_vals) if real_vals else 0.0

    # Dynamic range: max / min (clamped at 0.01 to avoid inf)
    min_mu = max(min(sim_vals), 0.01)
    max_mu = max(sim_vals)
    dynamic_range = max_mu / min_mu

    # Coefficient of variation (normalized diversity measure)
    cv = std_sim / mean_sim if mean_sim > 0.01 else 0.0

    # Near-zero collapse count
    near_zero_threshold = 1.0
    num_near_zero = sum(1 for v in sim_vals if v < near_zero_threshold)
    zero_collapse_ratio = num_near_zero / n

    # -------------------------------------------------------------------
    # Spatial autocorrelation (simplified Moran's I on all nodes)
    # Higher I → predictions correlate with spatial proximity → good.
    # -------------------------------------------------------------------
    all_means = {**sim_means, **real_means}
    all_ids = list(all_means.keys())
    all_vals_list = [all_means[nid] for nid in all_ids]
    grand_mean = sum(all_vals_list) / len(all_vals_list) if all_vals_list else 0.0

    # Build inverse-distance weight matrix (k-nearest approach: top 5 neighbours)
    K_NEIGHBOURS = 5
    morans_num = 0.0
    morans_denom = sum((v - grand_mean) ** 2 for v in all_vals_list)
    total_w = 0.0

    if morans_denom > 0 and len(all_ids) > 1:
        for i, nid_i in enumerate(all_ids):
            lat_i, lon_i = coords.get(nid_i, (0, 0))
            dists = []
            for j, nid_j in enumerate(all_ids):
                if i == j:
                    continue
                lat_j, lon_j = coords.get(nid_j, (0, 0))
                d = haversine_km(lat_i, lon_i, lat_j, lon_j)
                dists.append((d, j))
            dists.sort()
            for d, j in dists[:K_NEIGHBOURS]:
                w = 1.0 / max(d, 0.01)
                morans_num += w * (all_vals_list[i] - grand_mean) * (
                    all_vals_list[j] - grand_mean
                )
                total_w += w

        N = len(all_ids)
        morans_i = (N / total_w) * (morans_num / morans_denom) if total_w > 0 else 0.0
    else:
        morans_i = 0.0

    # -------------------------------------------------------------------
    # Nearest-real-node correlation for simulated nodes
    # For each simulated node, find the nearest real node and compare μ values.
    # -------------------------------------------------------------------
    sim_nearest_pairs: List[Tuple[float, float]] = []
    for sid in sim_ids:
        if sid not in sim_means or sid not in coords:
            continue
        s_lat, s_lon = coords[sid]
        best_dist = float("inf")
        best_real_mu = 0.0
        for rid in real_ids:
            if rid not in real_means or rid not in coords:
                continue
            r_lat, r_lon = coords[rid]
            d = haversine_km(s_lat, s_lon, r_lat, r_lon)
            if d < best_dist:
                best_dist = d
                best_real_mu = real_means[rid]
        sim_nearest_pairs.append((sim_means[sid], best_real_mu))

    # Pearson correlation between sim μ and nearest-real μ
    if len(sim_nearest_pairs) > 2:
        xs = [p[0] for p in sim_nearest_pairs]
        ys = [p[1] for p in sim_nearest_pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        real_sim_corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0.0
    else:
        real_sim_corr = None

    return {
        "sim_mu_mean": round(mean_sim, 4),
        "sim_mu_std": round(std_sim, 4),
        "sim_mu_variance": round(var_sim, 4),
        "sim_dynamic_range": round(dynamic_range, 2),
        "sim_cv": round(cv, 4),
        "real_mu_mean": round(mean_real, 4),
        "num_sim_nodes": n,
        "num_near_zero_sim": num_near_zero,
        "zero_collapse_ratio": round(zero_collapse_ratio, 4),
        "spatial_autocorr_morans_i": round(morans_i, 4),
        "real_sim_nearest_corr": round(real_sim_corr, 4) if real_sim_corr is not None else None,
    }


# ---------------------------------------------------------------------------
# Config diff extraction
# ---------------------------------------------------------------------------

def extract_arch_summary(registry_entry: Dict) -> Dict[str, Any]:
    """Pull the key architectural parameters from a registry train_config."""
    cfg = registry_entry.get("train_config", {})
    model = cfg.get("model", {})
    loss = cfg.get("loss", {})
    data = cfg.get("data", {})
    training = cfg.get("training", {})
    return {
        "hidden_dim": model.get("hidden_dim"),
        "num_layers": model.get("num_layers"),
        "embedding_dim": model.get("embedding_dim"),
        "num_heads": model.get("num_heads"),
        "batch_size": training.get("batch_size"),
        "w_kriging": loss.get("w_kriging"),
        "w_impute": loss.get("w_impute"),
        "eval_mode": cfg.get("eval_mode", "temporal"),
        "inductive_holdout_frac": data.get("inductive_holdout_frac", "n/a"),
        "spatial_holdout_prob": data.get("spatial_holdout_prob"),
        "data_range": f"{data.get('start_date', '?')} → {data.get('end_date', '?')}",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[Dict], run_ids: List[str]) -> None:
    """Print a formatted comparison table to stdout."""
    # Header
    col_w = 22
    header_cols = ["Metric"] + run_ids
    header = " | ".join(f"{c:>{col_w}}" for c in header_cols)
    print("\n" + "=" * len(header))
    print("SPATIAL INTERPOLATION QUALITY COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Spatial metrics rows
    metric_keys = [
        ("sim_mu_mean", "Sim μ Mean"),
        ("sim_mu_std", "Sim μ Std Dev"),
        ("sim_mu_variance", "Sim μ Variance"),
        ("sim_dynamic_range", "Sim Dynamic Range"),
        ("sim_cv", "Sim Coeff of Var"),
        ("real_mu_mean", "Real μ Mean"),
        ("num_sim_nodes", "Num Sim Nodes"),
        ("num_near_zero_sim", "Near-Zero Sim Nodes"),
        ("zero_collapse_ratio", "Zero Collapse Ratio"),
        ("spatial_autocorr_morans_i", "Moran's I (autocorr)"),
        ("real_sim_nearest_corr", "Real-Sim Nearest Corr"),
    ]

    for key, label in metric_keys:
        row = [f"{label:>{col_w}}"]
        for r in results:
            metrics = r.get("spatial_metrics", {})
            val = metrics.get(key)
            if val is None:
                row.append(f"{'N/A':>{col_w}}")
            elif isinstance(val, int):
                row.append(f"{val:>{col_w}}")
            else:
                row.append(f"{val:>{col_w}.4f}")
        print(" | ".join(row))

    print("-" * len(header))

    # Architecture summary rows
    arch_keys = [
        ("hidden_dim", "Hidden Dim"),
        ("num_layers", "Num Layers"),
        ("embedding_dim", "Embedding Dim"),
        ("batch_size", "Batch Size"),
        ("w_kriging", "w_kriging"),
        ("w_impute", "w_impute"),
        ("eval_mode", "Eval Mode"),
        ("inductive_holdout_frac", "Inductive Holdout"),
        ("spatial_holdout_prob", "Spatial Holdout"),
        ("data_range", "Data Range"),
    ]

    print(f"\n{'ARCHITECTURE COMPARISON':>{col_w}}")
    print("-" * len(header))
    for key, label in arch_keys:
        row = [f"{label:>{col_w}}"]
        for r in results:
            arch = r.get("arch_summary", {})
            val = arch.get(key, "N/A")
            row.append(f"{str(val):>{col_w}}")
        print(" | ".join(row))

    print("-" * len(header))

    # Registry metrics
    print(f"\n{'TRAINING METRICS (from registry)':>{col_w}}")
    print("-" * len(header))
    reg_keys = [
        ("val_rmse", "Val RMSE"),
        ("val_mae", "Val MAE"),
        ("val_zero_acc", "Val Zero Acc"),
    ]
    for key, label in reg_keys:
        row = [f"{label:>{col_w}}"]
        for r in results:
            ms = r.get("registry_metrics", {})
            val = ms.get(key)
            if val is None:
                row.append(f"{'N/A':>{col_w}}")
            else:
                row.append(f"{val:>{col_w}.4f}")
        print(" | ".join(row))

    # Number of nodes
    row_nodes = [f"{'Total Nodes':>{col_w}}"]
    row_sim = [f"{'Simulated Nodes':>{col_w}}"]
    for r in results:
        row_nodes.append(f"{r.get('num_nodes', 'N/A'):>{col_w}}")
        row_sim.append(f"{r.get('num_simulated_nodes', 'N/A'):>{col_w}}")
    print(" | ".join(row_nodes))
    print(" | ".join(row_sim))

    print("=" * len(header))


def print_diagnosis(results: List[Dict], run_ids: List[str]) -> None:
    """Print diagnostic insights based on the computed metrics."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    # Find best/worst by key spatial metrics
    best_cv = None
    best_cv_run = None
    best_morans = None
    best_morans_run = None
    worst_collapse = None
    worst_collapse_run = None

    for r, rid in zip(results, run_ids):
        m = r.get("spatial_metrics", {})
        cv = m.get("sim_cv")
        mi = m.get("spatial_autocorr_morans_i")
        zc = m.get("zero_collapse_ratio")

        if cv is not None and (best_cv is None or cv > best_cv):
            best_cv = cv
            best_cv_run = rid
        if mi is not None and (best_morans is None or mi > best_morans):
            best_morans = mi
            best_morans_run = rid
        if zc is not None and (worst_collapse is None or zc > worst_collapse):
            worst_collapse = zc
            worst_collapse_run = rid

    print(f"\n  Best spatial diversity (CV):     {best_cv_run} (CV = {best_cv:.4f})")
    print(f"  Best spatial autocorrelation:    {best_morans_run} (Moran's I = {best_morans:.4f})")
    print(f"  Worst zero-collapse:            {worst_collapse_run} (ratio = {worst_collapse:.4f})")

    # Architectural insights
    print("\n  KEY ARCHITECTURAL DIFFERENCES:")
    for r, rid in zip(results, run_ids):
        arch = r.get("arch_summary", {})
        m = r.get("spatial_metrics", {})
        params = f"h={arch.get('hidden_dim')}, L={arch.get('num_layers')}, e={arch.get('embedding_dim')}"
        sim_n = r.get("num_simulated_nodes", "?")
        cv = m.get("sim_cv", 0)
        print(f"    {rid}: {params}, bs={arch.get('batch_size')}, sim_nodes={sim_n}, CV={cv:.3f}")

    print("\n  CONCLUSION:")
    if best_cv_run:
        best_r = next(r for r, rid in zip(results, run_ids) if rid == best_cv_run)
        arch = best_r.get("arch_summary", {})
        print(f"    → {best_cv_run} produces the most spatially diverse simulated-node")
        print(f"      predictions (arch: h={arch.get('hidden_dim')}, L={arch.get('num_layers')})")
        print(f"      This is the architecture most capable of spatial interpolation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare spatial interpolation quality across model runs."
    )
    parser.add_argument(
        "run_ids",
        nargs="*",
        help="Run IDs to compare (e.g. dark-brook-37 crisp-puddle-47).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all runs in the registry.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to analyse (YYYY-MM-DD). Defaults to 2026-05-15.",
    )
    args = parser.parse_args()

    # Fetch registry
    registry = fetch_registry()
    registry_by_id = {r["run_id"]: r for r in registry["runs"]}

    if args.all:
        run_ids = list(registry_by_id.keys())
    elif args.run_ids:
        run_ids = args.run_ids
    else:
        parser.error("Provide run IDs or use --all.")

    # Validate run IDs against registry
    for rid in run_ids:
        if rid not in registry_by_id:
            logger.error(f"Run '{rid}' not found in registry. Available: {list(registry_by_id.keys())}")
            sys.exit(1)

    analysis_date = args.date or "2026-05-15"
    logger.info(f"Comparing {len(run_ids)} runs on date {analysis_date}")

    results: List[Dict] = []
    for rid in run_ids:
        logger.info(f"\nProcessing: {rid}")
        reg = registry_by_id[rid]

        # Check the date is available
        available = reg.get("available_dates", [])
        date_to_use = analysis_date
        if analysis_date not in available:
            if available:
                date_to_use = available[len(available) // 2]  # pick middle date
                logger.warning(
                    f"  {analysis_date} not available for {rid}; using {date_to_use}"
                )
            else:
                logger.error(f"  No dates available for {rid}")
                continue

        spatial = fetch_spatial(rid)
        page = fetch_page(rid, date_to_use)

        if spatial is None or page is None:
            logger.error(f"  Could not fetch data for {rid}")
            results.append({
                "run_id": rid,
                "error": "fetch_failed",
            })
            continue

        metrics = compute_spatial_metrics(spatial, page)
        arch = extract_arch_summary(reg)

        results.append({
            "run_id": rid,
            "date_analysed": date_to_use,
            "spatial_metrics": metrics,
            "arch_summary": arch,
            "registry_metrics": reg.get("metrics_summary", {}),
            "num_nodes": reg.get("num_nodes"),
            "num_simulated_nodes": reg.get("num_simulated_nodes"),
        })

    if not results:
        logger.error("No runs could be analysed.")
        sys.exit(1)

    valid_results = [r for r in results if "error" not in r]
    valid_ids = [r["run_id"] for r in valid_results]

    print_comparison_table(valid_results, valid_ids)
    print_diagnosis(valid_results, valid_ids)


if __name__ == "__main__":
    main()
