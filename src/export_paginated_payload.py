"""
export_paginated_payload.py

Multi-run, paginated payload generator for the Geospatial Digital Twin (Phase 1).

Refactor of `export_ui_payload.py` that emits the multi-run / per-day layout:

    frontend/public/data/runs/
    ├── run_registry.json                 (built separately by build_run_registry.py)
    └── {run_id}/                          e.g. feasible-capybara-22
        ├── spatial.json                   nodes + edge topology + run metadata (NO timeseries)
        └── {YYYY-MM-DD}/
            ├── page.json                  96-step timeseries + edge attention + day metrics
            └── nodes/node_{id}.json        per-node deep analytics (day-scoped)

The heavy lifting (model loading, ZINB extraction, attention extraction, calibration
metrics, ego-graph construction, semantic tags) is *imported* from `export_ui_payload`
rather than duplicated, so the two scripts cannot drift.

Design notes
------------
* For each calendar day D in the analysis window we load a 2-day dataset window
  [D - 1 day, D + 1 day) so the model has seq_len look-back context for the first
  intervals of D. We then run inference only on the windows whose *target timestamp*
  falls on D (explicit date grouping — more robust than the half/skip heuristic in
  the legacy script).
* The node set, static features and graph topology are fixed ONCE per run (from the
  first day's dataset) and reused for every day, so the per-run spatial.json is stable.
  Each day asserts its location set matches the canonical set.
* Memory is bounded: only a 2-day window is resident at a time (the parquet read uses
  pushdown date filters), and the dataset object is released between days.

Usage:
    python src/export_paginated_payload.py \
        --artifact entity/resilient-iot/resilient-iot-model:latest \
        [--config configs/analysis/analysis.yaml] \
        [--start_date 2026-05-01] [--end_date 2026-06-01] \
        [--output_dir frontend/public/data/runs] \
        [--all_artifacts] [--dry_run] \
        [--upload --wandb_entity <entity> --wandb_project riot-smoke]
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
import yaml

# Allow direct execution from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

# Importing export_ui_payload also runs its module-level .env loader + logging setup
# (idempotent). We reuse its helpers; we never call its main()/parse_args().
from export_ui_payload import (  # noqa: E402
    load_sensor_coords,
    select_artifact_from_wandb,
    load_model_from_wandb,
    build_model_and_adapter,
    compute_std_dev,

    extract_timestep_attention,
    load_semantic_tags,
    compute_pit_histogram,
    compute_zero_inflation_reliability,
    compute_two_extremes,
    build_ego_graph_edges,
    SENSORS_PATH,
)
from simulated_nodes import SimulatedNodeInjector
from dataset import RealIoTDataset, apply_spatial_holdout  # noqa: E402
from db_connector import create_connector, DataConnector  # noqa: E402
from harness import (  # noqa: E402
    ModelAdapter,
    load_holdout_manifest,
    manifest_sha,
    mask_from_manifest,
    validate_predictions,
)
from pipeline import ResilientIoTPipeline  # noqa: E402

logger = logging.getLogger("export_paginated_payload")

DEFAULT_CONFIG_PATH = "configs/analysis/analysis.yaml"
DEFAULT_OUTPUT_DIR = "frontend/public/data/runs"

# W&B collection coordinates for --all_artifacts discovery (DEC-3).
WANDB_PROJECT = "resilient-iot-v2"
WANDB_COLLECTION = "resilient-iot-model-v2"

# Candidate validation-metric keys to lift from the producer run's W&B summary.
METRIC_KEYS = ("val_rmse", "val_mae", "val_zero_acc", "val_loss", "rmse", "mae")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate multi-run paginated UI payloads for the Digital Twin."
    )
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                   help="Analysis-window YAML (default: configs/analysis/analysis.yaml).")
    p.add_argument("--start_date", type=str, default=None,
                   help="First analysis day, inclusive (YYYY-MM-DD). Overrides config.")
    p.add_argument("--end_date", type=str, default=None,
                   help="Last analysis day, exclusive (YYYY-MM-DD). Overrides config.")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Root output directory for paginated payloads.")
    p.add_argument("--artifact", type=str, default=None,
                   help="Single W&B artifact ref (entity/project/name:version).")
    p.add_argument("--all_artifacts", action="store_true",
                   help="Generate payloads for every artifact in the model collection.")
    p.add_argument("--veh_class", type=str, default=None, help="Vehicle class (default person).")
    p.add_argument("--dir", type=str, default=None, help="Direction filter (default all).")
    p.add_argument("--seq_len", type=int, default=None, help="Sequence length (default 12).")
    p.add_argument("--num_simulated", type=int, default=None,
                   help="Number of synthetic zero-shot nodes (default 5).")
    p.add_argument("--dry_run", action="store_true",
                   help="Resolve artifacts + date range, print, and exit (no inference).")
    p.add_argument("--upload", action="store_true",
                   help="Also upload each generated run tree to W&B as an artifact.")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity for --upload (defaults to the logged-in default entity).")
    p.add_argument("--wandb_project", type=str, default="riot-smoke",
                   help="W&B project for --upload (default riot-smoke for isolation).")
    p.add_argument("--query_project", type=str, default=WANDB_PROJECT,
                   help="W&B project to query for model artifacts.")
    p.add_argument("--query_collection", type=str, default=WANDB_COLLECTION,
                   help="W&B collection to query for model artifacts.")
    p.add_argument("--data_source", type=str, default=None, choices=["database", "parquet"],
                   help="Data source: database or parquet. Overrides config.")
    p.add_argument("--parquet_path", type=str, default=None,
                   help="Path to Parquet file if data_source='parquet'. Overrides config.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading (D-1 reconciliation)
# ---------------------------------------------------------------------------

def load_analysis_config(path: str, args: argparse.Namespace) -> Dict:
    """
    Resolve the analysis window from (in priority order): CLI flags > config file >
    hardcoded fallbacks.

    The user's config (configs/analysis/analysis.yaml) uses flat keys:
        holdout_start: 2026-05-01
        holdout_end:   2026-06-01
    We also accept the Architect's nested form for forward-compatibility:
        analysis: {start_date, end_date, veh_class, dir, seq_len, num_simulated}
    """
    cfg: Dict = {}
    if path and os.path.exists(path):
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.info(f"Loaded analysis config from {path}: {cfg}")
    else:
        raise FileNotFoundError(
            f"CRITICAL: analysis config not found at {path!r}. "
            "Pass --config with the correct path."
        )

    nested = cfg.get("analysis", {}) if isinstance(cfg, dict) else {}

    def pick(cli_val, *keys, default):
        if cli_val is not None:
            return cli_val
        for src in (cfg, nested):
            for k in keys:
                if isinstance(src, dict) and src.get(k) is not None:
                    return src[k]
        return default

    start = pick(args.start_date, "holdout_start", "start_date", default=None)
    end = pick(args.end_date, "holdout_end", "end_date", default=None)
    if start is None:
        raise ValueError("CRITICAL: start_date is required — set it via --start_date or config holdout_start.")
    if end is None:
        raise ValueError("CRITICAL: end_date is required — set it via --end_date or config holdout_end.")

    resolved = {
        "start_date": str(start),
        "end_date": str(end),
        "veh_class": pick(args.veh_class, "veh_class", default="person"),
        "dir": pick(args.dir, "dir", default="all"),
        "seq_len": int(pick(args.seq_len, "seq_len", default=12)),
        "num_simulated": int(pick(args.num_simulated, "num_simulated", default=5)),
        "data_source": pick(args.data_source, "data_source", default=None),
        "parquet_path": pick(args.parquet_path, "parquet_path", default=None),
        "export": pick(None, "export", default={}),
    }
    # Tripwire: window must be non-empty and well-ordered.
    assert dt.date.fromisoformat(resolved["start_date"]) < dt.date.fromisoformat(resolved["end_date"]), (
        f"TRIPWIRE: start_date {resolved['start_date']} must precede "
        f"end_date {resolved['end_date']} (exclusive)."
    )
    logger.info(f"Resolved analysis config: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Artifact resolution
# ---------------------------------------------------------------------------

def resolve_artifacts(args: argparse.Namespace) -> List[str]:
    """Return the list of fully-qualified W&B artifact refs to process."""
    if args.artifact:
        return [args.artifact]
    if args.all_artifacts:
        api = wandb.Api()
        entity = api.default_entity
        collection_path = f"{entity}/{args.query_project}/{args.query_collection}"
        logger.info(f"Discovering all artifacts in collection {collection_path} …")
        collection = api.artifact_collection("model", collection_path)
        refs = []
        for art in collection.artifacts():
            version = art.name.split(":")[-1]
            refs.append(f"{collection_path}:{version}")
        if not refs:
            raise ValueError(f"No artifacts found in collection {collection_path}")
        logger.info(f"Resolved {len(refs)} artifacts: {refs}")
        return refs
    # Interactive single selection (mirrors export_ui_payload behaviour).
    return [select_artifact_from_wandb(project=args.query_project, collection_name=args.query_collection)]


def load_model_and_meta(artifact_name: str) -> Tuple[str, str, str, str, Dict]:
    """
    Download the artifact and resolve its filesystem run_id + W&B run metadata.

    Returns (checkpoint_path, config_path, wandb_run_id, run_name, metrics_summary).
    run_name is the producer run's human name (e.g. feasible-capybara-22) and becomes
    the on-disk run directory; falls back to the wandb_run_id when unavailable.
    """
    checkpoint_path, config_path, wandb_run_id = load_model_from_wandb(artifact_name)

    run_name = wandb_run_id
    api = wandb.Api()
    producer_run = api.artifact(artifact_name).logged_by()
    assert producer_run is not None, f"FATAL: Artifact {artifact_name} has no producer run. Manually uploaded artifacts are strictly forbidden."

    run_name = producer_run.name or wandb_run_id
    metrics_summary: Dict = {}

    summary = dict(producer_run.summary) if producer_run.summary else {}
    for k in METRIC_KEYS:
        if k in summary and isinstance(summary[k], (int, float)):
            metrics_summary[k] = round(float(summary[k]), 6)

    run_id = sanitize_run_id(run_name)
    return checkpoint_path, config_path, wandb_run_id, run_id, metrics_summary


def sanitize_run_id(name: str) -> str:
    """Filesystem-safe run id (keep wandb-style names like 'feasible-capybara-22')."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(name)).strip("-")
    return safe or "unknown-run"


# ---------------------------------------------------------------------------
# Model construction (port of export_ui_payload.main lines 810-875)
# ---------------------------------------------------------------------------

def build_model(
    checkpoint_path: str,
    config_path: str,
    *,
    num_nodes_total: int,
    static_feat_dim: int,
    time_series_dim: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[ModelAdapter, Dict, str]:
    """Resolve model_type from the co-located config and build a ModelAdapter
    (GNN or recurrent baseline) via the shared builder, so this exporter serves
    any model family through the Predictions contract."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"CRITICAL: Co-located config not found at {config_path}. "
            "Every checkpoint must ship with an adjacent config.yaml."
        )

    logger.info(f"Resolving adjacent configuration from: {config_path}")
    with open(config_path, "r") as f:
        loaded_config_data = yaml.safe_load(f) or {}
    model_type = loaded_config_data.get("model_type", "riot_gnn") # TEMPORARY FALLBACK ADDED FOR BACKWARD COMPATIBILITY - PLEASE REVIEW
    model_cfg = loaded_config_data.get("model", {})
    logger.info(f"Loaded model_type={model_type}, hyperparameters: {model_cfg}")

    import subprocess
    git_commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()

    adapter = build_model_and_adapter(
        model_type, checkpoint_path, model_cfg,
        num_nodes_total=num_nodes_total, static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim, seq_len=seq_len, device=device,
    )
    return adapter, loaded_config_data, git_commit_hash


# ---------------------------------------------------------------------------
# Node metadata (port of export_ui_payload.main lines 766-791)
# ---------------------------------------------------------------------------

def build_nodes_meta(
    real_location_names: List[str],
    sim_injector: SimulatedNodeInjector,
    sensor_coords: Dict[str, Dict[str, float]],
    holdout_locations: Optional[set] = None,
) -> List[Dict]:
    """node_role — three-tier display taxonomy for the frontend:
      observed:  normal sensor (prediction + ground truth)
      holdout:   real sensor masked throughout training — genuine zero-shot
                 prediction WITH ground truth available for comparison
      simulated: zero-baseline synthetic node (prediction, no ground truth)
    is_simulated is kept for backward compatibility, derived from node_role."""
    holdout_locations = holdout_locations or set()
    nodes_meta: List[Dict] = []
    for i, loc in enumerate(real_location_names):
        if loc not in sensor_coords:
            raise KeyError(f"CRITICAL: sensor '{loc}' has no coordinates in sensor_coords")
        coords = sensor_coords[loc]
        role = "holdout" if loc in holdout_locations else "observed"
        nodes_meta.append({
            "id": i,
            "location_name": loc,
            "lat": coords["lat"],
            "lon": coords["lon"],
            "node_role": role,
            "is_simulated": False,
        })
    num_nodes_real = len(real_location_names)
    for j, meta in enumerate(sim_injector.nodes_meta):
        meta_copy = dict(meta)
        meta_copy["id"] = num_nodes_real + j
        meta_copy["node_role"] = "simulated"
        nodes_meta.append(meta_copy)
    return nodes_meta


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def daterange(start_date: str, end_date: str):
    """Yield date objects from start (inclusive) to end (exclusive)."""
    cur = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    one = dt.timedelta(days=1)
    while cur < end:
        yield cur
        cur += one


def select_day_window_indices(dataset: RealIoTDataset, seq_len: int, target_date: dt.date) -> List[int]:
    """
    Return dataset window indices whose forecast target timestamp falls on target_date.

    dataset[idx] forecasts the timestamp at dataset.times[idx + seq_len]; we group by
    that timestamp's calendar date.
    """
    indices: List[int] = []
    times = dataset.times
    for idx in range(len(dataset)):
        ts = times[idx + seq_len]
        # ts is a numpy datetime64 / pandas Timestamp — normalise to date.
        ts_date = np.datetime64(ts, "D").astype("datetime64[D]").astype(dt.date) \
            if isinstance(ts, np.datetime64) else ts.date()
        if ts_date == target_date:
            indices.append(idx)
    return indices


# ---------------------------------------------------------------------------
# Per-day inference (port of export_ui_payload.main lines 890-1087)
# ---------------------------------------------------------------------------

def run_day_inference(
    adapter: ModelAdapter,
    dataset: RealIoTDataset,
    window_indices: List[int],
    *,
    combined_static: torch.Tensor,
    num_nodes_real: int,
    sim_injector: SimulatedNodeInjector,
    seq_len: int,
    device: torch.device,
    location_ids: List[str],
    holdout_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """Run inference for one day and return the per-node / per-edge accumulators.

    Runs through adapter.predict() (the Predictions contract), so a baseline
    without uncertainty/graph heads yields null theta/pi/uncertainty and an empty
    edge set — never fabricated ZINB or attention values.

    holdout_mask ([num_nodes_total] bool): inductive holdout nodes from the run's
    holdout_manifest.json. Their dynamic features are blanked exactly as during
    training, so their displayed predictions are genuine zero-shot reconstructions."""
    caps = adapter.capabilities
    num_nodes_total = num_nodes_real + sim_injector.num_nodes
    num_timesteps = len(window_indices)
    assert num_timesteps > 0, "TRIPWIRE: day has zero inference windows."

    traffic_mean = float(dataset.traffic_mean)
    traffic_scale = float(dataset.traffic_std)
    assert traffic_scale > 0, f"TRIPWIRE: dataset.traffic_std={traffic_scale} must be > 0."

    timeseries_data: List[List[Dict]] = [[] for _ in range(num_nodes_total)]
    ground_truth_series: List[List] = [[] for _ in range(num_nodes_total)]
    imputed_series: List[List[float]] = [[] for _ in range(num_nodes_total)]
    mask_series: List[List[bool]] = [[] for _ in range(num_nodes_total)]
    theta_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
    pi_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
    edge_attention_data: Dict[Tuple[int, int], List[float]] = {}
    edge_index_cached: Optional[torch.Tensor] = None

    global_imputation_burden: List[float] = []
    global_healing_quality: List[float] = []
    attention_entropy_series: List[float] = []
    tripwire_audit_trail: List[Dict] = []

    with torch.no_grad():
        for t, window_idx in enumerate(window_indices):
            raw_item = dataset[window_idx]

            combined_item = sim_injector.inject_dynamic_item(raw_item)
            combined_dynamic = combined_item["dynamic_features"].unsqueeze(0)

            if holdout_mask is not None:
                combined_dynamic = apply_spatial_holdout(combined_dynamic, holdout_mask)

            batch = {
                "static_features": combined_static,
                "dynamic_features": combined_dynamic,
                "structural_masks": combined_item["structural_masks"].unsqueeze(0),
                "simulated_nodes_mask": combined_item["simulated_nodes_mask"],
                "location_ids": location_ids,
            }
            preds = adapter.predict(batch)
            validate_predictions(caps, preds)

            for tw in preds.tripwires:
                tw_copy = dict(tw)
                tw_copy["timestep"] = t
                tripwire_audit_trail.append(tw_copy)

            mu = preds.mu[0]
            assert not torch.isnan(mu).any(), f"t={t}: NaN in mu"

            if caps.uncertainty:
                theta = preds.theta[0]
                std_dev = compute_std_dev(preds.mu, preds.theta)[0]
                assert not torch.isnan(std_dev).any(), f"t={t}: NaN in std_dev"
            else:
                theta = None
                std_dev = None

            pi_raw = preds.pi[0] if caps.zero_inflation else None

            for node_id in range(num_nodes_total):
                timeseries_data[node_id].append({
                    "t": t,
                    "prediction": round(float(mu[node_id]), 4),
                    "uncertainty": (None if std_dev is None
                                    else round(float(std_dev[node_id]), 4)),
                })
                if caps.uncertainty:
                    theta_data[node_id].append(round(float(theta[node_id]), 4))
                if caps.zero_inflation:
                    pi_data[node_id].append(round(float(pi_raw[node_id]), 6))

            imputed_features = preds.imputed
            assert imputed_features.shape == (1, num_nodes_total, seq_len, time_series_dim), \
                f"t={t}: unexpected imputed_features shape {tuple(imputed_features.shape)}"
            assert not torch.isnan(imputed_features).any(), f"t={t}: NaN in imputed_features"

            imputed_last_scaled = imputed_features[0, :, -1, 0]
            imputed_last_raw = imputed_last_scaled * traffic_scale + traffic_mean
            real_mask_last = real_node_mask[:, -1, 0]

            gt_index = window_idx + seq_len - 1
            assert 0 <= gt_index < dataset.unscaled_traffic.shape[0], \
                f"t={t}: gt_index={gt_index} out of bounds"
            real_gt = dataset.unscaled_traffic[gt_index]

            mask_np = real_mask_last.cpu().numpy().astype(bool)
            imputed_np = imputed_last_raw[:num_nodes_real].cpu().numpy()
            mse = float(np.mean((imputed_np[mask_np] - real_gt[mask_np]) ** 2)) if mask_np.any() else 0.0
            global_imputation_burden.append(round(float(mask_np.mean()), 4))
            global_healing_quality.append(round(mse, 4))

            for node_id in range(num_nodes_real):
                imputed_series[node_id].append(round(float(imputed_last_raw[node_id]), 4))
                ground_truth_series[node_id].append(round(max(0.0, float(real_gt[node_id])), 4))
                mask_series[node_id].append(bool(real_mask_last[node_id]))

            for j in range(num_simulated):
                node_id = num_nodes_real + j
                imputed_series[node_id].append(round(float(imputed_last_raw[node_id]), 4))
                ground_truth_series[node_id].append(None)
                mask_series[node_id].append(True)

            # Attention/edges only exist for graph models; baselines leave the
            # edge accumulators empty and skip attention entropy entirely.
            if caps.graph:
                alpha = preds.attention
                edge_index = preds.edge_index
                num_edges = edge_index.shape[1]
                if edge_index_cached is None:
                    edge_index_cached = edge_index.cpu()
                    for e in range(num_edges):
                        src = int(edge_index_cached[0, e])
                        tgt = int(edge_index_cached[1, e])
                        edge_attention_data[(src, tgt)] = []

                per_edge_attn = extract_timestep_attention(alpha, num_edges, seq_len)
                assert not torch.isnan(per_edge_attn).any(), f"t={t}: NaN in attention weights"
                entropy = -float(torch.sum(per_edge_attn * torch.log(per_edge_attn + 1e-6)).item())
                attention_entropy_series.append(round(entropy, 4))

                for e in range(num_edges):
                    src = int(edge_index_cached[0, e])
                    tgt = int(edge_index_cached[1, e])
                    edge_attention_data[(src, tgt)].append(round(float(per_edge_attn[e]), 6))

    if caps.graph:
        assert edge_index_cached is not None, "TRIPWIRE: no edges accumulated — empty graph?"
    return {
        "capabilities": caps,
        "num_timesteps": num_timesteps,
        "timeseries_data": timeseries_data,
        "ground_truth_series": ground_truth_series,
        "imputed_series": imputed_series,
        "mask_series": mask_series,
        "theta_data": theta_data,
        "pi_data": pi_data,
        "edge_attention_data": edge_attention_data,
        "edge_index_cached": edge_index_cached,
        "global_imputation_burden": global_imputation_burden,
        "global_healing_quality": global_healing_quality,
        "attention_entropy_series": attention_entropy_series,
        "tripwire_audit_trail": tripwire_audit_trail,
    }


def compute_day_metrics(acc: Dict, num_nodes_total: int, num_nodes_real: int) -> Dict:
    """Compute day-scoped PIT histogram, reliability diagram and per-node
    two-extremes. PIT/reliability are ZINB diagnostics — null for a baseline
    without uncertainty/zero-inflation heads."""
    caps = acc["capabilities"]
    timeseries_data = acc["timeseries_data"]
    if caps.uncertainty and caps.zero_inflation:
        pit_histogram = compute_pit_histogram(
            ground_truth_series=acc["ground_truth_series"],
            mu_series=[[ts["prediction"] for ts in timeseries_data[n]] for n in range(num_nodes_total)],
            theta_series=acc["theta_data"],
            pi_series=acc["pi_data"],
            num_real_nodes=num_nodes_real,
        )
    else:
        pit_histogram = None
    if caps.zero_inflation:
        zero_inflation_reliability = compute_zero_inflation_reliability(
            ground_truth_series=acc["ground_truth_series"],
            pi_series=acc["pi_data"],
            num_real_nodes=num_nodes_real,
        )
    else:
        zero_inflation_reliability = None
    two_extremes_per_node: List[Dict] = []
    for node_id in range(num_nodes_total):
        two_extremes_per_node.append(compute_two_extremes(
            ground_truth=acc["ground_truth_series"][node_id],
            mu_predictions=[ts["prediction"] for ts in timeseries_data[node_id]],
        ))
    return {
        "pit_histogram": pit_histogram,
        "zero_inflation_reliability": zero_inflation_reliability,
        "two_extremes_per_node": two_extremes_per_node,
    }


# ---------------------------------------------------------------------------
# Alpha-node leaderboard (port of export_ui_payload.main lines 1151-1167)
# ---------------------------------------------------------------------------

def build_alpha_leaderboard(edge_attention_data: Dict[Tuple[int, int], List[float]],
                            nodes_meta: List[Dict], num_nodes_total: int) -> List[Dict]:
    node_out_influence = {i: 0.0 for i in range(num_nodes_total)}
    for (src, _tgt), attn_seq in edge_attention_data.items():
        node_out_influence[src] += sum(attn_seq)
    leaderboard = [
        {
            "node_id": node_id,
            "total_outgoing_attention": round(total, 4),
            "location_name": nodes_meta[node_id]["location_name"]
            if node_id < len(nodes_meta) else f"Node {node_id}",
        }
        for node_id, total in node_out_influence.items()
    ]
    leaderboard.sort(key=lambda x: x["total_outgoing_attention"], reverse=True)
    return leaderboard


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_page_json(out_dir: str, run_id: str, date_str: str, acc: Dict, day_metrics: Dict,
                    *, seq_len: int, num_nodes_total: int, num_nodes_real: int,
                    nodes_meta: List[Dict]) -> str:
    """Write {run_id}/{date}/page.json — timeseries + edge attention + day metrics (§5.3)."""
    caps = acc["capabilities"]
    num_timesteps = acc["num_timesteps"]
    timeseries_data = acc["timeseries_data"]
    edge_attention_data = acc["edge_attention_data"]

    node_timeseries = {str(n): timeseries_data[n] for n in range(num_nodes_total)}
    # Graph sections are empty/null for baselines (no attention to serve).
    edge_attention = [
        {"source": src, "target": tgt, "attention_sequence": attn_seq}
        for (src, tgt), attn_seq in edge_attention_data.items()
    ]
    alpha_leaderboard = (
        build_alpha_leaderboard(edge_attention_data, nodes_meta, num_nodes_total)
        if caps.graph else None
    )

    page = {
        "date": date_str,
        "timesteps": num_timesteps,
        "seq_len": seq_len,
        "node_timeseries": node_timeseries,
        "edge_attention": edge_attention,
        "pit_histogram": day_metrics["pit_histogram"],
        "zero_inflation_reliability": day_metrics["zero_inflation_reliability"],
        "global_imputation_burden": acc["global_imputation_burden"],
        "global_healing_quality": acc["global_healing_quality"],
        "attention_entropy_series": acc["attention_entropy_series"],
        "tripwire_audit_trail": acc["tripwire_audit_trail"],
        "alpha_node_leaderboard": alpha_leaderboard,
    }

    # Schema tripwires (mirror legacy master_index checks, scoped per day).
    # PIT/reliability are null when the model has no ZINB head.
    if page["pit_histogram"] is not None:
        assert len(page["pit_histogram"]) == 10, "PIT histogram must have 10 bins"
    if page["zero_inflation_reliability"] is not None:
        assert len(page["zero_inflation_reliability"]) <= 10, "Reliability diagram > 10 bins"
    for n in range(num_nodes_total):
        assert len(node_timeseries[str(n)]) == num_timesteps, \
            f"page node {n} timeseries length mismatch"
    for edge in edge_attention:
        assert len(edge["attention_sequence"]) == num_timesteps, \
            f"edge ({edge['source']},{edge['target']}) attention length mismatch"

    day_dir = os.path.join(out_dir, run_id, date_str)
    os.makedirs(day_dir, exist_ok=True)
    page_path = os.path.join(day_dir, "page.json")
    with open(page_path, "w") as fh:
        json.dump(page, fh, separators=(",", ":"))  # minified — machine-fetched payload
    return page_path


def write_node_detail_jsons(out_dir: str, run_id: str, date_str: str, acc: Dict,
                            day_metrics: Dict, nodes_meta: List[Dict],
                            *, num_nodes_total: int) -> int:
    """Write {run_id}/{date}/nodes/node_{id}.json (port of legacy lines 1252-1342)."""
    caps = acc["capabilities"]
    num_timesteps = acc["num_timesteps"]
    timeseries_data = acc["timeseries_data"]
    theta_data = acc["theta_data"]
    pi_data = acc["pi_data"]
    mask_series = acc["mask_series"]
    imputed_series = acc["imputed_series"]
    ground_truth_series = acc["ground_truth_series"]
    edge_attention_data = acc["edge_attention_data"]
    edge_index_cached = acc["edge_index_cached"]
    two_extremes_per_node = day_metrics["two_extremes_per_node"]

    # 1-hop edge lookup per node — empty for graph-less baselines.
    node_edges: Dict[int, List[Dict]] = {i: [] for i in range(num_nodes_total)}
    if caps.graph:
        num_edges = edge_index_cached.shape[1]
        for e in range(num_edges):
            src = int(edge_index_cached[0, e])
            tgt = int(edge_index_cached[1, e])
            attn_seq = edge_attention_data.get((src, tgt), [])
            node_edges[src].append({"neighbor_id": tgt, "direction": "outgoing", "attention_sequence": attn_seq})
            node_edges[tgt].append({"neighbor_id": src, "direction": "incoming", "attention_sequence": attn_seq})

    node_coord_map: Dict[int, Tuple[float, float]] = {m["id"]: (m["lat"], m["lon"]) for m in nodes_meta}

    nodes_dir = os.path.join(out_dir, run_id, date_str, "nodes")
    os.makedirs(nodes_dir, exist_ok=True)

    import math
    for meta in nodes_meta:
        node_id = meta["id"]
        assert len(mask_series[node_id]) == num_timesteps, \
            f"TRIPWIRE: temporal misalignment for node {node_id}"

        # theta/std_dev/pi are null for models without the matching capability
        # (a baseline reports N/A, never a fabricated ZINB parameter).
        zinb_series: List[Dict] = []
        for t_idx in range(num_timesteps):
            entry = {
                "t": t_idx,
                "mu": timeseries_data[node_id][t_idx]["prediction"],
                "theta": theta_data[node_id][t_idx] if caps.uncertainty else None,
                "std_dev": timeseries_data[node_id][t_idx]["uncertainty"],
                "pi": pi_data[node_id][t_idx] if caps.zero_inflation else None,
                "is_missing": mask_series[node_id][t_idx],
            }
            assert not math.isnan(entry["mu"]), f"NaN mu node {node_id} t={t_idx}"
            if entry["theta"] is not None:
                assert not math.isnan(entry["theta"]), f"NaN theta node {node_id} t={t_idx}"
            if entry["std_dev"] is not None:
                assert not math.isnan(entry["std_dev"]), f"NaN std_dev node {node_id} t={t_idx}"
            if entry["pi"] is not None:
                assert not math.isnan(entry["pi"]), f"NaN pi node {node_id} t={t_idx}"
            assert isinstance(entry["is_missing"], bool), f"is_missing not bool node {node_id} t={t_idx}"
            zinb_series.append(entry)

        for v in imputed_series[node_id]:
            assert not math.isnan(v), f"NaN in node {node_id} imputed_series"

        if caps.graph:
            ego_edges = build_ego_graph_edges(
                node_id, edge_attention_data, node_coord_map, num_timesteps, top_k=top_k_ego_edges
            )
            for e in ego_edges:
                assert len(e["weight_sequence"]) == num_timesteps, \
                    f"ego_graph node {node_id}: weight_sequence length mismatch"
                assert all(w >= 0.0 for w in e["weight_sequence"]), f"Negative ego weight node {node_id}"
                assert e["direction"] in ("incoming", "outgoing"), f"Bad direction node {node_id}"
        else:
            ego_edges = []

        node_data = {
            "id": node_id,
            "location_name": meta["location_name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "node_role": meta.get("node_role", "simulated" if meta["is_simulated"] else "observed"),
            "is_simulated": meta["is_simulated"],
            "zinb_series": zinb_series,
            "ground_truth_series": ground_truth_series[node_id],
            "imputed_series": imputed_series[node_id],
            "mask_series": mask_series[node_id],
            "edges_1hop": node_edges[node_id],
            "two_extremes": two_extremes_per_node[node_id],
            "ego_graph_edges": ego_edges,
        }
        with open(os.path.join(nodes_dir, f"node_{node_id}.json"), "w") as fh:
            json.dump(node_data, fh, separators=(",", ":"))  # minified
    return num_nodes_total


def write_spatial_json(out_dir: str, run_id: str, nodes_meta: List[Dict],
                       edge_topology: List[Dict], semantic_tags: Dict[str, List[str]],
                       metadata: Dict) -> str:
    """Write {run_id}/spatial.json — nodes + edge topology + run metadata (§5.2)."""
    spatial_nodes = [
        {
            "id": m["id"],
            "location_name": m["location_name"],
            "lat": m["lat"],
            "lon": m["lon"],
            "node_role": m.get("node_role", "simulated" if m["is_simulated"] else "observed"),
            "is_simulated": m["is_simulated"],
            "semantic_tags": semantic_tags.get(m["location_name"], []),
        }
        for m in nodes_meta
    ]
    spatial = {"metadata": metadata, "nodes": spatial_nodes, "edge_topology": edge_topology}

    assert len(spatial_nodes) == metadata["num_nodes"], "spatial node count mismatch"
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    spatial_path = os.path.join(run_dir, "spatial.json")
    with open(spatial_path, "w") as fh:
        json.dump(spatial, fh, separators=(",", ":"))  # minified
    return spatial_path


def upload_run_artifact(out_dir: str, run_id: str, entity: Optional[str], project: str) -> None:
    """Log {out_dir}/{run_id}/ to W&B as a ui-payload artifact (opt-in archival)."""
    run_dir = os.path.join(out_dir, run_id)
    logger.info(f"Uploading {run_dir} to W&B project {project} (entity={entity or 'default'}) …")
    wb_run = wandb.init(entity=entity, project=project, job_type="ui-payload-export",
                        name=f"export-{run_id}", reinit=True)
    try:
        art = wandb.Artifact(name=run_id, type="ui-payload")
        art.add_dir(run_dir)
        wb_run.log_artifact(art)
    finally:
        wb_run.finish()


# ---------------------------------------------------------------------------
# Per-artifact generation
# ---------------------------------------------------------------------------

def generate_for_artifact(artifact_name: str, cfg: Dict, args: argparse.Namespace,
                          device: torch.device, sensor_coords: Dict,
                          connector: DataConnector) -> Dict:
    """Generate the full per-day payload tree for a single W&B artifact."""
    start_date, end_date = cfg["start_date"], cfg["end_date"]
    seq_len, num_simulated = cfg["seq_len"], cfg["num_simulated"]
    veh_class, direction = cfg["veh_class"], cfg["dir"]
    export_cfg = cfg["export"]
    num_timesteps_target: int = export_cfg["num_timesteps_target"]
    top_k_ego_edges: int = export_cfg["top_k_ego_edges"]
    ncl_lat_min: float = export_cfg["ncl_lat_min"]
    ncl_lat_max: float = export_cfg["ncl_lat_max"]
    ncl_lon_min: float = export_cfg["ncl_lon_min"]
    ncl_lon_max: float = export_cfg["ncl_lon_max"]
    num_simulated_nodes: int = export_cfg.get("num_simulated_nodes", 50)
    assert num_simulated_nodes <= 100, "Maximum simulated nodes is 100"
    mask_prob_export: float = export_cfg["mask_prob"]

    ckpt, cfg_path, wandb_run_id, run_id, metrics_summary = load_model_and_meta(artifact_name)
    logger.info(f"=== Run '{run_id}' (wandb_run_id={wandb_run_id}) from {artifact_name} ===")

    # HARNESS: inductive holdout manifest co-located with the checkpoint. When
    # present, those nodes are blanked at inference (zero-shot predictions) and
    # tagged node_role="holdout" for distinct frontend display.
    holdout_manifest = load_holdout_manifest(os.path.dirname(ckpt))
    holdout_locations = set(holdout_manifest["holdout_location_ids"]) if holdout_manifest else set()
    if holdout_manifest:
        logger.info(
            f"Holdout manifest found: {len(holdout_locations)} holdout nodes, "
            f"sha={manifest_sha(holdout_manifest)}"
        )

    dates = list(daterange(start_date, end_date))
    assert dates, "TRIPWIRE: empty analysis window."

    # --- Single full-window load (canonical node set / scaler / static / topology) ---
    # One dataset over [start - 1 day, end) gives a SINGLE sorted location set (the
    # union over the window), one scaler and one static-feature matrix shared by every
    # day. RealIoTDataset.reindex(...).fillna(0.0) zero-fills any sensor that is absent
    # on a given day, so days with a dropped sensor (e.g. 2026-05-19) are handled
    # gracefully instead of shifting node ids. The look-back day (start - 1) supplies
    # seq_len context for the first target day.
    full_ds = RealIoTDataset(
        connector,
        start_date=(dt.date.fromisoformat(start_date) - dt.timedelta(days=1)).isoformat(),
        end_date=end_date,
        seq_len=seq_len,
        target_veh_class=veh_class, target_dir=direction,
    )
    canonical_locations = list(full_ds.locations)
    num_nodes_real = full_ds.num_nodes
    num_nodes_total = num_nodes_real + num_simulated
    static_feat_dim = full_ds.static_feat_dim
    time_series_dim = full_ds.time_series_dim

    # Instantiate the SimulatedNodeInjector
    bbox = {
        'lat_min': ncl_lat_min,
        'lat_max': ncl_lat_max,
        'lon_min': ncl_lon_min,
        'lon_max': ncl_lon_max
    }
    sim_injector = SimulatedNodeInjector(
        num_nodes=num_simulated,
        seq_len=seq_len,
        bbox=bbox,
        static_scaler=full_ds.static_scaler,
        device=device
    )

    nodes_meta = build_nodes_meta(
        canonical_locations, sim_injector, sensor_coords,
        holdout_locations=holdout_locations,
    )
    location_ids = [m["location_name"] for m in nodes_meta]

    # [num_nodes_total] bool: manifest holdout over real nodes, False for simulated.
    holdout_mask_total: Optional[torch.Tensor] = None
    if holdout_manifest is not None:
        real_holdout_mask = mask_from_manifest(holdout_manifest, canonical_locations, device=device)
        holdout_mask_total = torch.cat([
            real_holdout_mask,
            torch.zeros(num_simulated, dtype=torch.bool, device=device),
        ])

    adapter, loaded_config_data, git_hash = build_model(
        ckpt, cfg_path,
        num_nodes_total=num_nodes_total, static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim, seq_len=seq_len, device=device,
    )
    caps = adapter.capabilities

    # Static features (real + simulated) → identical topology every day.
    real_static = full_ds.static_features.to(device)
    combined_static = sim_injector.inject_static(real_static)
    assert combined_static.shape == (num_nodes_total, static_feat_dim), \
        f"Static features shape mismatch: {combined_static.shape}"

    semantic_tags = load_semantic_tags(nodes_meta)
    edge_topology: Optional[List[Dict]] = None
    available_dates: List[str] = []

    for target_date in dates:
        date_str = target_date.isoformat()
        window_indices = select_day_window_indices(full_ds, seq_len, target_date)
        if not window_indices:
            logger.warning(f"  {date_str}: no inference windows — skipping day.")
            continue
        if len(window_indices) > num_timesteps_target:
            logger.info(f"  {date_str}: {len(window_indices)} windows; capping to {num_timesteps_target}.")
            window_indices = window_indices[:num_timesteps_target]

        logger.info(f"  {date_str}: {len(window_indices)} inference timesteps …")
        acc = run_day_inference(
            adapter, full_ds, window_indices,
            combined_static=combined_static, num_nodes_real=num_nodes_real,
            sim_injector=sim_injector, seq_len=seq_len,
            device=device,
            location_ids=location_ids,
            holdout_mask=holdout_mask_total,
        )
        day_metrics = compute_day_metrics(acc, num_nodes_total, num_nodes_real)

        write_page_json(args.output_dir, run_id, date_str, acc, day_metrics,
                        seq_len=seq_len, num_nodes_total=num_nodes_total,
                        num_nodes_real=num_nodes_real, nodes_meta=nodes_meta)
        write_node_detail_jsons(args.output_dir, run_id, date_str, acc, day_metrics,
                                nodes_meta, num_nodes_total=num_nodes_total)

        # Edge topology is fixed per run; empty for graph-less baselines.
        if edge_topology is None:
            ei = acc["edge_index_cached"]
            edge_topology = [] if ei is None else [
                {"source": int(ei[0, e]), "target": int(ei[1, e])}
                for e in range(ei.shape[1])
            ]
        available_dates.append(date_str)
        del acc, day_metrics

    assert available_dates, f"TRIPWIRE: run '{run_id}' produced no days."
    assert edge_topology is not None

    metadata = {
        "run_id": run_id,
        "wandb_run_id": wandb_run_id,
        "git_commit_hash": git_hash,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hyperparameters": loaded_config_data,
        "num_nodes": num_nodes_total,
        "num_real_nodes": num_nodes_real,
        "num_simulated_nodes": num_simulated,
        "num_holdout_nodes": len(holdout_locations),
        "holdout_manifest_sha": manifest_sha(holdout_manifest),
        "available_dates": available_dates,
        "veh_class": veh_class,
        "dir": direction,
        "seq_len": seq_len,
        "metrics_summary": metrics_summary,
    }
    spatial_path = write_spatial_json(args.output_dir, run_id, nodes_meta,
                                      edge_topology, semantic_tags, metadata)
    logger.info(f"  spatial.json written → {spatial_path} ({len(available_dates)} days)")

    if args.upload:
        upload_run_artifact(args.output_dir, run_id, args.wandb_entity, args.wandb_project)

    return {"run_id": run_id, "days": len(available_dates), "num_nodes": num_nodes_total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    cfg = load_analysis_config(args.config, args)
    artifacts = resolve_artifacts(args)
    dates = list(daterange(cfg["start_date"], cfg["end_date"]))

    logger.info(
        f"Plan: {len(artifacts)} artifact(s) × {len(dates)} day(s) "
        f"({cfg['start_date']} → {cfg['end_date']}, exclusive) → {args.output_dir}"
    )
    for a in artifacts:
        logger.info(f"  artifact: {a}")

    if args.dry_run:
        logger.info("--dry_run set; exiting before inference.")
        return

    connector = create_connector(cfg)
    sensor_coords = connector.load_sensor_metadata()
    logger.info(f"Loaded {len(sensor_coords)} sensor coordinates via connector")

    summaries = []
    try:
        for artifact_name in artifacts:
            summaries.append(
                generate_for_artifact(artifact_name, cfg, args, device, sensor_coords, connector)
            )
    finally:
        connector.close()

    logger.info(f"DONE. Generated {len(summaries)}/{len(artifacts)} run(s):")
    for s in summaries:
        logger.info(f"  {s['run_id']}: {s['days']} days, {s['num_nodes']} nodes")
    if not summaries:
        raise SystemExit("No runs generated — all artifacts failed.")


if __name__ == "__main__":
    main()
