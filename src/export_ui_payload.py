"""
export_ui_payload.py

Generates the static ui_payload.json for the Geospatial Digital Twin (Phase 1).

Runs 96 forward passes through the trained ResilientIoTPipeline (one per 15-min
interval of a 24-hour test day), collecting per-node ZINB parameters (mu, theta)
and per-edge GAT attention weights.  Five synthetic 'Simulated Nodes' are injected
into every batch to demonstrate zero-shot topological inference.

Output: data/processed/ui_payload.json

Usage:
    python src/export_ui_payload.py \
        [--start_date 2026-05-09] [--end_date 2026-05-11] \
        [--veh_class person] [--dir all] \
        [--seq_len 12] [--num_simulated 5] \
        [--checkpoint models/best_pipeline.pt] \
        [--artifact entity/project/model:latest]
"""

from networkx import generators
import argparse

import json
import logging
import math
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from scipy.special import betainc

# Allow direct execution from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

env_path = os.path.join(Path(__file__).parent.parent, '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"\''))

from baselines import GRUForecaster, LSTMForecaster
from dataset import RealIoTDataset, apply_spatial_holdout
from db_connector import create_connector
from features import StaticFeatureLoader, REQUIRED_KEYS
from simulated_nodes import SimulatedNodeInjector
from harness import (
    ModelAdapter,
    build_adapter,
    load_holdout_manifest,
    manifest_sha,
    mask_from_manifest,
    validate_predictions,
)
from pipeline import ResilientIoTPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants inferred from the trained checkpoint
# Derived from models/best_pipeline.pt weight shapes:
#   graph_learner.lin.weight  → [32, 5]  ⟹ embedding_dim=32, static_feat_dim=5
#   encoder.gat.lin_l.weight  → [64, 3]  ⟹ hidden_dim=64, time_series_dim=3
# ---------------------------------------------------------------------------
SENSORS_PATH = "data/traffic_sensors.json"
MASTER_INDEX_PATH = "frontend/public/data/master_index.json"
NODES_DIR = "frontend/public/data/nodes"
PLACES_CACHE_PATH = "data/external/places_cache.json"


# Export constants — loaded from configs/analysis/analysis.yaml at runtime.
# These sentinel values are replaced by config values in main(); do not use directly.
NUM_TIMESTEPS_TARGET: int = 96  # overridden from export.num_timesteps_target
TOP_K_EGO_EDGES: int = 10       # overridden from export.top_k_ego_edges


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ui_payload.json for the Geospatial Digital Twin"
    )
    # Load two days so we have seq_len context for day 2 predictions
    parser.add_argument("--start_date", type=str, default="2026-06-01",
                        help="Dataset start date (YYYY-MM-DD).  Load one day earlier than "
                             "the target day to supply seq_len look-back context.")
    parser.add_argument("--end_date", type=str, default="2026-06-03",
                        help="Dataset end date (YYYY-MM-DD, exclusive).")
    parser.add_argument("--veh_class", type=str, default="person",
                        help="Vehicle class filter used during training.")
    parser.add_argument("--dir", type=str, default="all",
                        help="Spatial direction filter used during training.")
    parser.add_argument("--seq_len", type=int, default=12,
                        help="Sequence length (must match trained model).")
    parser.add_argument("--num_simulated", type=int, default=5,
                        help="Number of synthetic zero-shot nodes to inject.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the trained model checkpoint. If not provided, W&B will be queried.")
    parser.add_argument("--artifact", type=str, default=None,
                        help="W&B artifact name (e.g., 'entity/project/name:latest'). "
                             "If provided, overrides --checkpoint.")
    parser.add_argument("--wandb_project", type=str, default="resilient-iot-v2",
                        help="W&B project name to query for model artifacts.")
    parser.add_argument("--wandb_collection", type=str, default="resilient-iot-model-v2",
                        help="W&B artifact collection name to query.")
    parser.add_argument("--config", type=str, default="configs/analysis/analysis.yaml",
                        help="Analysis config YAML (drives data_source and connector creation).")
    parser.add_argument("--output_dir", type=str, default="frontend/public/data",
                        help="Directory for master_index.json + nodes/. Defaults to the "
                             "served frontend location; point elsewhere for scratch exports.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sensor_coords(path: str) -> Dict[str, Dict[str, float]]:
    """Return {location_name: {lat, lon}} from traffic_sensors.json."""
    with open(path, "r") as fh:
        raw = json.load(fh)
    return {entry["location"]: {"lat": entry["lat"], "lon": entry["lon"]} for entry in raw}


def select_artifact_from_wandb(project: str = "resilient-iot-v2", collection_name: str = "resilient-iot-model-v2") -> str:
    """
    Connects to W&B, lists all available model artifacts in the configured project,
    and returns the selected or latest artifact name.
    """
    api = wandb.Api()
    entity = api.default_entity
    collection_path = f"{entity}/{project}/{collection_name}"

    logger.info(f"Connecting to W&B to list available model artifacts in project {project}...")
    try:
        collection = api.artifact_collection('model', collection_path)
        artifacts = list(collection.artifacts())
    except wandb.errors.CommError as e:
        raise RuntimeError(
            "Could not fetch model artifacts from Weights & Biases. Please ensure "
            "WANDB_API_KEY is configured and valid, or provide a local checkpoint path "
            "via --checkpoint."
        ) from e

    if not artifacts:
        raise ValueError(f"No model artifacts found in W&B collection: {collection_path}")

    # Display options
    print("\nAvailable W&B Model Artifacts:")
    print("--------------------------------")
    for i, art in enumerate(artifacts):
        aliases_str = f" ({', '.join(art.aliases)})" if art.aliases else ""
        print(f"[{i + 1}] {art.name}{aliases_str}")
    print("--------------------------------")

    # If interactive, ask the user to choose
    if sys.stdin.isatty():
        while True:
            try:
                user_input = input(f"Select an artifact (1-{len(artifacts)}) or press Enter for [1]: ").strip()
                if not user_input:
                    selected_idx = 0
                    break
                selected_idx = int(user_input) - 1
                if 0 <= selected_idx < len(artifacts):
                    break
                print(f"Please enter a number between 1 and {len(artifacts)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        # Non-interactive mode (e.g., CI or background task)
        raise ValueError(
            "Non-interactive session detected but no artifact was explicitly provided. "
            "Please specify an artifact using the --artifact flag to prevent silent fallbacks."
        )
    selected_art = artifacts[selected_idx]
    version = selected_art.name.split(':')[-1]
    full_name = f"{collection_path}:{version}"
    logger.info(f"Selected artifact: {full_name}")
    return full_name


def load_model_from_wandb(artifact_name: str) -> Tuple[str, str, str]:
    """
    Downloads a model artifact from W&B.
    Returns (checkpoint_path, config_path, wandb_run_id).
    """
    api = wandb.Api()
    logger.info(f"PHASE 2: Fetching artifact {artifact_name} from W&B...")
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    checkpoint_path = os.path.join(artifact_dir, "best_model.pt")
    config_path = os.path.join(artifact_dir, "config.yaml")

    # Get the run ID from the run that logged the artifact
    producer_run = artifact.logged_by()
    assert producer_run is not None, f"FATAL: Artifact {artifact_name} has no producer run. Manually uploaded artifacts are strictly forbidden."
    wandb_run_id = producer_run.id

    return checkpoint_path, config_path, wandb_run_id


def build_model_and_adapter(
    model_type: str,
    checkpoint_path: str,
    model_cfg: Dict,
    *,
    num_nodes_total: int,
    static_feat_dim: int,
    time_series_dim: int,
    seq_len: int,
    device: torch.device,
) -> ModelAdapter:
    """Instantiate the served model by family and wrap it in a ModelAdapter so
    the exporters depend only on the Predictions contract, never on a model's
    internals. model_type ('gnn' | 'lstm' | 'gru') is read from the co-located
    config.yaml; checkpoints predating the harness default to 'gnn'.

    Baselines (lstm/gru) are node-agnostic recurrent encoders — they carry no
    graph, uncertainty, or zero-inflation heads, so the adapter's capabilities
    tell the exporter to emit null (not fabricated) uncertainty/graph sections."""
    if model_type == "gnn":
        # The graph learner rejects a simulated-edge cache without a physical
        # edge cache, so only supply the simulated fallback when the physical
        # cache is present (learned-only GNNs set neither).
        edge_cache = model_cfg.get("edge_features_cache_path")
        sim_edge_cache = (
            (model_cfg.get("simulated_edge_features_cache_path")
             or "data/external/simulated_edge_features_cache.json")
            if edge_cache else None
        )
        model = ResilientIoTPipeline(
            num_nodes=num_nodes_total,
            static_feat_dim=static_feat_dim,
            time_series_dim=time_series_dim,
            seq_len=seq_len,
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            sparsity_threshold=model_cfg["sparsity_threshold"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            dropout=model_cfg["dropout"],
            dim_feedforward=model_cfg["dim_feedforward"],
            zinb_eps=model_cfg["zinb_eps"],
            edge_features_cache_path=edge_cache,
            simulated_edge_features_cache_path=sim_edge_cache,
        ).to(device)
    elif model_type in ("lstm", "gru"):
        baseline_cls = LSTMForecaster if model_type == "lstm" else GRUForecaster
        model = baseline_cls(
            time_series_dim=time_series_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r} (expected gnn | lstm | gru)")

    logger.info(f"Loading checkpoint from {checkpoint_path} (model_type={model_type})")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    adapter = build_adapter(model_type, model)
    logger.info(f"Model loaded (eval mode). Capabilities: {adapter.capabilities}")
    return adapter

def compute_std_dev(
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Standard deviation of the ZINB distribution:
        Var(Y) = (1 - pi) * (mu + mu² / theta) + pi * (1 - pi) * mu²
    If pi is None, evaluates the standard Negative Binomial variance:
        Var = mu + mu² / theta

    Args:
        mu:    [batch_size, num_nodes]  — mean parameter (already softplus'd + eps)
        theta: [batch_size, num_nodes]  — dispersion parameter (already softplus'd + eps)
        pi:    [batch_size, num_nodes]  — optional zero-inflation probability parameter

    Returns:
        std_dev: [batch_size, num_nodes]
    """
    nb_var = mu + (mu ** 2) / theta
    if pi is not None:
        variance = (1.0 - pi) * nb_var + pi * (1.0 - pi) * (mu ** 2)
    else:
        variance = nb_var
    # Tripwire: variance must be strictly positive (guaranteed by softplus+eps on mu, theta)
    assert (variance > 0).all(), "TRIPWIRE: Non-positive variance detected."
    return torch.sqrt(variance)


def extract_timestep_attention(
    alpha: torch.Tensor,
    num_edges: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Extract per-edge attention for the most recent time step of the sequence.

    The encoder batches (batch_size × seq_len) graphs end-to-end.  With
    batch_size=1 the layout is:
        alpha[0          : E]        → (sample 0, t=0)
        alpha[E          : 2E]       → (sample 0, t=1)
        …
        alpha[(T-1)*E    : T*E]      → (sample 0, t=T-1)   ← most recent

    We take the last slice and average across attention heads to yield a
    single scalar weight per edge.

    Args:
        alpha:     [(batch_size * seq_len * E), num_heads]
        num_edges: E (edges in the unbatched graph)
        seq_len:   T

    Returns:
        [num_edges]  — mean attention weight per edge for the last time step
    """
    last_slice = alpha[(seq_len - 1) * num_edges : seq_len * num_edges, :]  # [E, num_heads]
    return last_slice.mean(dim=-1)  # [E]


# ---------------------------------------------------------------------------
# Semantic tag helpers
# ---------------------------------------------------------------------------

def load_semantic_tags(
    nodes_meta: List[Dict],
    places_cache_path: str = PLACES_CACHE_PATH,
    sim_places_cache_path: str = "data/external/simulated_places_cache.json",
) -> Dict[str, List[str]]:
    """
    Build a {location_name: [tag, ...]} mapping from the Places cache.

    Source:
      1. StaticFeatureLoader (Places API) — live API features for simulated
         nodes + cached features for real nodes.
    """
    tags: Dict[str, List[str]] = {m["location_name"]: [] for m in nodes_meta}

    # Initialize loader; will raise ValueError/RuntimeError on auth/quota/missing key issues
    loader = StaticFeatureLoader(cache_path=places_cache_path)
    sim_loader = StaticFeatureLoader(cache_path=sim_places_cache_path)

    # Live Query for simulated nodes missing from cache
    for meta in nodes_meta:
        loc = meta["location_name"]
        if meta["is_simulated"] and loc not in sim_loader.cache:
            logger.info(f"Fetching live data for simulated node: {loc}")
            # Note: Using private method as requested in Task 3.2
            sim_loader.cache[loc] = sim_loader._fetch_location_data(meta["lat"], meta["lon"])
            sim_loader._save_cache()

    cache = dict(loader.cache)
    cache.update(sim_loader.cache)

    # Process features and append tags from Cache
    for loc, feats in cache.items():
        if loc not in tags:
            continue
        t = tags[loc]

        # Commercial density — mapped from features schema key `retail_count`
        cd = int(feats["retail_count"])
        if cd >= 15:
            t.append("high_commercial_density")
        elif cd >= 5:
            t.append("medium_commercial_density")
        elif cd > 0:
            t.append("low_commercial_density")

        # School zone
        if feats.get("is_school_zone"):
            t.append("school_zone")

        # Transit density — mapped from features schema key `transit_count`
        hp = int(feats["transit_count"])
        if hp >= 15:
            t.append("high_transit_density")
        elif hp >= 5:
            t.append("medium_transit_density")
        elif hp > 0:
            t.append("low_transit_density")

        # POI density — mapped from features schema key `total_poi_count`
        poi = int(feats["total_poi_count"])
        if poi >= 20:
            t.append("high_poi_density")
        elif poi >= 8:
            t.append("medium_poi_density")
        elif poi >= 1:
            t.append("low_poi_density")

    return tags


# ---------------------------------------------------------------------------
# Calibration metric helpers
# ---------------------------------------------------------------------------

def _zinb_randomized_pit(y_obs: float, mu: float, theta: float, pi_val: float) -> float:
    """
    Randomized Probability Integral Transform (PIT) for the discrete ZINB distribution.

    For discrete distributions, F(y) produces step functions which skew the PIT histogram.
    Randomized PIT draws a uniform random value between F(y-1) and F(y) to restore
    the continuous uniform property under the null hypothesis.

    F_NB(y; mu, theta) = betainc(theta, floor(y)+1, theta/(theta+mu))
    F_ZINB(y)          = pi + (1 - pi) * F_NB(y)
    """
    import random
    assert mu > 0.0,              f"_zinb_randomized_pit: mu={mu} must be > 0"
    assert theta > 0.0,           f"_zinb_randomized_pit: theta={theta} must be > 0"
    pi_f = float(pi_val)
    assert 0.0 < pi_f < 1.0,     f"_zinb_randomized_pit: pi={pi_f} must be in (0, 1)"
    k = int(math.floor(y_obs))
    p_nb = theta / (theta + mu)

    # Calculate upper bound F(y)
    cdf_nb_upper = float(betainc(theta, k + 1, p_nb))
    f_upper = pi_f + (1.0 - pi_f) * cdf_nb_upper

    # Calculate lower bound F(y-1)
    if k == 0:
        f_lower = 0.0
    else:
        cdf_nb_lower = float(betainc(theta, k, p_nb))
        f_lower = pi_f + (1.0 - pi_f) * cdf_nb_lower

    # Draw uniform random value between F(y-1) and F(y)
    u = random.uniform(f_lower, f_upper)

    # Clip just in case of floating point inaccuracies
    u = max(0.0, min(1.0, u))
    return u


def compute_pit_histogram(
    ground_truth_series: List[List],
    mu_series: List[List[float]],
    theta_series: List[List[float]],
    pi_series: List[List[float]],
    num_real_nodes: int,
    num_bins: int = 10,
) -> List[Dict]:
    """
    Compute the Probability Integral Transform histogram for the ZINB distribution.

    Iterates over all real nodes and all timesteps where ground truth is a valid
    (non-None, non-NaN) count.  Returns a list of num_bins dicts with:
      { "bin_center": float, "density": float }
    where density = fraction of PIT values in that bin (sums to ~1.0).

    A perfectly calibrated model produces a uniform histogram (density ≈ 0.1 per bin).
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(num_bins, dtype=np.int64)

    for node_id in range(num_real_nodes):
        gt_seq = ground_truth_series[node_id]
        mu_seq = mu_series[node_id]
        th_seq = theta_series[node_id]
        pi_seq = pi_series[node_id]
        assert len(gt_seq) == len(mu_seq) == len(th_seq) == len(pi_seq), (
            f"PIT: length mismatch for node {node_id}: "
            f"gt={len(gt_seq)} mu={len(mu_seq)} theta={len(th_seq)} pi={len(pi_seq)}"
        )
        for t_idx, (gt, mu_val, th_val, pi_val) in enumerate(
            zip(gt_seq, mu_seq, th_seq, pi_seq)
        ):
            if gt is None or math.isnan(gt):
                continue

            # Prevent perfectly rounded 0.0s from tripping the strictly-positive assert
            safe_mu = max(1e-6, float(mu_val))
            safe_th = max(1e-6, float(th_val))
            safe_pi = max(1e-6, min(1.0 - 1e-6, float(pi_val)))

            u = _zinb_randomized_pit(float(gt), safe_mu, safe_th, safe_pi)
            bin_idx = min(int(u * num_bins), num_bins - 1)
            counts[bin_idx] += 1

    total = int(counts.sum())
    assert total > 0, "TRIPWIRE: PIT histogram has zero valid observations — check ground truth data"

    return [
        {"bin_center": round(float(bin_centers[i]), 4), "density": round(float(counts[i] / total), 6)}
        for i in range(num_bins)
    ]


def compute_zero_inflation_reliability(
    ground_truth_series: List[List],
    pi_series: List[List[float]],
    num_real_nodes: int,
    num_bins: int = 10,
    mu_series: Optional[List[List[float]]] = None,
    theta_series: Optional[List[List[float]]] = None,
) -> List[Dict]:
    """
    Compute the zero-inflation reliability diagram.

    For each valid (non-None, non-NaN) observation, calculates the theoretical probability
    of observing zero: P(Y = 0) = pi + (1 - pi) * (theta / (theta + mu))^theta.
    When mu_series/theta_series are not supplied, falls back to using the mixture weight pi directly.
    Bins this probability into num_bins equal-width buckets across [0, 1] and computes the actual
    fraction of true zeros in each bucket.

    Returns a list of dicts (one per non-empty bin):
      { "pi_center": float, "observed_zero_fraction": float, "count": int }

    A perfectly calibrated model has all points on the y = x diagonal.
    Bins with zero observations are omitted to avoid charting artifacts.
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    zero_counts = np.zeros(num_bins, dtype=np.int64)
    total_counts = np.zeros(num_bins, dtype=np.int64)

    use_full_zinb = mu_series is not None and theta_series is not None

    for node_id in range(num_real_nodes):
        gt_seq = ground_truth_series[node_id]
        pi_seq = pi_series[node_id]
        mu_seq = mu_series[node_id] if use_full_zinb else None
        th_seq = theta_series[node_id] if use_full_zinb else None

        assert len(gt_seq) == len(pi_seq), (
            f"Reliability: length mismatch node {node_id}: gt={len(gt_seq)} pi={len(pi_seq)}"
        )
        for idx, (gt, pi_val) in enumerate(zip(gt_seq, pi_seq)):
            if gt is None or math.isnan(gt):
                continue
            pi_f = float(pi_val)
            assert 0.0 <= pi_f <= 1.0, f"Reliability: pi={pi_f} outside [0,1]"

            if use_full_zinb:
                mu_f = float(mu_seq[idx])
                th_f = float(th_seq[idx])
                nb_p0 = (th_f / (th_f + mu_f)) ** th_f
                p_zero = pi_f + (1.0 - pi_f) * nb_p0
            else:
                p_zero = pi_f

            p_zero = max(0.0, min(1.0, p_zero))
            bin_idx = min(int(p_zero * num_bins), num_bins - 1)
            total_counts[bin_idx] += 1
            if gt == 0.0:  # noqa: comparing float to 0.0 is intentional (raw count)
                zero_counts[bin_idx] += 1

    result = []
    for i in range(num_bins):
        n = int(total_counts[i])
        if n == 0:
            continue
        result.append({
            "pi_center": round(float(bin_centers[i]), 4),
            "observed_zero_fraction": round(float(zero_counts[i] / n), 6),
            "count": n,
        })
    return result


def compute_two_extremes(
    ground_truth: List,
    mu_predictions: List[float],
) -> Dict:
    """
    Compute MAE and RMSE under two missing-data strategies.

    Drop mode: exclude timesteps where ground truth is None or NaN.
    Fill mode: replace None/NaN ground truths with 0.0 before computing.

    Args:
        ground_truth:   per-timestep observed counts (float | None, may include NaN)
        mu_predictions: per-timestep ZINB mean predictions

    Returns dict with keys: mae_drop_nan, rmse_drop_nan (None if no valid GT),
    mae_fill_nan, rmse_fill_nan.
    """
    assert len(ground_truth) == len(mu_predictions), (
        f"two_extremes: length mismatch gt={len(ground_truth)} pred={len(mu_predictions)}"
    )

    # Drop mode
    drop_preds, drop_gts = [], []
    for mu_val, gt in zip(mu_predictions, ground_truth):
        if gt is None or (isinstance(gt, float) and math.isnan(gt)):
            continue
        drop_preds.append(mu_val)
        drop_gts.append(float(gt))

    if drop_preds:
        arr_p = np.array(drop_preds, dtype=np.float64)
        arr_g = np.array(drop_gts, dtype=np.float64)
        residuals = arr_p - arr_g
        mae_drop: Optional[float] = float(np.mean(np.abs(residuals)))
        rmse_drop: Optional[float] = float(np.sqrt(np.mean(residuals ** 2)))
        assert mae_drop >= 0.0, f"TRIPWIRE: negative MAE_drop={mae_drop}"
        assert rmse_drop >= 0.0, f"TRIPWIRE: negative RMSE_drop={rmse_drop}"
    else:
        mae_drop = None
        rmse_drop = None

    # Fill mode: replace None/NaN with 0.0
    fill_gts = np.array(
        [0.0 if (gt is None or (isinstance(gt, float) and math.isnan(gt))) else float(gt)
         for gt in ground_truth],
        dtype=np.float64,
    )
    assert not np.isnan(fill_gts).any(), "TRIPWIRE: NaN in fill_gts after substitution"
    arr_fill_p = np.array(mu_predictions, dtype=np.float64)
    residuals_fill = arr_fill_p - fill_gts
    mae_fill = float(np.mean(np.abs(residuals_fill)))
    rmse_fill = float(np.sqrt(np.mean(residuals_fill ** 2)))
    assert mae_fill >= 0.0, f"TRIPWIRE: negative MAE_fill={mae_fill}"
    assert rmse_fill >= 0.0, f"TRIPWIRE: negative RMSE_fill={rmse_fill}"

    return {
        "mae_drop_nan": round(mae_drop, 4) if mae_drop is not None else None,
        "rmse_drop_nan": round(rmse_drop, 4) if rmse_drop is not None else None,
        "mae_fill_nan": round(mae_fill, 4),
        "rmse_fill_nan": round(rmse_fill, 4),
    }


def build_ego_graph_edges(
    node_id: int,
    edge_attention_data: Dict[Tuple[int, int], List[float]],
    node_coord_map: Dict[int, Tuple[float, float]],
    num_timesteps: int,
    top_k: int = 10,
) -> List[Dict]:
    """
    Extract top-K incoming and top-K outgoing GAT attention edges for node_id.

    Returns the full per-timestep weight_sequence (length = num_timesteps) so the
    frontend can rank edges dynamically at the active timestep instead of using a
    static mean scalar.

    Edges where either endpoint has NaN coordinates are skipped (external data
    quality issue, not a model failure — logged at WARNING level).

    Args:
        node_id:             the focal node
        edge_attention_data: {(src, tgt): List[float]} — full attention history
        node_coord_map:      {node_id: (lat, lon)}
        num_timesteps:       expected length of every attention sequence (tripwire)
        top_k:               max edges returned per direction

    Returns:
        List of up to 2*top_k dicts, each with keys:
        source_id, target_id, direction, weight_sequence,
        source_lat, source_lon, target_lat, target_lon
    """
    incoming: List[Dict] = []
    outgoing: List[Dict] = []

    for (src, tgt), attn_seq in edge_attention_data.items():
        if src != node_id and tgt != node_id:
            continue
        direction = "incoming" if tgt == node_id else "outgoing"

        if not attn_seq:
            logger.warning(
                f"ego_graph node {node_id}: empty attention sequence for edge "
                f"({src},{tgt}) — skipped"
            )
            continue

        # Offensive tripwire: every sequence must span the full inference window
        assert len(attn_seq) == num_timesteps, (
            f"TRIPWIRE: edge ({src},{tgt}) attention_sequence length "
            f"{len(attn_seq)} != expected {num_timesteps}"
        )

        mean_w = float(np.mean(attn_seq))
        assert not math.isnan(mean_w), (
            f"TRIPWIRE: NaN mean attention for edge ({src},{tgt})"
        )

        if src not in node_coord_map:
            raise KeyError(f"CRITICAL: ego_graph node {node_id}: source node {src} has no coordinates")
        if tgt not in node_coord_map:
            raise KeyError(f"CRITICAL: ego_graph node {node_id}: target node {tgt} has no coordinates")
        src_lat, src_lon = node_coord_map[src]
        tgt_lat, tgt_lon = node_coord_map[tgt]

        entry = {
            "source_id": src,
            "target_id": tgt,
            "direction": direction,
            "weight_sequence": [round(w, 6) for w in attn_seq],
            "_mean_w": mean_w,  # transient sort key — stripped before return
            "source_lat": src_lat,
            "source_lon": src_lon,
            "target_lat": tgt_lat,
            "target_lon": tgt_lon,
        }
        (incoming if direction == "incoming" else outgoing).append(entry)

    incoming.sort(key=lambda x: x["_mean_w"], reverse=True)
    outgoing.sort(key=lambda x: x["_mean_w"], reverse=True)

    result = incoming[:top_k] + outgoing[:top_k]
    for e in result:
        del e["_mean_w"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()
    device = torch.device("cpu")  # CPU runner as specified
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 0. Build connector from config
    # ------------------------------------------------------------------
    analysis_cfg: dict = {}
    if os.path.exists(args.config):
        import yaml as _yaml
        with open(args.config) as _fh:
            analysis_cfg = _yaml.safe_load(_fh) or {}
    else:
        raise FileNotFoundError(
            f"CRITICAL: analysis config not found at {args.config!r}. "
            "Pass --config with the correct path."
        )

    export_cfg = analysis_cfg["export"]
    num_timesteps_target: int = export_cfg["num_timesteps_target"]
    top_k_ego_edges: int = export_cfg["top_k_ego_edges"]
    ncl_lat_min: float = export_cfg["ncl_lat_min"]
    ncl_lat_max: float = export_cfg["ncl_lat_max"]
    ncl_lon_min: float = export_cfg["ncl_lon_min"]
    ncl_lon_max: float = export_cfg["ncl_lon_max"]
    num_simulated_nodes: int = export_cfg.get("num_simulated_nodes", 50)
    assert num_simulated_nodes <= 100, "Maximum simulated nodes is 100"
    mask_prob_export: float = export_cfg["mask_prob"]

    connector = create_connector(analysis_cfg)

    # ------------------------------------------------------------------
    # 1. Load real-world sensor coordinates
    # ------------------------------------------------------------------
    sensor_coords = connector.load_sensor_metadata()
    logger.info(f"Loaded {len(sensor_coords)} sensor coordinates via connector")

    # ------------------------------------------------------------------
    # 2. Load dataset for the test day
    #
    #    Using two-day window (default 2024-05-09 → 2024-05-11) so that
    #    seq_len look-back history exists for day-2 predictions.
    #    Windows for the target day (day 2) start at index:
    #        skip = T_day1 - seq_len   (last seq_len steps of day 1 form context)
    #    where T_day1 ≈ 96 for a full day.
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.start_date} → {args.end_date}")
    dataset = RealIoTDataset(
        connector,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=args.seq_len,
        target_veh_class=args.veh_class,
        target_dir=args.dir,
    )

    # Inverse-transform parameters for channel 0 (traffic count). StandardScaler
    # stores mean_ and scale_ as length-time_series_dim vectors.
    # x_raw = x_scaled * scale_[0] + mean_[0]
    traffic_mean: float = float(dataset.traffic_scaler.mean_[0])
    traffic_scale: float = float(dataset.traffic_scaler.scale_[0])
    assert traffic_scale > 0, (
        f"TRIPWIRE: dataset.scaler.scale_[0]={traffic_scale} must be > 0 "
        f"to inverse-transform without divergence."
    )
    logger.info(
        f"Traffic-channel inverse-transform: mean={traffic_mean:.4f} "
        f"scale={traffic_scale:.4f}"
    )

    num_nodes_real: int = dataset.num_nodes
    static_feat_dim: int = dataset.static_feat_dim
    time_series_dim: int = dataset.time_series_dim

    # Allow args override, fallback to config
    num_simulated: int = args.num_simulated if args.num_simulated is not None else num_simulated_nodes
    num_nodes_total: int = num_nodes_real + num_simulated

    logger.info(
        f"Dataset: {len(dataset)} windows | "
        f"real nodes={num_nodes_real} | "
        f"simulated nodes={num_simulated} | "
        f"total nodes={num_nodes_total}"
    )

    # ------------------------------------------------------------------
    # 3. Determine which 96 windows map to the target day (day 2)
    #
    #    total_times = len(dataset) + seq_len
    #    If two days were loaded: total_times ≈ 192.
    #    Windows at indices [skip, skip+96) predict day-2 timestamps.
    #
    #    Fallback: if fewer windows exist, use all available.
    # ------------------------------------------------------------------
    total_times = len(dataset) + args.seq_len
    # Number of timestamps in day 1 is roughly half of total_times
    # (may vary if data is sparse).  We estimate T_day1 as the number of
    # 15-min intervals between start_date and the midpoint.
    half = total_times // 2
    skip = max(0, half - args.seq_len)  # first window whose prediction lands in day 2

    max_windows = min(len(dataset) - skip, num_timesteps_target)
    if max_windows <= 0:
        # Single-day dataset: use all available windows
        skip = 0
        max_windows = min(len(dataset), num_timesteps_target)

    num_timesteps = max_windows
    logger.info(
        f"Using windows [{skip}, {skip + num_timesteps}) "
        f"→ {num_timesteps} inference timesteps"
    )

    # ------------------------------------------------------------------
    # 4. Build combined node metadata list (real + simulated)
    # ------------------------------------------------------------------
    # Real nodes: sorted location names from the dataset
    real_location_names: List[str] = dataset.locations  # already sorted

    nodes_meta: List[Dict] = []
    for i, loc in enumerate(real_location_names):
        if loc not in sensor_coords:
            raise KeyError(f"CRITICAL: sensor '{loc}' has no coordinates in sensor_coords")
        coords = sensor_coords[loc]
        nodes_meta.append({
            "id": i,
            "location_name": loc,
            "lat": coords["lat"],
            "lon": coords["lon"],
            "is_simulated": False,
        })

    # Instantiate the SimulatedNodeInjector
    bbox = {
        'lat_min': ncl_lat_min,
        'lat_max': ncl_lat_max,
        'lon_min': ncl_lon_min,
        'lon_max': ncl_lon_max
    }
    sim_injector = SimulatedNodeInjector(
        num_nodes=num_simulated,
        seq_len=args.seq_len,
        bbox=bbox,
        static_scaler=dataset.static_scaler,
        device=device
    )

    for j, meta in enumerate(sim_injector.nodes_meta):
        meta["id"] = num_nodes_real + j
        nodes_meta.append(meta)

    location_ids = [m["location_name"] for m in nodes_meta]

    # ------------------------------------------------------------------
    # 5. Load Co-Located Configuration and Initialise Model
    # ------------------------------------------------------------------
    wandb_run_id = "unknown"
    loaded_config_data = {}
    checkpoint_path = args.checkpoint
    artifact_name = args.artifact

    if not checkpoint_path and not artifact_name:
        artifact_name = select_artifact_from_wandb(
            project=args.wandb_project,
            collection_name=args.wandb_collection
        )

    if artifact_name:
        checkpoint_path, config_path, wandb_run_id = load_model_from_wandb(artifact_name)
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.yaml")

    # HARNESS: inductive holdout manifest co-located with the checkpoint. When
    # present, those nodes are blanked at inference (zero-shot predictions) and
    # tagged node_role="holdout" for distinct frontend display.
    holdout_manifest = load_holdout_manifest(os.path.dirname(checkpoint_path))
    holdout_locations = set(holdout_manifest["holdout_location_ids"]) if holdout_manifest else set()
    for meta in nodes_meta:
        if meta["is_simulated"]:
            meta["node_role"] = "simulated"
        else:
            meta["node_role"] = "holdout" if meta["location_name"] in holdout_locations else "observed"
    holdout_mask_total: Optional[torch.Tensor] = None
    if holdout_manifest is not None:
        logger.info(
            f"Holdout manifest found: {len(holdout_locations)} holdout nodes, "
            f"sha={manifest_sha(holdout_manifest)}"
        )
        real_holdout_mask = mask_from_manifest(holdout_manifest, real_location_names, device=device)
        holdout_mask_total = torch.cat([
            real_holdout_mask,
            torch.zeros(num_simulated, dtype=torch.bool, device=device),
        ])

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"CRITICAL: Co-located config not found at {config_path}. "
            "Every checkpoint must ship with an adjacent config.yaml."
        )

    logger.info(f"ARCHITECT TWINNING: Resolving adjacent configuration from: {config_path}")
    with open(config_path, "r") as f:
        loaded_config_data = yaml.safe_load(f)
    model_type = loaded_config_data["model_type"]
    model_cfg = loaded_config_data["model"]
    logger.info(f"Loaded model_type={model_type}, hyperparameters: {model_cfg}")

    # Infer W&B Run ID from folder namespace if not already set by artifact
    if wandb_run_id == "unknown":
        checkpoint_dir = os.path.dirname(checkpoint_path)
        folder_parts = [p for p in checkpoint_dir.split(os.sep) if p]
        if len(folder_parts) >= 1:
            wandb_run_id = folder_parts[-1]

    # Get Git metadata dynamically — fails fast if the repo is not accessible.
    import subprocess
    git_commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()

    # HARNESS: build the served model by family and wrap in a ModelAdapter. The
    # exporter below reads adapter.capabilities to decide which payload sections
    # are genuinely available (baselines emit null uncertainty/graph, never
    # fabricated numbers).
    adapter = build_model_and_adapter(
        model_type, checkpoint_path, model_cfg,
        num_nodes_total=num_nodes_total, static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim, seq_len=args.seq_len, device=device,
    )
    caps = adapter.capabilities
    logger.info("Model loaded and set to eval mode.")

    # ------------------------------------------------------------------
    # 6. Build static features for simulated nodes once (reused every step)
    # ------------------------------------------------------------------
    real_static: torch.Tensor = dataset.static_features.to(device)   # [num_nodes_real, static_feat_dim]
    combined_static = sim_injector.inject_static(real_static)

    assert combined_static.shape == (num_nodes_total, static_feat_dim), \
        f"Static features shape mismatch: {combined_static.shape}"


    # ------------------------------------------------------------------
    # 7. 24-hour inference loop
    # ------------------------------------------------------------------
    # Accumulators — pre-allocate for efficiency
    # timeseries_data[node_id] = list of {t, prediction, uncertainty}
    timeseries_data: List[List[Dict]] = [[] for _ in range(num_nodes_total)]
    mu_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
    # Imputation analytics: three parallel arrays per node, each length=num_timesteps.
    # ground_truth_series uses (number | null) — None for simulated nodes (no physical sensor).
    ground_truth_series: List[List] = [[] for _ in range(num_nodes_total)]
    imputed_series: List[List[float]] = [[] for _ in range(num_nodes_total)]
    mask_series: List[List[bool]] = [[] for _ in range(num_nodes_total)]
    theta_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
    pi_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
    edge_attention_data: Dict[Tuple[int, int], List[float]] = {}
    edge_index_cached: torch.Tensor | None = None

    global_imputation_burden: List[float] = []
    global_healing_quality: List[float] = []
    attention_entropy_series: List[float] = []
    tripwire_audit_trail: List[Dict] = []

    logger.info(f"Starting 24-hour inference loop ({num_timesteps} timesteps) …")

    with torch.no_grad():
        for t, window_idx in enumerate(range(skip, skip + num_timesteps)):
            raw_item = dataset[window_idx]

            # Inject simulated dynamic features and masks
            combined_item = sim_injector.inject_dynamic_item(raw_item)

            combined_dynamic = combined_item["dynamic_features"]
            combined_dynamic = combined_dynamic.unsqueeze(0)

            # HARNESS: blank inductive holdout nodes exactly as during training so
            # their displayed predictions are genuine zero-shot reconstructions.
            if holdout_mask_total is not None:
                combined_dynamic = apply_spatial_holdout(combined_dynamic, holdout_mask_total)

            batch = {
                "static_features": combined_static,
                "dynamic_features": combined_dynamic,
                "structural_masks": combined_item["structural_masks"].unsqueeze(0),
                "simulated_nodes_mask": combined_item["simulated_nodes_mask"],
                "location_ids": location_ids,
            }

            # HARNESS: adapter.predict returns the Predictions contract. Absent
            # capabilities are None (theta/pi/attention for a baseline) — the
            # capability-gated tripwires below fire only for claimed quantities.
            preds = adapter.predict(batch)
            validate_predictions(caps, preds)

            # Record tripwire logs (baselines emit none — the tuple is empty)
            for tw in preds.tripwires:
                tw_copy = dict(tw)
                tw_copy["timestep"] = t
                tripwire_audit_trail.append(tw_copy)

            # -------- Point forecast (every model) --------
            mu: torch.Tensor = preds.mu[0]  # [num_nodes_total]
            assert not torch.isnan(mu).any(), f"t={t}: NaN in mu"

            # -------- Zero-inflation probability — gated on the capability --------
            pi_raw = preds.pi[0] if caps.zero_inflation else None

            # -------- Uncertainty (ZINB dispersion) — gated on the capability --------
            if caps.uncertainty:
                theta = preds.theta[0]                                                    # [num_nodes_total]
                std_dev = compute_std_dev(preds.mu, preds.theta, preds.pi)[0]            # [num_nodes_total]
                assert not torch.isnan(std_dev).any(), f"t={t}: NaN in std_dev"
            else:
                theta = None
                std_dev = None

            # For ZINB models, E[Y] = mu * (1 - pi). For standard models, E[Y] = mu.
            if caps.zero_inflation and pi_raw is not None:
                point_pred = mu * (1.0 - pi_raw)
            else:
                point_pred = mu

            # Store per-node prediction (+ uncertainty/theta/pi when available).
            # A baseline reports uncertainty=None rather than a fabricated band.
            for node_id in range(num_nodes_total):
                timeseries_data[node_id].append({
                    "t": t,
                    "prediction": round(float(point_pred[node_id]), 4),
                    "uncertainty": (None if std_dev is None
                                    else round(float(std_dev[node_id]), 4)),
                })
                mu_data[node_id].append(round(float(mu[node_id]), 4))
                if caps.uncertainty:
                    theta_data[node_id].append(round(float(theta[node_id]), 4))
                if caps.zero_inflation:
                    pi_data[node_id].append(round(float(pi_raw[node_id]), 6))

            # -------- Imputation analytics (length-96 arrays per node) --------
            # imputed: [1, num_nodes_total, seq_len, time_series_dim] (both families)
            imputed_features: torch.Tensor = preds.imputed
            assert imputed_features.shape == (
                1, num_nodes_total, args.seq_len, time_series_dim
            ), (
                f"t={t}: unexpected imputed_features shape "
                f"{tuple(imputed_features.shape)} (expected "
                f"(1, {num_nodes_total}, {args.seq_len}, {time_series_dim}))"
            )
            assert not torch.isnan(imputed_features).any(), \
                f"t={t}: NaN in imputed_features"

            # Take the LAST step of the reconstructed window, traffic channel (0).
            # Inverse-transform from scaled space → raw counts:
            #   x_raw = x_scaled * scale_[0] + mean_[0]
            imputed_last_scaled = imputed_features[0, :, -1, 0]  # [num_nodes_total]
            imputed_last_raw = imputed_last_scaled * traffic_scale + traffic_mean

            # Mask at the last step, traffic channel — boolean per node
            real_mask_last = raw_item["structural_masks"][:, -1, 0]  # [num_nodes_real]

            # Ground truth at the absolute timestamp = window_idx + seq_len - 1
            # (last step inside the input window). Pull directly from
            # dataset.unscaled_traffic which is already in raw-count space.
            gt_index = window_idx + args.seq_len - 1
            assert 0 <= gt_index < dataset.unscaled_traffic.shape[0], \
                f"t={t}: gt_index={gt_index} out of bounds"
            real_gt = dataset.unscaled_traffic[gt_index].cpu().numpy()  # numpy [num_nodes_real]

            mask_np = real_mask_last.cpu().numpy().astype(bool)
            imputed_np = imputed_last_raw[:num_nodes_real].cpu().numpy()
            if mask_np.any():
                mse = np.mean((imputed_np[mask_np] - real_gt[mask_np])**2)
            else:
                mse = 0.0
            global_imputation_burden.append(round(float(mask_np.mean()), 4))
            global_healing_quality.append(round(float(mse), 4))

            for node_id in range(num_nodes_real):
                imputed_series[node_id].append(
                    round(float(imputed_last_raw[node_id]), 4)
                )
                # clamp(min=0) on display-only ground truth (matches the dataset's
                # target_forecast clamp; raw counts cannot be negative).
                ground_truth_series[node_id].append(
                    round(max(0.0, float(real_gt[node_id])), 4)
                )
                mask_series[node_id].append(bool(real_mask_last[node_id]))

            # Simulated nodes: every input was NaN → fully imputed by definition.
            # No physical sensor → no ground truth (encode as None / null in JSON).
            for j in range(num_simulated):
                node_id = num_nodes_real + j
                imputed_series[node_id].append(
                    round(float(imputed_last_raw[node_id]), 4)
                )
                ground_truth_series[node_id].append(None)
                mask_series[node_id].append(True)

            # -------- Attention weights — gated on the graph capability --------
            # Baselines have no graph: edge_index/attention are None, so the
            # edge topology and attention payload sections stay empty (rendered
            # as an absent graph on the frontend, not a fabricated one).
            if caps.graph:
                alpha: torch.Tensor = preds.attention      # [batch*seq_len*E, heads]
                edge_index: torch.Tensor = preds.edge_index  # [2, E]
                num_edges = edge_index.shape[1]

                # Cache edge_index from first timestep (topology is static —
                # derived from fixed static_features, identical across timesteps)
                if edge_index_cached is None:
                    edge_index_cached = edge_index.cpu()
                    for e in range(num_edges):
                        src = int(edge_index_cached[0, e])
                        tgt = int(edge_index_cached[1, e])
                        edge_attention_data[(src, tgt)] = []

                per_edge_attn = extract_timestep_attention(alpha, num_edges, args.seq_len)  # [E]
                assert not torch.isnan(per_edge_attn).any(), f"t={t}: NaN in attention weights"

                entropy = -float(torch.sum(per_edge_attn * torch.log(per_edge_attn + 1e-6)).item())
                attention_entropy_series.append(round(entropy, 4))

                for e in range(num_edges):
                    src = int(edge_index_cached[0, e])
                    tgt = int(edge_index_cached[1, e])
                    edge_attention_data[(src, tgt)].append(round(float(per_edge_attn[e]), 6))
            else:
                num_edges = 0

            if (t + 1) % 10 == 0 or t == num_timesteps - 1:
                logger.info(f"  Timestep {t + 1}/{num_timesteps} complete | edges={num_edges}")

    logger.info("Inference loop complete.")

    # ------------------------------------------------------------------
    # 7b. Compute calibration metrics (global + per-node)
    #
    # PIT and zero-inflation reliability are distributional diagnostics of the
    # ZINB head; a baseline without theta/pi reports them as null (N/A), never
    # as an empty-but-present diagram.
    # ------------------------------------------------------------------
    if caps.uncertainty and caps.zero_inflation:
        logger.info("Computing PIT histogram …")
        pit_histogram = compute_pit_histogram(
            ground_truth_series=ground_truth_series,
            mu_series=mu_data,
            theta_series=theta_data,
            pi_series=pi_data,
            num_real_nodes=num_nodes_real,
        )
        logger.info(f"  PIT histogram computed ({sum(b['density'] for b in pit_histogram):.4f} total density)")
    else:
        pit_histogram = None
        logger.info("  PIT histogram: null (model has no ZINB uncertainty head)")

    if caps.zero_inflation:
        logger.info("Computing zero-inflation reliability …")
        zero_inflation_reliability = compute_zero_inflation_reliability(
            ground_truth_series=ground_truth_series,
            pi_series=pi_data,
            num_real_nodes=num_nodes_real,
            mu_series=mu_data if caps.uncertainty else None,
            theta_series=theta_data if caps.uncertainty else None,
        )
        logger.info(f"  Reliability diagram: {len(zero_inflation_reliability)} non-empty bins")
    else:
        zero_inflation_reliability = None
        logger.info("  Reliability diagram: null (model has no zero-inflation head)")

    logger.info("Computing per-node Two-Extremes missingness bracket …")
    two_extremes_per_node: List[Dict] = []
    for node_id in range(num_nodes_total):
        mu_preds = [ts["prediction"] for ts in timeseries_data[node_id]]
        te = compute_two_extremes(
            ground_truth=ground_truth_series[node_id],
            mu_predictions=mu_preds,
        )
        two_extremes_per_node.append(te)
    logger.info("  Two-Extremes computed for all nodes.")

    # ------------------------------------------------------------------
    # 8. Load semantic tags from Places cache
    # ------------------------------------------------------------------
    logger.info("Loading semantic tags …")
    semantic_tags = load_semantic_tags(nodes_meta)
    tagged_count = sum(1 for v in semantic_tags.values() if v)
    logger.info(f"  {tagged_count}/{len(real_location_names)} sensors have semantic tags")

    # ------------------------------------------------------------------
    # 9. Build 1-hop edge lookup per node — gated on the graph capability
    #
    # Baselines produce no graph, so every node carries an empty edge list and
    # the leaderboard is null (there is no attention to rank), rather than a
    # fabricated all-zero graph.
    # ------------------------------------------------------------------
    node_edges: Dict[int, List[Dict]] = {i: [] for i in range(num_nodes_total)}
    if caps.graph:
        logger.info("Building 1-hop edge lookup …")
        assert edge_index_cached is not None, "TRIPWIRE: No edges accumulated — is the graph empty?"
        num_edges: int = edge_index_cached.shape[1]
        for e in range(num_edges):
            src = int(edge_index_cached[0, e])
            tgt = int(edge_index_cached[1, e])
            attn_seq = edge_attention_data.get((src, tgt), [])
            node_edges[src].append({
                "neighbor_id": tgt,
                "direction": "outgoing",
                "attention_sequence": attn_seq,
            })
            node_edges[tgt].append({
                "neighbor_id": src,
                "direction": "incoming",
                "attention_sequence": attn_seq,
            })

        # 9b. Alpha-Node Leaderboard (outgoing-attention ranking).
        logger.info("Computing Alpha-Node Leaderboard …")
        node_out_influence = {i: 0.0 for i in range(num_nodes_total)}
        for (src, tgt), attn_seq in edge_attention_data.items():
            node_out_influence[src] += sum(attn_seq)
        alpha_node_leaderboard: Optional[List[Dict]] = [
            {
                "node_id": node_id,
                "total_outgoing_attention": round(total_attn, 4),
                "location_name": nodes_meta[node_id]["location_name"] if node_id < len(nodes_meta) else f"Node {node_id}"
            }
            for node_id, total_attn in node_out_influence.items()
        ]
        alpha_node_leaderboard.sort(key=lambda x: x["total_outgoing_attention"], reverse=True)
    else:
        logger.info("Graph sections: null/empty (model has no graph capability)")
        alpha_node_leaderboard = None

    # ------------------------------------------------------------------
    # 10. Build and write master_index.json
    #     (lightweight: timeseries + semantic_tags only, no imputation arrays)
    # ------------------------------------------------------------------
    logger.info("Writing master_index.json …")
    master_nodes: List[Dict] = []
    for meta in nodes_meta:
        node_id = meta["id"]
        master_nodes.append({
            "id": node_id,
            "location_name": meta["location_name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "node_role": meta.get("node_role", "simulated" if meta["is_simulated"] else "observed"),
            "is_simulated": meta["is_simulated"],
            "timeseries": timeseries_data[node_id],
            "semantic_tags": semantic_tags.get(meta["location_name"], []),
            "two_extremes": two_extremes_per_node[node_id],
        })

    edges_payload: List[Dict] = [
        {"source": src, "target": tgt, "attention_sequence": attn_seq}
        for (src, tgt), attn_seq in edge_attention_data.items()
    ]

    master_index: Dict = {
        "metadata": {
            "num_nodes": num_nodes_total,
            "num_real_nodes": num_nodes_real,
            "num_simulated_nodes": num_simulated,
            "num_holdout_nodes": len(holdout_locations),
            "holdout_manifest_sha": manifest_sha(holdout_manifest),
            "timesteps": num_timesteps,
            "seq_len": args.seq_len,
            "veh_class": args.veh_class,
            "dir": args.dir,
            "start_date": args.start_date,
            "end_date": args.end_date,
            # --- ARTIFACT TWINNING INJECTION ---
            "wandb_run_id": wandb_run_id,
            "git_commit_hash": git_commit_hash,
            "hyperparameters": loaded_config_data
        },
        "nodes": master_nodes,
        "edges": edges_payload,
        "pit_histogram": pit_histogram,
        "zero_inflation_reliability": zero_inflation_reliability,
        "global_imputation_burden": global_imputation_burden,
        "global_healing_quality": global_healing_quality,
        "attention_entropy_series": attention_entropy_series,
        "tripwire_audit_trail": tripwire_audit_trail,
        "alpha_node_leaderboard": alpha_node_leaderboard,
    }

    # Schema assertions before write
    assert len(master_index["nodes"]) == num_nodes_total, "Node count mismatch in master_index"
    sim_nodes_found = [n for n in master_index["nodes"] if n["is_simulated"]]
    assert len(sim_nodes_found) == num_simulated, \
        f"Expected {num_simulated} simulated nodes, found {len(sim_nodes_found)}"
    # PIT / reliability are null for models without a ZINB head; when present
    # they must satisfy the fixed-bin schema.
    if master_index["pit_histogram"] is not None:
        assert len(master_index["pit_histogram"]) == 10, \
            f"PIT histogram must have 10 bins, got {len(master_index['pit_histogram'])}"
    if master_index["zero_inflation_reliability"] is not None:
        assert len(master_index["zero_inflation_reliability"]) <= 10, \
            "Reliability diagram must have at most 10 bins"
    for node in master_index["nodes"]:
        assert len(node["timeseries"]) == num_timesteps, \
            f"master_index node {node['id']} timeseries length mismatch"
        assert isinstance(node["semantic_tags"], list), \
            f"Node {node['id']} semantic_tags must be a list"
        te = node["two_extremes"]
        assert te is not None, f"Node {node['id']} missing two_extremes"
        assert te["mae_fill_nan"] >= 0.0, f"Node {node['id']} mae_fill_nan is negative"
        assert te["rmse_fill_nan"] >= 0.0, f"Node {node['id']} rmse_fill_nan is negative"
    for edge in master_index["edges"]:
        assert len(edge["attention_sequence"]) == num_timesteps, \
            f"Edge ({edge['source']},{edge['target']}) attention_sequence length mismatch"

    # Output location: served frontend dir by default; --output_dir for scratch
    # exports that must not clobber the committed payload.
    master_index_path = os.path.join(args.output_dir, "master_index.json")
    nodes_dir = os.path.join(args.output_dir, "nodes")

    os.makedirs(os.path.dirname(master_index_path) or ".", exist_ok=True)
    with open(master_index_path, "w") as fh:
        json.dump(master_index, fh, indent=2)
    logger.info(
        f"master_index.json written  "
        f"({os.path.getsize(master_index_path) / 1024:.1f} KB) | "
        f"nodes={num_nodes_total} | edges={len(edges_payload)}"
    )

    # ------------------------------------------------------------------
    # 11. Build and write per-node node_{id}.json
    #     (deep data: zinb_series + imputation arrays + 1-hop edges)
    # ------------------------------------------------------------------
    logger.info(f"Writing {num_nodes_total} per-node files to {nodes_dir}/ …")
    os.makedirs(nodes_dir, exist_ok=True)

    # Coordinate lookup for ego-graph edge embedding (built once, reused per node)
    node_coord_map: Dict[int, Tuple[float, float]] = {
        m["id"]: (m["lat"], m["lon"]) for m in nodes_meta
    }

    for meta in nodes_meta:
        node_id = meta["id"]

        # Temporal alignment tripwire: mask_series and zinb entries must all be
        # exactly num_timesteps long. theta/pi are only accumulated for models
        # with the corresponding capability, so gate their length checks.
        assert len(mask_series[node_id]) == num_timesteps, (
            f"TRIPWIRE: temporal misalignment for node {node_id}: "
            f"mask_series={len(mask_series[node_id])}, expected={num_timesteps}"
        )
        if caps.uncertainty:
            assert len(theta_data[node_id]) == num_timesteps, \
                f"TRIPWIRE: theta misalignment for node {node_id}"
        if caps.zero_inflation:
            assert len(pi_data[node_id]) == num_timesteps, \
                f"TRIPWIRE: pi misalignment for node {node_id}"

        # Build zinb_series. theta/std_dev/pi are null for models without the
        # matching capability (a baseline reports N/A, never a fabricated ZINB
        # parameter); NaN tripwires apply only to genuinely-present values.
        zinb_series: List[Dict] = []
        for t_idx in range(num_timesteps):
            entry: Dict = {
                "t": t_idx,
                "mu": timeseries_data[node_id][t_idx]["prediction"],
                "theta": theta_data[node_id][t_idx] if caps.uncertainty else None,
                "std_dev": timeseries_data[node_id][t_idx]["uncertainty"],
                "pi": pi_data[node_id][t_idx] if caps.zero_inflation else None,
                "is_missing": mask_series[node_id][t_idx],
            }
            assert not math.isnan(entry["mu"]), \
                f"NaN in node {node_id} zinb mu at t={t_idx}"
            if entry["theta"] is not None:
                assert not math.isnan(entry["theta"]), \
                    f"NaN in node {node_id} zinb theta at t={t_idx}"
            if entry["std_dev"] is not None:
                assert not math.isnan(entry["std_dev"]), \
                    f"NaN in node {node_id} zinb std_dev at t={t_idx}"
            if entry["pi"] is not None:
                assert not math.isnan(entry["pi"]), \
                    f"NaN in node {node_id} zinb pi at t={t_idx}"
            assert isinstance(entry["is_missing"], bool), \
                f"is_missing must be bool for node {node_id} at t={t_idx}"
            zinb_series.append(entry)

        for v in imputed_series[node_id]:
            assert not math.isnan(v), f"NaN in node {node_id} imputed_series"

        # Localized ego-graph: top-K incoming + outgoing attention edges. Absent
        # for graph-less baselines (empty list — the frontend renders no ego graph).
        if caps.graph:
            ego_edges = build_ego_graph_edges(
                node_id, edge_attention_data, node_coord_map, num_timesteps,
                top_k=top_k_ego_edges,
            )
            assert isinstance(ego_edges, list)
            for e in ego_edges:
                assert len(e["weight_sequence"]) == num_timesteps, (
                    f"ego_graph node {node_id}: weight_sequence length mismatch"
                )
                assert all(w >= 0.0 for w in e["weight_sequence"]), \
                    f"Negative weight in node {node_id} ego_graph: {e}"
                assert e["direction"] in ("incoming", "outgoing"), \
                    f"Invalid direction in node {node_id} ego_graph: {e['direction']}"
        else:
            ego_edges = []

        node_data: Dict = {
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

        node_path = os.path.join(nodes_dir, f"node_{node_id}.json")
        with open(node_path, "w") as fh:
            json.dump(node_data, fh, indent=2)

    logger.info(
        f"Per-node files written | "
        f"nodes={num_nodes_total} | "
        f"timesteps={num_timesteps}"
    )
    logger.info("export_ui_payload.py COMPLETE.")


if __name__ == "__main__":
    main()
