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

from dataset import RealIoTDataset
from features import StaticFeatureLoader
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
CHECKPOINT_PATH = "models/resilient-iot/feasible-capybara-22/best_model.pt"
SENSORS_PATH = "data/traffic_sensors.json"
PARQUET_PATH = "data/processed/master_dataset_15T.parquet"
MASTER_INDEX_PATH = "frontend/public/data/master_index.json"
NODES_DIR = "frontend/public/data/nodes"
PLACES_CACHE_PATH = "data/external/places_cache.json"


# Newcastle city bounding box (WGS84) for simulated node placement
NCL_LAT_MIN, NCL_LAT_MAX = 54.968, 55.012
NCL_LON_MIN, NCL_LON_MAX = -1.660, -1.576

# Five deterministic simulated node positions within the bounding box
SIMULATED_NODE_COORDS: List[Tuple[float, float]] = [
    (54.978, -1.595),
    (55.005, -1.580),
    (54.977851, -1.612579), # Saint Marys Place
    (54.976248, -1.610691), # John Dobson Street Bottom
    (54.973199, -1.611128), # Pilgrim Street / Market Street
]

NUM_TIMESTEPS_TARGET = 96  # 24 h × 4 per hour
TOP_K_EGO_EDGES = 10      # top K incoming + top K outgoing per node in ego_graph_edges


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ui_payload.json for the Geospatial Digital Twin"
    )
    # Load two days so we have seq_len context for day 2 predictions
    parser.add_argument("--start_date", type=str, default="2025-05-10",
                        help="Dataset start date (YYYY-MM-DD).  Load one day earlier than "
                             "the target day to supply seq_len look-back context.")
    parser.add_argument("--end_date", type=str, default="2025-05-12",
                        help="Dataset end date (YYYY-MM-DD, exclusive).")
    parser.add_argument("--veh_class", type=str, default="person",
                        help="Vehicle class filter used during training.")
    parser.add_argument("--dir", type=str, default="all",
                        help="Spatial direction filter used during training.")
    parser.add_argument("--seq_len", type=int, default=12,
                        help="Sequence length (must match trained model).")
    parser.add_argument("--num_simulated", type=int, default=5,
                        help="Number of synthetic zero-shot nodes to inject.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help=f"Path to the trained model checkpoint (default: {CHECKPOINT_PATH})")
    parser.add_argument("--artifact", type=str, default=None,
                        help="W&B artifact name (e.g., 'entity/project/name:latest'). "
                             "If provided, overrides --checkpoint.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sensor_coords(path: str) -> Dict[str, Dict[str, float]]:
    """Return {location_name: {lat, lon}} from traffic_sensors.json."""
    with open(path, "r") as fh:
        raw = json.load(fh)
    return {entry["location"]: {"lat": entry["lat"], "lon": entry["lon"]} for entry in raw}


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

    # Get the run ID from the artifact's producer
    wandb_run_id = "unknown"
    if artifact.producer:
        wandb_run_id = artifact.producer.id

    return checkpoint_path, config_path, wandb_run_id


def build_simulated_static(
    real_static: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Create static features for n simulated nodes.

    Uses the mean of the real nodes' (already-scaled) static features so that
    the graph learner sees plausible, in-distribution embeddings.

    Args:
        real_static: [num_real_nodes, static_feat_dim]
        n: number of simulated nodes to create

    Returns:
        [n, static_feat_dim]
    """
    mean_static = real_static.mean(dim=0, keepdim=True)  # [1, static_feat_dim]
    return mean_static.expand(n, -1)  # [n, static_feat_dim]


def compute_std_dev(mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Variance of the Negative Binomial marginal of the ZINB distribution:
        Var = mu + mu² / theta

    Args:
        mu:    [batch_size, num_nodes]  — mean parameter (already softplus'd + eps)
        theta: [batch_size, num_nodes]  — dispersion parameter (already softplus'd + eps)

    Returns:
        std_dev: [batch_size, num_nodes]
    """
    variance = mu + (mu ** 2) / theta
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

    # Live Query for simulated nodes missing from cache
    for meta in nodes_meta:
        loc = meta["location_name"]
        if meta["is_simulated"] and loc not in loader.cache:
            logger.info(f"Fetching live data for simulated node: {loc}")
            # Note: Using private method as requested in Task 3.2
            loader.cache[loc] = loader._fetch_location_data(meta["lat"], meta["lon"])
            loader._save_cache()

    cache = loader.cache

    # Process features and append tags from Cache
    for loc, feats in cache.items():
        if loc not in tags:
            continue
        t = tags[loc]

        # Commercial density
        cd = int(feats.get("commercial_density", 0))
        if cd >= 15:
            t.append("high_commercial_density")
        elif cd >= 5:
            t.append("medium_commercial_density")
        elif cd > 0:
            t.append("low_commercial_density")

        # School zone
        if feats.get("is_school_zone"):
            t.append("school_zone")

        # Transit density
        hp = int(feats.get("highway_proximity", 0))
        if hp >= 15:
            t.append("high_transit_density")
        elif hp >= 5:
            t.append("medium_transit_density")
        elif hp > 0:
            t.append("low_transit_density")

        # POI density (migrated from template CSV to live cache)
        poi = int(feats.get("nearby_poi_count", 0))
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

def _zinb_cdf(y_obs: float, mu: float, theta: float, pi_val: float) -> float:
    """
    ZINB CDF at observed count y_obs.

    F_NB(y; mu, theta) = betainc(theta, floor(y)+1, theta/(theta+mu))
    F_ZINB(y)          = pi + (1 - pi) * F_NB(y)

    Both mu and theta are guaranteed > 0 by upstream tripwires.
    """
    assert mu > 0.0,    f"_zinb_cdf: mu={mu} must be > 0"
    assert theta > 0.0, f"_zinb_cdf: theta={theta} must be > 0"
    assert 0.0 < pi_val < 1.0, f"_zinb_cdf: pi={pi_val} must be in (0,1)"
    k = int(math.floor(y_obs))
    p_nb = theta / (theta + mu)
    cdf_nb = float(betainc(theta, k + 1, p_nb))
    u = pi_val + (1.0 - pi_val) * cdf_nb
    assert 0.0 <= u <= 1.0, f"_zinb_cdf: PIT value {u} outside [0,1] — parametric error"
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
            u = _zinb_cdf(float(gt), float(mu_val), float(th_val), float(pi_val))
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
) -> List[Dict]:
    """
    Compute the zero-inflation reliability diagram.

    For each valid (non-None, non-NaN) observation, pairs the predicted pi (zero-inflation
    probability) with whether the observed count is a true structural zero (y_obs == 0).
    Bins pi into num_bins equal-width buckets across [0, 1] and computes the actual
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

    for node_id in range(num_real_nodes):
        gt_seq = ground_truth_series[node_id]
        pi_seq = pi_series[node_id]
        assert len(gt_seq) == len(pi_seq), (
            f"Reliability: length mismatch node {node_id}: gt={len(gt_seq)} pi={len(pi_seq)}"
        )
        for gt, pi_val in zip(gt_seq, pi_seq):
            if gt is None or math.isnan(gt):
                continue
            assert 0.0 < pi_val < 1.0, f"Reliability: pi={pi_val} outside (0,1)"
            bin_idx = min(int(pi_val * num_bins), num_bins - 1)
            total_counts[bin_idx] += 1
            if gt == 0.0:
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
    top_k: int = TOP_K_EGO_EDGES,
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

        src_lat, src_lon = node_coord_map.get(src, (float("nan"), float("nan")))
        tgt_lat, tgt_lon = node_coord_map.get(tgt, (float("nan"), float("nan")))

        if any(math.isnan(v) for v in (src_lat, src_lon, tgt_lat, tgt_lon)):
            logger.warning(
                f"ego_graph node {node_id}: NaN coordinates for edge ({src},{tgt}) — skipped"
            )
            continue

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
    # 1. Load real-world sensor coordinates
    # ------------------------------------------------------------------
    logger.info(f"Loading sensor coordinates from {SENSORS_PATH}")
    sensor_coords = load_sensor_coords(SENSORS_PATH)

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
    # mask_prob=0.15 (training default) so node_mask + target_impute carry meaningful
    # signal: the model's imputation reconstructs the masked timesteps and the UI
    # can visualise the "self-healing" behaviour at the masked positions.
    dataset = RealIoTDataset(
        parquet_path=PARQUET_PATH,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=args.seq_len,
        mask_prob=0.15,
        target_veh_class=args.veh_class,
        target_dir=args.dir,
    )

    # Inverse-transform parameters for channel 0 (traffic count). StandardScaler
    # stores mean_ and scale_ as length-time_series_dim vectors.
    # x_raw = x_scaled * scale_[0] + mean_[0]
    traffic_mean: float = float(dataset.scaler.mean_[0])
    traffic_scale: float = float(dataset.scaler.scale_[0])
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
    num_simulated: int = args.num_simulated
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

    max_windows = min(len(dataset) - skip, NUM_TIMESTEPS_TARGET)
    if max_windows <= 0:
        # Single-day dataset: use all available windows
        skip = 0
        max_windows = min(len(dataset), NUM_TIMESTEPS_TARGET)

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
        coords = sensor_coords.get(loc, {"lat": float("nan"), "lon": float("nan")})
        nodes_meta.append({
            "id": i,
            "location_name": loc,
            "lat": coords["lat"],
            "lon": coords["lon"],
            "is_simulated": False,
        })

    for j in range(num_simulated):
        node_id = num_nodes_real + j
        lat, lon = SIMULATED_NODE_COORDS[j]
        nodes_meta.append({
            "id": node_id,
            "location_name": f"Simulated_Node_{j}",
            "lat": lat,
            "lon": lon,
            "is_simulated": True,
        })

    # ------------------------------------------------------------------
    # 5. Load Co-Located Configuration and Initialise Model
    # ------------------------------------------------------------------
    wandb_run_id = "unknown"
    loaded_config_data = {}
    checkpoint_path = args.checkpoint

    if args.artifact:
        checkpoint_path, config_path, wandb_run_id = load_model_from_wandb(args.artifact)
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.yaml")

    # Defaults in case no config.yaml is found (backward-compatibility fallback)
    model_cfg = {
        "embedding_dim": 16,
        "hidden_dim": 32,
        "sparsity_threshold": 0.5,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "dim_feedforward": None,
        "zinb_eps": 1e-6
    }

    if os.path.exists(config_path):
        logger.info(f"ARCHITECT TWINNING: Resolving adjacent configuration from: {config_path}")
        with open(config_path, "r") as f:
            loaded_config_data = yaml.safe_load(f)
        if "model" in loaded_config_data:
            model_cfg.update(loaded_config_data["model"])
            logger.info(f"Loaded model hyperparameters: {model_cfg}")

        # Infer W&B Run ID from folder namespace if not already set by artifact
        if wandb_run_id == "unknown":
            checkpoint_dir = os.path.dirname(checkpoint_path)
            folder_parts = [p for p in checkpoint_dir.split(os.sep) if p]
            if len(folder_parts) >= 1:
                wandb_run_id = folder_parts[-1]
    else:
        logger.warning(f"No co-located config found at {config_path}! Using defaults. WARNING: Potential shape mismatch risk.")

    # Get Git metadata dynamically
    git_commit_hash = "unknown"
    try:
        import subprocess
        git_commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass

    logger.info(f"Initialising ResilientIoTPipeline ...")
    model = ResilientIoTPipeline(
        num_nodes=num_nodes_total,
        static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim,
        seq_len=args.seq_len,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        sparsity_threshold=model_cfg["sparsity_threshold"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        dim_feedforward=model_cfg["dim_feedforward"],
        zinb_eps=model_cfg["zinb_eps"]
    ).to(device)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # ------------------------------------------------------------------
    # 6. Build static features for simulated nodes once (reused every step)
    # ------------------------------------------------------------------
    real_static: torch.Tensor = dataset.static_features.to(device)   # [num_nodes_real, 5]
    sim_static: torch.Tensor = build_simulated_static(real_static, num_simulated)  # [5, 5]
    combined_static: torch.Tensor = torch.cat([real_static, sim_static], dim=0)    # [total, 5]

    assert combined_static.shape == (num_nodes_total, 5), \
        f"Static features shape mismatch: {combined_static.shape}"

    # ------------------------------------------------------------------
    # 7. 24-hour inference loop
    # ------------------------------------------------------------------
    # Accumulators — pre-allocate for efficiency
    # timeseries_data[node_id] = list of {t, prediction, uncertainty}
    timeseries_data: List[List[Dict]] = [[] for _ in range(num_nodes_total)]
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

            # Real node dynamic features: [num_nodes_real, seq_len, time_series_dim]
            real_dynamic: torch.Tensor = raw_item["dynamic_features"].to(device)
            assert real_dynamic.shape == (num_nodes_real, args.seq_len, time_series_dim), \
                f"t={t}: unexpected real_dynamic shape {real_dynamic.shape}"

            # Real node boolean mask: [num_nodes_real, seq_len, time_series_dim]
            real_node_mask: torch.Tensor = raw_item["node_mask"].to(device)
            assert real_node_mask.shape == real_dynamic.shape, (
                f"t={t}: node_mask shape {real_node_mask.shape} != "
                f"dynamic shape {real_dynamic.shape}"
            )

            # Simulated node dynamic features: all NaN → zeros via nan_to_num in pipeline
            sim_dynamic: torch.Tensor = torch.full(
                (num_simulated, args.seq_len, time_series_dim),
                float("nan"),
                device=device,
            )

            # Combined: [num_nodes_total, seq_len, time_series_dim]
            combined_dynamic = torch.cat([real_dynamic, sim_dynamic], dim=0)

            # Add batch dimension: [1, num_nodes_total, seq_len, time_series_dim]
            combined_dynamic = combined_dynamic.unsqueeze(0)

            batch = {
                "static_features": combined_static,
                "dynamic_features": combined_dynamic,
            }

            # Forward pass (analysis mode: returns graph + ZINB params + imputation, no loss)
            result = model(batch, return_analysis=True)

            # Record tripwire logs
            for tw in result.get("tripwire_logs", []):
                tw_copy = dict(tw)
                tw_copy["timestep"] = t
                tripwire_audit_trail.append(tw_copy)

            # -------- ZINB parameters --------
            # mu, theta: [1, num_nodes_total]  (batch_size=1)
            mu: torch.Tensor = result["mu"][0]       # [num_nodes_total]
            theta: torch.Tensor = result["theta"][0]  # [num_nodes_total]
            std_dev: torch.Tensor = compute_std_dev(
                result["mu"], result["theta"]
            )[0]  # [num_nodes_total]

            # ZINB non-negative tripwires (Offensive Programming)
            assert (mu > 0).all(), \
                f"t={t}: ZINB tripwire — mu has non-positive values: {mu.min()}"
            assert (theta > 0).all(), \
                f"t={t}: ZINB tripwire — theta has non-positive values: {theta.min()}"

            # NaN tripwire on outputs
            assert not torch.isnan(mu).any(), f"t={t}: NaN in mu"
            assert not torch.isnan(std_dev).any(), f"t={t}: NaN in std_dev"

            # pi — zero-inflation probability from sigmoid: (0, 1) strictly
            pi_raw: torch.Tensor = result["pi"][0]  # [num_nodes_total]
            assert (pi_raw > 0).all(), \
                f"t={t}: TRIPWIRE — pi has zero/negative values: {pi_raw.min()}"
            assert (pi_raw < 1).all(), \
                f"t={t}: TRIPWIRE — pi has saturated-to-1 values: {pi_raw.max()}"
            assert not torch.isnan(pi_raw).any(), f"t={t}: NaN in pi"

            # Store per-node prediction, uncertainty, theta, and pi
            for node_id in range(num_nodes_total):
                timeseries_data[node_id].append({
                    "t": t,
                    "prediction": round(float(mu[node_id]), 4),
                    "uncertainty": round(float(std_dev[node_id]), 4),
                })
                theta_data[node_id].append(round(float(theta[node_id]), 4))
                pi_data[node_id].append(round(float(pi_raw[node_id]), 6))

            # -------- Imputation analytics (length-96 arrays per node) --------
            # imputed_features: [1, num_nodes_total, seq_len, time_series_dim]
            imputed_features: torch.Tensor = result["imputed_features"]
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
            real_mask_last = real_node_mask[:, -1, 0]  # [num_nodes_real]

            # Ground truth at the absolute timestamp = window_idx + seq_len - 1
            # (last step inside the input window). Pull directly from
            # dataset.unscaled_traffic which is already in raw-count space.
            gt_index = window_idx + args.seq_len - 1
            assert 0 <= gt_index < dataset.unscaled_traffic.shape[0], \
                f"t={t}: gt_index={gt_index} out of bounds"
            real_gt = dataset.unscaled_traffic[gt_index]  # numpy [num_nodes_real]

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

            # -------- Attention weights --------
            alpha: torch.Tensor = result["attention_weights"]  # [batch*seq_len*E, heads]
            edge_index: torch.Tensor = result["edge_index"]    # [2, E]
            num_edges = edge_index.shape[1]

            # Cache edge_index from first timestep (topology is static — derived
            # from fixed static_features, deterministic across all timesteps)
            if edge_index_cached is None:
                edge_index_cached = edge_index.cpu()
                # Pre-initialise attention accumulator for every edge
                for e in range(num_edges):
                    src = int(edge_index_cached[0, e])
                    tgt = int(edge_index_cached[1, e])
                    edge_attention_data[(src, tgt)] = []

            # Average attention across heads for the last step in the sequence
            per_edge_attn = extract_timestep_attention(alpha, num_edges, args.seq_len)  # [E]
            assert not torch.isnan(per_edge_attn).any(), f"t={t}: NaN in attention weights"

            entropy = -float(torch.sum(per_edge_attn * torch.log(per_edge_attn + 1e-6)).item())
            attention_entropy_series.append(round(entropy, 4))

            for e in range(num_edges):
                src = int(edge_index_cached[0, e])
                tgt = int(edge_index_cached[1, e])
                edge_attention_data[(src, tgt)].append(round(float(per_edge_attn[e]), 6))

            if (t + 1) % 10 == 0 or t == num_timesteps - 1:
                logger.info(f"  Timestep {t + 1}/{num_timesteps} complete | edges={num_edges}")

    logger.info("Inference loop complete.")

    # ------------------------------------------------------------------
    # 7b. Compute calibration metrics (global + per-node)
    # ------------------------------------------------------------------
    logger.info("Computing PIT histogram …")
    pit_histogram = compute_pit_histogram(
        ground_truth_series=ground_truth_series,
        mu_series=[[ts["prediction"] for ts in timeseries_data[n]] for n in range(num_nodes_total)],
        theta_series=theta_data,
        pi_series=pi_data,
        num_real_nodes=num_nodes_real,
    )
    logger.info(f"  PIT histogram computed ({sum(b['density'] for b in pit_histogram):.4f} total density)")

    logger.info("Computing zero-inflation reliability …")
    zero_inflation_reliability = compute_zero_inflation_reliability(
        ground_truth_series=ground_truth_series,
        pi_series=pi_data,
        num_real_nodes=num_nodes_real,
    )
    logger.info(f"  Reliability diagram: {len(zero_inflation_reliability)} non-empty bins")

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
    # 9. Build 1-hop edge lookup per node
    # ------------------------------------------------------------------
    logger.info("Building 1-hop edge lookup …")
    assert edge_index_cached is not None, "TRIPWIRE: No edges accumulated — is the graph empty?"
    num_edges: int = edge_index_cached.shape[1]
    node_edges: Dict[int, List[Dict]] = {i: [] for i in range(num_nodes_total)}
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

    # ------------------------------------------------------------------
    # 9b. Compute Alpha-Node Leaderboard
    # ------------------------------------------------------------------
    logger.info("Computing Alpha-Node Leaderboard …")
    node_out_influence = {i: 0.0 for i in range(num_nodes_total)}
    for (src, tgt), attn_seq in edge_attention_data.items():
        node_out_influence[src] += sum(attn_seq)

    alpha_node_leaderboard = [
        {
            "node_id": node_id,
            "total_outgoing_attention": round(total_attn, 4),
            "location_name": nodes_meta[node_id]["location_name"] if node_id < len(nodes_meta) else f"Node {node_id}"
        }
        for node_id, total_attn in node_out_influence.items()
    ]
    alpha_node_leaderboard.sort(key=lambda x: x["total_outgoing_attention"], reverse=True)

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
    assert len(master_index["pit_histogram"]) == 10, \
        f"PIT histogram must have 10 bins, got {len(master_index['pit_histogram'])}"
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

    os.makedirs(os.path.dirname(MASTER_INDEX_PATH), exist_ok=True)
    with open(MASTER_INDEX_PATH, "w") as fh:
        json.dump(master_index, fh, indent=2)
    logger.info(
        f"master_index.json written  "
        f"({os.path.getsize(MASTER_INDEX_PATH) / 1024:.1f} KB) | "
        f"nodes={num_nodes_total} | edges={len(edges_payload)}"
    )

    # ------------------------------------------------------------------
    # 11. Build and write per-node node_{id}.json
    #     (deep data: zinb_series + imputation arrays + 1-hop edges)
    # ------------------------------------------------------------------
    logger.info(f"Writing {num_nodes_total} per-node files to {NODES_DIR}/ …")
    os.makedirs(NODES_DIR, exist_ok=True)

    # Coordinate lookup for ego-graph edge embedding (built once, reused per node)
    node_coord_map: Dict[int, Tuple[float, float]] = {
        m["id"]: (m["lat"], m["lon"]) for m in nodes_meta
    }

    for meta in nodes_meta:
        node_id = meta["id"]

        # Temporal alignment tripwire: mask_series, pi_data, and zinb entries
        # must all be exactly num_timesteps long before we interleave them.
        assert len(mask_series[node_id]) == len(pi_data[node_id]) == num_timesteps, (
            f"TRIPWIRE: temporal misalignment for node {node_id}: "
            f"mask_series={len(mask_series[node_id])}, "
            f"pi_data={len(pi_data[node_id])}, expected={num_timesteps}"
        )

        # Build zinb_series with per-timestep NaN tripwires (now includes pi + is_missing)
        zinb_series: List[Dict] = []
        for t_idx in range(num_timesteps):
            entry: Dict = {
                "t": t_idx,
                "mu": timeseries_data[node_id][t_idx]["prediction"],
                "theta": theta_data[node_id][t_idx],
                "std_dev": timeseries_data[node_id][t_idx]["uncertainty"],
                "pi": pi_data[node_id][t_idx],
                "is_missing": mask_series[node_id][t_idx],
            }
            assert not math.isnan(entry["mu"]), \
                f"NaN in node {node_id} zinb mu at t={t_idx}"
            assert not math.isnan(entry["theta"]), \
                f"NaN in node {node_id} zinb theta at t={t_idx}"
            assert not math.isnan(entry["std_dev"]), \
                f"NaN in node {node_id} zinb std_dev at t={t_idx}"
            assert not math.isnan(entry["pi"]), \
                f"NaN in node {node_id} zinb pi at t={t_idx}"
            assert isinstance(entry["is_missing"], bool), \
                f"is_missing must be bool for node {node_id} at t={t_idx}"
            zinb_series.append(entry)

        for v in imputed_series[node_id]:
            assert not math.isnan(v), f"NaN in node {node_id} imputed_series"

        # Post-build schema check: every entry must carry pi and is_missing
        for entry in zinb_series:
            assert "pi" in entry and "is_missing" in entry, \
                f"Node {node_id}: zinb_series entry missing pi or is_missing"

        # Localized ego-graph: top-K incoming + outgoing edges with full weight_sequence
        ego_edges = build_ego_graph_edges(
            node_id, edge_attention_data, node_coord_map, num_timesteps
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

        node_data: Dict = {
            "id": node_id,
            "location_name": meta["location_name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "is_simulated": meta["is_simulated"],
            "zinb_series": zinb_series,
            "ground_truth_series": ground_truth_series[node_id],
            "imputed_series": imputed_series[node_id],
            "mask_series": mask_series[node_id],
            "edges_1hop": node_edges[node_id],
            "two_extremes": two_extremes_per_node[node_id],
            "ego_graph_edges": ego_edges,
        }

        node_path = os.path.join(NODES_DIR, f"node_{node_id}.json")
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
