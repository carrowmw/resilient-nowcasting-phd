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
        [--start_date 2024-05-09] [--end_date 2024-05-11] \
        [--veh_class person] [--dir all] \
        [--seq_len 12] [--num_simulated 5]
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Allow direct execution from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

from dataset import RealIoTDataset
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
#   graph_learner.lin.weight  → [32, 2]  ⟹ embedding_dim=32, static_feat_dim=2
#   encoder.gat.lin_l.weight  → [64, 3]  ⟹ hidden_dim=64, time_series_dim=3
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "models/best_pipeline.pt"
SENSORS_PATH = "data/traffic_sensors.json"
PARQUET_PATH = "data/processed/master_dataset_15T.parquet"
MASTER_INDEX_PATH = "frontend/public/data/master_index.json"
NODES_DIR = "frontend/public/data/nodes"
PLACES_CACHE_PATH = "data/external/places_cache_test.json"
PLACES_TEMPLATE_PATH = "data/external/google_places_template.csv"

HIDDEN_DIM = 64
EMBEDDING_DIM = 32

# Newcastle city bounding box (WGS84) for simulated node placement
NCL_LAT_MIN, NCL_LAT_MAX = 54.968, 55.012
NCL_LON_MIN, NCL_LON_MAX = -1.660, -1.576

# Five deterministic simulated node positions within the bounding box
SIMULATED_NODE_COORDS: List[Tuple[float, float]] = [
    (54.975, -1.645),
    (54.985, -1.630),
    (54.990, -1.610),
    (54.978, -1.595),
    (55.005, -1.580),
]

NUM_TIMESTEPS_TARGET = 96  # 24 h × 4 per hour


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ui_payload.json for the Geospatial Digital Twin"
    )
    # Load two days so we have seq_len context for day 2 predictions
    parser.add_argument("--start_date", type=str, default="2024-05-09",
                        help="Dataset start date (YYYY-MM-DD).  Load one day earlier than "
                             "the target day to supply seq_len look-back context.")
    parser.add_argument("--end_date", type=str, default="2024-05-11",
                        help="Dataset end date (YYYY-MM-DD, exclusive).")
    parser.add_argument("--veh_class", type=str, default="person",
                        help="Vehicle class filter used during training.")
    parser.add_argument("--dir", type=str, default="all",
                        help="Spatial direction filter used during training.")
    parser.add_argument("--seq_len", type=int, default=12,
                        help="Sequence length (must match trained model).")
    parser.add_argument("--num_simulated", type=int, default=5,
                        help="Number of synthetic zero-shot nodes to inject.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sensor_coords(path: str) -> Dict[str, Dict[str, float]]:
    """Return {location_name: {lat, lon}} from traffic_sensors.json."""
    with open(path, "r") as fh:
        raw = json.load(fh)
    return {entry["location"]: {"lat": entry["lat"], "lon": entry["lon"]} for entry in raw}


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
    locations: List[str],
    places_cache_path: str = PLACES_CACHE_PATH,
    template_path: str = PLACES_TEMPLATE_PATH,
) -> Dict[str, List[str]]:
    """
    Build a {location_name: [tag, ...]} mapping from two Places data sources.

    Sources:
      1. places_cache_test.json — live API features: commercial_density,
         is_school_zone, highway_proximity.
      2. google_places_template.csv — zone_type, nearby_poi_count.

    Simulated nodes are not in either source and receive an empty list.
    """
    tags: Dict[str, List[str]] = {loc: [] for loc in locations}

    # Source 1: Live Places API cache
    if Path(places_cache_path).exists():
        with open(places_cache_path) as f:
            cache: Dict[str, Dict] = json.load(f)
        for loc, feats in cache.items():
            if loc not in tags:
                continue
            t = tags[loc]
            cd = int(feats.get("commercial_density", 0))
            if cd >= 15:
                t.append("high_commercial_density")
            elif cd >= 5:
                t.append("medium_commercial_density")
            elif cd > 0:
                t.append("low_commercial_density")
            if feats.get("is_school_zone"):
                t.append("school_zone")
            hp = int(feats.get("highway_proximity", 0))
            if hp >= 15:
                t.append("high_transit_density")
            elif hp >= 5:
                t.append("medium_transit_density")
            elif hp > 0:
                t.append("low_transit_density")

    # Source 2: Template CSV (zone_type, nearby_poi_count)
    if Path(template_path).exists():
        with open(template_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loc = row.get("location_id", "").strip()
                if loc not in tags:
                    continue
                t = tags[loc]
                zone = row.get("zone_type", "").strip()
                if zone:
                    t.append(zone.lower().replace(" ", "_") + "_zone")
                try:
                    poi = int(row.get("nearby_poi_count", 0) or 0)
                except ValueError:
                    poi = 0
                if poi >= 20:
                    t.append("high_poi_density")
                elif poi >= 8:
                    t.append("medium_poi_density")
                elif poi >= 1:
                    t.append("low_poi_density")

    return tags


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
    # 5. Initialise model
    #
    #    Parameters are shape-independent of num_nodes (no per-node
    #    embeddings exist in the architecture), so we can safely
    #    initialise with num_nodes_total and load the checkpoint weights.
    # ------------------------------------------------------------------
    logger.info("Initialising ResilientIoTPipeline …")
    model = ResilientIoTPipeline(
        num_nodes=num_nodes_total,
        static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim,
        seq_len=args.seq_len,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # ------------------------------------------------------------------
    # 6. Build static features for simulated nodes once (reused every step)
    # ------------------------------------------------------------------
    real_static: torch.Tensor = dataset.static_features.to(device)   # [num_nodes_real, 2]
    sim_static: torch.Tensor = build_simulated_static(real_static, num_simulated)  # [5, 2]
    combined_static: torch.Tensor = torch.cat([real_static, sim_static], dim=0)    # [total, 2]

    assert combined_static.shape == (num_nodes_total, static_feat_dim), \
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
    # edge_attention_data: filled once edge topology is known
    edge_attention_data: Dict[Tuple[int, int], List[float]] = {}
    edge_index_cached: torch.Tensor | None = None

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

            # Store per-node prediction, uncertainty, and theta
            for node_id in range(num_nodes_total):
                timeseries_data[node_id].append({
                    "t": t,
                    "prediction": round(float(mu[node_id]), 4),
                    "uncertainty": round(float(std_dev[node_id]), 4),
                })
                theta_data[node_id].append(round(float(theta[node_id]), 4))

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

            for e in range(num_edges):
                src = int(edge_index_cached[0, e])
                tgt = int(edge_index_cached[1, e])
                edge_attention_data[(src, tgt)].append(round(float(per_edge_attn[e]), 6))

            if (t + 1) % 10 == 0 or t == num_timesteps - 1:
                logger.info(f"  Timestep {t + 1}/{num_timesteps} complete | edges={num_edges}")

    logger.info("Inference loop complete.")

    # ------------------------------------------------------------------
    # 8. Load semantic tags from Places cache + template CSV
    # ------------------------------------------------------------------
    logger.info("Loading semantic tags …")
    semantic_tags = load_semantic_tags(real_location_names)
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
        },
        "nodes": master_nodes,
        "edges": edges_payload,
    }

    # Schema assertions before write
    assert len(master_index["nodes"]) == num_nodes_total, "Node count mismatch in master_index"
    sim_nodes_found = [n for n in master_index["nodes"] if n["is_simulated"]]
    assert len(sim_nodes_found) == num_simulated, \
        f"Expected {num_simulated} simulated nodes, found {len(sim_nodes_found)}"
    for node in master_index["nodes"]:
        assert len(node["timeseries"]) == num_timesteps, \
            f"master_index node {node['id']} timeseries length mismatch"
        assert isinstance(node["semantic_tags"], list), \
            f"Node {node['id']} semantic_tags must be a list"
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

    for meta in nodes_meta:
        node_id = meta["id"]

        # Build zinb_series with per-timestep NaN tripwires
        zinb_series: List[Dict] = []
        for t_idx in range(num_timesteps):
            entry: Dict = {
                "t": t_idx,
                "mu": timeseries_data[node_id][t_idx]["prediction"],
                "theta": theta_data[node_id][t_idx],
                "std_dev": timeseries_data[node_id][t_idx]["uncertainty"],
            }
            assert not math.isnan(entry["mu"]), \
                f"NaN in node {node_id} zinb mu at t={t_idx}"
            assert not math.isnan(entry["theta"]), \
                f"NaN in node {node_id} zinb theta at t={t_idx}"
            assert not math.isnan(entry["std_dev"]), \
                f"NaN in node {node_id} zinb std_dev at t={t_idx}"
            zinb_series.append(entry)

        for v in imputed_series[node_id]:
            assert not math.isnan(v), f"NaN in node {node_id} imputed_series"

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
