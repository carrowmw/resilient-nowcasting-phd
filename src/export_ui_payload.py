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
OUTPUT_PATH = "data/processed/ui_payload.json"

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
    dataset = RealIoTDataset(
        parquet_path=PARQUET_PATH,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=args.seq_len,
        mask_prob=0.0,          # no masking during inference
        target_veh_class=args.veh_class,
        target_dir=args.dir,
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

            # Forward pass (analysis mode: returns graph + ZINB params, no loss)
            result = model(batch, return_analysis=True)

            # -------- ZINB parameters --------
            # mu, theta: [1, num_nodes_total]  (batch_size=1)
            mu: torch.Tensor = result["mu"][0]       # [num_nodes_total]
            theta: torch.Tensor = result["theta"][0]  # [num_nodes_total]
            std_dev: torch.Tensor = compute_std_dev(
                result["mu"], result["theta"]
            )[0]  # [num_nodes_total]

            # NaN tripwire on outputs
            assert not torch.isnan(mu).any(), f"t={t}: NaN in mu"
            assert not torch.isnan(std_dev).any(), f"t={t}: NaN in std_dev"

            # Store per-node prediction & uncertainty
            for node_id in range(num_nodes_total):
                timeseries_data[node_id].append({
                    "t": t,
                    "prediction": round(float(mu[node_id]), 4),
                    "uncertainty": round(float(std_dev[node_id]), 4),
                })

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
    # 8. Assemble JSON payload
    # ------------------------------------------------------------------
    logger.info("Assembling JSON payload …")

    nodes_payload: List[Dict] = []
    for meta in nodes_meta:
        node_id = meta["id"]
        nodes_payload.append({
            "id": node_id,
            "location_name": meta["location_name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "is_simulated": meta["is_simulated"],
            "timeseries": timeseries_data[node_id],
        })

    edges_payload: List[Dict] = []
    for (src, tgt), attn_seq in edge_attention_data.items():
        edges_payload.append({
            "source": src,
            "target": tgt,
            "attention_sequence": attn_seq,
        })

    payload = {
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
        "nodes": nodes_payload,
        "edges": edges_payload,
    }

    # ------------------------------------------------------------------
    # 9. Schema validation (fast assertions before writing)
    # ------------------------------------------------------------------
    assert len(payload["nodes"]) == num_nodes_total, "Node count mismatch in payload"
    # Verify simulated nodes are present and flagged
    sim_nodes_found = [n for n in payload["nodes"] if n["is_simulated"]]
    assert len(sim_nodes_found) == num_simulated, \
        f"Expected {num_simulated} simulated nodes, found {len(sim_nodes_found)}"
    # Verify timeseries length
    for node in payload["nodes"]:
        assert len(node["timeseries"]) == num_timesteps, \
            f"Node {node['id']} has {len(node['timeseries'])} timesteps, expected {num_timesteps}"
    # Verify edge attention sequence length
    for edge in payload["edges"]:
        assert len(edge["attention_sequence"]) == num_timesteps, \
            f"Edge ({edge['source']},{edge['target']}) has wrong attention_sequence length"

    logger.info(
        f"Schema validation passed | "
        f"nodes={num_nodes_total} | "
        f"edges={len(edges_payload)} | "
        f"timesteps={num_timesteps}"
    )

    # ------------------------------------------------------------------
    # 10. Write output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
    logger.info(f"Payload written to {OUTPUT_PATH}  ({file_size_mb:.2f} MB)")
    logger.info("export_ui_payload.py COMPLETE.")


if __name__ == "__main__":
    main()
