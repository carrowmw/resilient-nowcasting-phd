"""
export_ui_payload.py

Shared RIoT v5 inference library for the Geospatial Digital Twin payloads.

The paginated exporter (export_paginated_payload.py) drives everything; this module
owns the parts that must match training exactly:

  * build_inference_dataset — the run's RealIoTDataset re-built over
    [config start_date, analysis end) with the train slice pinned to the run's own
    (train_t_end), so the traffic/weather scalers, seasonal profiles, gap banks and
    node_missing_rate are identical to training while the analysis window is
    fully out-of-sample.
  * build_v5_context — BatchPreparer (eval holdout = manifest inductive set,
    artificial gaps disabled), model_factory.build_model, strict checkpoint load
    with scaler tripwires and the v5 RIoTAdapter.
  * predictive-distribution helpers (quantiles, mid-PIT, CRPS) for every v5
    likelihood, reliability bins and GAT attention capture.

v4 is retired: there are no simulated nodes (v5's sensor embedding, profiles and
outage rates are per real sensor), no 3-channel inputs and no ZINB-only maths.

Usage (single run, single day — the paginated exporter is the normal entry point):
    python src/export_ui_payload.py --run v5_full-s42 --date 2026-05-01
"""

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

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

from dataset import RealIoTDataset  # noqa: E402
from harness import (  # noqa: E402
    ModelAdapter,
    build_adapter,
    mask_from_manifest,
    select_network_locations,
)
from harness.batch_prep import BatchPreparer  # noqa: E402
from likelihoods import predictive_pmf_grid  # noqa: E402
from losses import TrafficScale  # noqa: E402
from model_factory import build_model  # noqa: E402
from run_config import resolve_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PLACES_CACHE_PATH = "data/external/places_cache.json"

# Central 80% predictive interval shown as the forecast band.
INTERVAL_LEVELS = (0.1, 0.9)
PIT_BINS = 10
RELIABILITY_BINS = 10


# ---------------------------------------------------------------------------
# W&B artifacts
# ---------------------------------------------------------------------------

@dataclass
class RunArtifact:
    run_name: str
    artifact_ref: str
    checkpoint_path: str
    config_path: str
    manifest_path: Optional[str]
    wandb_run_id: Optional[str]
    git_commit: Optional[str]


def fetch_run_artifact(run_name: str, project: str, collection: str,
                       entity: Optional[str] = None) -> RunArtifact:
    """Download the newest model artifact in `collection` produced by the W&B run
    named `run_name`. Seed replicates share a collection, so the producer run's
    name — not the collection alias — identifies the checkpoint."""
    import wandb

    api = wandb.Api()
    entity = entity or api.default_entity
    collection_path = f"{entity}/{project}/{collection}"
    best = None
    for art in api.artifact_collection("model", collection_path).artifacts():
        producer = art.logged_by()
        if producer is None or producer.name != run_name:
            continue
        version = int(art.version.lstrip("v"))
        if best is None or version > best[0]:
            best = (version, art, producer)
    if best is None:
        raise ValueError(f"CRITICAL: no artifact in {collection_path} was produced by run {run_name!r}")
    _, art, producer = best
    artifact_dir = art.download()
    manifest_path = os.path.join(artifact_dir, "holdout_manifest.json")
    git = (producer.metadata or {}).get("git") or {}
    return RunArtifact(
        run_name=run_name,
        artifact_ref=f"{collection_path}:{art.version}",
        checkpoint_path=os.path.join(artifact_dir, "best_model.pt"),
        config_path=os.path.join(artifact_dir, "config.yaml"),
        manifest_path=manifest_path if os.path.exists(manifest_path) else None,
        wandb_run_id=producer.id,
        git_commit=git.get("commit") or producer.config.get("git_commit"),
    )


def load_run_config(config_path: str) -> Dict:
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh)
    assert raw.get("pipeline_version") == 5, \
        f"CRITICAL: {config_path} is not a v5 run config (pipeline_version={raw.get('pipeline_version')})"
    return resolve_config(raw)


# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------

def semantic_tags_from_cache(location_names: List[str],
                             cache_path: str = PLACES_CACHE_PATH) -> Dict[str, List[str]]:
    """{location_name: [tag, ...]} from the cached Places features (no API calls)."""
    with open(cache_path, "r") as fh:
        cache = json.load(fh)
    tags: Dict[str, List[str]] = {}
    for loc in location_names:
        feats = cache.get(loc)
        t: List[str] = []
        if feats is not None:
            for key, label, cuts in (
                ("retail_count", "commercial_density", (15, 5, 1)),
                ("transit_count", "transit_density", (15, 5, 1)),
                ("total_poi_count", "poi_density", (20, 8, 1)),
            ):
                v = int(feats[key])
                if v >= cuts[0]:
                    t.append(f"high_{label}")
                elif v >= cuts[1]:
                    t.append(f"medium_{label}")
                elif v >= cuts[2]:
                    t.append(f"low_{label}")
            if feats.get("is_school_zone"):
                t.append("school_zone")
        tags[loc] = t
    return tags


# ---------------------------------------------------------------------------
# Dataset faithful to the run's training scalers / profiles
# ---------------------------------------------------------------------------

class CachingConnector:
    """Memoises connector reads so several runs (and the two-pass dataset build)
    hit the database once per distinct query. Returns copies: RealIoTDataset
    treats its inputs as owned frames."""

    def __init__(self, connector):
        self._c = connector
        self._cache: Dict[Tuple, pd.DataFrame] = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key].copy()

    def load_traffic(self, start, end, veh_class, direction):
        return self._get(("traffic", start, end, veh_class, direction),
                         lambda: self._c.load_traffic(start, end, veh_class, direction))

    def load_weather(self, start, end, locations):
        return self._get(("weather", start, end, tuple(sorted(locations))),
                         lambda: self._c.load_weather(start, end, locations))

    def load_static_features(self, locations):
        return self._get(("static", tuple(sorted(locations))),
                         lambda: self._c.load_static_features(locations))

    def load_sensor_metadata(self):
        return self._c.load_sensor_metadata()

    def close(self):
        self._c.close()


def dataset_key(config: Dict) -> Tuple:
    """Data options that change the dataset tensors (runs sharing a key share a dataset)."""
    d = config["data"]
    return (d["start_date"], d["end_date"], d["seq_len"], d["target_veh_class"], d["target_dir"],
            config["train_split"], d["ablate_masks"], d["log1p_inputs"], d["complete_weather"],
            d["train_node_count"], d["train_node_seed"], d["inductive_holdout_frac"],
            d["inductive_holdout_seed"], d["val_station_count"], d["val_station_seed"],
            d["val_station_strategy"], d["val_station_min_coverage"])


def _network_filter(config: Dict):
    data_cfg = config["data"]
    if data_cfg["train_node_count"] is None:
        return None
    return partial(
        select_network_locations,
        holdout_frac=data_cfg["inductive_holdout_frac"],
        holdout_seed=data_cfg["inductive_holdout_seed"],
        train_node_count=data_cfg["train_node_count"],
        train_node_seed=data_cfg["train_node_seed"],
    )


def build_inference_dataset(config: Dict, connector, analysis_end: str) -> RealIoTDataset:
    """RealIoTDataset over [config start_date, analysis_end) whose fitted tables
    (scalers, profiles, banks, node_missing_rate) come from exactly the training
    run's train slice [0, int(train_split * T_train)).

    T_train is the number of timesteps before the config end_date; the train split
    of the extended dataset is chosen so int(split * T_ext) lands on the same index.
    """
    data_cfg = config["data"]
    train_end = pd.Timestamp(data_cfg["end_date"])
    assert pd.Timestamp(analysis_end) > train_end, \
        f"CRITICAL: analysis end {analysis_end} must lie after the training end {train_end.date()}"
    common = dict(
        seq_len=data_cfg["seq_len"],
        target_veh_class=data_cfg["target_veh_class"],
        target_dir=data_cfg["target_dir"],
        ablate_masks=data_cfg["ablate_masks"],
        log1p_inputs=data_cfg["log1p_inputs"],
        complete_weather=data_cfg["complete_weather"],
        location_filter=_network_filter(config),
    )

    def _build(split: float) -> RealIoTDataset:
        return RealIoTDataset(connector, data_cfg["start_date"], analysis_end, train_split=split, **common)

    ds = _build(config["train_split"])
    times = pd.DatetimeIndex(ds.times)
    if times.tz is not None:
        # The connector's `dt < end_date` bound is evaluated in the DB session zone (UTC).
        train_end = train_end.tz_localize("UTC").tz_convert(times.tz)
    t_train = int((times < train_end).sum())
    train_t_end = int(config["train_split"] * t_train)
    if ds.train_t_end != train_t_end:
        ds = _build((train_t_end + 0.5) / len(times))
    assert ds.train_t_end == train_t_end, \
        f"CRITICAL: train slice mismatch ({ds.train_t_end} != {train_t_end})"
    logger.info(
        f"Inference dataset: {len(ds.times)} steps × {ds.num_nodes} nodes, train slice "
        f"[0, {train_t_end}) of the {t_train} training-range steps (seq_len={ds.seq_len})"
    )
    return ds


# ---------------------------------------------------------------------------
# Model / adapter context
# ---------------------------------------------------------------------------

class _NoGaps:
    """GapMasker stand-in: inference sees only the real missingness."""

    def __init__(self, seq_len: int, device):
        self.seq_len, self.device = seq_len, device

    def sample(self, B, N, gen=None):
        return torch.zeros(B, N, self.seq_len + 1, dtype=torch.bool, device=self.device)


@dataclass
class V5Context:
    run_name: str
    config: Dict
    dataset: RealIoTDataset
    preparer: BatchPreparer
    model: torch.nn.Module
    adapter: ModelAdapter
    scale: TrafficScale
    holdout_mask: Optional[torch.Tensor]   # [N] bool, manifest inductive set
    location_ids: List[str]
    likelihood: str
    quantile_levels: Tuple[float, ...]
    device: torch.device

    @property
    def seq_len(self) -> int:
        return self.config["data"]["seq_len"]


def build_v5_context(run_name: str, config: Dict, dataset: RealIoTDataset, checkpoint_path: str,
                     manifest: Optional[Dict], device: torch.device) -> V5Context:
    data_cfg = config["data"]
    assert dataset.seq_len == data_cfg["seq_len"] and dataset.log1p_inputs == data_cfg["log1p_inputs"], \
        "CRITICAL: dataset was built for another data config"
    location_ids = list(dataset.locations)

    holdout_mask = None
    unseen_mask = None
    if config["eval_mode"] == "inductive":
        assert manifest is not None, f"CRITICAL: inductive run {run_name} has no holdout manifest"
        holdout_mask = mask_from_manifest(manifest, location_ids, device=device)
        assert int(holdout_mask.sum()) == len(manifest["holdout_location_ids"]), \
            "CRITICAL: manifest holdout stations missing from the inference dataset"
        assert manifest["num_nodes"] == len(location_ids), \
            f"CRITICAL: manifest has {manifest['num_nodes']} nodes, dataset {len(location_ids)}"
        unseen_mask = holdout_mask
        val_ids = manifest.get("val_location_ids") or []
        if val_ids:
            val_mask = mask_from_manifest({"holdout_location_ids": val_ids}, location_ids, device=device)
            unseen_mask = holdout_mask | val_mask

    preparer = BatchPreparer(
        config, dataset, device,
        inductive_mask=unseen_mask,
        loss_node_mask=None if unseen_mask is None else ~unseen_mask,
        eval_holdout_mask=holdout_mask,
        static_features=dataset.static_features.to(device),
        location_ids=location_ids,
    )
    preparer.masker = _NoGaps(data_cfg["seq_len"], device)
    preparer.legacy_mask = False
    scale = TrafficScale(
        torch.tensor(dataset.traffic_mean, device=device),
        torch.tensor(dataset.traffic_std, device=device),
        data_cfg["log1p_inputs"],
    )

    model = build_model(
        config, dataset, device,
        covariate_dim=preparer.covariates.dim,
        edge_features_cache_path=config["model"]["edge_features_cache_path"],
    )
    state = torch.load(checkpoint_path, map_location=device)
    # Tripwire: the checkpoint's scaler buffers must equal this dataset's train-slice fit.
    for key, value in (("traffic_mean", dataset.traffic_mean), ("traffic_std", dataset.traffic_std)):
        if key in state:
            ckpt_v = float(state[key])
            assert math.isclose(ckpt_v, value, rel_tol=1e-5, abs_tol=1e-6), (
                f"CRITICAL: {run_name}: checkpoint {key}={ckpt_v:.6f} but the rebuilt train slice "
                f"gives {value:.6f} — inference would not match training"
            )
    if "node_missing_rate" in state:
        drift = (state["node_missing_rate"].cpu() - dataset.node_missing_rate).abs().max().item()
        assert drift < 1e-5, f"CRITICAL: {run_name}: node_missing_rate drift {drift:.2e} vs checkpoint"
    model.load_state_dict(state, strict=True)
    model.eval()

    adapter = build_adapter(config["model_type"], model, config, scale=scale, covariates=preparer.covariates)
    likelihood = config["loss"]["forecast_likelihood"]
    return V5Context(
        run_name=run_name, config=config, dataset=dataset, preparer=preparer, model=model,
        adapter=adapter, scale=scale, holdout_mask=holdout_mask, location_ids=location_ids,
        likelihood=likelihood, quantile_levels=tuple(config["model"]["quantiles"] or ()),
        device=device,
    )


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

@contextmanager
def capture_gat_attention(model: torch.nn.Module):
    """Collect (edge_index_used, alpha) from the encoder's GATv2 layer. The encoder
    discards the edge index PyG actually used (missing sources dropped, self-loops
    added), without which alpha cannot be mapped back onto edges."""
    store: Dict[str, torch.Tensor] = {}
    gat = getattr(getattr(model, "encoder", None), "gat", None)
    if gat is None:
        yield store
        return

    def hook(_module, _inputs, output):
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], tuple):
            store["edge_index"], store["alpha"] = output[1]

    handle = gat.register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


def last_step_attention(store: Dict[str, torch.Tensor], base_edges: torch.Tensor,
                        B: int, T: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-window attention at the last look-back step.

    Returns (edge_att [B, E] with NaN where the edge was dropped, entropy [B] =
    mean over target nodes of the attention entropy, self-loops included since the
    softmax normalises over them)."""
    E = base_edges.shape[1]
    att = np.full((B, E), np.nan, dtype=np.float64)
    entropy = np.full(B, np.nan, dtype=np.float64)
    if "alpha" not in store:
        return att, entropy
    ei = store["edge_index"].cpu().numpy()
    alpha = store["alpha"].float().mean(dim=-1).cpu().numpy()                  # mean over heads
    g = ei[1] // N                                                              # graph of the target row
    b, t = g // T, g % T
    last = t == T - 1
    src_n, dst_n, b_l, a_l = ei[0][last] % N, ei[1][last] % N, b[last], alpha[last]

    lookup = {int(s) * N + int(d): e for e, (s, d) in enumerate(base_edges.cpu().numpy().T)}
    for s, d, bb, a in zip(src_n, dst_n, b_l, a_l):
        e = lookup.get(int(s) * N + int(d))
        if e is not None and s != d:
            att[bb, e] = a
    # Entropy of each target's incoming distribution, averaged over targets.
    key = b_l * N + dst_n
    plogp = -a_l * np.log(np.clip(a_l, 1e-12, None))
    ent_sum = np.bincount(key, weights=plogp, minlength=B * N).reshape(B, N)
    has_in = np.bincount(key, minlength=B * N).reshape(B, N) > 0
    for bb in range(B):
        if has_in[bb].any():
            entropy[bb] = ent_sum[bb][has_in[bb]].mean()
    return att, entropy


# ---------------------------------------------------------------------------
# Predictive distribution helpers
# ---------------------------------------------------------------------------

def pmf_grid(likelihood: str, mu, theta, pi, max_count: int) -> torch.Tensor:
    """[..., K + 1] pmf of the count distribution scored on observed slots (the
    outage mixture's count component is NB — evaluate_v5 uses the same)."""
    return predictive_pmf_grid(likelihood, mu.float(), theta.float(),
                               None if pi is None else pi.float(), max_count)


def quantiles_from_pmf(pmf: torch.Tensor, levels) -> torch.Tensor:
    """[..., L] smallest k with CDF(k) >= level."""
    cdf = torch.cumsum(pmf, dim=-1)
    out = [(cdf < lvl).sum(dim=-1).clamp(max=pmf.shape[-1] - 1) for lvl in levels]
    return torch.stack(out, dim=-1).float()


def mid_pit_and_crps(pmf: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mid-PIT and CRPS of integer observations y [...] under pmf [..., K + 1]."""
    K = pmf.shape[-1] - 1
    ys = y.round().long().clamp(0, K)
    cdf = torch.cumsum(pmf, dim=-1)
    p_y = pmf.gather(-1, ys.unsqueeze(-1)).squeeze(-1)
    below = torch.where(ys > 0, cdf.gather(-1, (ys - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
                        torch.zeros_like(p_y))
    k = torch.arange(K + 1, device=pmf.device)
    crps = ((cdf - (k >= ys.unsqueeze(-1)).float()) ** 2).sum(dim=-1)
    return below + 0.5 * p_y, crps


def pit_histogram(pit: np.ndarray) -> Optional[List[Dict]]:
    if pit.size == 0:
        return None
    counts = np.bincount(np.clip((pit * PIT_BINS).astype(int), 0, PIT_BINS - 1), minlength=PIT_BINS)
    total = counts.sum()
    return [{"bin_center": round((i + 0.5) / PIT_BINS, 3), "density": round(float(c) / total, 6)}
            for i, c in enumerate(counts)]


def reliability_bins(pred: np.ndarray, event: np.ndarray) -> Optional[List[Dict]]:
    """Equal-width reliability diagram; empty bins are omitted."""
    if pred.size == 0:
        return None
    idx = np.clip((pred * RELIABILITY_BINS).astype(int), 0, RELIABILITY_BINS - 1)
    out = []
    for i in range(RELIABILITY_BINS):
        m = idx == i
        n = int(m.sum())
        if n == 0:
            continue
        out.append({
            "bin_center": round((i + 0.5) / RELIABILITY_BINS, 3),
            "mean_predicted": round(float(pred[m].mean()), 6),
            "observed_frac": round(float(event[m].mean()), 6),
            "count": n,
        })
    return out


def build_ego_graph_edges(
    node_id: int,
    edge_attention: Dict[Tuple[int, int], List[Optional[float]]],
    node_coord_map: Dict[int, Tuple[float, float]],
    num_timesteps: int,
    top_k: int = 10,
) -> List[Dict]:
    """Top-K incoming and top-K outgoing GAT edges of node_id, ranked by mean
    attention over the timesteps where the edge was active (None = dropped)."""
    incoming: List[Tuple[float, Dict]] = []
    outgoing: List[Tuple[float, Dict]] = []
    for (src, tgt), seq in edge_attention.items():
        if node_id not in (src, tgt) or src == tgt:
            continue
        assert len(seq) == num_timesteps, \
            f"TRIPWIRE: edge ({src},{tgt}) has {len(seq)} weights, expected {num_timesteps}"
        active = [w for w in seq if w is not None]
        score = float(np.mean(active)) if active else 0.0
        src_lat, src_lon = node_coord_map[src]
        tgt_lat, tgt_lon = node_coord_map[tgt]
        direction = "incoming" if tgt == node_id else "outgoing"
        entry = {
            "source_id": src, "target_id": tgt, "direction": direction, "weight_sequence": seq,
            "source_lat": src_lat, "source_lon": src_lon, "target_lat": tgt_lat, "target_lon": tgt_lon,
        }
        (incoming if direction == "incoming" else outgoing).append((score, entry))
    incoming.sort(key=lambda x: x[0], reverse=True)
    outgoing.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in incoming[:top_k]] + [e for _, e in outgoing[:top_k]]


# ---------------------------------------------------------------------------
# Thin CLI: one run, one day (delegates to the paginated exporter)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export one v5 run for one day (see export_paginated_payload.py)")
    parser.add_argument("--run", required=True, help="W&B run name, e.g. v5_full-s42")
    parser.add_argument("--date", required=True, help="Target day YYYY-MM-DD")
    parser.add_argument("--output_dir", default="artifacts/payload_staging/runs")
    args = parser.parse_args()
    day = dt.date.fromisoformat(args.date)

    import export_paginated_payload
    sys.argv = [
        "export_paginated_payload.py", "--runs", args.run,
        "--start_date", day.isoformat(), "--end_date", (day + dt.timedelta(days=1)).isoformat(),
        "--output_dir", args.output_dir,
    ]
    export_paginated_payload.main()


if __name__ == "__main__":
    main()
