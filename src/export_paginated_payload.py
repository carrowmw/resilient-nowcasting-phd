"""
export_paginated_payload.py

RIoT v5 multi-run, paginated payload generator for the Geospatial Digital Twin.

    artifacts/payload_staging/runs/
    ├── run_registry.json                  (scripts/build_run_registry.py)
    └── {run_id}/                           e.g. v5_full-s42
        ├── spatial.json                    nodes + learned topology + run metadata
        └── {YYYY-MM-DD}/
            ├── page.json                   columnar per-node forecast series + day analytics
            └── nodes/node_{id}.json         per-node distribution / self-healing / ego graph

Schema: frontend/src/types/payload.ts + registry.ts (schema_version 5).

Run selection (default): the reference v5_full-s42 plus the best (top_k - 1) runs by
val_obs_all_rmse from docs/experiments/riot_v5/data/runs.csv, excluding the naive
floors, the v4 replicas and the transductive-profile oracle. Checkpoints come from
W&B (the run's wandb.collection, producer run = the run name).

Inference reproduces training: the dataset is re-built over [train start, analysis
end) with the run's own train slice (scalers / profiles / outage rates verified
against the checkpoint buffers); batches go through the run's BatchPreparer with
the manifest inductive stations blanked (zero-shot forecasts, ground truth kept for
scoring) and no artificial gaps; outputs come from the v5 RIoTAdapter.

Usage:
    python src/export_paginated_payload.py [--top_k 10] [--runs v5_full-s42 ...]
        [--config configs/analysis/analysis.yaml] [--start_date 2026-05-01] [--end_date 2026-06-01]
        [--output_dir artifacts/payload_staging/runs] [--dry_run]
"""

import argparse
import datetime as dt
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

# Importing export_ui_payload runs its .env loader + logging setup.
from export_ui_payload import (  # noqa: E402
    INTERVAL_LEVELS,
    CachingConnector,
    RunArtifact,
    V5Context,
    build_ego_graph_edges,
    build_inference_dataset,
    build_v5_context,
    capture_gat_attention,
    dataset_key,
    fetch_run_artifact,
    last_step_attention,
    load_run_config,
    mid_pit_and_crps,
    pit_histogram,
    pmf_grid,
    quantiles_from_pmf,
    reliability_bins,
    semantic_tags_from_cache,
)
from dataset import custom_collate_real  # noqa: E402
from db_connector import create_connector  # noqa: E402
from harness import load_holdout_manifest, manifest_sha, validate_predictions  # noqa: E402

logger = logging.getLogger("export_paginated_payload")

SCHEMA_VERSION = 5
DEFAULT_CONFIG_PATH = "configs/analysis/analysis.yaml"
DEFAULT_OUTPUT_DIR = "artifacts/payload_staging/runs"
DEFAULT_RUNS_CSV = "docs/experiments/riot_v5/data/runs.csv"
LOCAL_RUNS_DIR = "models/resilient-iot-v5"

REFERENCE_RUN = "v5_full-s42"
EXCLUDED_PREFIXES = ("naive_", "v4r_")
# Oracle: the transductive profile prior is fitted on the holdout stations' own data.
EXCLUDED_EXPERIMENTS = ("e3_3a_profile_transductive",)
SELECTION_RULE = (
    "v5_full-s42 reference + best by val_obs_all_rmse "
    "(excluding naive_*, v4r_* and the transductive-profile oracle)"
)

TEST_METRIC_KEYS = (
    "val_obs_all_rmse", "test_obs_all_rmse", "test_obs_all_mae", "test_obs_trans_rmse",
    "test_obs_ind_rmse", "test_peak_rmse", "test_so_12_forecast_rmse", "test_crps",
    "test_collapse_rate", "test_real_gap_mass_ratio", "test_q90_coverage", "test_pzero_missing_auc",
)

# Windows per forward pass: B * seq_len graphs go through the GAT at once.
GRAPHS_PER_BATCH = 1152
# collapse_rate only counts missing slots whose own profile expects real traffic.
COLLAPSE_PROFILE_MIN = 10.0


# ---------------------------------------------------------------------------
# CLI / config / run plan
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate RIoT v5 paginated UI payloads")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Analysis config (window + export settings)")
    p.add_argument("--start_date", default=None, help="First target day (YYYY-MM-DD); default holdout_start")
    p.add_argument("--end_date", default=None, help="Exclusive end day; default holdout_end")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--runs", nargs="+", default=None, help="Explicit W&B run names (overrides --top_k)")
    p.add_argument("--top_k", type=int, default=10, help="Reference run + (top_k - 1) best by validation RMSE")
    p.add_argument("--runs_csv", default=DEFAULT_RUNS_CSV)
    p.add_argument("--data_source", default=None, help="Override the analysis config data_source")
    p.add_argument("--parquet_path", default=None)
    p.add_argument("--dry_run", action="store_true", help="Print the run plan and exit")
    return p.parse_args(argv)


def load_analysis_config(args: argparse.Namespace) -> Dict:
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    start = str(args.start_date or cfg["holdout_start"])
    end = str(args.end_date or cfg["holdout_end"])
    assert dt.date.fromisoformat(start) < dt.date.fromisoformat(end), \
        f"TRIPWIRE: start_date {start} must precede end_date {end} (exclusive)"
    resolved = {
        "start_date": start,
        "end_date": end,
        "data_source": args.data_source or cfg["data_source"],
        "num_timesteps_target": int(cfg["export"]["num_timesteps_target"]),
        "top_k_ego_edges": int(cfg["export"]["top_k_ego_edges"]),
    }
    parquet_path = args.parquet_path or cfg.get("parquet_path")
    if parquet_path:
        resolved["parquet_path"] = parquet_path
    logger.info(f"Analysis window {start} → {end} (exclusive), data_source={resolved['data_source']}")
    return resolved


def _num(v) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def select_runs(runs_csv: str, top_k: int, explicit: Optional[List[str]] = None) -> List[Dict]:
    """Ordered run plan: [{run, experiment, group, description, rank, is_reference, test_metrics}]."""
    df = pd.read_csv(runs_csv)
    if explicit:
        missing = sorted(set(explicit) - set(df["run"]))
        assert not missing, f"CRITICAL: runs not in {runs_csv}: {missing}"
        rows = [df[df["run"] == r].iloc[0] for r in explicit]
    else:
        assert top_k >= 1, "CRITICAL: top_k must be >= 1"
        ref = df[df["run"] == REFERENCE_RUN]
        assert len(ref) == 1, f"CRITICAL: reference run {REFERENCE_RUN} not in {runs_csv}"
        pool = df[
            (df["run"] != REFERENCE_RUN)
            & ~df["experiment"].str.startswith(EXCLUDED_PREFIXES)
            & ~df["experiment"].isin(EXCLUDED_EXPERIMENTS)
            & df["val_obs_all_rmse"].notna()
        ].sort_values("val_obs_all_rmse", kind="stable")
        rows = [ref.iloc[0]] + [r for _, r in pool.head(top_k - 1).iterrows()]
    return [{
        "run": row["run"],
        "experiment": row["experiment"],
        "group": row["group"],
        "description": row["desc"],
        "rank": rank,
        "is_reference": row["run"] == REFERENCE_RUN,
        "test_metrics": {k: _num(row.get(k)) for k in TEST_METRIC_KEYS},
    } for rank, row in enumerate(rows, start=1)]


def local_run_config(run: str) -> Dict:
    path = os.path.join(LOCAL_RUNS_DIR, run, "config.yaml")
    assert os.path.exists(path), f"CRITICAL: {path} missing — cannot resolve the W&B collection for {run}"
    return load_run_config(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _round_list(a: np.ndarray, nd: int = 2) -> List[Optional[float]]:
    """JSON-safe list: NaN/inf → null."""
    return [round(float(v), nd) if np.isfinite(v) else None for v in np.asarray(a, dtype=np.float64)]


def infer_windows(ctx: V5Context, window_idx: np.ndarray) -> Dict:
    """Batched v5 inference for dataset windows `window_idx` (target = window + seq_len).

    Per-target arrays are [W, N]; interval [W, N, 2]; quantiles [W, N, Q];
    attention [W, E]; entropy [W]."""
    ds, device = ctx.dataset, ctx.device
    T, N = ctx.seq_len, ds.num_nodes
    K = ctx.config["evaluation"]["crps_max_count"]
    bsz = max(1, GRAPHS_PER_BATCH // T)
    gen = torch.Generator(device=device).manual_seed(ctx.config["evaluation"]["eval_mask_seed"])
    parts: Dict[str, List[np.ndarray]] = defaultdict(list)
    tripwires = []
    edge_index = None

    def keep(name, tensor):
        parts[name].append(tensor.float().cpu().numpy())

    for start in range(0, len(window_idx), bsz):
        idx = window_idx[start:start + bsz]
        B = len(idx)
        batch = custom_collate_real([ds[int(i)] for i in idx])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            prepared = ctx.preparer.prepare(batch, train=False, gen=gen)
            with capture_gat_attention(ctx.model) as store:
                preds = ctx.adapter.predict(prepared)
            validate_predictions(ctx.adapter.capabilities, preds)
            assert preds.point is not None, "CRITICAL: v5 adapter returned no point forecast"
            y = prepared["target_forecast"].float()

            keep("point", preds.point)
            keep("mu", preds.mu)
            parts["observed"].append(prepared["target_observed"].cpu().numpy())
            for name, val in (("theta", preds.theta), ("pi", preds.pi), ("q", preds.outage_q),
                              ("p_zero", preds.p_zero), ("quantiles", preds.quantiles)):
                if val is not None:
                    keep(name, val)
            if preds.imputed_raw is not None:
                keep("imputed_last", preds.imputed_raw[..., -1])
            if preds.theta is not None:
                pmf = pmf_grid(ctx.likelihood, preds.mu, preds.theta, preds.pi, K)
                keep("interval", quantiles_from_pmf(pmf, INTERVAL_LEVELS))
                pit, crps = mid_pit_and_crps(pmf, y)
                keep("pit", pit)
                keep("crps", crps)
                parts["in_grid"].append((y <= K).cpu().numpy())
                del pmf
            keep("profile", ctx.preparer.covariates.profile_raw(prepared["time_idx"], None)[..., -1])
            keep("input_missing_frac", prepared["input_missing"].float().mean(-1))

            if preds.edge_index is not None:
                ei = preds.edge_index.cpu()
                if edge_index is None:
                    edge_index = ei
                assert torch.equal(edge_index, ei), "CRITICAL: learned graph changed between windows"
                att, ent = last_step_attention(store, ei, B, T, N)
                parts["attention"].append(att)
                parts["entropy"].append(ent)
            # Model tripwires carry node + type but not the window: tag with the batch's span.
            for ev in preds.tripwires:
                tripwires.append((int(start), int(start + B - 1), int(ev["node_id"]), str(ev["event_type"])))

    out: Dict = {k: np.concatenate(v) for k, v in parts.items()}
    t_target = window_idx + T
    observed = out["observed"]
    truth = ds.unscaled_traffic[t_target].numpy().astype(np.float64)
    out["truth"] = np.where(observed, truth, np.nan)
    out["gap_steps"] = ds.run_length[t_target].numpy()
    out["input_truth_last"] = np.where(ds.observed[t_target - 1].numpy(),
                                       ds.unscaled_traffic[t_target - 1].numpy(), np.nan)
    out["edge_index"] = edge_index
    out["tripwires"] = tripwires
    out["window_idx"] = window_idx
    out["quantile_levels"] = tuple(ctx.quantile_levels)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricSums:
    """Sum/count accumulator for WindowMetrics + Calibration (one day or the whole window)."""

    def __init__(self):
        self.s: Dict[str, float] = defaultdict(float)
        self.c: Dict[str, float] = defaultdict(float)
        self.lists: Dict[str, List[np.ndarray]] = defaultdict(list)

    def _add(self, name: str, values: np.ndarray, mask: np.ndarray) -> None:
        self.s[name] += float(np.asarray(values)[mask].sum())
        self.c[name] += float(mask.sum())

    def _mean(self, name: str) -> Optional[float]:
        return self.s[name] / self.c[name] if self.c.get(name) else None

    def update(self, r: Dict, holdout: np.ndarray) -> None:
        obs = r["observed"]
        miss = ~obs
        y = np.nan_to_num(r["truth"])
        point = r["point"]
        err = point - y
        ind = np.broadcast_to(holdout[None, :], obs.shape)
        self.s["n_targets"] += obs.size
        self.s["n_observed"] += float(obs.sum())
        for name, m in (("all", obs), ("trans", obs & ~ind), ("ind", obs & ind)):
            self._add(f"{name}_se", err ** 2, m)
            self._add(f"{name}_ae", np.abs(err), m)
        self._add("bias", err, obs)
        if "interval" in r:
            lo, hi = r["interval"][..., 0], r["interval"][..., 1]
            self._add("interval80", ((y >= lo) & (y <= hi)).astype(float), obs)
            scored = obs & r["in_grid"]
            self._add("crps", r["crps"], scored)
            self.lists["pit"].append(r["pit"][scored])
        if "quantiles" in r:
            for i, lvl in enumerate(r["quantile_levels"]):
                self._add(f"q{int(round(lvl * 100))}", (y <= r["quantiles"][..., i]).astype(float), obs)
        self._add("collapse", (point < 1.0).astype(float), miss & (r["profile"] > COLLAPSE_PROFILE_MIN))
        self.s["gap_point"] += float(point[miss].sum())
        self.s["gap_profile"] += float(r["profile"][miss].sum())
        if "p_zero" in r:
            self.lists["p_zero"].append(r["p_zero"].ravel())
            self.lists["p_zero_event"].append((miss | (y == 0)).ravel().astype(float))
        if "q" in r:
            self.lists["outage"].append(r["q"].ravel())
            self.lists["outage_event"].append(miss.ravel().astype(float))

    def window_metrics(self) -> Dict:
        def rnd(v, nd=4):
            return None if v is None else round(float(v), nd)

        def rmse(k):
            m = self._mean(k)
            return None if m is None else round(float(np.sqrt(m)), 4)

        n_t, n_o = int(self.s["n_targets"]), int(self.s["n_observed"])
        gp = self.s["gap_profile"]
        return {
            "n_targets": n_t,
            "n_observed": n_o,
            "observed_frac": round(n_o / n_t, 4) if n_t else 0.0,
            "obs_all_rmse": rmse("all_se"), "obs_all_mae": rnd(self._mean("all_ae")),
            "obs_trans_rmse": rmse("trans_se"), "obs_trans_mae": rnd(self._mean("trans_ae")),
            "obs_ind_rmse": rmse("ind_se"), "obs_ind_mae": rnd(self._mean("ind_ae")),
            "bias": rnd(self._mean("bias")),
            "interval80_coverage": rnd(self._mean("interval80")),
            "q50_coverage": rnd(self._mean("q50")),
            "q90_coverage": rnd(self._mean("q90")),
            "crps": rnd(self._mean("crps")),
            "collapse_rate": rnd(self._mean("collapse")),
            "real_gap_mass_ratio": rnd(self.s["gap_point"] / gp) if gp > 0 else None,
        }

    def calibration(self) -> Dict:
        def cat(name):
            return np.concatenate(self.lists[name]) if self.lists.get(name) else np.zeros(0)
        return {
            "pit_histogram": pit_histogram(cat("pit")),
            "p_zero_reliability": reliability_bins(cat("p_zero"), cat("p_zero_event")),
            "outage_reliability": reliability_bins(cat("outage"), cat("outage_event")),
        }


def node_day_metrics(r: Dict, n: int) -> Dict:
    obs = r["observed"][:, n]
    point = r["point"][:, n]
    y = r["truth"][:, n]
    k = int(obs.sum())
    out = {
        "n_observed": k, "rmse": None, "mae": None, "bias": None, "interval80_coverage": None,
        "peak_point": round(float(point.max()), 2), "mean_point": round(float(point.mean()), 2),
        "mean_truth": None,
    }
    if k:
        err = point[obs] - y[obs]
        out.update({
            "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "mae": round(float(np.abs(err).mean()), 3),
            "bias": round(float(err.mean()), 3),
            "mean_truth": round(float(y[obs].mean()), 2),
        })
        if "interval" in r:
            lo, hi = r["interval"][obs, n, 0], r["interval"][obs, n, 1]
            out["interval80_coverage"] = round(float(((y[obs] >= lo) & (y[obs] <= hi)).mean()), 4)
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), allow_nan=False)


def day_edge_attention(r: Dict) -> Optional[Dict]:
    """{(src, dst): [w | None] * W} over the learned (non-self-loop) edges, or None."""
    if r["edge_index"] is None:
        return None
    ei = r["edge_index"].numpy()
    return {
        (int(ei[0, e]), int(ei[1, e])): _round_list(r["attention"][:, e], 5)
        for e in range(ei.shape[1]) if ei[0, e] != ei[1, e]
    }


def build_page(ctx: V5Context, day: dt.date, r: Dict, nodes: List[Dict], day_sums: MetricSums,
               edge_att: Optional[Dict]) -> Dict:
    ds = ctx.dataset
    W, N = r["point"].shape
    q_idx = {round(lvl, 4): i for i, lvl in enumerate(r["quantile_levels"])}

    def quantile_series(level, n):
        if "quantiles" not in r or level not in q_idx:
            return None
        return _round_list(r["quantiles"][:, n, q_idx[level]])

    node_series, node_metrics = {}, {}
    for n in range(N):
        node_series[str(n)] = {
            "point": _round_list(r["point"][:, n]),
            "lo": _round_list(r["interval"][:, n, 0], 0) if "interval" in r else None,
            "hi": _round_list(r["interval"][:, n, 1], 0) if "interval" in r else None,
            "q50": quantile_series(0.5, n),
            "q90": quantile_series(0.9, n),
            "p_zero": _round_list(r["p_zero"][:, n], 4) if "p_zero" in r else None,
            "outage_q": _round_list(r["q"][:, n], 4) if "q" in r else None,
            "profile": _round_list(r["profile"][:, n]),
            "truth": _round_list(r["truth"][:, n], 0),
            "gap_steps": [int(v) for v in r["gap_steps"][:, n]],
        }
        node_metrics[str(n)] = node_day_metrics(r, n)

    obs = r["observed"]
    miss = ~obs
    n_obs = obs.sum(1)
    truth_sum = np.where(obs, np.nan_to_num(r["truth"]), 0.0).sum(1)
    gap_profile = np.where(miss, r["profile"], 0.0).sum(1)
    gap_point = np.where(miss, r["point"], 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_truth = np.where(n_obs > 0, truth_sum / n_obs, np.nan)
        gap_ratio = np.where(gap_profile > 0, gap_point / gap_profile, np.nan)
    network = {
        "observed_frac": _round_list(obs.mean(1), 4),
        "mean_point": _round_list(r["point"].mean(1)),
        "mean_truth": _round_list(mean_truth),
        "real_gap_mass_ratio": _round_list(gap_ratio, 4),
        "attention_entropy": _round_list(r["entropy"], 4) if "entropy" in r else None,
    }

    leaderboard = None
    if edge_att is not None:
        out_sum = np.zeros(N)
        for (src, _dst), seq in edge_att.items():
            out_sum[src] += sum(w for w in seq if w is not None)
        leaderboard = [{"node_id": int(i), "location_name": nodes[i]["location_name"],
                        "total_outgoing_attention": round(float(out_sum[i]), 4)}
                       for i in np.argsort(-out_sum, kind="stable")[:10]]

    tripwires = [{"timestep": a, "timestep_end": b, "node_id": nid, "event_type": et,
                  "location_name": nodes[nid]["location_name"]}
                 for a, b, nid, et in sorted(set(r["tripwires"]))]

    return {
        "schema_version": SCHEMA_VERSION,
        "date": day.isoformat(),
        "timesteps": W,
        "seq_len": ctx.seq_len,
        "step_minutes": int(ds.step_minutes),
        "timestamps": [pd.Timestamp(ds.times[i]).isoformat() for i in r["window_idx"] + ctx.seq_len],
        "node_series": node_series,
        "node_metrics": node_metrics,
        "network": network,
        "day_metrics": day_sums.window_metrics(),
        "calibration": day_sums.calibration(),
        "edge_attention": None if edge_att is None else [
            {"source": s, "target": d, "weights": seq} for (s, d), seq in edge_att.items()
        ],
        "alpha_node_leaderboard": leaderboard,
        "tripwire_audit_trail": tripwires,
    }


def build_node_detail(ctx: V5Context, day: dt.date, r: Dict, n: int, meta: Dict, node_metrics: Dict,
                      edge_att: Optional[Dict], coord: Dict, top_k_ego: int) -> Dict:
    W = r["point"].shape[0]
    edges_1hop, ego = [], []
    if edge_att is not None:
        for (s, d), seq in edge_att.items():
            if n not in (s, d):
                continue
            edges_1hop.append({"neighbor_id": s if d == n else d,
                               "direction": "incoming" if d == n else "outgoing",
                               "attention_sequence": seq})
        ego = build_ego_graph_edges(n, edge_att, coord, W, top_k_ego)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": n,
        "location_name": meta["location_name"],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "node_role": meta["node_role"],
        "date": day.isoformat(),
        "distribution": {
            "likelihood": ctx.likelihood,
            "mu": _round_list(r["mu"][:, n], 3),
            "theta": _round_list(r["theta"][:, n], 4) if "theta" in r else None,
            "pi": _round_list(r["pi"][:, n], 4) if "pi" in r else None,
            "q": _round_list(r["q"][:, n], 4) if "q" in r else None,
        },
        "imputed_last": _round_list(r["imputed_last"][:, n]) if "imputed_last" in r else None,
        "input_truth_last": _round_list(r["input_truth_last"][:, n], 0),
        "input_missing_frac": _round_list(r["input_missing_frac"][:, n], 3),
        "edges_1hop": edges_1hop,
        "ego_graph_edges": ego,
        "metrics": node_metrics,
    }


def write_day(out_dir: str, ctx: V5Context, run_id: str, day: dt.date, r: Dict, nodes: List[Dict],
              holdout: np.ndarray, top_k_ego: int, window: MetricSums) -> None:
    day_sums = MetricSums()
    day_sums.update(r, holdout)
    window.update(r, holdout)
    edge_att = day_edge_attention(r)
    page = build_page(ctx, day, r, nodes, day_sums, edge_att)
    day_dir = os.path.join(out_dir, run_id, day.isoformat())
    _dump(os.path.join(day_dir, "page.json"), page)
    coord = {m["id"]: (m["lat"], m["lon"]) for m in nodes}
    for n, meta in enumerate(nodes):
        detail = build_node_detail(ctx, day, r, n, meta, page["node_metrics"][str(n)], edge_att, coord, top_k_ego)
        _dump(os.path.join(day_dir, "nodes", f"node_{n}.json"), detail)


# ---------------------------------------------------------------------------
# Per-run generation
# ---------------------------------------------------------------------------

def window_observation(ds, targets: np.ndarray) -> Tuple[np.ndarray, List[Optional[str]]]:
    """Per node: observed target slots in the analysis window, and the ISO timestamp
    of its last observation at or before the window end (None = never in the dataset).
    A node with zero observed slots is a "ghost": forecast and scored nowhere."""
    observed = ds.observed.numpy()
    counts = observed[targets].sum(0).astype(int)
    upto = observed[: int(targets.max()) + 1]
    last = upto.shape[0] - 1 - np.argmax(upto[::-1], axis=0)
    seen = upto.any(0)
    stamps = [pd.Timestamp(ds.times[int(t)]).isoformat() if ok else None for t, ok in zip(last, seen)]
    return counts, stamps


def node_table(location_ids: List[str], sensor_coords: Dict, holdout_ids) -> List[Dict]:
    nodes = []
    for i, loc in enumerate(location_ids):
        assert loc in sensor_coords, f"CRITICAL: sensor {loc!r} has no coordinates in sensor_metadata"
        nodes.append({"id": i, "location_name": loc,
                      "lat": float(sensor_coords[loc]["lat"]), "lon": float(sensor_coords[loc]["lon"]),
                      "node_role": "holdout" if loc in holdout_ids else "observed"})
    return nodes


def build_spatial_metadata(ctx: V5Context, plan: Dict, art: RunArtifact, manifest: Optional[Dict],
                           cfg: Dict, available_dates: List[str], window: MetricSums) -> Dict:
    caps = ctx.adapter.capabilities
    data_cfg = ctx.config["data"]
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": ctx.config["pipeline_version"],
        "run_id": plan["run"],
        "experiment": plan["experiment"],
        "group": plan["group"],
        "description": plan["description"],
        "model_type": ctx.config["model_type"],
        "forecast_likelihood": ctx.likelihood,
        "quantile_levels": list(ctx.quantile_levels),
        "capabilities": {
            "uncertainty": caps.uncertainty, "zero_inflation": caps.zero_inflation,
            "imputation": caps.imputation, "kriging": caps.kriging, "graph": caps.graph,
            "outage": caps.outage, "quantiles": bool(ctx.quantile_levels),
        },
        "seq_len": ctx.seq_len,
        "wandb_run_id": art.wandb_run_id,
        "wandb_artifact": art.artifact_ref,
        "git_commit_hash": art.git_commit,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "registry": {"rank": plan["rank"], "is_reference": plan["is_reference"],
                     "selection_rule": SELECTION_RULE},
        "hyperparameters": ctx.config,
        "num_nodes": ctx.dataset.num_nodes,
        "num_holdout_nodes": 0 if manifest is None else len(manifest["holdout_location_ids"]),
        "holdout_manifest_sha": manifest_sha(manifest),
        "available_dates": available_dates,
        "start_date": cfg["start_date"],
        "end_date": cfg["end_date"],
        "veh_class": data_cfg["target_veh_class"],
        "dir": data_cfg["target_dir"],
        "step_minutes": int(ctx.dataset.step_minutes),
        "test_metrics": plan["test_metrics"],
        "window_metrics": window.window_metrics(),
    }


def generate_run(ctx: V5Context, plan: Dict, art: RunArtifact, manifest: Optional[Dict], cfg: Dict,
                 out_dir: str, sensor_coords: Dict) -> Dict:
    ds = ctx.dataset
    run_id = plan["run"]
    T = ctx.seq_len
    holdout_ids = set(manifest["holdout_location_ids"]) if manifest else set()
    nodes = node_table(ctx.location_ids, sensor_coords, holdout_ids)
    holdout = np.array([m["node_role"] == "holdout" for m in nodes])

    # Days are UTC calendar days (the v4 convention); timestamps carry their offset.
    times = pd.DatetimeIndex(ds.times)
    target_day = (times.tz_convert("UTC").tz_localize(None) if times.tz is not None else times).normalize()
    window = MetricSums()
    available_dates: List[str] = []
    window_targets: List[np.ndarray] = []
    edge_index = None
    day = dt.date.fromisoformat(cfg["start_date"])
    end = dt.date.fromisoformat(cfg["end_date"])
    while day < end:
        targets = np.where(target_day == pd.Timestamp(day))[0]
        targets = targets[targets >= T][:cfg["num_timesteps_target"]]
        if len(targets) == 0:
            logger.warning(f"  {run_id} {day}: no forecast targets — skipped")
        else:
            r = infer_windows(ctx, targets - T)
            write_day(out_dir, ctx, run_id, day, r, nodes, holdout, cfg["top_k_ego_edges"], window)
            if edge_index is None:
                edge_index = r["edge_index"]
            available_dates.append(day.isoformat())
            window_targets.append(targets)
            logger.info(f"  {run_id} {day}: {len(targets)} targets, observed {r['observed'].mean():.1%}")
        day += dt.timedelta(days=1)
    assert available_dates, f"TRIPWIRE: run {run_id} produced no days in {cfg['start_date']}..{cfg['end_date']}"

    window_observed, last_observed = window_observation(ds, np.concatenate(window_targets))
    for m, k, last in zip(nodes, window_observed, last_observed):
        m["window_observed"] = int(k)
        m["last_observed"] = last
    metadata = build_spatial_metadata(ctx, plan, art, manifest, cfg, available_dates, window)
    metadata["num_ghost_nodes"] = int((window_observed == 0).sum())
    tags = semantic_tags_from_cache(ctx.location_ids)
    topology = [] if edge_index is None else [
        {"source": int(s), "target": int(d)} for s, d in edge_index.numpy().T if s != d
    ]
    _dump(os.path.join(out_dir, run_id, "spatial.json"), {
        "metadata": metadata,
        "nodes": [dict(m, semantic_tags=tags[m["location_name"]]) for m in nodes],
        "edge_topology": topology,
    })
    wm = metadata["window_metrics"]
    logger.info(
        f"=== {run_id}: {len(available_dates)} days, obs_all_rmse={wm['obs_all_rmse']} "
        f"ind_rmse={wm['obs_ind_rmse']} cov80={wm['interval80_coverage']} crps={wm['crps']} "
        f"gap_mass_ratio={wm['real_gap_mass_ratio']} ghosts={metadata['num_ghost_nodes']}"
    )
    return {"run_id": run_id, "days": len(available_dates), "window_metrics": wm}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args(argv)
    cfg = load_analysis_config(args)
    plan = select_runs(args.runs_csv, args.top_k, args.runs)

    configs = {p["run"]: local_run_config(p["run"]) for p in plan}
    logger.info(f"Plan ({SELECTION_RULE}):")
    for p in plan:
        c = configs[p["run"]]
        logger.info(f"  #{p['rank']:>2} {p['run']:<36} {c['loss']['forecast_likelihood']:<15} "
                    f"seq_len={c['data']['seq_len']:<3} val_obs_all_rmse={p['test_metrics']['val_obs_all_rmse']}")
    if args.dry_run:
        return

    # Group runs sharing a data config so each (~400 MB) inference dataset is built once.
    first_rank: Dict = {}
    for p in plan:
        first_rank.setdefault(dataset_key(configs[p["run"]]), p["rank"])
    order = sorted(plan, key=lambda p: (first_rank[dataset_key(configs[p["run"]])], p["rank"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, output → {args.output_dir}")
    connector = CachingConnector(create_connector(cfg))
    summaries = []
    try:
        sensor_coords = connector.load_sensor_metadata()
        ds_key, dataset = None, None
        for p in order:
            local = configs[p["run"]]
            art = fetch_run_artifact(p["run"], local["wandb"]["project"], local["wandb"]["collection"])
            config = load_run_config(art.config_path)
            assert config["wandb"]["run_name"] == p["run"], \
                f"CRITICAL: artifact config run_name {config['wandb']['run_name']!r} != {p['run']!r}"
            key = dataset_key(config)
            if key != ds_key:
                dataset = None
                gc.collect()
                dataset = build_inference_dataset(config, connector, cfg["end_date"])
                ds_key = key
            manifest = load_holdout_manifest(os.path.dirname(art.checkpoint_path))
            ctx = build_v5_context(p["run"], config, dataset, art.checkpoint_path, manifest, device)
            logger.info(f"=== #{p['rank']} {p['run']} ({art.artifact_ref}) {ctx.adapter.capabilities}")
            summaries.append(generate_run(ctx, p, art, manifest, cfg, args.output_dir, sensor_coords))
            del ctx
            gc.collect()
    finally:
        connector.close()

    logger.info(f"DONE. {len(summaries)}/{len(plan)} run(s) → {args.output_dir}")
    for s in summaries:
        logger.info(f"  {s['run_id']}: {s['days']} days, obs_all_rmse={s['window_metrics']['obs_all_rmse']}")


if __name__ == "__main__":
    main()
