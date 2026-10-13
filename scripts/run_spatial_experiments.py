#!/usr/bin/env python3
"""
Config-driven experiment runner for the spatial track (thesis Chapter 6).

Mirrors scripts/run_quality_experiments.py: YAML config, resumable registry,
per-job persistence. Differences, both deliberate:
  - SCOPED job SHA: each job is keyed by the hash of only the config subsets
    that affect it (data/nodes/temporal/inputs + its graph variant + its
    model spec + seed), so editing wandb.project no longer invalidates
    finished jobs.
  - Capability gating: attention passes run iff the model declares
    emits_attention — never name-prefix string matching.

Per job (= model × graph variant × seed, one full-network fit):
  runs/<job_key>/predictions.parquet   test-split long-format predictions
  runs/<job_key>/results.csv           per-(node, horizon) metrics, val + test
  runs/<job_key>/manifest.json         spec, graph sha, fit report, git hash
  runs/<job_key>/attention.parquet     final-layer attention (capable models,
                                       when run.attention.enabled)

Usage:
  python scripts/run_spatial_experiments.py --config configs/spatial/spatial_smoke.yaml
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from spatial.contract import GraphSpec, assert_predictions_valid  # noqa: E402
from spatial.evaluate import evaluate_split, mase_scale, predictions_frame  # noqa: E402
from spatial.graphs import build_graph, graph_manifest  # noqa: E402
from spatial.loader import GraphSeriesStore  # noqa: E402
from spatial.registry import build_forecaster  # noqa: E402
from spatial.splits import build_graph_splits, count_usable_targets  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_spatial")


class Job(NamedTuple):
    job_key: str
    stage: str
    model_key: str
    graph_variant: str
    seed: int


def _sha16(payload: dict) -> str:
    return hashlib.sha256(
        yaml.safe_dump(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def job_sha(config: dict, model_key: str, graph_variant: str, seed: int) -> str:
    """Scoped hash: only the config subsets this job's result depends on."""
    return _sha16({
        "data": config["data"],
        "data_source": config["data_source"],
        "nodes": config["nodes"],
        "temporal": config["temporal"],
        "inputs": config["inputs"],
        "graph_variant": {graph_variant: config["graph"]["variants"][graph_variant]},
        "model": {model_key: config["models"][model_key]},
        "seed": int(seed),
    })


def expand_jobs(config: dict) -> List[Job]:
    """Deterministic job list from run.matrix; duplicate keys are a config bug."""
    jobs: List[Job] = []
    variants = config["graph"]["variants"]
    models = config["models"]
    for block in config["run"]["matrix"]:
        for model_key in block["models"]:
            assert model_key in models, \
                f"CRITICAL: matrix references unknown model {model_key!r}"
            for graph_variant in block["graphs"]:
                assert graph_variant in variants, \
                    f"CRITICAL: matrix references unknown graph {graph_variant!r}"
                for seed in block["seeds"]:
                    jobs.append(Job(
                        job_key=f"{model_key}_{graph_variant}_s{seed}",
                        stage=block["tag"], model_key=model_key,
                        graph_variant=graph_variant, seed=int(seed),
                    ))
    keys = [j.job_key for j in jobs]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"CRITICAL: duplicate job keys in run.matrix: {sorted(dupes)}"
    return jobs


def _git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _attention_frames(record) -> Dict[str, pd.DataFrame]:
    src, dst = record.edge_index
    heads = record.alpha_mean.shape[1]
    main = pd.DataFrame({"src_idx": src, "dst_idx": dst,
                         "alpha": record.alpha_flat()})
    for h in range(heads):
        main[f"alpha_head{h}"] = record.alpha_mean[:, h]
    by_hour = pd.DataFrame(record.alpha_by_hour.T,
                           columns=[f"hour_{h:02d}" for h in range(24)])
    by_hour.insert(0, "src_idx", src)
    by_hour.insert(1, "dst_idx", dst)
    return {"attention.parquet": main, "attention_by_hour.parquet": by_hour}


def run_job(job: Job, context, splits, graph: GraphSpec, graph_sha: str,
            mase_per_node: np.ndarray, config: dict, out_dir: Path) -> dict:
    t0 = time.time()
    spec = config["models"][job.model_key]
    forecaster = build_forecaster(
        job.model_key, spec, seed=job.seed,
        device=config["run"].get("device"), inputs_cfg=config["inputs"],
    )
    report = forecaster.fit(splits["train"], splits["val"], graph)

    results_frames = []
    preds_test = None
    for split_name in ("val", "test"):
        preds = forecaster.predict(splits[split_name], graph)
        assert_predictions_valid(splits[split_name], preds, job.job_key)
        res = evaluate_split(splits[split_name], preds, mase_per_node)
        results_frames.append(res)
        if split_name == "test":
            preds_test = predictions_frame(
                splits["test"], preds, model=job.model_key,
                graph_variant=job.graph_variant, seed=job.seed)

    job_dir = out_dir / "runs" / job.job_key
    job_dir.mkdir(parents=True, exist_ok=True)
    preds_test.to_parquet(job_dir / "predictions.parquet", index=False)
    results = pd.concat(results_frames, ignore_index=True)
    results.insert(0, "model", job.model_key)
    results.insert(1, "graph_variant", job.graph_variant)
    results.insert(2, "seed", job.seed)
    results.to_csv(job_dir / "results.csv", index=False)

    att_cfg = config["run"].get("attention", {"enabled": False})
    attention_written = False
    if (att_cfg.get("enabled") and forecaster.capabilities.emits_attention
            and job.model_key in att_cfg.get("models", [])):
        record = forecaster.predict_attention(
            splits["test"], graph, max_windows=int(att_cfg["max_windows"]))
        for fname, frame in _attention_frames(record).items():
            frame.to_parquet(job_dir / fname, index=False)
        attention_written = True

    manifest = {
        "job_key": job.job_key,
        "stage": job.stage,
        "model": job.model_key,
        "model_spec": dict(spec),
        "graph_variant": job.graph_variant,
        "graph_sha": graph_sha,
        "seed": job.seed,
        "horizons": list(context.horizons),
        "n_nodes": context.n_nodes,
        "fit_report": {"fitted": report.fitted, "epochs_run": report.epochs_run,
                       "best_val_loss": report.best_val_loss, "notes": report.notes},
        "diagnostics": getattr(forecaster, "diagnostics", {}),
        "capabilities": vars(forecaster.capabilities),
        "attention_written": attention_written,
        "params": forecaster.get_params(),
        "git_hash": _git_hash(),
        "job_sha": job_sha(config, job.model_key, job.graph_variant, job.seed),
        "wall_seconds": round(time.time() - t0, 1),
    }
    if forecaster.capabilities.learns_topology:
        manifest["resolved_graph"] = forecaster.resolved_graph(context)
    with open(job_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def maybe_wandb_log(config: dict, manifest: dict, results_csv: Path) -> None:
    wb = config.get("wandb", {})
    if not wb.get("enabled", False):
        return
    import wandb
    run = wandb.init(project=wb["project"], name=manifest["job_key"],
                     mode=wb.get("mode", "offline"),
                     config={"manifest": manifest})
    df = pd.read_csv(results_csv)
    test = df[df["split"] == "test"]
    run.log({
        "results": wandb.Table(dataframe=df),
        "median_mae_raw_test": float(test["mae_raw"].median()),
        "median_mase_test": float(test["mase"].median()),
    })
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["run"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / "registry.json"
    registry: Dict[str, str] = (
        json.loads(registry_path.read_text()) if registry_path.exists() else {}
    )

    np.random.seed(int(config["run"].get("base_seed", 42)))

    # ---- context (once) ----
    store = GraphSeriesStore(config)
    context = store.build_context()
    store.node_report().to_csv(out_dir / "node_report.csv", index=False)
    t_cfg = config["temporal"]
    splits = build_graph_splits(
        context, stride=int(t_cfg.get("stride", 1)),
        train_frac=float(t_cfg["train_frac"]), val_frac=float(t_cfg["val_frac"]),
        purge_boundaries=bool(t_cfg["purge_boundaries"]),
    )
    pd.concat([count_usable_targets(s) for s in splits.values()]).to_csv(
        out_dir / "usable_targets.csv", index=False)
    mase_per_node = mase_scale(context, splits["train"])
    with open(out_dir / "context_manifest.json", "w") as f:
        json.dump({
            "node_ids": context.node_ids,
            "n_nodes": context.n_nodes,
            "grid": {"start": str(context.grid_index[0]),
                     "end": str(context.grid_index[-1]),
                     "freq": context.freq, "steps": len(context.grid_index)},
            "window": context.window,
            "horizons": list(context.horizons),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "scaler_mean": context.scaler_mean.tolist(),
            "scaler_std": context.scaler_std.tolist(),
            "mase_scale": mase_per_node.tolist(),
            "git_hash": _git_hash(),
        }, f, indent=2)
    logger.info(f"Context: {context.n_nodes} nodes, "
                f"{len(context.grid_index)} steps, splits "
                f"{ {k: len(v) for k, v in splits.items()} }")

    # ---- graphs (once per referenced variant) ----
    jobs = expand_jobs(config)
    graph_dir = out_dir / "graphs"
    graph_dir.mkdir(exist_ok=True)
    cache_path = config["nodes"]["edge_features_cache"]
    graphs: Dict[str, GraphSpec] = {}
    graph_shas: Dict[str, str] = {}
    for variant in sorted({j.graph_variant for j in jobs}):
        spec = build_graph(variant, config["graph"]["variants"][variant],
                           context.node_ids, cache_path)
        man = graph_manifest(spec, context.node_ids)
        with open(graph_dir / f"{variant}_manifest.json", "w") as f:
            json.dump(man, f, indent=2)
        graphs[variant] = spec
        graph_shas[variant] = man["sha16"]
        logger.info(f"Graph {variant}: {man['n_edges_directed']} directed edges, "
                    f"sha {man['sha16']}")

    # ---- jobs ----
    n_done = n_skipped = 0
    for i, job in enumerate(jobs):
        sha = job_sha(config, job.model_key, job.graph_variant, job.seed)
        if registry.get(job.job_key) == sha:
            n_skipped += 1
            logger.info(f"[{i + 1}/{len(jobs)}] {job.job_key} — up to date, skipped")
            continue
        logger.info(f"[{i + 1}/{len(jobs)}] {job.job_key} — running (stage {job.stage})")
        manifest = run_job(job, context, splits, graphs[job.graph_variant],
                           graph_shas[job.graph_variant], mase_per_node,
                           config, out_dir)
        registry[job.job_key] = sha
        registry_path.write_text(json.dumps(registry, indent=2))
        # W&B AFTER registry marking: a wandb crash must not rerun a finished job.
        maybe_wandb_log(config, manifest,
                        out_dir / "runs" / job.job_key / "results.csv")
        n_done += 1
        logger.info(f"    done in {manifest['wall_seconds']}s "
                    f"(epochs {manifest['fit_report']['epochs_run']})")

    logger.info(f"Finished: {n_done} run, {n_skipped} skipped, "
                f"{len(jobs)} total. Output: {out_dir}")


if __name__ == "__main__":
    main()
