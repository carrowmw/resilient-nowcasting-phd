#!/usr/bin/env python3
"""Outlier Event Analysis — end-to-end runner.

Pulls inference payloads from Cloudflare R2 + training curves from W&B, runs the
eight outlier tests, and writes figures, a results JSON and a markdown report to
``outputs/``.

Usage
-----
    python run_analysis.py                 # full 31-day analysis (default run)
    python run_analysis.py --quick         # first 3 days, for a fast sanity pass
    python run_analysis.py --run-id NAME   # analyse a specific R2 run
    python run_analysis.py --refresh       # ignore cache and re-download
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oea import analysis, config, data_io, loading, report  # noqa: E402


def parse_args() -> config.AnalysisConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default=None, help="R2 run id (default: registry default)")
    p.add_argument("--max-days", type=int, default=None, help="limit number of dates")
    p.add_argument("--quick", action="store_true", help="shortcut for --max-days 3")
    p.add_argument("--max-workers", type=int, default=24, help="download concurrency")
    p.add_argument("--refresh", action="store_true", help="ignore cache, re-download")
    p.add_argument("--skip-wandb", action="store_true", help="skip W&B training-curve pull")
    p.add_argument("--no-simulated", action="store_true",
                   help="do not download simulated-node files (uses page predictions only)")
    a = p.parse_args()
    cfg = config.AnalysisConfig(
        run_id=a.run_id or config.DEFAULT_RUN_ID,
        max_days=3 if a.quick else a.max_days,
        max_workers=a.max_workers,
        refresh=a.refresh,
        skip_wandb=a.skip_wandb,
        include_simulated_nodes=not a.no_simulated,
    )
    return cfg


def main() -> int:
    np.random.seed(42)                       # reproducible bootstraps / jitter
    cfg = parse_args()
    config.ensure_dirs()
    t0 = time.time()

    # 1. Resolve run + dates from the registry / spatial metadata.
    registry = data_io.load_registry(refresh=cfg.refresh)
    if cfg.run_id == config.DEFAULT_RUN_ID and registry.get("default_run_id"):
        cfg.run_id = registry["default_run_id"]
    reg_entry = next((r for r in registry.get("runs", []) if r["run_id"] == cfg.run_id), {})
    cfg.wandb_run_id = reg_entry.get("wandb_run_id")

    spatial = data_io.load_spatial(cfg.run_id, refresh=cfg.refresh)
    meta = spatial["metadata"]
    dates = list(meta.get("available_dates", []))
    if cfg.max_days:
        dates = dates[: cfg.max_days]
    node_ids = [n["id"] for n in spatial["nodes"]]
    if not cfg.include_simulated_nodes:
        node_ids = [n["id"] for n in spatial["nodes"] if not n.get("is_simulated")]

    print(f"[run] run_id={cfg.run_id} wandb={cfg.wandb_run_id} "
          f"dates={len(dates)} nodes={len(node_ids)}", file=sys.stderr)

    # 2. Warm the cache (concurrent).
    data_io.prefetch(cfg.run_id, dates, node_ids,
                     max_workers=cfg.max_workers, refresh=cfg.refresh)

    # 3. Build dataset + shared outlier artefacts.
    ds = loading.build_dataset(cfg)
    print(f"[run] dataset built: T={ds.T} real={len(ds.real_ids)} "
          f"holdout={len(ds.holdout_ids)} sim={len(ds.sim_ids)}", file=sys.stderr)
    shared = analysis.compute_shared(ds)
    print(f"[run] {int(shared.n_flagged.sum())} outlier steps, "
          f"{len(shared.events)} events", file=sys.stderr)

    # 4. Run the eight tests.
    tests = [
        ("test1", analysis.test1_catalogue),
        ("test2", analysis.test2_zinb),
        ("test3", analysis.test3_propagation),
        ("test4", analysis.test4_buildup),
        ("test5", analysis.test5_kriging),
        ("test6", analysis.test6_tripwire),
        ("test7", analysis.test7_scale),
        ("test8", analysis.test8_two_extremes),
    ]
    results = {}
    for key, fn in tests:
        ts = time.time()
        results[key] = fn(ds, shared)
        print(f"[run] {key} done ({time.time()-ts:.1f}s)", file=sys.stderr)

    # 5. W&B training curves (optional, non-fatal).
    wandb_run = None
    if not cfg.skip_wandb and cfg.wandb_run_id:
        wandb_run = data_io.load_wandb_run(cfg.wandb_run_id)

    # 6. Emit outputs.
    run_meta = {
        "run_id": cfg.run_id,
        "wandb_run_id": cfg.wandb_run_id,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": reg_entry.get("git_commit", ""),
        "n_dates": len(ds.dates),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report.write_outputs(ds, results, wandb_run, shared, run_meta)
    print(f"[run] wrote {config.REPORT_PATH} and outputs/results.json "
          f"in {run_meta['elapsed_sec']}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
