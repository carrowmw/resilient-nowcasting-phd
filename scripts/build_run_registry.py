"""
build_run_registry.py

Scan a paginated-payload output directory for per-run spatial.json files and emit
run_registry.json (the index the frontend loads first to populate the model selector).

Schema: RunRegistry in frontend/src/types/registry.ts (schema_version 5). Runs are
ordered by their registry rank (1 = the v5_full-s42 reference, then by validation
RMSE); the reference run is the default selection.

Usage:
    python scripts/build_run_registry.py [runs_dir]

Defaults to artifacts/payload_staging/runs.
"""

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_run_registry")

DEFAULT_RUNS_DIR = "artifacts/payload_staging/runs"
SCHEMA_VERSION = 5

# RunInfo fields copied verbatim from spatial.json metadata (all required).
RUN_INFO_KEYS = (
    "run_id", "experiment", "group", "description", "model_type", "forecast_likelihood",
    "quantile_levels", "capabilities", "seq_len", "wandb_run_id", "wandb_artifact",
    "git_commit_hash", "generated_at",
)
ENTRY_KEYS = (
    "registry", "available_dates", "num_nodes", "num_holdout_nodes", "num_ghost_nodes", "holdout_manifest_sha",
    "test_metrics", "window_metrics", "hyperparameters",
)


def build_registry(runs_dir: str) -> Dict:
    """Build the registry dict from every {run}/spatial.json under runs_dir."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"runs_dir not found: {runs_dir}")

    runs = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        spatial_path = run_dir / "spatial.json"
        if not spatial_path.exists():
            logger.warning(f"Skipping {run_dir.name}: no spatial.json")
            continue
        meta = json.loads(spatial_path.read_text())["metadata"]
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"CRITICAL: {spatial_path} has schema_version={meta.get('schema_version')}, "
                f"expected {SCHEMA_VERSION} — regenerate it with src/export_paginated_payload.py"
            )
        missing = [k for k in RUN_INFO_KEYS + ENTRY_KEYS if k not in meta]
        if missing:
            raise KeyError(f"CRITICAL: {spatial_path} metadata missing {missing}")
        if meta["run_id"] != run_dir.name:
            raise ValueError(f"CRITICAL: {spatial_path} run_id {meta['run_id']!r} != directory {run_dir.name!r}")

        on_disk = sorted(d.name for d in run_dir.iterdir() if d.is_dir() and (d / "page.json").exists())
        if on_disk != sorted(meta["available_dates"]):
            raise ValueError(f"CRITICAL: {run_dir.name}: available_dates disagree with the day folders on disk")

        entry = {k: meta[k] for k in RUN_INFO_KEYS}
        entry.update({k: meta[k] for k in ENTRY_KEYS})
        entry["analysis_period"] = {"start_date": meta["start_date"], "end_date": meta["end_date"]}
        runs.append(entry)

    runs.sort(key=lambda r: r["registry"]["rank"])
    ranks = [r["registry"]["rank"] for r in runs]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"CRITICAL: duplicate registry ranks {ranks} — runs from different exports mixed")

    reference = [r["run_id"] for r in runs if r["registry"]["is_reference"]]
    default = reference[0] if reference else (runs[0]["run_id"] if runs else None)
    rules = {r["registry"]["selection_rule"] for r in runs}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "default_run_id": default,
        "selection_rule": rules.pop() if len(rules) == 1 else "mixed",
        "runs": runs,
    }


def main() -> None:
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUNS_DIR
    registry = build_registry(runs_dir)
    out_path = Path(runs_dir) / "run_registry.json"
    out_path.write_text(json.dumps(registry, separators=(",", ":"), allow_nan=False))
    logger.info(
        f"Wrote {out_path} | {len(registry['runs'])} run(s) | "
        f"default_run_id={registry['default_run_id']}"
    )


if __name__ == "__main__":
    main()
