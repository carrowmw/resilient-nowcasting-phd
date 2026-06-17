"""
build_run_registry.py

Scan a paginated-payload output directory for per-run spatial.json files and emit
run_registry.json (the index the frontend loads first to populate the model selector).

Usage:
    python scripts/build_run_registry.py [runs_dir]

Defaults to frontend/public/data/runs.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_run_registry")

DEFAULT_RUNS_DIR = "frontend/public/data/runs"


def build_registry(runs_dir: str) -> Dict:
    """Build the registry dict from every {run}/spatial.json under runs_dir."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"runs_dir not found: {runs_dir}")

    registry: Dict = {"default_run_id": None, "runs": []}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        spatial_path = run_dir / "spatial.json"
        if not spatial_path.exists():
            logger.warning(f"Skipping {run_dir.name}: no spatial.json")
            continue
        spatial = json.loads(spatial_path.read_text())
        meta = spatial.get("metadata", {})

        # Available dates: prefer metadata; fall back to scanning day folders.
        dates = meta.get("available_dates")
        if not dates:
            dates = sorted(
                d.name for d in run_dir.iterdir()
                if d.is_dir() and (d / "page.json").exists()
            )

        registry["runs"].append({
            "run_id": run_dir.name,
            "wandb_run_id": meta.get("wandb_run_id", "unknown"),
            "wandb_run_name": run_dir.name,
            "git_commit": meta.get("git_commit_hash", "unknown"),
            "generated_at": meta.get("generated_at", "unknown"),
            "analysis_period": {
                "start_date": dates[0] if dates else None,
                "end_date": dates[-1] if dates else None,
            },
            "train_config": meta.get("hyperparameters", {}),
            "metrics_summary": meta.get("metrics_summary", {}),
            "available_dates": dates,
            "num_nodes": meta.get("num_nodes", 0),
            "num_simulated_nodes": meta.get("num_simulated_nodes", 0),
        })

    if registry["runs"]:
        registry["default_run_id"] = registry["runs"][0]["run_id"]
    return registry


def main() -> None:
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUNS_DIR
    registry = build_registry(runs_dir)
    out_path = Path(runs_dir) / "run_registry.json"
    out_path.write_text(json.dumps(registry, indent=2))
    logger.info(
        f"Wrote {out_path} | {len(registry['runs'])} run(s) | "
        f"default_run_id={registry['default_run_id']}"
    )


if __name__ == "__main__":
    main()
