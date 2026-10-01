#!/usr/bin/env python3
"""
Multi-GPU unattended runner for the RIoT pipeline-v5 experiment suite.

Reads configs/train/v5/suite_manifest.yaml (scripts/riotv5/generate_v5_configs.py)
and dispatches every job to a pool of GPU worker slots (default: GPUs 0 1 2 3, one
job per GPU). Each job runs `src/train.py --config <cfg> --run-name <name>` with
CUDA_VISIBLE_DEVICES pinned to its GPU.

Scheduling: tier 1 (headline thesis results) -> tier 2 (screens / mechanisms) ->
tier 3 (extra seeds); within a tier the most expensive jobs start first so the
suite does not end on one long straggler.

Resume: a completed job writes logs/v5_suite/status/<name>.json; re-launching the
suite skips it (use --force to re-run). Failed jobs are retried on the next launch.

Outputs (logs/v5_suite/):
  runs/<name>.log                per-job stdout/stderr
  status/<name>.json             per-job status + path of its results.json
  (--use_mock_data runs use logs/v5_suite_mock/ instead, never mixed with real runs)
  v5_suite_runs.csv              one row per run: key test metrics
  v5_suite_summary.csv           one row per experiment: mean / std over seeds
  tables/<group>.tex             LaTeX table per experiment group (mean +- std)
  suite_<timestamp>.log          dispatcher log

Usage:
  python scripts/riotv5/run_v5_suite.py --gpus 0 1 2 3              # everything
  python scripts/riotv5/run_v5_suite.py --tiers 1                    # headline runs only
  python scripts/riotv5/run_v5_suite.py --groups thesis_v4 p7_v5_suite --seeds 42
  python scripts/riotv5/run_v5_suite.py --dry-run                    # print the plan
  python scripts/riotv5/run_v5_suite.py --summarise-only             # rebuild CSV / LaTeX
  python scripts/riotv5/run_v5_suite.py --gpus 0 --use_mock_data --smoke --jobs v5_full-s42
"""
import argparse
import csv
import fnmatch
import json
import logging
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "configs" / "train" / "v5" / "suite_manifest.yaml"
LOGS_DIR = PROJECT_ROOT / "logs" / "v5_suite"
RUN_LOGS = LOGS_DIR / "runs"
STATUS_DIR = LOGS_DIR / "status"
TABLES_DIR = LOGS_DIR / "tables"
SMOKE_DIR = LOGS_DIR / "smoke_configs"
MODELS_DIR = PROJECT_ROOT / "models"

# Metrics carried into the CSV / LaTeX summaries (keys of results.json v5_test_metrics,
# except the "best_val" / "best_epoch" / "test_loss" bookkeeping fields).
SUMMARY_METRICS = [
    "obs_all_rmse", "obs_all_mae", "obs_trans_rmse", "obs_ind_rmse",
    "peak_rmse", "peak_underpred_rate", "collapse_rate", "real_gap_mass_ratio",
    "so_1_forecast_rmse", "so_12_forecast_rmse", "so_96_forecast_rmse",
    "so_12_mass_ratio", "so_96_mass_ratio", "so_12_impute_rmse",
    "crps", "q90_coverage", "nll_obs", "low_tol_acc", "night_short_gap_zero_rate",
    "kriging_raw_rmse", "impute_raw_rmse",
    "zir_obs_zero_rate", "zir_night_missing_rate", "zir_day_missing_rate",
    "mask_flip_dpoint", "node_nrmse_median_trans", "node_nrmse_median_ind",
    "ind_corr_mean", "ind_pred_pairwise_corr_mean", "ind_true_pairwise_corr_mean",
]
# Columns of the per-group LaTeX tables (subset; the CSV has everything).
LATEX_METRICS = [
    ("obs_trans_rmse", "RMSE$_{tr}$"), ("obs_ind_rmse", "RMSE$_{ind}$"), ("peak_rmse", "Peak RMSE"),
    ("so_12_forecast_rmse", "SO@12"), ("so_96_forecast_rmse", "SO@96"), ("so_96_mass_ratio", "Mass@96"),
    ("collapse_rate", "Collapse"), ("crps", "CRPS"), ("kriging_raw_rmse", "Krig. RMSE"),
]


def parse_args():
    ap = argparse.ArgumentParser(description="Run the RIoT v5 experiment suite on multiple GPUs")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--jobs-per-gpu", type=int, default=1,
                    help="concurrent jobs per GPU (1 is safest; 2 if memory allows)")
    ap.add_argument("--groups", nargs="+", default=None, help="experiment groups to include")
    ap.add_argument("--tiers", nargs="+", type=int, default=None, help="tiers to include (1 2 3)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="seeds to include")
    ap.add_argument("--jobs", nargs="+", default=None, help="run names or glob patterns (e.g. 'v5_*-s42')")
    ap.add_argument("--force", action="store_true", help="re-run jobs that already completed")
    ap.add_argument("--stagger", type=float, default=45.0,
                    help="seconds between job starts (spreads the database load of dataset construction)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise-only", action="store_true", help="only rebuild CSV / LaTeX from status files")
    ap.add_argument("--use_mock_data", action="store_true", help="synthetic city instead of the database")
    ap.add_argument("--smoke", action="store_true",
                    help="with --use_mock_data: 2 epochs, 12 nodes, W&B disabled (pipeline check only)")
    return ap.parse_args()


def use_mock_dirs():
    """Mock / smoke runs keep their own status and summaries (never mixed with real runs)."""
    global LOGS_DIR, RUN_LOGS, STATUS_DIR, TABLES_DIR, SMOKE_DIR
    LOGS_DIR = PROJECT_ROOT / "logs" / "v5_suite_mock"
    RUN_LOGS, STATUS_DIR = LOGS_DIR / "runs", LOGS_DIR / "status"
    TABLES_DIR, SMOKE_DIR = LOGS_DIR / "tables", LOGS_DIR / "smoke_configs"


def setup_logger() -> logging.Logger:
    for d in (LOGS_DIR, RUN_LOGS, STATUS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("v5_suite")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOGS_DIR / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_jobs() -> List[Dict]:
    assert MANIFEST.exists(), (
        f"{MANIFEST} not found — run `python scripts/riotv5/generate_v5_configs.py` first"
    )
    with open(MANIFEST) as f:
        return yaml.safe_load(f)["jobs"]


def select_jobs(jobs: List[Dict], args) -> List[Dict]:
    out = []
    for j in jobs:
        if args.groups and j["group"] not in args.groups:
            continue
        if args.tiers and j["tier"] not in args.tiers:
            continue
        if args.seeds and j["seed"] not in args.seeds:
            continue
        if args.jobs and not any(fnmatch.fnmatch(j["name"], p) for p in args.jobs):
            continue
        out.append(j)
    return sorted(out, key=lambda j: (j["tier"], -j["cost"], j["name"]))


def status_path(name: str) -> Path:
    return STATUS_DIR / f"{name}.json"


def read_status(name: str) -> Optional[Dict]:
    p = status_path(name)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def smoke_config(job: Dict) -> Path:
    """A shrunken copy of the job config for --smoke (structure unchanged)."""
    with open(PROJECT_ROOT / job["config"]) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["num_nodes"] = 12
    cfg["training"]["epochs"] = 2
    cfg["training"]["early_stop_patience"] = 1
    cfg["loss"]["impute_pretrain_epochs"] = min(cfg["loss"]["impute_pretrain_epochs"], 1)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    path = SMOKE_DIR / f"{job['name']}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def find_results(job: Dict, project: str, start_ts: float) -> Optional[Path]:
    """train.py writes models/<project>/<run name>/results.json under an active W&B
    run, models/local/<run name>_<timestamp>/results.json when W&B is disabled."""
    direct = MODELS_DIR / project / job["name"] / "results.json"
    if direct.exists() and direct.stat().st_mtime >= start_ts:
        return direct
    local = sorted(
        (p for p in (MODELS_DIR / "local").glob(f"{job['name']}_*/results.json") if p.stat().st_mtime >= start_ts),
        key=lambda p: p.stat().st_mtime,
    )
    return local[-1] if local else None


# Active child processes for clean cancellation.
_active: List[subprocess.Popen] = []
_active_lock = threading.Lock()
_stop = threading.Event()


def _signal_handler(signum, frame):
    print("\n[!] Termination signal received — stopping workers and active runs...", flush=True)
    _stop.set()
    with _active_lock:
        for p in _active:
            p.terminate()
    sys.exit(1)


def run_job(job: Dict, gpu: int, args, project: str, logger: logging.Logger) -> Dict:
    cfg_path = smoke_config(job) if args.smoke else PROJECT_ROOT / job["config"]
    log_path = RUN_LOGS / f"{job['name']}.log"
    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "train.py"),
           "--config", str(cfg_path), "--run-name", job["name"]]
    if args.use_mock_data:
        cmd.append("--use_mock_data")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    if args.smoke:
        env["WANDB_MODE"] = "disabled"

    start_ts = time.time()
    logger.info(f"==> [GPU {gpu}] start  {job['name']}  (tier {job['tier']}, cost {job['cost']}) — {job['desc']}")
    with open(log_path, "w") as log_f:
        log_f.write(f"=== {job['name']} on GPU {gpu} at {datetime.now()} ===\n$ {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
        with _active_lock:
            _active.append(proc)
        returncode = proc.wait()
        with _active_lock:
            _active.remove(proc)

    elapsed = time.time() - start_ts
    results = find_results(job, project, start_ts) if returncode == 0 else None
    status = "COMPLETED" if returncode == 0 and results is not None else (
        "FAILED (no results.json)" if returncode == 0 else f"FAILED (exit {returncode})"
    )
    record = {
        **{k: job[k] for k in ("name", "experiment", "group", "seed", "tier", "desc", "config")},
        "status": status, "gpu": gpu, "duration_sec": round(elapsed, 1),
        "finished": datetime.now().isoformat(), "results": str(results) if results else None,
        "log": str(log_path), "mock": args.use_mock_data,
    }
    with open(status_path(job["name"]), "w") as f:
        json.dump(record, f, indent=2)
    mark = "✓" if status == "COMPLETED" else "✗"
    logger.info(f"{mark}   [GPU {gpu}] {status:<10} {job['name']} in {elapsed / 60:.1f} min"
                + ("" if status == "COMPLETED" else f" — see {log_path}"))
    return record


class StartGate:
    """Spaces job starts by `stagger` seconds across all workers."""

    def __init__(self, stagger: float):
        self.stagger = stagger
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            delay = self.last + self.stagger - time.time()
            if delay > 0:
                time.sleep(delay)
            self.last = time.time()


def worker(gpu: int, jobs: "queue.Queue[Dict]", gate: StartGate, args, project, logger, done: List[Dict], lock):
    while not _stop.is_set():
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        gate.wait()
        rec = run_job(job, gpu, args, project, logger)
        with lock:
            done.append(rec)
            summarise(load_jobs(), logger, quiet=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _load_metrics(record: Dict) -> Dict:
    with open(record["results"]) as f:
        res = json.load(f)
    test = res["v5_test_metrics"] or {}
    row = {m: test.get(m) for m in SUMMARY_METRICS}
    row["best_epoch"] = res["best_epoch"]
    row["selection_metric"] = res["selection_metric"]
    row["holdout_manifest_sha"] = res["holdout_manifest_sha"]
    return row


def _fmt(v, p=3):
    return "" if v is None else f"{v:.{p}f}"


def summarise(all_jobs: List[Dict], logger: logging.Logger, quiet: bool = False):
    runs = []
    for j in all_jobs:
        rec = read_status(j["name"])
        if rec is None or rec["status"] != "COMPLETED":
            continue
        runs.append({**rec, **_load_metrics(rec)})

    run_fields = ["name", "experiment", "group", "seed", "tier", "status", "duration_sec", "best_epoch",
                  "selection_metric", "holdout_manifest_sha", *SUMMARY_METRICS]
    with open(LOGS_DIR / "v5_suite_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    # Aggregate over seeds per experiment, in generator order.
    order, by_exp = [], {}
    for j in all_jobs:
        if j["experiment"] not in by_exp:
            order.append(j["experiment"])
            by_exp[j["experiment"]] = {"group": j["group"], "desc": j["desc"], "runs": []}
    for r in runs:
        by_exp[r["experiment"]]["runs"].append(r)

    agg_rows = []
    for exp in order:
        e = by_exp[exp]
        if not e["runs"]:
            continue
        row = {"experiment": exp, "group": e["group"], "desc": e["desc"], "n_seeds": len(e["runs"]),
               "holdout_sha_consistent": len({r["holdout_manifest_sha"] for r in e["runs"]}) == 1}
        for m in SUMMARY_METRICS:
            vals = [r[m] for r in e["runs"] if r[m] is not None]
            row[f"{m}_mean"] = statistics.fmean(vals) if vals else None
            row[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        agg_rows.append(row)

    agg_fields = ["experiment", "group", "desc", "n_seeds", "holdout_sha_consistent"] + [
        f"{m}_{s}" for m in SUMMARY_METRICS for s in ("mean", "std")
    ]
    with open(LOGS_DIR / "v5_suite_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for row in agg_rows:
            w.writerow({k: (_fmt(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    shas = {r["holdout_manifest_sha"] for r in runs}
    if len(shas) > 1:
        logger.warning(f"[!] {len(shas)} distinct holdout manifests across runs — "
                       "comparisons are only paired within one manifest")

    groups = {}
    for row in agg_rows:
        groups.setdefault(row["group"], []).append(row)
    for group, rows in groups.items():
        _write_latex(group, rows)

    if not quiet:
        logger.info("=" * 110)
        logger.info(f"{'experiment':<30} {'n':>2} {'RMSE_tr':>9} {'RMSE_ind':>9} {'peak':>9} "
                    f"{'SO@12':>9} {'SO@96':>9} {'mass96':>7} {'collapse':>8} {'CRPS':>8}")
        for row in agg_rows:
            g = lambda m, p=2: _fmt(row[f"{m}_mean"], p) or "--"  # noqa: E731
            logger.info(f"{row['experiment']:<30} {row['n_seeds']:>2} {g('obs_trans_rmse'):>9} "
                        f"{g('obs_ind_rmse'):>9} {g('peak_rmse'):>9} {g('so_12_forecast_rmse'):>9} "
                        f"{g('so_96_forecast_rmse'):>9} {g('so_96_mass_ratio', 3):>7} "
                        f"{g('collapse_rate', 3):>8} {g('crps', 3):>8}")
        logger.info("=" * 110)
        logger.info(f"Per-run CSV:     {LOGS_DIR / 'v5_suite_runs.csv'}")
        logger.info(f"Per-experiment:  {LOGS_DIR / 'v5_suite_summary.csv'}")
        logger.info(f"LaTeX tables:    {TABLES_DIR}/")


def _write_latex(group: str, rows: List[Dict]):
    def cell(row, m):
        mean, std = row[f"{m}_mean"], row[f"{m}_std"]
        if mean is None:
            return "--"
        return f"{mean:.3g}" if std is None else f"{mean:.3g} $\\pm$ {std:.2g}"

    body = "\n".join(
        "        " + row["experiment"].replace("_", "\\_") + f" ({row['n_seeds']})"
        + "".join(f" & {cell(row, m)}" for m, _ in LATEX_METRICS) + " \\\\"
        for row in rows
    )
    head = " & ".join(f"\\textbf{{{label}}}" for _, label in LATEX_METRICS)
    content = (
        f"% Auto-generated by scripts/riotv5/run_v5_suite.py on {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        "\\begin{table}[htbp]\n    \\centering\\scriptsize\n"
        f"    \\caption{{RIoT v5 suite, group \\texttt{{{group.replace('_', chr(92) + '_')}}} "
        "(test split, observed targets; mean $\\pm$ std over seeds, n in brackets)}}\n"
        f"    \\label{{tab:v5-{group.replace('_', '-')}}}\n"
        f"    \\begin{{tabular}}{{l{'r' * len(LATEX_METRICS)}}}\n        \\toprule\n"
        f"        \\textbf{{Configuration}} & {head} \\\\\n        \\midrule\n{body}\n"
        "        \\bottomrule\n    \\end{tabular}\n\\end{table}\n"
    )
    with open(TABLES_DIR / f"{group}.tex", "w") as f:
        f.write(content)


def main():
    args = parse_args()
    assert not args.smoke or args.use_mock_data, "--smoke requires --use_mock_data"
    if args.use_mock_data:
        use_mock_dirs()
    logger = setup_logger()
    with open(MANIFEST) as f:
        project = yaml.safe_load(f)["wandb_project"]
    all_jobs = load_jobs()

    if args.summarise_only:
        summarise(all_jobs, logger)
        return

    selected = select_jobs(all_jobs, args)
    pending, skipped = [], []
    for j in selected:
        rec = read_status(j["name"])
        done = rec is not None and rec["status"] == "COMPLETED"
        (skipped if done and not args.force else pending).append(j)

    total_cost = sum(j["cost"] for j in pending)
    slots = [g for g in args.gpus for _ in range(args.jobs_per_gpu)]
    logger.info("=" * 90)
    logger.info(" RIoT pipeline-v5 experiment suite")
    logger.info(f" GPUs: {args.gpus} x {args.jobs_per_gpu} slot(s) | selected {len(selected)} | "
                f"already complete {len(skipped)} | to run {len(pending)}")
    logger.info(f" Relative cost {total_cost:.1f} v4-full-run units "
                f"(~{total_cost / max(len(args.gpus), 1):.1f} units of wall time on {len(args.gpus)} GPU(s))")
    logger.info(f" Mock data: {args.use_mock_data} | smoke: {args.smoke} | W&B project: {project}")
    logger.info("=" * 90)
    for i, j in enumerate(pending, 1):
        logger.info(f" {i:>3}. [t{j['tier']} {j['group']:<13}] {j['name']:<34} cost {j['cost']:<5} {j['desc']}")

    if args.dry_run or not pending:
        if not pending:
            logger.info("Nothing to run.")
        summarise(all_jobs, logger, quiet=args.dry_run)
        return

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    q: "queue.Queue[Dict]" = queue.Queue()
    for j in pending:
        q.put(j)
    gate = StartGate(0.0 if args.use_mock_data else args.stagger)
    done: List[Dict] = []
    lock = threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(g, q, gate, args, project, logger, done, lock), daemon=True)
               for g in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    n_ok = sum(r["status"] == "COMPLETED" for r in done)
    logger.info(f"Suite finished in {(time.time() - t0) / 3600:.2f} h: {n_ok}/{len(done)} completed.")
    failed = [r["name"] for r in done if r["status"] != "COMPLETED"]
    if failed:
        logger.info(f"Failed ({len(failed)}): {' '.join(failed)} — re-launch to retry (completed jobs are skipped).")
    summarise(all_jobs, logger)


if __name__ == "__main__":
    main()
