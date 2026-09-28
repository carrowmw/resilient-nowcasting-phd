#!/usr/bin/env python3
"""
Server Multi-GPU Batch Execution Suite for RIoT v4 Models and Ablations.

Dispatches training and ablation runs across available GPUs (default: GPU 0 and GPU 1)
using a thread-safe worker queue. Each worker sets CUDA_VISIBLE_DEVICES for strict
hardware isolation, logs to dedicated log files in logs/v4_suite/, extracts the
final test partition metrics from results_manifest.json, runs export_ui_payload.py,
and compiles a summary CSV and LaTeX table for Chapter 7 Table 7.3.

Usage:
    # Run all 8 jobs across GPU 0 and GPU 1:
    python scripts/run_v4_server_suite.py --gpus 0 1

    # Run on a single GPU (e.g., GPU 0):
    python scripts/run_v4_server_suite.py --gpus 0

    # Dry-run inspection without launching subprocesses:
    python scripts/run_v4_server_suite.py --dry-run

    # Local smoke-test with mock data:
    python scripts/run_v4_server_suite.py --gpus 0 --use_mock_data
"""

import argparse
import csv
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "v4_suite"
MODELS_DIR = PROJECT_ROOT / "models" / "resilient-iot-v4"
PAYLOAD_DIR = PROJECT_ROOT / "public" / "data" / "v4"

# Default ablation and comparison suite matrix (ordered by priority)
SUITE_JOBS = [
    {
        "name": "riot-v4-full-inductive",
        "desc": "RIoT v4 Full Model (Ablation Ceiling)",
        "config": "configs/train/v4/train_v4_full_inductive.yaml",
        "category": "core",
    },
    {
        "name": "lstm-v4-baseline-inductive",
        "desc": "Non-Spatial LSTM Baseline",
        "config": "configs/train/v4/train_v4_baseline_lstm.yaml",
        "category": "baseline",
    },
    {
        "name": "riot-v4-ablation-no-impute",
        "desc": "Ablation 1: No Reconstruction Loss (w_impute=0.0)",
        "config": "configs/train/v4/train_v4_ablation_no_impute.yaml",
        "category": "ablation",
    },
    {
        "name": "riot-v4-ablation-no-kriging",
        "desc": "Ablation 2: No Kriging Loss (w_kriging=0.0)",
        "config": "configs/train/v4/train_v4_ablation_no_kriging.yaml",
        "category": "ablation",
    },
    {
        "name": "riot-v4-ablation-no-masks",
        "desc": "Ablation 3: No Missingness Indicator Masks",
        "config": "configs/train/v4/train_v4_ablation_no_masks.yaml",
        "category": "ablation",
    },
    {
        "name": "riot-v4-ablation-no-pe",
        "desc": "Ablation 4: No Positional Encoding",
        "config": "configs/train/v4/train_v4_ablation_no_pe.yaml",
        "category": "ablation",
    },
    {
        "name": "riot-v4-tuning-small-batch",
        "desc": "Tuning: Small Batch Size (32)",
        "config": "configs/train/v4/train_v4_spatial_small_batch.yaml",
        "category": "tuning",
    },
    {
        "name": "riot-v4-tuning-high-dropout",
        "desc": "Tuning: High Dropout Regularization (0.3)",
        "config": "configs/train/v4/train_v4_spatial_high_dropout.yaml",
        "category": "tuning",
    },
]


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("v4_server_suite")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run RIoT v4 multi-GPU batch suite")
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1],
        help="List of GPU IDs to use (e.g., --gpus 0 1)",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        type=str,
        default=None,
        help="Subset of job names to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned execution order and exit without running",
    )
    parser.add_argument(
        "--use_mock_data",
        action="store_true",
        help="Run fast smoke test with synthetic mock data",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip running export_ui_payload.py after each job",
    )
    return parser.parse_args()


# Active child processes for clean cancellation
active_processes: List[subprocess.Popen] = []
process_lock = threading.Lock()


def register_process(proc: subprocess.Popen):
    with process_lock:
        active_processes.append(proc)


def unregister_process(proc: subprocess.Popen):
    with process_lock:
        if proc in active_processes:
            active_processes.remove(proc)


def signal_handler(signum, frame):
    print("\n[!] Termination signal received. Stopping active worker processes...")
    with process_lock:
        for proc in active_processes:
            try:
                proc.terminate()
            except Exception:
                pass
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_job(
    job: Dict,
    gpu_id: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict:
    """Execute a single training job and optional export on a specific GPU."""
    name = job["name"]
    cfg_path = PROJECT_ROOT / job["config"]
    log_path = LOGS_DIR / f"{name}_gpu{gpu_id}.log"

    result = {
        "name": name,
        "desc": job["desc"],
        "category": job["category"],
        "gpu_id": gpu_id,
        "status": "PENDING",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_sec": 0,
        "test_rmse": None,
        "test_mae": None,
        "test_zero_acc": None,
        "test_kriging_rmse": None,
        "test_kriging_mae": None,
        "test_loss": None,
        "best_val_loss": None,
    }

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train.py"),
        "--config",
        str(cfg_path),
        "--run-name",
        name,
    ]
    if args.use_mock_data:
        cmd.append("--use_mock_data")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    start_ts = time.time()
    logger.info(f"==> [GPU {gpu_id}] Starting: {name} ({job['desc']})")
    logger.info(f"    Logging output to: {log_path}")

    with open(log_path, "w") as log_f:
        log_f.write(f"=== Job {name} on GPU {gpu_id} started at {datetime.now()} ===\n")
        log_f.write(f"Command: {' '.join(cmd)}\n\n")
        log_f.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            register_process(proc)
            returncode = proc.wait()
            unregister_process(proc)

            elapsed = time.time() - start_ts
            result["duration_sec"] = round(elapsed, 1)
            result["end_time"] = datetime.now().isoformat()

            if returncode == 0:
                result["status"] = "COMPLETED"
                logger.info(
                    f"✓ [GPU {gpu_id}] Completed: {name} in {elapsed/60:.1f}m"
                )
            else:
                result["status"] = f"FAILED (exit code {returncode})"
                logger.error(
                    f"✗ [GPU {gpu_id}] FAILED: {name} (exit code {returncode}). See {log_path}"
                )

        except Exception as e:
            elapsed = time.time() - start_ts
            result["duration_sec"] = round(elapsed, 1)
            result["status"] = f"ERROR: {str(e)}"
            logger.error(f"✗ [GPU {gpu_id}] Exception running {name}: {e}")

    # Extract metrics from results.json (RESULTS_MANIFEST_FILENAME)
    manifest_path = MODELS_DIR / name / "results.json"
    if not manifest_path.exists():
        candidates = list((PROJECT_ROOT / "models").glob("**/results.json"))
        recent = [p for p in candidates if p.stat().st_mtime >= start_ts - 5]
        if recent:
            recent.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            manifest_path = recent[0]

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as mf:
                manifest_data = json.load(mf)

            result["best_val_loss"] = manifest_data.get("val_loss_total")
            test_m = manifest_data.get("test_metrics") or {}
            result["test_rmse"] = test_m.get("rmse")
            result["test_mae"] = test_m.get("mae")
            result["test_zero_acc"] = test_m.get("zero_acc")
            result["test_kriging_rmse"] = test_m.get("kriging_rmse")
            result["test_kriging_mae"] = test_m.get("kriging_mae")
            result["test_loss"] = test_m.get("test_loss")
        except Exception as e:
            logger.warning(f"Could not read manifest at {manifest_path}: {e}")

    # Run UI payload export if completed and not skipped
    if result["status"] == "COMPLETED" and not args.skip_export and not args.use_mock_data:
        checkpoint_path = MODELS_DIR / name / "best_model.pt"
        if not checkpoint_path.exists():
            chk_candidates = list((PROJECT_ROOT / "models").glob("**/best_model.pt"))
            recent_chk = [p for p in chk_candidates if p.stat().st_mtime >= start_ts - 5]
            if recent_chk:
                recent_chk.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                checkpoint_path = recent_chk[0]

        if checkpoint_path.exists():
            export_payload_path = PAYLOAD_DIR / f"{name}_payload.json"
            export_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "export_ui_payload.py"),
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(export_payload_path),
                "--quiet",
            ]
            logger.info(f"    [GPU {gpu_id}] Exporting UI payload for {name}...")
            try:
                subprocess.run(
                    export_cmd,
                    env=env,
                    cwd=str(PROJECT_ROOT),
                    check=True,
                    capture_output=True,
                )
                logger.info(f"    [GPU {gpu_id}] Exported UI payload -> {export_payload_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"    [GPU {gpu_id}] Payload export failed for {name}: {e.stderr.decode()[:200]}")

    return result


def worker_loop(
    gpu_id: int,
    job_queue: queue.Queue,
    results_list: List[Dict],
    results_lock: threading.Lock,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    """Worker thread dedicated to one GPU."""
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        res = run_job(job, gpu_id, args, logger)
        with results_lock:
            results_list.append(res)
            # Update summary CSV after each job completes
            write_summary_csv(results_list)

        job_queue.task_done()


def write_summary_csv(results: List[Dict]):
    csv_path = LOGS_DIR / "v4_suite_summary.csv"
    fields = [
        "name",
        "category",
        "desc",
        "status",
        "gpu_id",
        "duration_sec",
        "test_rmse",
        "test_mae",
        "test_zero_acc",
        "test_kriging_rmse",
        "test_kriging_mae",
        "test_loss",
        "best_val_loss",
        "start_time",
        "end_time",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fields})


def generate_latex_table(results: List[Dict]) -> str:
    """Generate LaTeX snippet matching Thesis Chapter 7 Table 7.3."""
    latex_path = LOGS_DIR / "tab_exp3_ablation.tex"

    def _fmt(v, p=2):
        return "--" if v is None else f"{v:.{p}f}"

    rows = []
    for r in results:
        label = r["desc"]
        rmse = _fmt(r.get("test_rmse"))
        mae = _fmt(r.get("test_mae"))
        zero_acc = _fmt(r.get("test_zero_acc"), 4)
        k_rmse = _fmt(r.get("test_kriging_rmse"))
        rows.append(f"        {label} & {rmse} & {mae} & {zero_acc} & {k_rmse} \\\\")

    content = f"""% Auto-generated by scripts/run_v4_server_suite.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
\\begin{{table}}[htbp]
    \\centering
    \\caption{{Experiment 3: RIoT v4 Model and Ablation Test Results (Evaluated on Held-Out Test Set)}}
    \\label{{tab:exp3-ablation}}
    \\begin{{tabular}}{{lrrrr}}
        \\toprule
        \\textbf{{Configuration}} & \\textbf{{Test RMSE}} & \\textbf{{Test MAE}} & \\textbf{{Zero Acc}} & \\textbf{{Kriging RMSE}} \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""
    with open(latex_path, "w") as f:
        f.write(content)
    return content


def print_ascii_summary(results: List[Dict], logger: logging.Logger):
    logger.info("\n" + "=" * 95)
    logger.info("  RIoT v4 BATCH EXECUTION SUITE - FINAL RESULTS SUMMARY")
    logger.info("=" * 95)
    header = (
        f"{'Job Name':<30} | {'Status':<11} | {'Time':<6} | "
        f"{'T-RMSE':<7} | {'T-MAE':<6} | {'ZeroAcc':<7} | {'KrigRMSE':<8}"
    )
    logger.info(header)
    logger.info("-" * 95)

    def _fmt(v, p=2):
        return "N/A" if v is None else f"{v:.{p}f}"

    for r in results:
        duration_m = f"{r['duration_sec']/60:.1f}m"
        line = (
            f"{r['name']:<30} | "
            f"{r['status']:<11} | "
            f"{duration_m:<6} | "
            f"{_fmt(r.get('test_rmse')):<7} | "
            f"{_fmt(r.get('test_mae')):<6} | "
            f"{_fmt(r.get('test_zero_acc'), 4):<7} | "
            f"{_fmt(r.get('test_kriging_rmse')):<8}"
        )
        logger.info(line)
    logger.info("=" * 95)
    logger.info(f"CSV Summary saved to:   {LOGS_DIR / 'v4_suite_summary.csv'}")
    logger.info(f"LaTeX Table saved to:  {LOGS_DIR / 'tab_exp3_ablation.tex'}")


def main():
    args = parse_args()
    logger = setup_logger()

    # Filter jobs if subset requested
    jobs_to_run = SUITE_JOBS
    if args.jobs:
        jobs_to_run = [j for j in SUITE_JOBS if j["name"] in args.jobs]
        if not jobs_to_run:
            logger.error(f"None of requested jobs {args.jobs} matched available jobs.")
            sys.exit(1)

    logger.info("=" * 75)
    logger.info(" RIoT v4 Multi-GPU Unattended Execution Suite")
    logger.info(f" Target GPUs:     {args.gpus}")
    logger.info(f" Total Jobs:      {len(jobs_to_run)}")
    logger.info(f" Mock Data Mode:  {args.use_mock_data}")
    logger.info(f" Skip UI Export:  {args.skip_export}")
    logger.info("=" * 75)

    for i, j in enumerate(jobs_to_run, 1):
        logger.info(f"  {i}. [{j['category'].upper():<8}] {j['name']} ({j['desc']})")
    logger.info("=" * 75)

    if args.dry_run:
        logger.info("[Dry Run] Exiting without executing jobs.")
        return

    job_queue = queue.Queue()
    for j in jobs_to_run:
        job_queue.put(j)

    results: List[Dict] = []
    results_lock = threading.Lock()
    threads: List[threading.Thread] = []

    suite_start = time.time()

    for gpu_id in args.gpus:
        t = threading.Thread(
            target=worker_loop,
            args=(gpu_id, job_queue, results, results_lock, args, logger),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all worker threads to complete
    for t in threads:
        t.join()

    total_time = time.time() - suite_start
    logger.info(f"\nAll suite jobs finished in {total_time/3600:.2f} hours ({total_time/60:.1f} minutes).")

    # Generate final outputs
    write_summary_csv(results)
    generate_latex_table(results)
    print_ascii_summary(results, logger)


if __name__ == "__main__":
    main()
