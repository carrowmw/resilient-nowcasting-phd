#!/usr/bin/env python3
"""
Orchestrator for the quality-aware per-sensor experiments (MDPI rebuild).

Run matrix: sensors (from the viability manifest) x models x horizons x
interpolation variants (x feature ablations). One job = (model, horizon,
variant[, ablation]) across all sensors; each job persists:

    <output_dir>/runs/<job_key>/predictions.parquet   per-window forecasts (test split)
    <output_dir>/runs/<job_key>/results.csv           per-sensor metrics
    <output_dir>/runs/<job_key>/manifest.json         params + git hash + config sha + failures

plus, per interpolation variant:
    <output_dir>/quality_table_<variant>.parquet
and once:
    <output_dir>/sensor_manifest.json                 the sensor-selection record

The runner is resumable: completed job_keys are recorded in
<output_dir>/registry.json keyed by the config sha; rerunning skips them.

Usage (inside the container):
    python scripts/run_quality_experiments.py --config configs/quality/quality_smoke.yaml
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from quality.contract import ALL_FEATURE_GROUPS
from quality.evaluate import evaluate_split, predictions_frame
from quality.interpolation import apply_interpolation
from quality.loader import SensorSeriesStore
from quality.per_sensor_dataset import build_sensor_splits
from quality.periodogram import compute_periodogram_features
from quality.quality_metrics import (
    build_quality_table,
    build_sensor_manifest,
    find_consecutive_sequences,
    save_sensor_manifest,
    _git_hash,
)
from quality.registry import build_forecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_quality_experiments")


def config_sha(config: dict) -> str:
    canonical = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class VariantStore:
    """
    Store adapter serving interpolated series (and their masks) for one
    interpolation method, backed by the base SensorSeriesStore.
    """

    def __init__(self, base: SensorSeriesStore, method: str, interp_cfg: dict):
        self._base = base
        self.method = method
        self._cfg = interp_cfg
        self._cache = {}

    def series_ids(self):
        return self._base.series_ids()

    def expected_steps_in_window(self):
        return self._base.expected_steps_in_window()

    def get_series_and_mask(self, series_id: str):
        if series_id in self._cache:
            return self._cache[series_id]
        raw = self._base.get_series(series_id)
        filled, mask = apply_interpolation(
            raw,
            self.method,
            night_hours=tuple(self._cfg["night_hours"]),
            spline_order=int(self._cfg["spline_order"]),
            knn_neighbors=int(self._cfg["knn_neighbors"]),
            max_gap_steps=int(self._cfg.get("max_gap_steps", 0)),
        )
        self._cache[series_id] = (filled, mask)
        return filled, mask

    def get_series(self, series_id: str):
        return self.get_series_and_mask(series_id)[0]

    def drop_from_cache(self, series_id: str):
        self._cache.pop(series_id, None)
        self._base.drop_from_cache(series_id)


def select_sensors(quality_table: pd.DataFrame, run_cfg: dict) -> list:
    viable = quality_table[quality_table["viable"]].index.tolist()
    assert viable, "CRITICAL: no viable sensors under the current viability rule"
    spec = run_cfg["sensors"]
    if spec == "all":
        return viable
    if isinstance(spec, int):
        assert spec >= 1, f"CRITICAL: run.sensors={spec} must be >= 1"
        return viable[:spec]
    assert isinstance(spec, list), f"CRITICAL: run.sensors must be 'all', int, or list"
    missing = set(spec) - set(quality_table.index)
    assert not missing, f"CRITICAL: unknown sensors requested: {sorted(missing)}"
    not_viable = set(spec) - set(viable)
    assert not not_viable, f"CRITICAL: requested sensors fail viability: {sorted(not_viable)}"
    return spec


def add_periodogram_features(quality_table: pd.DataFrame, store, sensors: list,
                             pg_cfg: dict) -> pd.DataFrame:
    rows = {}
    for sid in sensors:
        series = store.get_series(sid)
        spans = find_consecutive_sequences(series, min_length=1)
        rows[sid] = compute_periodogram_features(
            series, spans,
            avg_seq_len_hours=float(quality_table.loc[sid, "avg_seq_len_hours"]),
            top_k=int(pg_cfg["top_k"]),
        )
        store.drop_from_cache(sid)
    pg_df = pd.DataFrame.from_dict(rows, orient="index")
    pg_df.index.name = "sensor_id"
    return quality_table.join(pg_df, how="left")


def run_job(
    job_key: str,
    model_key: str,
    horizon: int,
    variant_store: VariantStore,
    sensors: list,
    config: dict,
    feature_groups_override=None,
    ablation_key: str = "",
) -> dict:
    q_cfg = config["quality"]
    run_cfg = config["run"]
    out_dir = Path(run_cfg["output_dir"]) / "runs" / job_key
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frames, result_rows, failures = [], [], []
    t0 = time.time()
    for i, sensor_id in enumerate(sensors):
        try:
            series, mask = variant_store.get_series_and_mask(sensor_id)
            _, splits = build_sensor_splits(
                series, mask,
                window=int(q_cfg["window"]),
                horizon=horizon,
                stride=int(q_cfg.get("stride", 1)),
                train_frac=float(q_cfg["train_frac"]),
                val_frac=float(q_cfg["val_frac"]),
                purge_boundaries=bool(q_cfg.get("purge_boundaries", True)),
            )
            forecaster = build_forecaster(
                model_key,
                config["models"][model_key],
                seed=int(run_cfg["seed"]),
                device=run_cfg.get("device"),
                feature_groups_override=feature_groups_override,
            )
            fit_report = forecaster.fit(splits["train"], splits["val"])
            preds = forecaster.predict(splits["test"])
            metrics = evaluate_split(splits["test"], preds)
            pred_frames.append(predictions_frame(
                splits["test"], preds,
                model=model_key, horizon=horizon,
                interpolation=variant_store.method, ablation=ablation_key,
            ))
            result_rows.append({
                "sensor_id": sensor_id,
                "model": model_key,
                "horizon": horizon,
                "interpolation": variant_store.method,
                "ablation": ablation_key,
                "epochs_run": fit_report.epochs_run,
                "best_val_loss": fit_report.best_val_loss,
                **metrics,
                **{f"diag_{k}": v for k, v in getattr(forecaster, "diagnostics", {}).items()},
            })
        except Exception as exc:  # recorded loudly in the manifest, never silently
            logger.error(f"[{job_key}] sensor {sensor_id} FAILED: {exc}")
            failures.append({
                "sensor_id": sensor_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            })
        finally:
            variant_store.drop_from_cache(sensor_id)
        if (i + 1) % 10 == 0 or (i + 1) == len(sensors):
            logger.info(f"[{job_key}] {i + 1}/{len(sensors)} sensors "
                        f"({len(failures)} failures, {time.time() - t0:.0f}s)")

    assert result_rows, f"CRITICAL: job {job_key} produced no results ({len(failures)} failures)"
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "results.csv", index=False)

    manifest = {
        "job_key": job_key,
        "model": model_key,
        "model_spec": config["models"][model_key],
        "feature_groups_override": list(feature_groups_override) if feature_groups_override else None,
        "ablation": ablation_key,
        "horizon": horizon,
        "interpolation": variant_store.method,
        "n_sensors_ok": len(result_rows),
        "n_sensors_failed": len(failures),
        "failures": failures,
        "git_hash": _git_hash(),
        "config_sha": config_sha(config),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    if failures:
        logger.warning(f"[{job_key}] completed with {len(failures)} sensor failures — see manifest")
    return manifest


def maybe_wandb_log(config: dict, manifest: dict, results_csv: Path):
    wb_cfg = config.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return
    import wandb
    run = wandb.init(
        project=wb_cfg["project"],
        name=manifest["job_key"],
        mode=wb_cfg.get("mode", "offline"),
        config={"manifest": {k: v for k, v in manifest.items() if k != "failures"}},
    )
    results = pd.read_csv(results_csv)
    run.log({
        "per_sensor_results": wandb.Table(dataframe=results),
        "median_rmse_raw": float(results["rmse_raw"].median()),
        "median_mae_raw": float(results["mae_raw"].median()),
        "n_sensors_ok": manifest["n_sensors_ok"],
        "n_sensors_failed": manifest["n_sensors_failed"],
    })
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    sha = config_sha(config)
    run_cfg = config["run"]
    out_root = Path(run_cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(run_cfg["seed"]))
    torch.manual_seed(int(run_cfg["seed"]))

    registry_path = out_root / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}

    connector = None
    if config["data_source"] == "database":
        from db_connector import create_connector
        connector = create_connector(config)
    base_store = SensorSeriesStore(config, connector=connector)

    # ---- Phase A artifacts: quality tables per variant + sensor manifest ----
    variants = list(run_cfg["interpolation_variants"])
    assert variants and variants[0] == "none", \
        "CRITICAL: interpolation_variants must start with 'none' (defines the sensor set)"
    variant_stores, quality_tables = {}, {}
    for method in variants:
        vs = VariantStore(base_store, method, config["interpolation"])
        table_path = out_root / f"quality_table_{method}.parquet"
        if table_path.exists() and registry.get(f"quality_table_{method}") == sha:
            table = pd.read_parquet(table_path)
        else:
            logger.info(f"Building quality table for variant '{method}' …")
            table = build_quality_table(vs, config)
            if method == "none":
                sensors_tmp = table[table["viable"]].index.tolist()
                table = add_periodogram_features(table, vs, sensors_tmp, config["periodogram"])
            table.to_parquet(table_path)
            registry[f"quality_table_{method}"] = sha
            registry_path.write_text(json.dumps(registry, indent=2))
        variant_stores[method] = vs
        quality_tables[method] = table

    manifest = build_sensor_manifest(quality_tables["none"], config)
    save_sensor_manifest(manifest, str(out_root / "sensor_manifest.json"))
    sensors = select_sensors(quality_tables["none"], run_cfg)
    logger.info(f"Sensor set: {len(sensors)} of {len(quality_tables['none'])} candidates")

    # ---- Job matrix -----------------------------------------------------
    jobs = []
    for method in variants:
        for model_key in run_cfg["models"]:
            assert model_key in config["models"], \
                f"CRITICAL: run.models entry {model_key!r} not defined under models:"
            for horizon in run_cfg["horizons"]:
                jobs.append((f"{model_key}_h{horizon}_{method}", model_key, int(horizon),
                             method, None, ""))
    if run_cfg.get("run_ablations", False):
        abl_cfg = config["ablation"]
        for abl_key, groups in abl_cfg["feature_sets"].items():
            for horizon in run_cfg["horizons"]:
                jobs.append((
                    f"{abl_cfg['model']}_h{horizon}_none_abl-{abl_key}",
                    abl_cfg["model"], int(horizon), "none", list(groups), abl_key,
                ))

    logger.info(f"Job matrix: {len(jobs)} jobs x {len(sensors)} sensors")
    for job_key, model_key, horizon, method, groups, abl_key in jobs:
        if registry.get(job_key) == sha:
            logger.info(f"[{job_key}] already completed for this config — skipping")
            continue
        manifest = run_job(
            job_key, model_key, horizon, variant_stores[method], sensors, config,
            feature_groups_override=groups, ablation_key=abl_key,
        )
        maybe_wandb_log(config, manifest, out_root / "runs" / job_key / "results.csv")
        registry[job_key] = sha
        registry_path.write_text(json.dumps(registry, indent=2))

    if connector is not None:
        connector.close()
    logger.info("All jobs complete. Next: scripts/aggregate_quality_results.py")


if __name__ == "__main__":
    main()
