#!/usr/bin/env python3
"""
Aggregates quality-track run artifacts into the paper's result tables,
statistical analyses and figures.

Inputs  (produced by scripts/run_quality_experiments.py):
    <output_dir>/runs/<job_key>/{predictions.parquet, results.csv, manifest.json}
    <output_dir>/quality_table_<variant>.parquet
    <output_dir>/sensor_manifest.json

Outputs (under <output_dir>/analysis/):
    results.parquet              tidy per-(sensor, model, horizon, variant, ablation) metrics
                                 joined with quality indicators — the paper's master table
    correlations_<model>_h<h>.csv        Spearman + FDR (+ partial correlations)
    threshold_<model>_h<h>.json          segmented-regression threshold test
    dm_tests_h<h>.csv                    per-sensor Diebold-Mariano tests
    rf_exploratory_<model>_h<h>.json     exploratory RF quality→error model
    tables/table_metrics_h<h>.csv        sensors x models metric table (with sensor IDs)
    tables/table_interpolation.csv       variant x model, all-targets vs real-only
    tables/table_ablation.csv            feature-ablation comparison
    figures/*.png                        threshold scatter, importances, periodogram

Usage:
    python scripts/aggregate_quality_results.py --config configs/quality/quality.yaml
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import yaml

from quality.stats import (
    dm_matrix,
    partial_spearman,
    rf_quality_to_error,
    segmented_threshold_fit,
    spearman_with_fdr,
)
from quality.plotting import plot_error_vs_quality, plot_importances

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("aggregate_quality_results")

QUALITY_PREDICTORS = [
    "completeness_active",
    "avg_seq_len_hours",
    "std_seq_len_hours",
    "seq_count",
]


def load_runs(out_root: Path):
    runs_dir = out_root / "runs"
    assert runs_dir.exists(), f"CRITICAL: no runs directory at {runs_dir}"
    results, predictions = [], []
    for job_dir in sorted(runs_dir.iterdir()):
        if not (job_dir / "manifest.json").exists():
            continue
        results.append(pd.read_csv(job_dir / "results.csv"))
        predictions.append(pd.read_parquet(job_dir / "predictions.parquet"))
    assert results, f"CRITICAL: no completed jobs found under {runs_dir}"
    res = pd.concat(results, ignore_index=True)
    preds = pd.concat(predictions, ignore_index=True)
    res["ablation"] = res["ablation"].fillna("")
    preds["ablation"] = preds["ablation"].fillna("")
    logger.info(f"Loaded {len(results)} jobs: {len(res)} result rows, {len(preds):,} forecasts")
    return res, preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_root = Path(config["run"]["output_dir"])
    analysis_dir = out_root / "analysis"
    (analysis_dir / "tables").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "figures").mkdir(parents=True, exist_ok=True)

    results, predictions = load_runs(out_root)
    quality = pd.read_parquet(out_root / f"quality_table_none.parquet")

    # ---- master table ----------------------------------------------------
    master = results.merge(
        quality.reset_index()[["sensor_id", *QUALITY_PREDICTORS,
                               *[c for c in quality.columns if c.startswith("ls_") or c == "diurnal_capture_ratio"]]],
        on="sensor_id", how="left",
    )
    master.to_parquet(analysis_dir / "results.parquet", index=False)
    logger.info(f"Master table: {len(master)} rows → {analysis_dir / 'results.parquet'}")

    horizons = sorted(results["horizon"].unique())
    base_mask = (results["interpolation"] == "none") & (results["ablation"] == "")
    models = sorted(results.loc[base_mask, "model"].unique())
    lstm_models = [m for m in models if m.startswith("lstm")]
    baseline_models = [m for m in models if not m.startswith("lstm")]

    # ---- per-horizon metric tables (sensor IDs preserved) ------------------
    for h in horizons:
        sub = master[base_mask & (master["horizon"] == h)]
        table = sub.pivot_table(
            index="sensor_id", columns="model",
            values=["rmse_raw", "mae_raw"],
        ).round(3)
        table.columns = [f"{metric}__{model}" for metric, model in table.columns]
        table = table.join(
            quality[["completeness_active", "avg_seq_len_hours"]].round(3)
        )
        table.to_csv(analysis_dir / "tables" / f"table_metrics_h{h}.csv")

    # ---- correlations + threshold + RF per (model, horizon) ---------------
    for h in horizons:
        for model in models:
            sub = master[base_mask & (master["horizon"] == h) & (master["model"] == model)]
            if len(sub) < 10:
                logger.warning(f"Skipping stats for {model} h{h}: only {len(sub)} sensors")
                continue
            sub = sub.set_index("sensor_id")
            qdf = sub[QUALITY_PREDICTORS]
            err = sub["rmse_raw"]

            corr = spearman_with_fdr(qdf, err)
            # Partial correlations: the headline claim is seq-length ⊥ completeness.
            r_seq, p_seq = partial_spearman(
                qdf["avg_seq_len_hours"].to_numpy(), err.to_numpy(),
                qdf[["completeness_active"]].to_numpy(),
            )
            r_comp, p_comp = partial_spearman(
                qdf["completeness_active"].to_numpy(), err.to_numpy(),
                qdf[["avg_seq_len_hours"]].to_numpy(),
            )
            corr = pd.concat([corr, pd.DataFrame([
                {"metric": "avg_seq_len_hours | completeness", "n": len(sub),
                 "rho": r_seq, "p": p_seq, "p_fdr": np.nan, "significant_fdr": np.nan},
                {"metric": "completeness | avg_seq_len_hours", "n": len(sub),
                 "rho": r_comp, "p": p_comp, "p_fdr": np.nan, "significant_fdr": np.nan},
            ])], ignore_index=True)
            corr.to_csv(analysis_dir / f"correlations_{model}_h{h}.csv", index=False)

            fit = segmented_threshold_fit(
                qdf["avg_seq_len_hours"].to_numpy(), err.to_numpy()
            )
            with open(analysis_dir / f"threshold_{model}_h{h}.json", "w") as f:
                json.dump(asdict(fit), f, indent=2)
            plot_error_vs_quality(
                sub.reset_index(), "avg_seq_len_hours", "rmse_raw",
                str(analysis_dir / "figures" / f"threshold_{model}_h{h}.png"),
                segmented_fit=fit,
                xlabel="Average sequence length (hours)",
                ylabel="Test RMSE (counts)",
                title=f"{model}, horizon {h} steps",
            )

            if model in lstm_models:
                rf = rf_quality_to_error(qdf, sub["mae_raw"])
                with open(analysis_dir / f"rf_exploratory_{model}_h{h}.json", "w") as f:
                    json.dump(rf, f, indent=2)
                plot_importances(
                    rf["permutation_importance"],
                    str(analysis_dir / "figures" / f"rf_importance_{model}_h{h}.png"),
                )

    # ---- Diebold–Mariano: each LSTM vs every baseline ----------------------
    base_preds = predictions[
        (predictions["interpolation"] == "none") & (predictions["ablation"] == "")
    ]
    for h in horizons:
        dm_frames = []
        for focal in lstm_models:
            dm = dm_matrix(base_preds, baseline_models, focal, horizon=h)
            if not dm.empty:
                dm_frames.append(dm)
        if dm_frames:
            dm_all = pd.concat(dm_frames, ignore_index=True)
            dm_all.to_csv(analysis_dir / f"dm_tests_h{h}.csv", index=False)
            summary = dm_all.groupby(["focal", "baseline"]).agg(
                n_sensors=("sensor_id", "count"),
                frac_focal_better=("dm_stat", lambda s: float((s < 0).mean())),
                frac_significant=("p_value", lambda s: float((s < 0.05).mean())),
                median_dm=("dm_stat", "median"),
            ).reset_index()
            summary.to_csv(analysis_dir / f"dm_summary_h{h}.csv", index=False)

    # ---- interpolation comparison ------------------------------------------
    interp = master[master["ablation"] == ""]
    if interp["interpolation"].nunique() > 1:
        table = interp.groupby(["interpolation", "model", "horizon"]).agg(
            n_sensors=("sensor_id", "count"),
            median_rmse_raw=("rmse_raw", "median"),
            median_rmse_real_only=("rmse_raw_real_only", "median"),
            median_mae_raw=("mae_raw", "median"),
            median_mae_real_only=("mae_raw_real_only", "median"),
        ).round(3).reset_index()
        table.to_csv(analysis_dir / "tables" / "table_interpolation.csv", index=False)

    # ---- feature ablation ----------------------------------------------------
    abl = master[master["ablation"] != ""]
    if not abl.empty:
        table = abl.groupby(["ablation", "horizon"]).agg(
            n_sensors=("sensor_id", "count"),
            median_rmse_raw=("rmse_raw", "median"),
            median_mae_raw=("mae_raw", "median"),
        ).round(3).reset_index()
        table.to_csv(analysis_dir / "tables" / "table_ablation.csv", index=False)

    logger.info(f"Analysis complete → {analysis_dir}")


if __name__ == "__main__":
    main()
