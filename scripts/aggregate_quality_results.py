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
    correlations_mape_<model>_h<h>.csv   same, but with MAPE as the error metric
    threshold_<model>_h<h>.json          segmented-regression threshold test
    threshold_mape_<model>_h<h>.json     same, with MAPE
    dm_tests_h<h>.csv                    per-sensor Diebold-Mariano tests
    rf_exploratory_<model>_h<h>.json     exploratory RF quality→error model
    tables/table_metrics_h<h>.csv        sensors x models metric table (with sensor IDs)
    tables/table_interpolation.csv       variant x model, all-targets vs real-only
    tables/table_ablation.csv            feature-ablation comparison
    tables/table_sensor_summary.csv      per-sensor quality + scale + error overview
    figures/*.png                        threshold scatter (RMSE + MAPE), importances

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


def compute_posthoc_scale_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute MAPE, SMAPE, CV(RMSE) and mean_count per (sensor, model, horizon,
    interpolation, ablation) from stored per-window predictions.

    Uses max(|y|, 1) as MAPE denominator to handle zero-count targets.
    This function exists for backwards compatibility with runs that predate
    the evaluate.py change that adds these metrics natively.
    """
    group_cols = ["sensor_id", "model", "horizon", "interpolation", "ablation"]

    def _metrics(g: pd.DataFrame) -> pd.Series:
        y = g["y_true_raw"].to_numpy(dtype=float)
        p = g["y_pred_raw"].to_numpy(dtype=float)
        abs_err = np.abs(p - y)
        safe_denom = np.maximum(np.abs(y), 1.0)
        mape = float(np.mean(abs_err / safe_denom) * 100.0)
        smape_denom = np.abs(y) + np.abs(p)
        smape = float(np.mean(2.0 * abs_err / np.maximum(smape_denom, 1e-8)) * 100.0)
        mean_count = float(np.mean(y))
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        cv_rmse = float(rmse / max(mean_count, 1e-8) * 100.0)

        # Real-only subset
        if "target_interpolated" in g.columns:
            real = ~g["target_interpolated"].to_numpy(dtype=bool)
        else:
            real = np.ones(len(g), dtype=bool)

        if real.any():
            yr, pr = y[real], p[real]
            abs_err_r = np.abs(pr - yr)
            sd_r = np.maximum(np.abs(yr), 1.0)
            mape_ro = float(np.mean(abs_err_r / sd_r) * 100.0)
            mean_count_ro = float(np.mean(yr))
            rmse_ro = float(np.sqrt(np.mean((pr - yr) ** 2)))
            cv_rmse_ro = float(rmse_ro / max(mean_count_ro, 1e-8) * 100.0)
        else:
            mape_ro = mean_count_ro = cv_rmse_ro = float("nan")

        return pd.Series({
            "mape_raw": mape,
            "smape_raw": smape,
            "mean_count_raw": mean_count,
            "cv_rmse_raw": cv_rmse,
            "mape_raw_real_only": mape_ro,
            "mean_count_raw_real_only": mean_count_ro,
            "cv_rmse_raw_real_only": cv_rmse_ro,
        })

    # Filter to test-split predictions only
    test_preds = predictions[predictions["split"] == "test"]
    logger.info(f"Computing scale metrics from {len(test_preds):,} test predictions...")
    scale_df = test_preds.groupby(group_cols, sort=False).apply(_metrics, include_groups=False).reset_index()
    logger.info(f"Scale metrics computed for {len(scale_df)} (sensor, model, horizon, variant, ablation) combos")
    return scale_df


def _run_correlation_suite(sub, qdf, err_col, analysis_dir, model, h, suffix=""):
    """Run Spearman + partial correlations and save CSV. Returns the correlation df."""
    err = sub[err_col]
    corr = spearman_with_fdr(qdf, err)
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
    fname = f"correlations_{suffix}{model}_h{h}.csv" if suffix else f"correlations_{model}_h{h}.csv"
    corr.to_csv(analysis_dir / fname, index=False)
    return corr


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

    # ---- post-hoc scale metrics from predictions --------------------------
    scale_metrics = compute_posthoc_scale_metrics(predictions)
    merge_keys = ["sensor_id", "model", "horizon", "interpolation", "ablation"]
    # Drop columns that already exist in results to avoid _x/_y duplicates
    existing_scale_cols = [c for c in scale_metrics.columns if c not in merge_keys]
    for col in existing_scale_cols:
        if col in results.columns:
            results = results.drop(columns=[col])
    results = results.merge(scale_metrics, on=merge_keys, how="left")
    logger.info(f"Scale metrics merged: {results.columns.tolist()}")

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
        metric_cols = ["rmse_raw", "mae_raw", "mape_raw", "cv_rmse_raw", "mean_count_raw"]
        table = sub.pivot_table(
            index="sensor_id", columns="model",
            values=metric_cols,
        ).round(3)
        table.columns = [f"{metric}__{model}" for metric, model in table.columns]
        table = table.join(
            quality[["completeness_active", "avg_seq_len_hours"]].round(3)
        )
        table.to_csv(analysis_dir / "tables" / f"table_metrics_h{h}.csv")

    # ---- sensor summary table (paper Table 1 candidate) --------------------
    for h in horizons:
        sub = master[base_mask & (master["horizon"] == h) & (master["model"] == "lstm_mv")]
        if len(sub) == 0:
            continue
        sensor_tbl = sub.set_index("sensor_id")[[
            "completeness_active", "avg_seq_len_hours", "std_seq_len_hours",
            "seq_count", "mean_count_raw",
            "rmse_raw", "mae_raw", "mape_raw", "cv_rmse_raw",
        ]].round(3)
        sensor_tbl.to_csv(analysis_dir / "tables" / f"table_sensor_summary_h{h}.csv")

    # ---- correlations + threshold + RF per (model, horizon) ---------------
    for h in horizons:
        for model in models:
            sub = master[base_mask & (master["horizon"] == h) & (master["model"] == model)]
            if len(sub) < 10:
                logger.warning(f"Skipping stats for {model} h{h}: only {len(sub)} sensors")
                continue
            sub = sub.set_index("sensor_id")
            qdf = sub[QUALITY_PREDICTORS]

            # --- RMSE correlations + threshold (original) ---
            _run_correlation_suite(sub, qdf, "rmse_raw", analysis_dir, model, h)

            fit = segmented_threshold_fit(
                qdf["avg_seq_len_hours"].to_numpy(), sub["rmse_raw"].to_numpy()
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

            # --- MAPE correlations + threshold (new) ---
            if "mape_raw" in sub.columns and sub["mape_raw"].notna().sum() >= 10:
                _run_correlation_suite(sub, qdf, "mape_raw", analysis_dir, model, h, suffix="mape_")

                fit_mape = segmented_threshold_fit(
                    qdf["avg_seq_len_hours"].to_numpy(), sub["mape_raw"].to_numpy()
                )
                with open(analysis_dir / f"threshold_mape_{model}_h{h}.json", "w") as f:
                    json.dump(asdict(fit_mape), f, indent=2)
                plot_error_vs_quality(
                    sub.reset_index(), "avg_seq_len_hours", "mape_raw",
                    str(analysis_dir / "figures" / f"threshold_mape_{model}_h{h}.png"),
                    segmented_fit=fit_mape,
                    xlabel="Average sequence length (hours)",
                    ylabel="Test MAPE (%)",
                    title=f"{model}, horizon {h} steps (MAPE)",
                )

            # --- CV(RMSE) correlations + threshold (new) ---
            if "cv_rmse_raw" in sub.columns and sub["cv_rmse_raw"].notna().sum() >= 10:
                _run_correlation_suite(sub, qdf, "cv_rmse_raw", analysis_dir, model, h, suffix="cvrmse_")

                fit_cv = segmented_threshold_fit(
                    qdf["avg_seq_len_hours"].to_numpy(), sub["cv_rmse_raw"].to_numpy()
                )
                with open(analysis_dir / f"threshold_cvrmse_{model}_h{h}.json", "w") as f:
                    json.dump(asdict(fit_cv), f, indent=2)
                plot_error_vs_quality(
                    sub.reset_index(), "avg_seq_len_hours", "cv_rmse_raw",
                    str(analysis_dir / "figures" / f"threshold_cvrmse_{model}_h{h}.png"),
                    segmented_fit=fit_cv,
                    xlabel="Average sequence length (hours)",
                    ylabel="CV(RMSE) (%)",
                    title=f"{model}, horizon {h} steps (CV of RMSE)",
                )

            # --- mean_count vs error scatter (confound diagnostic) ---
            if "mean_count_raw" in sub.columns:
                plot_error_vs_quality(
                    sub.reset_index(), "mean_count_raw", "rmse_raw",
                    str(analysis_dir / "figures" / f"rmse_vs_meancount_{model}_h{h}.png"),
                    xlabel="Mean test count",
                    ylabel="Test RMSE (counts)",
                    title=f"{model} h{h}: RMSE vs traffic volume",
                )

            # --- RF exploratory (unchanged) ---
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
        agg_dict = {
            "n_sensors": ("sensor_id", "count"),
            "median_rmse_raw": ("rmse_raw", "median"),
            "median_rmse_real_only": ("rmse_raw_real_only", "median"),
            "median_mae_raw": ("mae_raw", "median"),
            "median_mae_real_only": ("mae_raw_real_only", "median"),
        }
        if "mape_raw" in interp.columns:
            agg_dict["median_mape_raw"] = ("mape_raw", "median")
        if "mape_raw_real_only" in interp.columns:
            agg_dict["median_mape_real_only"] = ("mape_raw_real_only", "median")
        if "cv_rmse_raw" in interp.columns:
            agg_dict["median_cv_rmse_raw"] = ("cv_rmse_raw", "median")
        table = interp.groupby(["interpolation", "model", "horizon"]).agg(
            **agg_dict
        ).round(3).reset_index()
        table.to_csv(analysis_dir / "tables" / "table_interpolation.csv", index=False)

    # ---- feature ablation ----------------------------------------------------
    abl = master[master["ablation"] != ""]
    if not abl.empty:
        abl_agg = {
            "n_sensors": ("sensor_id", "count"),
            "median_rmse_raw": ("rmse_raw", "median"),
            "median_mae_raw": ("mae_raw", "median"),
        }
        if "mape_raw" in abl.columns:
            abl_agg["median_mape_raw"] = ("mape_raw", "median")
        if "cv_rmse_raw" in abl.columns:
            abl_agg["median_cv_rmse_raw"] = ("cv_rmse_raw", "median")
        table = abl.groupby(["ablation", "horizon"]).agg(**abl_agg).round(3).reset_index()
        table.to_csv(analysis_dir / "tables" / "table_ablation.csv", index=False)

    logger.info(f"Analysis complete → {analysis_dir}")


if __name__ == "__main__":
    main()
