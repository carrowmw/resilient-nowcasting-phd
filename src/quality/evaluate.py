"""
Per-window prediction persistence and per-sensor metric computation.

Every model's forecasts for a given (sensor, horizon, variant) are made on the
identical window set and persisted per window — the precondition for paired
Diebold–Mariano tests and consistent cross-model tables.

Metrics are reported in raw count space (primary; comparable across sensors
and models) and scaled space (continuity with the original paper). When
interpolation is active, metrics are additionally computed on the subset of
windows whose target is a real observation ('real-only') so synthetic values
can never launder an apparent improvement.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quality.contract import PerSensorSplit

logger = logging.getLogger(__name__)


def _rmse_mae(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    assert y.shape == p.shape, f"CRITICAL: shape mismatch {y.shape} vs {p.shape}"
    err = p - y
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }


def _scale_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """MAPE, SMAPE, CV(RMSE), and mean count — scale-aware metrics.

    MAPE denominator uses max(|y|, 1) to handle zero-count targets without
    silently dropping them (the zero is a real observation in pedestrian data).
    """
    assert y.shape == p.shape, f"CRITICAL: shape mismatch {y.shape} vs {p.shape}"
    abs_err = np.abs(p - y)
    safe_denom = np.maximum(np.abs(y), 1.0)
    mape = float(np.mean(abs_err / safe_denom) * 100.0)
    smape_denom = np.abs(y) + np.abs(p)
    smape = float(np.mean(2.0 * abs_err / np.maximum(smape_denom, 1e-8)) * 100.0)
    mean_count = float(np.mean(y))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    cv_rmse = float(rmse / max(mean_count, 1e-8) * 100.0)
    return {
        "mape": mape,
        "smape": smape,
        "mean_count": mean_count,
        "cv_rmse": cv_rmse,
    }


def evaluate_split(split: PerSensorSplit, preds_scaled: np.ndarray) -> Dict[str, float]:
    """Aggregate metrics for one (sensor, model, horizon, variant, split)."""
    assert len(preds_scaled) == len(split), \
        f"CRITICAL: {len(preds_scaled)} predictions for {len(split)} windows"
    assert not np.isnan(preds_scaled).any(), "CRITICAL: NaN predictions reached evaluation"

    ctx = split.context
    y_raw = split.y_raw
    p_raw = ctx.inverse_transform(preds_scaled)

    metrics: Dict[str, float] = {}
    for k, v in _rmse_mae(y_raw, p_raw).items():
        metrics[f"{k}_raw"] = v
    for k, v in _scale_metrics(y_raw, p_raw).items():
        metrics[f"{k}_raw"] = v
    for k, v in _rmse_mae(split.y_scaled, preds_scaled).items():
        metrics[f"{k}_scaled"] = v

    real = ~split.target_interpolated
    metrics["n_windows"] = int(len(split))
    metrics["n_real_targets"] = int(real.sum())
    if real.any():
        for k, v in _rmse_mae(y_raw[real], p_raw[real]).items():
            metrics[f"{k}_raw_real_only"] = v
        for k, v in _scale_metrics(y_raw[real], p_raw[real]).items():
            metrics[f"{k}_raw_real_only"] = v
    else:
        metrics["rmse_raw_real_only"] = float("nan")
        metrics["mae_raw_real_only"] = float("nan")
        metrics["mape_raw_real_only"] = float("nan")
        metrics["smape_raw_real_only"] = float("nan")
        metrics["mean_count_raw_real_only"] = float("nan")
        metrics["cv_rmse_raw_real_only"] = float("nan")
    return metrics


def predictions_frame(
    split: PerSensorSplit,
    preds_scaled: np.ndarray,
    model: str,
    horizon: int,
    interpolation: str,
    ablation: Optional[str] = None,
) -> pd.DataFrame:
    """Tidy per-window forecast rows (the DM-test substrate)."""
    ctx = split.context
    return pd.DataFrame({
        "sensor_id": ctx.sensor_id,
        "model": model,
        "horizon": horizon,
        "interpolation": interpolation,
        "ablation": ablation if ablation is not None else "",
        "split": split.split,
        "origin_dt": split.origin_dt,
        "target_dt": split.target_dt,
        "seq_id": split.seq_id,
        "y_true_raw": split.y_raw,
        "y_pred_raw": ctx.inverse_transform(preds_scaled),
        "y_true_scaled": split.y_scaled,
        "y_pred_scaled": preds_scaled,
        "target_interpolated": split.target_interpolated,
    })
