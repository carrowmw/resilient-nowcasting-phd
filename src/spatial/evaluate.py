"""
Evaluation for the spatial track: per-(node, horizon) metrics on raw counts,
observed targets only.

MAPE is deliberately absent: person counts are zero-heavy and MAPE is
meaningless on them even with a max(|y|,1) denominator. MASE (scaled by the
in-sample daily-seasonal-naive MAE per node) is the scale-free metric —
MASE < 1 means "better than repeating yesterday".
"""
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from spatial.contract import GraphContext, GraphSplit

# One shared key list — metric producers and consumers can never drift.
METRIC_KEYS = ("rmse_raw", "mae_raw", "smape_raw", "mase",
               "rmse_scaled", "mae_scaled", "mean_count_raw", "n_targets")


def mase_scale(context: GraphContext, train: GraphSplit) -> np.ndarray:
    """
    [N] in-sample daily-seasonal-naive MAE per node over train-touched steps
    (pairs t vs t-96 where both observed). Nodes without a single valid pair
    get NaN — their MASE is honestly unreportable, never fabricated.
    """
    season = context.steps_per_day()
    last_touched = (int(train.window_start_idx.max()) + context.window - 1
                    + context.max_horizon)
    raw = context.values_raw[: last_touched + 1]
    obs = context.observed_mask[: last_touched + 1]
    assert raw.shape[0] > season, \
        "CRITICAL: train span shorter than one season — MASE undefined"
    err = np.abs(raw[season:] - raw[:-season])
    both = obs[season:] & obs[:-season]
    scale = np.full(context.n_nodes, np.nan)
    for j in range(context.n_nodes):
        e = err[both[:, j], j]
        if len(e) > 0 and e.mean() > 0:
            scale[j] = e.mean()
    return scale


def evaluate_split(split: GraphSplit, preds_scaled: np.ndarray,
                   mase_scale_per_node: np.ndarray) -> pd.DataFrame:
    """
    One row per (node_id, horizon). Metrics over observed targets only.
    (node, horizon) cells with zero observed targets get NaN metrics and
    n_targets=0 — present in the table, never silently dropped.
    """
    ctx = split.context
    n, N, H = len(split), ctx.n_nodes, len(ctx.horizons)
    assert preds_scaled.shape == (n, N, H), \
        f"CRITICAL: predictions shape {preds_scaled.shape} != ({n}, {N}, {H})"
    assert np.isfinite(preds_scaled).all(), "CRITICAL: non-finite predictions"
    assert mase_scale_per_node.shape == (N,)

    preds_raw = ctx.inverse_transform(preds_scaled)
    y_raw = split.y_raw
    y_scaled = split.y_scaled
    observed = split.target_observed

    rows = []
    for j, node in enumerate(ctx.node_ids):
        for k, h in enumerate(ctx.horizons):
            m = observed[:, j, k]
            row: Dict = {"node_id": node, "horizon": int(h), "split": split.split}
            if not m.any():
                row.update({key: (0 if key == "n_targets" else np.nan)
                            for key in METRIC_KEYS})
                rows.append(row)
                continue
            pr, tr = preds_raw[m, j, k], y_raw[m, j, k]
            ps, ts = preds_scaled[m, j, k], y_scaled[m, j, k]
            err_raw = pr - tr
            mae_raw = float(np.abs(err_raw).mean())
            denom = np.maximum((np.abs(pr) + np.abs(tr)) / 2.0, 1e-8)
            scale = mase_scale_per_node[j]
            row.update({
                "rmse_raw": float(np.sqrt((err_raw ** 2).mean())),
                "mae_raw": mae_raw,
                "smape_raw": float((np.abs(err_raw) / denom).mean()),
                "mase": float(mae_raw / scale) if np.isfinite(scale) else np.nan,
                "rmse_scaled": float(np.sqrt(((ps - ts) ** 2).mean())),
                "mae_scaled": float(np.abs(ps - ts).mean()),
                "mean_count_raw": float(tr.mean()),
                "n_targets": int(m.sum()),
            })
            rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df) == N * H
    return df


def aggregate_nodes(results: pd.DataFrame) -> pd.DataFrame:
    """Per horizon: median/IQR/mean across nodes (NaN-metric nodes excluded,
    exclusion count reported)."""
    out = []
    for h, grp in results.groupby("horizon"):
        row = {"horizon": int(h), "n_nodes": int(len(grp)),
               "n_nodes_no_targets": int((grp["n_targets"] == 0).sum())}
        for key in ("rmse_raw", "mae_raw", "smape_raw", "mase"):
            vals = grp[key].dropna()
            row[f"{key}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{key}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25)) \
                if len(vals) else np.nan
            row[f"{key}_mean"] = float(vals.mean()) if len(vals) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def predictions_frame(split: GraphSplit, preds_scaled: np.ndarray, model: str,
                      graph_variant: str, seed: int) -> pd.DataFrame:
    """
    Long-format per-target predictions for DM tests and post-hoc analysis.
    target_observed rides along so real-only metrics can always be recomputed.
    """
    ctx = split.context
    n, N, H = len(split), ctx.n_nodes, len(ctx.horizons)
    assert preds_scaled.shape == (n, N, H)
    preds_raw = ctx.inverse_transform(preds_scaled)

    origin = np.repeat(split.origin_dt.to_numpy(), N * H)
    target = np.repeat(split.target_dt[:, None, :], N, axis=1).reshape(-1)
    nodes = np.tile(np.repeat(np.asarray(ctx.node_ids), H), n)
    horizons = np.tile(np.asarray(ctx.horizons, dtype=np.int64), n * N)

    return pd.DataFrame({
        "model": model,
        "graph_variant": graph_variant,
        "seed": int(seed),
        "node_id": nodes,
        "horizon": horizons,
        "origin_dt": origin,
        "target_dt": target,
        "y_true_raw": split.y_raw.reshape(-1),
        "y_pred_raw": preds_raw.reshape(-1),
        "y_true_scaled": split.y_scaled.reshape(-1),
        "y_pred_scaled": preds_scaled.reshape(-1),
        "target_observed": split.target_observed.reshape(-1),
        "split": split.split,
    })
