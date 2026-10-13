"""
Statistical comparison framework for the spatial track.

Reuses the pure statistical machinery from the quality track
(quality.stats.diebold_mariano — HAC + Harvey–Leybourne–Newbold — and
bootstrap_ci); adds the node-level wrappers this track needs: per-node DM
with BH-FDR across nodes, node-resampled deltas, seed stability, ablation
tables.
"""
import logging
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from quality.stats import bootstrap_ci, diebold_mariano  # pure functions, reused

logger = logging.getLogger(__name__)

_JOIN = ["node_id", "target_dt"]


def dm_by_node(preds_a: pd.DataFrame, preds_b: pd.DataFrame, horizon: int,
               loss: str = "squared", min_n: int = 30) -> pd.DataFrame:
    """
    Per-node Diebold–Mariano on test-split predictions of two models at one
    horizon. Negative dm_stat → model A more accurate.

    Rows with n < min_n or a degenerate loss-differential series are excluded
    WITH a reason column — never silently.
    """
    frames = []
    for tag, df in (("a", preds_a), ("b", preds_b)):
        sel = df[(df["horizon"] == horizon) & (df["split"] == "test")
                 & df["target_observed"]]
        assert len(sel), f"CRITICAL: model {tag} has no observed test targets at h={horizon}"
        frames.append(sel[_JOIN + ["y_true_raw", "y_pred_raw"]])
    a, b = frames

    merged = a.merge(b, on=_JOIN, suffixes=("_a", "_b"))
    assert len(merged) == len(a) == len(b), (
        f"CRITICAL: models did not predict the identical window set "
        f"({len(a)} vs {len(b)} vs merged {len(merged)})"
    )
    np.testing.assert_allclose(
        merged["y_true_raw_a"], merged["y_true_raw_b"],
        err_msg="CRITICAL: models disagree on the ground truth — data drift")

    rows = []
    for node, grp in merged.groupby("node_id"):
        grp = grp.sort_values("target_dt")
        ea = (grp["y_pred_raw_a"] - grp["y_true_raw_a"]).to_numpy()
        eb = (grp["y_pred_raw_b"] - grp["y_true_raw_a"]).to_numpy()
        row: Dict = {"node_id": node, "n": len(grp), "excluded_reason": ""}
        if len(grp) < min_n:
            row["excluded_reason"] = f"n={len(grp)} < {min_n}"
            row.update({"dm_stat": np.nan, "p_value": np.nan, "mean_loss_diff": np.nan})
        else:
            try:
                dm = diebold_mariano(ea, eb, horizon=horizon, loss=loss)
                row.update({"dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
                            "mean_loss_diff": dm["mean_loss_diff"]})
            except AssertionError as exc:
                row["excluded_reason"] = f"degenerate: {exc}"
                row.update({"dm_stat": np.nan, "p_value": np.nan,
                            "mean_loss_diff": np.nan})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)


def dm_summary(dm_df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """BH-FDR across nodes → fraction significantly different / improved."""
    from statsmodels.stats.multitest import multipletests

    ok = dm_df[dm_df["excluded_reason"] == ""].copy()
    n_excluded = len(dm_df) - len(ok)
    if len(ok) == 0:
        return {"n_nodes": 0, "n_excluded": n_excluded,
                "frac_significant_fdr": np.nan, "frac_a_better": np.nan,
                "frac_a_better_significant": np.nan,
                "median_dm_stat": np.nan, "mean_loss_diff": np.nan}
    reject, p_fdr, _, _ = multipletests(ok["p_value"], alpha=alpha, method="fdr_bh")
    a_better = ok["dm_stat"] < 0
    return {
        "n_nodes": int(len(ok)),
        "n_excluded": int(n_excluded),
        "frac_significant_fdr": float(reject.mean()),
        "frac_a_better": float(a_better.mean()),
        "frac_a_better_significant": float((reject & a_better).mean()),
        "median_dm_stat": float(ok["dm_stat"].median()),
        "mean_loss_diff": float(ok["mean_loss_diff"].mean()),
    }


def paired_delta_ci(results_a: pd.DataFrame, results_b: pd.DataFrame,
                    metric: str, horizon: int, n_bootstrap: int = 2000,
                    seed: int = 42) -> Dict:
    """
    Bootstrap CI (nodes resampled jointly) on median(metric_a - metric_b)
    at one horizon. Negative delta → model A better.
    """
    a = results_a[results_a["horizon"] == horizon].set_index("node_id")[metric]
    b = results_b[results_b["horizon"] == horizon].set_index("node_id")[metric]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    assert len(joined) >= 5, \
        f"CRITICAL: only {len(joined)} nodes with finite {metric} at h={horizon}"
    delta = (joined["a"] - joined["b"]).to_numpy()
    out = bootstrap_ci(lambda d: float(np.median(d)), delta,
                       n_bootstrap=n_bootstrap, seed=seed)
    out.update({"metric": metric, "horizon": int(horizon), "n_nodes": len(joined)})
    return out


def seed_stability(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Across-seed spread of the node-median metric per (model, graph_variant,
    horizon). `results` needs columns: model, graph_variant, seed, horizon,
    node_id, <metric>.
    """
    med = (results.groupby(["model", "graph_variant", "seed", "horizon"])[metric]
           .median().reset_index())
    out = (med.groupby(["model", "graph_variant", "horizon"])[metric]
           .agg(mean="mean", sd="std", min="min", max="max",
                n_seeds="count").reset_index())
    return out


def ablation_table(agg: pd.DataFrame, champion_key: str, metric: str = "mae_raw_median",
                   keys: Sequence[str] = ("model", "graph_variant")) -> pd.DataFrame:
    """
    Δmetric vs the champion row, per horizon. `agg` needs one row per
    (model, graph_variant, horizon) — the aggregator's model_comparison table
    (seed-averaged).
    """
    key_col = agg[list(keys)].agg("|".join, axis=1)
    agg = agg.assign(_key=key_col)
    assert (agg["_key"] == champion_key).any(), \
        f"CRITICAL: champion {champion_key!r} not present in the table"
    rows = []
    for h, grp in agg.groupby("horizon"):
        champ = grp[grp["_key"] == champion_key]
        assert len(champ) == 1, \
            f"CRITICAL: champion {champion_key!r} ambiguous at h={h}"
        base = float(champ[metric].iloc[0])
        for _, r in grp.iterrows():
            rows.append({
                "horizon": int(h), "key": r["_key"], metric: float(r[metric]),
                f"delta_{metric}": float(r[metric]) - base,
                f"delta_{metric}_pct": 100.0 * (float(r[metric]) - base) / base
                if base != 0 else np.nan,
                "is_champion": r["_key"] == champion_key,
            })
    return pd.DataFrame(rows)
