# Tests for src/spatial/stats.py and attention analysis — DM wrappers, FDR, CIs.
import numpy as np
import pandas as pd
import pytest

from spatial.attention import (
    AttentionRecord,
    attention_entropy,
    attention_seed_agreement,
    attention_vs_distance,
    top_edges_table,
)
from spatial.stats import ablation_table, dm_by_node, dm_summary, paired_delta_ci


def _preds(model, sigma, n_nodes=4, n_windows=120, seed=0, bias=0.0):
    """Synthetic long-format predictions with controllable error scale.
    Ground truth is drawn from a FIXED seed so both models share it."""
    truth_rng = np.random.default_rng(1234)
    rng = np.random.default_rng(seed)
    dts = pd.date_range("2024-06-01", periods=n_windows, freq="15min")
    rows = []
    for j in range(n_nodes):
        truth = truth_rng.uniform(5, 50, n_windows)
        err = rng.normal(bias, sigma, n_windows)
        for t in range(n_windows):
            rows.append({
                "model": model, "graph_variant": "g", "seed": 0,
                "node_id": f"n{j}", "horizon": 1,
                "target_dt": dts[t], "split": "test", "target_observed": True,
                "y_true_raw": truth[t], "y_pred_raw": truth[t] + err[t],
            })
    return pd.DataFrame(rows)


def test_dm_positive_control():
    """Model A (small errors) must beat model B (large errors) on most nodes."""
    a = _preds("a", sigma=1.0, seed=1)
    b = _preds("b", sigma=4.0, seed=2)
    dm = dm_by_node(a, b, horizon=1)
    assert len(dm) == 4
    assert (dm["excluded_reason"] == "").all()
    assert (dm["dm_stat"] < 0).all()          # negative → A better
    summary = dm_summary(dm)
    assert summary["frac_a_better"] == 1.0
    assert summary["frac_significant_fdr"] == 1.0


def test_dm_negative_control():
    """Same error process → few/no significant nodes after FDR."""
    a = _preds("a", sigma=2.0, seed=3)
    b = _preds("b", sigma=2.0, seed=4)
    summary = dm_summary(dm_by_node(a, b, horizon=1))
    assert summary["frac_significant_fdr"] <= 0.25


def test_dm_identical_window_set_assert():
    a = _preds("a", sigma=1.0)
    b = _preds("b", sigma=1.0).iloc[:-10]     # drop rows → misaligned
    with pytest.raises(AssertionError, match="identical window set"):
        dm_by_node(a, b, horizon=1)


def test_dm_short_series_excluded_loudly():
    a = _preds("a", sigma=1.0, n_windows=20)
    b = _preds("b", sigma=2.0, n_windows=20)
    dm = dm_by_node(a, b, horizon=1, min_n=30)
    assert (dm["excluded_reason"].str.contains("< 30")).all()
    assert np.isnan(dm["dm_stat"]).all()
    assert dm_summary(dm)["n_nodes"] == 0


def test_paired_delta_ci():
    rng = np.random.default_rng(5)
    nodes = [f"n{j}" for j in range(20)]
    a = pd.DataFrame({"node_id": nodes, "horizon": 1,
                      "mae_raw": rng.uniform(2, 3, 20)})
    b = a.copy()
    b["mae_raw"] = a["mae_raw"] + 1.0          # B uniformly worse by 1
    out = paired_delta_ci(a, b, "mae_raw", horizon=1)
    assert out["point"] == pytest.approx(-1.0, abs=1e-9)
    assert out["ci_upper"] < 0                  # significantly better


def test_ablation_table():
    agg = pd.DataFrame({
        "model": ["gat", "gat", "lstm", "lstm"],
        "graph_variant": ["gabriel", "gabriel", "identity", "identity"],
        "horizon": [1, 2, 1, 2],
        "mae_raw_median": [2.0, 3.0, 2.5, 3.6],
    })
    tab = ablation_table(agg, champion_key="gat|gabriel")
    row = tab[(tab.horizon == 1) & (tab.key == "lstm|identity")].iloc[0]
    assert row["delta_mae_raw_median"] == pytest.approx(0.5)
    assert row["delta_mae_raw_median_pct"] == pytest.approx(25.0)
    with pytest.raises(AssertionError, match="not present"):
        ablation_table(agg, champion_key="ghost|graph")


# ------------------------------------------------------------------ attention

def _record(alpha_flat, edge_index, seed=0):
    E = edge_index.shape[1]
    return AttentionRecord(
        edge_index=edge_index,
        alpha_mean=np.repeat(alpha_flat[:, None], 2, axis=1).astype(np.float32),
        alpha_by_hour=np.full((24, E), np.nan, dtype=np.float32),
        n_windows=10, seed=seed,
    )


def test_attention_entropy_extremes():
    # Node 0 receives edges from 1 and 2: uniform vs peaked.
    ei = np.array([[1, 2, 0], [0, 0, 1]], dtype=np.int64)
    uniform = attention_entropy(_record(np.array([0.5, 0.5, 1.0]), ei), n_nodes=3)
    peaked = attention_entropy(_record(np.array([0.999, 0.001, 1.0]), ei), n_nodes=3)
    n0u = uniform[uniform.node_idx == 0]["entropy_norm"].iloc[0]
    n0p = peaked[peaked.node_idx == 0]["entropy_norm"].iloc[0]
    assert n0u == pytest.approx(1.0, abs=1e-6)
    assert n0p < 0.05
    # In-degree < 2 → NaN, honestly.
    assert np.isnan(uniform[uniform.node_idx == 2]["entropy_norm"].iloc[0])


def test_attention_vs_distance_and_top_edges():
    ei = np.array([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=np.int64)  # 2 real + 2 loops
    rec = _record(np.array([0.9, 0.2, 1.0, 1.0]), ei)
    dist = np.array([[0.0, 100.0], [100.0, 0.0]])
    df = attention_vs_distance(rec, dist, ["a", "b"])
    assert len(df) == 2 and (df["distance_m"] == 100.0).all()
    top = top_edges_table(rec, ["a", "b"], k=1)
    assert top.iloc[0]["src"] == "a" and top.iloc[0]["alpha"] == pytest.approx(0.9)


def test_attention_seed_agreement():
    ei = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    a = _record(np.array([0.9, 0.5, 0.1]), ei, seed=0)
    b = _record(np.array([0.8, 0.6, 0.2]), ei, seed=1)   # same ranking
    out = attention_seed_agreement([a, b])
    assert out["mean_spearman"] == pytest.approx(1.0)
    c = _record(np.array([0.1, 0.5, 0.9]), ei, seed=2)   # reversed
    out2 = attention_seed_agreement([a, c])
    assert out2["mean_spearman"] == pytest.approx(-1.0)
    with pytest.raises(AssertionError, match="edge set"):
        attention_seed_agreement([a, _record(np.array([0.5, 0.5]),
                                             np.array([[0, 1], [1, 0]]))])
