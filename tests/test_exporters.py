"""RIoT v5 payload exporter tests.

Pure-maths checks for the predictive-distribution helpers and attention mapping,
the top-k run selection against the committed results table, and an end-to-end
contract test: a tiny v5 model on the SyntheticConnector city goes through the
real exporter path (build_v5_context → infer_windows → writers) and every emitted
JSON object must carry exactly the fields of its TypeScript interface in
frontend/src/types/{payload,registry}.ts.
"""
import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "riotv5"))

import export_paginated_payload as epp  # noqa: E402
from export_ui_payload import (  # noqa: E402
    build_ego_graph_edges,
    build_v5_context,
    last_step_attention,
    mid_pit_and_crps,
    pmf_grid,
    quantiles_from_pmf,
    reliability_bins,
)
from build_run_registry import build_registry  # noqa: E402
from harness.holdout import make_fixed_holdout_mask, save_holdout_manifest  # noqa: E402
from model_factory import build_model  # noqa: E402
from run_config import resolve_config  # noqa: E402
from train import build_dataset  # noqa: E402

import generate_v5_configs as gen  # noqa: E402

NODES = 8
TS_DIR = ROOT / "frontend" / "src" / "types"


# --------------------------------------------------------------------------- #
# TypeScript contract                                                           #
# --------------------------------------------------------------------------- #
def ts_interfaces() -> dict:
    """{interface: set(field names)} with `extends` resolved, from the TS types."""
    raw = {}
    for path in (TS_DIR / "payload.ts", TS_DIR / "registry.ts"):
        src = path.read_text()
        for m in re.finditer(r"export interface (\w+)(?: extends (\w+))? \{(.*?)\n\}", src, re.S):
            fields = set(re.findall(r"^\s+(\w+)\??:", m.group(3), re.M))
            raw[m.group(1)] = (m.group(2), fields)

    def resolve(name):
        parent, fields = raw[name]
        return fields | (resolve(parent) if parent else set())
    return {name: resolve(name) for name in raw}


def assert_fields(obj: dict, interface: str, ifaces: dict, open_ok=False):
    expected = ifaces[interface]
    if open_ok:
        assert expected <= set(obj), f"{interface}: missing {expected - set(obj)}"
    else:
        assert set(obj) == expected, (
            f"{interface}: extra {set(obj) - expected}, missing {expected - set(obj)}"
        )


# --------------------------------------------------------------------------- #
# Predictive-distribution helpers                                               #
# --------------------------------------------------------------------------- #
def _nb_pmf_bruteforce(mu, theta, K):
    k = np.arange(K + 1)
    logp = (math.lgamma(theta) * -1 + np.array([math.lgamma(x + theta) - math.lgamma(x + 1) for x in k])
            + theta * math.log(theta / (theta + mu)) + k * math.log(mu / (theta + mu)))
    return np.exp(logp)


@pytest.mark.parametrize("lik", ["nb", "zinb", "ztnb", "outage_mixture"])
def test_pmf_grid_is_normalised_for_every_likelihood(lik):
    mu = torch.tensor([[0.5, 4.0, 40.0]])
    theta = torch.tensor([[0.7, 2.0, 5.0]])
    pi = torch.tensor([[0.3, 0.1, 0.05]]) if lik == "zinb" else None
    pmf = pmf_grid(lik, mu, theta, pi, 2048)
    assert pmf.shape == (1, 3, 2049)
    assert torch.allclose(pmf.sum(-1), torch.ones(1, 3), atol=1e-4)


def test_quantiles_pit_and_crps_match_bruteforce():
    mu, theta, K = 6.0, 1.5, 400
    ref = _nb_pmf_bruteforce(mu, theta, K)
    pmf = pmf_grid("nb", torch.tensor([mu]), torch.tensor([theta]), None, K)
    # The last bin absorbs the residual tail mass.
    np.testing.assert_allclose(pmf[0, :K].numpy(), ref[:K], rtol=1e-3, atol=1e-7)

    cdf = np.cumsum(ref)
    q = quantiles_from_pmf(pmf, (0.1, 0.5, 0.9))[0].numpy()
    np.testing.assert_array_equal(q, [np.argmax(cdf >= lvl) for lvl in (0.1, 0.5, 0.9)])

    for y in (0, 3, 17):
        pit, crps = mid_pit_and_crps(pmf, torch.tensor([float(y)]))
        below = cdf[y - 1] if y > 0 else 0.0
        assert pit.item() == pytest.approx(below + 0.5 * ref[y], rel=1e-3)
        assert crps.item() == pytest.approx(((cdf - (np.arange(K + 1) >= y)) ** 2).sum(), rel=1e-3)


def test_reliability_bins_omit_empty_bins_and_count_everything():
    pred = np.array([0.05, 0.07, 0.95, 0.99, 0.5])
    event = np.array([0, 0, 1, 1, 1], dtype=float)
    bins = reliability_bins(pred, event)
    assert [b["bin_center"] for b in bins] == [0.05, 0.55, 0.95]
    assert sum(b["count"] for b in bins) == 5
    assert reliability_bins(np.zeros(0), np.zeros(0)) is None


# --------------------------------------------------------------------------- #
# Attention mapping                                                             #
# --------------------------------------------------------------------------- #
def test_last_step_attention_maps_rows_and_skips_dropped_and_self_loops():
    B, T, N = 2, 3, 3
    base = torch.tensor([[0, 1, 2], [1, 2, 0]])           # 0→1, 1→2, 2→0
    rows, alphas = [], []
    for b in range(B):
        for t in range(T):
            off = (b * T + t) * N
            # edge 2→0 dropped in batch 1; self-loops on every node
            edges = [(0, 1, 0.25), (1, 2, 0.6)] + ([(2, 0, 0.4)] if b == 0 else [])
            edges += [(n, n, 0.75 if n == 1 else (0.6 if n == 0 and b == 0 else 1.0 if n == 0 else 0.4))
                      for n in range(N)]
            for s, d, a in edges:
                rows.append((s + off, d + off))
                alphas.append([a, a])                        # two identical heads
    store = {"edge_index": torch.tensor(rows).T, "alpha": torch.tensor(alphas)}
    att, ent = last_step_attention(store, base, B, T, N)
    np.testing.assert_allclose(att[0], [0.25, 0.6, 0.4])
    assert att[1, 0] == pytest.approx(0.25) and att[1, 1] == pytest.approx(0.6) and np.isnan(att[1, 2])
    assert np.isfinite(ent).all()


def test_ego_graph_ranks_on_active_weights_and_keeps_nulls():
    coords = {0: (1.0, 1.0), 1: (2.0, 2.0), 2: (3.0, 3.0)}
    att = {(1, 0): [None, 0.9], (2, 0): [0.2, 0.3], (0, 2): [None, None], (0, 0): [1.0, 1.0]}
    ego = build_ego_graph_edges(0, att, coords, 2, top_k=1)
    assert [(e["source_id"], e["target_id"]) for e in ego] == [(1, 0), (0, 2)]
    assert ego[0]["weight_sequence"] == [None, 0.9]


# --------------------------------------------------------------------------- #
# Run selection                                                                 #
# --------------------------------------------------------------------------- #
def test_select_runs_reference_first_then_validation_rmse_without_oracles():
    plan = epp.select_runs(str(ROOT / epp.DEFAULT_RUNS_CSV), 10)
    runs = [p["run"] for p in plan]
    assert runs[0] == "v5_full-s42" and plan[0]["is_reference"]
    assert [p["rank"] for p in plan] == list(range(1, 11))
    assert not any(r.startswith(("naive_", "v4r_", "e3_3a_profile_transductive")) for r in runs)
    vals = [p["test_metrics"]["val_obs_all_rmse"] for p in plan[1:]]
    assert vals == sorted(vals)
    df = pd.read_csv(ROOT / epp.DEFAULT_RUNS_CSV)
    eligible = df[~df["experiment"].str.startswith(epp.EXCLUDED_PREFIXES)
                  & ~df["experiment"].isin(epp.EXCLUDED_EXPERIMENTS) & (df["run"] != "v5_full-s42")]
    assert vals[-1] <= eligible["val_obs_all_rmse"].nsmallest(9).max() + 1e-12


# --------------------------------------------------------------------------- #
# End-to-end payload contract on the synthetic city                             #
# --------------------------------------------------------------------------- #
def _small_v5_cfg(name):
    exp = next(e for e in gen.EXPERIMENTS if e["name"] == name)
    cfg = copy.deepcopy(gen.derive(exp["base"], exp["overrides"]))
    cfg["model"].update(num_nodes=NODES, embedding_dim=8, hidden_dim=16, num_layers=1, num_heads=2,
                        edge_features_cache_path=None)
    cfg["training"]["num_workers"] = 0
    cfg["data_source"] = "parquet"
    return resolve_config(cfg)


@pytest.fixture(scope="module", params=["v5_full", "e1_2_obs_impute"])
def exported(request, tmp_path_factory):
    """Run the exporter's inference + writers for one synthetic day."""
    cfg = _small_v5_cfg(request.param)
    ds = build_dataset(cfg, use_mock_data=True)
    tmp = tmp_path_factory.mktemp(request.param)
    torch.manual_seed(0)
    from harness.batch_prep import BatchPreparer
    covariate_dim = BatchPreparer(cfg, ds, torch.device("cpu"), inductive_mask=None, loss_node_mask=None,
                                  eval_holdout_mask=None, static_features=ds.static_features,
                                  location_ids=list(ds.locations)).covariates.dim
    model = build_model(cfg, ds, torch.device("cpu"), covariate_dim=covariate_dim, edge_features_cache_path=None)
    torch.save(model.state_dict(), tmp / "best_model.pt")
    ind = make_fixed_holdout_mask(NODES, cfg["data"]["inductive_holdout_frac"], seed=42)
    save_holdout_manifest(str(tmp), list(ds.locations), ind, 42, cfg["data"]["inductive_holdout_frac"])
    manifest = json.loads((tmp / "holdout_manifest.json").read_text())

    ctx = build_v5_context(f"{request.param}-s42", cfg, ds, str(tmp / "best_model.pt"), manifest,
                           torch.device("cpu"))
    day = pd.DatetimeIndex(ds.times).normalize()[-1].date()
    run_cfg = {"start_date": day.isoformat(), "end_date": (day + pd.Timedelta(days=1)).isoformat(),
               "num_timesteps_target": 96, "top_k_ego_edges": 3}
    plan = {"run": f"{request.param}-s42", "experiment": request.param, "group": "test", "description": "d",
            "rank": 1, "is_reference": request.param == "v5_full",
            "test_metrics": {k: None for k in epp.TEST_METRIC_KEYS}}
    art = epp.RunArtifact(plan["run"], "e/p/c:v0", str(tmp / "best_model.pt"), "", None, "abc", None)
    coords = {loc: {"lat": 55.0 + i * 0.001, "lon": -1.6} for i, loc in enumerate(ds.locations)}
    out = tmp / "runs"
    epp.generate_run(ctx, plan, art, manifest, run_cfg, str(out), coords)
    run_dir = out / plan["run"]
    return {
        "ctx": ctx, "cfg": cfg, "out": out, "day": day.isoformat(), "ind": ind,
        "spatial": json.loads((run_dir / "spatial.json").read_text()),
        "page": json.loads((run_dir / day.isoformat() / "page.json").read_text()),
        "node": json.loads((run_dir / day.isoformat() / "nodes" / "node_0.json").read_text()),
    }


def test_payload_matches_typescript_contract(exported):
    ifaces = ts_interfaces()
    page, node, spatial = exported["page"], exported["node"], exported["spatial"]
    assert_fields(page, "TemporalPage", ifaces)
    assert_fields(page["network"], "NetworkSeries", ifaces)
    assert_fields(page["day_metrics"], "WindowMetrics", ifaces)
    assert_fields(page["calibration"], "Calibration", ifaces)
    for series in page["node_series"].values():
        assert_fields(series, "NodeSeries", ifaces)
    for m in page["node_metrics"].values():
        assert_fields(m, "NodeDayMetrics", ifaces)
    for e in page["edge_attention"]:
        assert_fields(e, "EdgeAttention", ifaces)
    for e in page["alpha_node_leaderboard"]:
        assert_fields(e, "AlphaNodeEntry", ifaces)
    for t in page["tripwire_audit_trail"]:
        assert_fields(t, "TripwireEntry", ifaces, open_ok=True)

    assert_fields(node, "NodeDetailPayload", ifaces)
    assert_fields(node["distribution"], "DistributionSeries", ifaces)
    assert_fields(node["metrics"], "NodeDayMetrics", ifaces)
    for e in node["edges_1hop"]:
        assert_fields(e, "HopEdge", ifaces)
    for e in node["ego_graph_edges"]:
        assert_fields(e, "AttentionEdge", ifaces)

    assert set(spatial) == {"metadata", "nodes", "edge_topology"}
    assert_fields(spatial["metadata"], "SpatialMetadata", ifaces)
    assert_fields(spatial["metadata"]["capabilities"], "Capabilities", ifaces)
    assert_fields(spatial["metadata"]["registry"], "RegistryPlacement", ifaces)
    assert_fields(spatial["metadata"]["test_metrics"], "TestMetrics", ifaces)
    for n in spatial["nodes"]:
        assert_fields(n, "SpatialNode", ifaces)

    registry = build_registry(str(exported["out"]))
    assert_fields(registry, "RunRegistry", ifaces)
    assert_fields(registry["runs"][0], "RunRegistryEntry", ifaces)


def test_payload_values_are_capability_gated_and_consistent(exported):
    page, node, ctx = exported["page"], exported["node"], exported["ctx"]
    lik = exported["cfg"]["loss"]["forecast_likelihood"]
    W = page["timesteps"]
    assert W == len(page["timestamps"]) == 96 - 0 and page["schema_version"] == 5
    s = page["node_series"]["0"]
    assert all(len(v) == W for v in s.values() if v is not None)
    # lo ≤ hi everywhere; truth null exactly where the dataset has no observation
    assert all(lo <= hi for lo, hi in zip(s["lo"], s["hi"]))
    ds = ctx.dataset
    t_idx = [list(pd.DatetimeIndex(ds.times)).index(pd.Timestamp(ts)) for ts in page["timestamps"]]
    obs = ds.observed[t_idx, 0].numpy()
    assert [v is not None for v in s["truth"]] == list(obs)
    assert s["gap_steps"] == ds.run_length[t_idx, 0].tolist()

    caps = exported["spatial"]["metadata"]["capabilities"]
    assert caps["outage"] == (lik == "outage_mixture") and caps["zero_inflation"] == (lik == "zinb")
    assert (s["outage_q"] is not None) == caps["outage"]
    assert (node["distribution"]["pi"] is not None) == (lik == "zinb")
    assert (s["q90"] is not None) == caps["quantiles"]
    assert (page["calibration"]["outage_reliability"] is not None) == caps["outage"]
    assert page["day_metrics"]["n_targets"] == W * NODES

    # Holdout nodes are blanked for the whole look-back window.
    roles = {n["id"]: n["node_role"] for n in exported["spatial"]["nodes"]}
    assert [roles[i] == "holdout" for i in range(NODES)] == exported["ind"].tolist()
    for i in range(NODES):
        if roles[i] == "holdout":
            nd = json.loads((exported["out"] / ctx.run_name / exported["day"] / "nodes" / f"node_{i}.json").read_text())
            assert all(v == 1.0 for v in nd["input_missing_frac"])


def test_scaler_tripwire_rejects_a_mismatched_checkpoint(exported, tmp_path):
    ctx = exported["ctx"]
    state = ctx.model.state_dict()
    state["traffic_mean"] = state["traffic_mean"] + 0.5
    torch.save(state, tmp_path / "best_model.pt")
    manifest = json.loads(json.dumps({"num_nodes": NODES, "holdout_location_ids":
                                      [l for l, h in zip(ctx.location_ids, exported["ind"].tolist()) if h]}))
    with pytest.raises(AssertionError, match="traffic_mean"):
        build_v5_context("x", ctx.config, ctx.dataset, str(tmp_path / "best_model.pt"), manifest,
                         torch.device("cpu"))


def test_window_observation_counts_slots_and_finds_ghosts():
    class _DS:
        times = pd.date_range("2026-05-01", periods=6, freq="15min")
        observed = torch.tensor([
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],   # after the window: must not count
        ], dtype=torch.bool)

    counts, last = epp.window_observation(_DS, np.array([3, 4]))
    assert counts.tolist() == [0, 2, 0]                     # nodes 0 and 2 are ghosts in the window
    assert last == [_DS.times[2].isoformat(), _DS.times[4].isoformat(), None]   # 2 never reported


def test_spatial_nodes_carry_window_observation(exported):
    nodes = exported["spatial"]["nodes"]
    page = exported["page"]
    for n in nodes:
        truth = page["node_series"][str(n["id"])]["truth"]
        assert n["window_observed"] == sum(v is not None for v in truth)   # one-day window
        if n["window_observed"] > 0:
            assert n["last_observed"] is not None and n["last_observed"] <= page["timestamps"][-1]
    assert exported["spatial"]["metadata"]["num_ghost_nodes"] == sum(n["window_observed"] == 0 for n in nodes)
