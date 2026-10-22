"""
v7-stage invariants (docs/experiments/riot_v6/riot_v6_seed_report.md §9.2): stratified
validation stations, origin-only weather extremes, the rec. 6 stop list, the rec. 2
gate simulation and the generated v7 stage.

Uses the SyntheticConnector city (no database).
"""
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "riotv5"))
sys.path.insert(0, str(ROOT / "tests"))

from harness.holdout import inductive_holdout_ids, stratified_validation_station_ids  # noqa: E402
from run_config import resolve_config  # noqa: E402
from train import build_node_masks, chronological_purged_split, split_bounds, station_window_stats  # noqa: E402

import generate_v5_configs as gen  # noqa: E402
import generate_v7_configs as gen7  # noqa: E402
import posthoc_v7_eval as posthoc7  # noqa: E402
import run_v5_suite  # noqa: E402
from test_v6 import CPU, prepare, small  # noqa: E402
from test_v6 import dataset, p2b_cfg  # noqa: E402,F401  (fixtures)


# --------------------------------------------------------------------------- #
# Config keys                                                                  #
# --------------------------------------------------------------------------- #
def test_v6_configs_backfill_v7_keys_to_legacy():
    raw = yaml.safe_load(open(ROOT / "configs/train/v6/g3_v6/v6_base__s42.yaml"))
    assert "val_station_strategy" not in raw["data"]
    cfg = resolve_config(raw)
    assert cfg["data"]["val_station_strategy"] == "random"
    assert cfg["data"]["weather_extreme_step"] == "target"


@pytest.mark.parametrize("data", [
    {"val_station_strategy": "stratified"},                      # no validation stations
    {"weather_extreme_step": "origin"},                          # no weather extremes
    {"val_station_count": 3, "val_station_min_coverage": 1.5},   # coverage outside [0, 1]
    {"val_station_strategy": "quantile"},                        # unknown enum
])
def test_invalid_v7_combinations_crash(data):
    cfg = copy.deepcopy(gen.P2B)
    cfg["data"].update(data)
    with pytest.raises(AssertionError):
        resolve_config(cfg)


def test_split_bounds_match_the_split(dataset):
    seq = dataset.seq_len
    tr, va, te = chronological_purged_split(dataset, 0.7, 0.15, seq + 1)
    b = split_bounds(len(dataset), 0.7, 0.15, seq + 1)
    assert (len(tr), va.indices[0], va.indices[-1] + 1, te.indices[0], te.indices[-1] + 1) == b


# --------------------------------------------------------------------------- #
# Stratified validation stations                                               #
# --------------------------------------------------------------------------- #
LOCS = [f"s{i:02d}" for i in range(81)]
HELD = inductive_holdout_ids(LOCS, 0.2, 42)
LEVEL = {loc: float(i) for i, loc in enumerate(LOCS)}
COVERAGE = {loc: (0.0 if i % 4 == 0 else 0.9) for i, loc in enumerate(LOCS)}


def test_stratified_stations_cover_every_level_stratum():
    ids = stratified_validation_station_ids(LOCS, HELD, 8, 0, LEVEL, COVERAGE, 0.25)
    assert len(ids) == 8 and not set(ids) & set(HELD)
    assert all(COVERAGE[s] >= 0.25 for s in ids), "uncovered station drawn"
    pool = sorted((s for s in LOCS if s not in HELD and COVERAGE[s] >= 0.25), key=LEVEL.get)
    strata = np.array_split(np.array(pool), 8)
    assert sorted(sum(s in stratum for s in ids) for stratum in strata) == [1] * 8


def test_stratified_stations_are_seed_fixed():
    a = stratified_validation_station_ids(LOCS, HELD, 8, 0, LEVEL, COVERAGE, 0.25)
    assert a == stratified_validation_station_ids(LOCS, HELD, 8, 0, LEVEL, COVERAGE, 0.25)
    assert a != stratified_validation_station_ids(LOCS, HELD, 8, 1, LEVEL, COVERAGE, 0.25)
    with pytest.raises(AssertionError):
        stratified_validation_station_ids(LOCS, HELD, 8, 0, LEVEL, COVERAGE, 0.95)   # empty pool


def test_station_window_stats_use_the_validation_window(p2b_cfg, dataset):
    level, cov = station_window_stats(p2b_cfg, dataset)
    seq = p2b_cfg["data"]["seq_len"]
    _, vs, ve, _, _ = split_bounds(len(dataset), p2b_cfg["train_split"], p2b_cfg["val_split"], seq + 1)
    expect = dataset.observed[vs + seq:ve + seq].float().mean(0)
    loc = dataset.locations[0]
    assert cov[loc] == pytest.approx(float(expect[0]))
    assert level[loc] == pytest.approx(float(dataset.node_train_mean[0]))


def test_stratified_node_masks(dataset):
    cfg = small(gen.P2B, val_station_count=2, val_station_strategy="stratified", val_station_min_coverage=0.0)
    locs = list(dataset.locations)
    m = build_node_masks(cfg, locs, dataset.all_locations, CPU, dataset)
    assert int(m.val_stations.sum()) == 2 and not bool((m.val_stations & m.test_holdout).any())
    assert torch.equal(m.unseen, m.val_stations | m.test_holdout)
    with pytest.raises(AssertionError):
        build_node_masks(cfg, locs, dataset.all_locations, CPU)          # needs the dataset


# --------------------------------------------------------------------------- #
# Origin-only weather extremes                                                 #
# --------------------------------------------------------------------------- #
def test_origin_weather_repeats_the_last_input_step(dataset):
    target = small(gen.P2B, weather_extreme_features=True)
    origin = small(gen.P2B, weather_extreme_features=True, weather_extreme_step="origin")
    masks = build_node_masks(target, list(dataset.locations), dataset.all_locations, CPU)
    bt, _ = prepare(target, dataset, masks, idx=list(range(100, 400, 10)))
    bo, _ = prepare(origin, dataset, masks, idx=list(range(100, 400, 10)))
    wt, wo = bt["covariates"][..., -4:], bo["covariates"][..., -4:]
    torch.testing.assert_close(wo[:, :, :-1], wt[:, :, :-1])
    torch.testing.assert_close(wo[:, :, -1], wt[:, :, -2])
    assert not torch.equal(wo[:, :, -1], wt[:, :, -1]), "test windows never change weather state"
    torch.testing.assert_close(bo["covariates"][..., :-4], bt["covariates"][..., :-4])


# --------------------------------------------------------------------------- #
# Generated stage                                                              #
# --------------------------------------------------------------------------- #
def test_v7_stage_jobs_are_valid():
    jobs = gen7.build_jobs()
    assert len({j["name"] for j in jobs}) == len(jobs) == 60
    assert sorted({j["tier"] for j in jobs}) == [1, 2, 3, 4, 5]
    for j in jobs:
        cfg = resolve_config(j["cfg"])
        d, t = cfg["data"], cfg["training"]
        assert cfg["wandb"]["project"] == "resilient-iot-v7" and d["inductive_holdout_seed"] == 42
        assert t["lr_schedule"] == "cosine"
        if j["tier"] == gen7.GATED_TIER:
            assert d["val_station_count"] == 8 and d["val_station_strategy"] == "stratified"
            assert j["seed"] in gen7.SEEDS_GATED
        else:
            assert d["val_station_count"] == 0 and j["seed"] in gen7.SEEDS
    exp = {j["experiment"]: resolve_config(j["cfg"]) for j in jobs}
    # Every rec. 1/3/4/5 arm differs from its control in the intended keys only.
    base = exp["v7_base"]

    def diff(a, b):
        return {f"{s}.{k}" for s in ("data", "model", "loss", "training") for k in a[s] if a[s][k] != b[s][k]} \
            - {"training.random_seed"}
    assert diff(exp["v7_profile_static"], base) == {"data.profile_prior"}
    assert diff(exp["v7_block"], base) == {"data.mask_strategy"}
    assert diff(exp["v7_static_block"], base) == {"data.profile_prior", "data.mask_strategy"}
    assert diff(exp["v7_weather_target"], base) == {"data.weather_extreme_features"}
    assert diff(exp["v7_weather_origin"], exp["v7_weather_target"]) == {"data.weather_extreme_step"}
    assert diff(exp["v7_base_e150"], base) == {"training.epochs", "training.min_epochs",
                                               "training.early_stop_patience"}


@pytest.mark.parametrize("override", [
    {"model_type": "lstm"},
    {"data.seq_len": 48},
    {"model.gat_drop_missing_sources": True},
    {"loss.w_artificial_outage": 1.0},
    {"loss.w_kriging": 0.0},
    {"loss.kriging_observed_only": False},
])
def test_rec6_stop_list_rejects_stopped_arms(override):
    cfg = gen.derive(gen7.V7, override) if "." in next(iter(override)) else {**gen7.V7, **override}
    assert gen7.stop_list_violations(cfg)
    assert not gen7.stop_list_violations(gen7.V7)


def test_smoke_config_keeps_v7_jobs_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(run_v5_suite, "SMOKE_DIR", tmp_path)
    for j in gen7.build_jobs()[::6]:
        path = tmp_path / "job.yaml"
        path.write_text(yaml.safe_dump(j["cfg"]))
        cfg = yaml.safe_load(open(run_v5_suite.smoke_config({"name": j["name"], "config": str(path)})))
        resolved = resolve_config(cfg)
        assert resolved["training"]["min_epochs"] <= resolved["training"]["epochs"] == 2
        assert math.isfinite(resolved["training"]["lr"])


# --------------------------------------------------------------------------- #
# Rec. 2 gate simulation                                                       #
# --------------------------------------------------------------------------- #
def _write_valcheck(tmp_path, n_runs=30, n_st=24, sign=1.0, seed=0):
    rng = np.random.RandomState(seed)
    (tmp_path / "valcheck").mkdir()
    quality = rng.rand(n_runs)                         # one latent model quality per run
    for r in range(n_runs):
        stations = {}
        for s in range(n_st):
            lvl = 1.0 + s
            val_rmse = lvl * (1.0 + sign * quality[r]) + 0.05 * rng.randn()
            test_rmse = lvl * (1.0 + quality[r]) + 0.05 * rng.randn()
            stations[f"h{s:02d}"] = {"level": lvl, "val_coverage": 0.0 if s % 3 == 0 else 0.8,
                                     "test_coverage": 0.8, "val_se": val_rmse ** 2 * 100, "val_n": 100.0,
                                     "test_se": test_rmse ** 2 * 100, "test_n": 100.0}
        json.dump({"run": f"v6_x{r}-s42", "experiment": f"v6_x{r}", "stations": stations},
                  open(tmp_path / "valcheck" / f"v6_x{r}-s42.json", "w"))


def _gate_args(tmp_path, **kw):
    base = dict(out_dir=str(tmp_path), out_name="gate.json", count=8, min_coverage=0.25, threshold=0.3,
                min_positive=0.8, draws=100, seed=0, min_runs=20, min_count=4)
    return SimpleNamespace(**{**base, **kw})


def test_valgate_passes_when_validation_tracks_heldout(tmp_path):
    _write_valcheck(tmp_path, sign=1.0)
    assert posthoc7.cmd_valgate(_gate_args(tmp_path)) == 0
    out = json.load(open(tmp_path / "gate.json"))
    assert out["n_covered"] == 16 and out["rules"]["all/stratified8"]["rho_median"] > 0.9


def test_valgate_fails_when_validation_is_anticorrelated(tmp_path):
    _write_valcheck(tmp_path, sign=-0.9)
    assert posthoc7.cmd_valgate(_gate_args(tmp_path)) == 1


def test_valgate_undecided_without_enough_runs(tmp_path):
    _write_valcheck(tmp_path, n_runs=5)
    assert posthoc7.cmd_valgate(_gate_args(tmp_path)) == 2


def test_valcheck_shards_keep_data_groups_together():
    items = [(f"r{i}", f"k{i % 3}") for i in range(10)] + [("lone", "k9")]
    shards = posthoc7.shard_runs(items, 3)
    assert sorted(r for s in shards for r, _ in s) == sorted(r for r, _ in items)
    for key in {k for _, k in items}:
        assert sum(any(k == key for _, k in s) for s in shards) == 1
