"""
v6-stage invariants (docs/experiments/riot_v5/riot_v5_seed_report.md §14.2): training
protocol keys, inductive validation stations, the evaluator's ignore mask, weather-extreme
covariates, the residual-to-profile head, the NB ensemble and the generated v6 stage.

Uses the SyntheticConnector city (no database).
"""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "riotv5"))

from harness.batch_prep import BatchPreparer  # noqa: E402
from harness.contract import Capabilities, Predictions  # noqa: E402
from harness.evaluate_v5 import V5Evaluator  # noqa: E402
from harness.holdout import inductive_holdout_ids, validation_station_ids  # noqa: E402
from losses import TrafficScale  # noqa: E402
from model_factory import build_model  # noqa: E402
from run_config import resolve_config  # noqa: E402
from train import build_dataset, build_node_masks, cosine_lr_factor, custom_collate_real  # noqa: E402

import generate_v5_configs as gen  # noqa: E402
import generate_v6_configs as gen6  # noqa: E402
import posthoc_v6_eval as posthoc  # noqa: E402
import run_v5_suite  # noqa: E402

NODES = 10
CPU = torch.device("cpu")


def small(cfg: dict, **data) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(num_nodes=NODES, embedding_dim=8, hidden_dim=16, num_layers=1, num_heads=2)
    cfg["training"]["num_workers"] = 0
    cfg["data_source"] = "parquet"
    cfg["data"].update(data)
    return resolve_config(cfg)


@pytest.fixture(scope="module")
def p2b_cfg():
    return small(gen.P2B, val_station_count=2)


@pytest.fixture(scope="module")
def dataset(p2b_cfg):
    return build_dataset(p2b_cfg, use_mock_data=True)


def prepare(cfg, dataset, masks, idx=(200, 204), seed=0):
    preparer = BatchPreparer(
        cfg, dataset, CPU, inductive_mask=masks.unseen, loss_node_mask=~masks.unseen, eval_holdout_mask=masks.unseen,
        static_features=dataset.static_features, location_ids=list(dataset.locations),
    )
    rows = range(*idx) if isinstance(idx, tuple) else idx
    raw = custom_collate_real([dataset[i] for i in rows])
    gen_ = torch.Generator(device=CPU).manual_seed(seed)
    return preparer.prepare(raw, train=False, gen=gen_), preparer


# --------------------------------------------------------------------------- #
# Config keys                                                                  #
# --------------------------------------------------------------------------- #
def test_v5_suite_configs_backfill_v6_keys_to_legacy():
    raw = yaml.safe_load(open(ROOT / "configs/train/v5/p7_v5_suite/v5_full__s42.yaml"))
    assert "lr_schedule" not in raw["training"]
    cfg = resolve_config(raw)
    assert cfg["training"]["lr_schedule"] == "none" and cfg["training"]["min_epochs"] == 0
    assert cfg["data"]["val_station_count"] == 0 and not cfg["data"]["weather_extreme_features"]
    assert not cfg["model"]["residual_to_profile"]


@pytest.mark.parametrize("section,key,value", [
    ("training", "min_epochs", 500),                       # beyond the budget
    ("training", "warmup_epochs", 3),                      # warmup without a schedule
    ("model", "residual_to_profile", True),                # no profile to anchor on (P2B)
])
def test_invalid_v6_combinations_crash(section, key, value):
    cfg = copy.deepcopy(gen.P2B)
    cfg[section][key] = value
    with pytest.raises(AssertionError):
        resolve_config(cfg)


def test_weather_extremes_need_complete_weather():
    cfg = copy.deepcopy(gen.V4R)
    cfg["data"]["weather_extreme_features"] = True
    with pytest.raises(AssertionError):
        resolve_config(cfg)


def test_cosine_schedule_warms_up_then_decays_to_floor():
    f = [cosine_lr_factor(e, epochs=100, warmup=3, floor=0.02) for e in range(100)]
    assert f[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert all(a >= b for a, b in zip(f[3:], f[4:])), "cosine phase must not increase"
    assert f[-1] == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Validation stations                                                          #
# --------------------------------------------------------------------------- #
def test_validation_stations_are_disjoint_and_seed_fixed():
    locs = [f"s{i:02d}" for i in range(81)]
    held = inductive_holdout_ids(locs, 0.2, 42)
    a = validation_station_ids(locs, held, 5, 0)
    assert len(a) == 5 and not set(a) & set(held)
    assert a == validation_station_ids(locs, held, 5, 0)
    assert a != validation_station_ids(locs, held, 5, 1)
    with pytest.raises(AssertionError):
        validation_station_ids(locs, held, 81 - len(held), 0)


def test_node_masks_partition_the_unseen_set(p2b_cfg, dataset):
    m = build_node_masks(p2b_cfg, list(dataset.locations), dataset.all_locations, CPU)
    assert int(m.val_stations.sum()) == 2
    assert not bool((m.val_stations & m.test_holdout).any())
    assert torch.equal(m.unseen, m.val_stations | m.test_holdout)


def test_evaluator_ignore_mask_removes_nodes_from_every_metric(p2b_cfg, dataset):
    masks = build_node_masks(p2b_cfg, list(dataset.locations), dataset.all_locations, CPU)
    # Windows whose target is observed at the validation stations (the synthetic city has outages).
    seq = p2b_cfg["data"]["seq_len"]
    rows = [i for i in range(len(dataset) - 1) if bool(dataset.observed[i + seq, masks.val_stations].all())][:8]
    assert rows, "synthetic city has no window with observed validation-station targets"
    batch, preparer = prepare(p2b_cfg, dataset, masks, idx=rows)
    scale = TrafficScale(torch.tensor(dataset.traffic_mean), torch.tensor(dataset.traffic_std), False)
    y = batch["target_forecast"].float()
    # Error 1 on scored nodes, 1000 on ignored (validation) stations.
    point = y + 1.0 + 999.0 * masks.val_stations.view(1, -1).float()
    preds = Predictions(mu=point, point=point)
    ev = V5Evaluator(dataset, preparer.covariates, Capabilities(), p2b_cfg, masks.test_holdout, scale, CPU,
                     ignore_mask=masks.val_stations)
    ev.update(preds, batch)
    out = ev.summary()
    assert out["obs_all_rmse"] == pytest.approx(1.0)
    assert out["obs_trans_rmse"] == pytest.approx(1.0)
    assert out.get("peak_rmse") is None or out["peak_rmse"] == pytest.approx(1.0)
    # Without the ignore mask the validation stations would count as transductive.
    ev2 = V5Evaluator(dataset, preparer.covariates, Capabilities(), p2b_cfg, masks.test_holdout, scale, CPU)
    ev2.update(preds, batch)
    assert ev2.summary()["obs_trans_rmse"] > 10.0


# --------------------------------------------------------------------------- #
# New inputs / heads                                                           #
# --------------------------------------------------------------------------- #
def test_weather_extreme_channels(dataset):
    base = small(gen.P2B, calendar_features=True)
    ext = small(gen.P2B, calendar_features=True, weather_extreme_features=True)
    masks = build_node_masks(ext, list(dataset.locations), dataset.all_locations, CPU)
    b0, p0 = prepare(base, dataset, masks)
    b1, p1 = prepare(ext, dataset, masks)
    assert p1.covariates.dim == p0.covariates.dim + 4
    extra = b1["covariates"][..., -4:]
    assert set(extra.unique().tolist()) <= {0.0, 1.0}
    assert float(extra[:, masks.unseen].abs().sum()) == 0.0, "blanked nodes must not see weather"
    torch.testing.assert_close(b1["covariates"][..., :-4], b0["covariates"])


def test_residual_head_starts_at_the_profile(dataset):
    cfg = small(gen6.V6, profile_prior="inductive_neighbour", lag_features=[], val_station_count=0)
    cfg["model"]["residual_to_profile"] = True
    cfg = resolve_config(cfg)
    masks = build_node_masks(cfg, list(dataset.locations), dataset.all_locations, CPU)
    batch, preparer = prepare(cfg, dataset, masks)
    assert batch["profile_target_raw"] is not None
    model = build_model(cfg, dataset, CPU, covariate_dim=preparer.covariates.dim, edge_features_cache_path=None)
    last = [m for m in model.decoders.forecast_layer.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight[0].zero_()
        last.bias[0] = 0.0
    model.eval()
    out = model(batch, return_outputs=True)
    torch.testing.assert_close(out["mu"], batch["profile_target_raw"] + 1.0, atol=1e-4, rtol=1e-4)


def test_nb_moment_match():
    mu1, th1 = torch.tensor([[10.0, 50.0]]), torch.tensor([[2.0, 5.0]])
    mu2, th2 = torch.tensor([[30.0, 50.0]]), torch.tensor([[4.0, 5.0]])

    def p(mu, th):
        return Predictions(mu=mu, theta=th, point=mu,
                           nll_params={"likelihood": "nb", "mu": mu, "theta": th, "pi": None, "q": None})

    same = posthoc.nb_moment_match([p(mu1, th1), p(mu1, th1)], [1, 1])
    torch.testing.assert_close(same.mu, mu1)
    torch.testing.assert_close(same.theta, th1)
    ens = posthoc.nb_moment_match([p(mu1, th1), p(mu2, th2)], [1, 1])
    m = (mu1 + mu2) / 2
    var = 0.5 * (mu1 + mu1 ** 2 / th1 + mu1 ** 2) + 0.5 * (mu2 + mu2 ** 2 / th2 + mu2 ** 2) - m ** 2
    torch.testing.assert_close(ens.mu, m)
    torch.testing.assert_close(ens.mu + ens.mu ** 2 / ens.theta, var)


# --------------------------------------------------------------------------- #
# Generated stage                                                              #
# --------------------------------------------------------------------------- #
def test_v6_stage_jobs_are_valid_and_protocol_complete():
    jobs = gen6.build_jobs()
    assert len({j["name"] for j in jobs}) == len(jobs) == 78
    assert sorted({j["tier"] for j in jobs}) == [1, 2, 3, 4, 5, 6]
    for j in jobs:
        cfg = resolve_config(j["cfg"])
        t = cfg["training"]
        assert t["lr_schedule"] == "cosine" and t["min_epochs"] == 40 and t["early_stop_patience"] == 20
        assert cfg["wandb"]["project"] == "resilient-iot-v6"
        if j["experiment"] != "v6p_v5f_sched":
            assert cfg["data"]["val_station_count"] == 5 and cfg["data"]["val_station_seed"] == 0
        assert cfg["data"]["inductive_holdout_seed"] == 42


def test_ensemble_partners_share_the_evaluation_batch():
    jobs = {j["name"]: j["cfg"] for j in gen6.build_jobs()}
    for seed in gen6.SEEDS:
        a, b = jobs[f"v6_base-s{seed}"], jobs[f"v6_lstm-s{seed}"]
        diff = {k for k in a["data"] if a["data"][k] != b["data"][k]}
        assert diff <= posthoc.ENSEMBLE_DATA_KEYS_FREE, diff
        assert a["loss"]["forecast_likelihood"] == "nb" and b["model"]["baseline_head"] == "nb"


def test_smoke_config_keeps_v6_jobs_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(run_v5_suite, "SMOKE_DIR", tmp_path)
    for j in gen6.build_jobs()[::3]:
        path = tmp_path / "job.yaml"
        path.write_text(yaml.safe_dump(j["cfg"]))
        cfg = yaml.safe_load(open(run_v5_suite.smoke_config({"name": j["name"], "config": str(path)})))
        resolved = resolve_config(cfg)
        assert resolved["training"]["min_epochs"] <= resolved["training"]["epochs"] == 2
        assert math.isfinite(resolved["training"]["lr"])
