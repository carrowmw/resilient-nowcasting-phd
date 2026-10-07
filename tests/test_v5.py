"""
Pipeline-v5 invariants: legacy resolution, loss hygiene, encoding, likelihood
arithmetic, adapters and the generated experiment suite.

Uses the SyntheticConnector city (train.build_dataset(..., use_mock_data=True)),
so no database is needed.
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

from harness.adapters import build_adapter  # noqa: E402
from harness.batch_prep import BatchPreparer  # noqa: E402
from harness.holdout import inductive_holdout_ids, make_fixed_holdout_mask, select_network_locations  # noqa: E402
from likelihoods import nb_log_p0, nb_logpmf, ztnb_logpmf  # noqa: E402
from losses import TrafficScale, masked_mean, mixture_terms  # noqa: E402
from model_factory import build_model  # noqa: E402
from run_config import LEGACY_V4_EQUIVALENTS, resolve_config  # noqa: E402
from train import build_dataset, custom_collate_real, load_config  # noqa: E402

import generate_v5_configs as gen  # noqa: E402

NODES = 8
WEIGHTS = {"w_impute": 1.0, "w_forecast": 1.0, "w_kriging": 0.5}


def small(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["model"]["num_nodes"] = NODES
    cfg["model"]["embedding_dim"] = 8
    cfg["model"]["hidden_dim"] = 16
    cfg["model"]["num_layers"] = 1
    cfg["model"]["num_heads"] = 2
    cfg["training"]["num_workers"] = 0
    cfg["data_source"] = "parquet"
    return resolve_config(cfg)


@pytest.fixture(scope="module")
def v4r_cfg():
    return small(gen.V4R)


@pytest.fixture(scope="module")
def dataset(v4r_cfg):
    return build_dataset(v4r_cfg, use_mock_data=True)


def prepared_batch(cfg, dataset, train=False, seed=0, so_length=None):
    """A device-prepared batch exactly as train.py builds it (CPU)."""
    device = torch.device("cpu")
    ind = make_fixed_holdout_mask(NODES, cfg["data"]["inductive_holdout_frac"], seed=42)
    preparer = BatchPreparer(
        cfg, dataset, device, inductive_mask=ind, loss_node_mask=~ind, eval_holdout_mask=ind,
        static_features=dataset.static_features, location_ids=list(dataset.locations),
    )
    raw = custom_collate_real([dataset[i] for i in range(200, 204)])
    raw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    gen_ = torch.Generator(device=device).manual_seed(seed)
    return preparer.prepare(raw, train=train, gen=gen_, so_length=so_length), preparer, ind


# --------------------------------------------------------------------------- #
# Config resolution and the generated suite                                     #
# --------------------------------------------------------------------------- #
def test_v4_replica_matches_legacy_resolution():
    """The suite's V4R base must be a v4 config in v5 form: every learning-relevant
    key equal to what resolve_config gives the original v4 YAML."""
    legacy = resolve_config(load_config(str(ROOT / "configs/train/v4/train_v4_full_inductive.yaml")))
    replica = resolve_config(gen.V4R)
    for section in ("data", "model", "loss", "selection", "evaluation"):
        for key, value in legacy[section].items():
            if section == "model" and key.endswith("cache_path"):
                continue  # infra.yaml, merged at load time
            assert replica[section][key] == value, f"{section}.{key}: {replica[section][key]!r} != {value!r}"
    for key in ("epochs", "batch_size", "lr", "grad_clip_max_norm", "early_stop_patience", "random_seed"):
        assert replica["training"][key] == legacy["training"][key]
    assert replica["training"]["grad_cosine_every"] > 0  # logging only


def test_generated_suite_is_complete_and_valid():
    jobs = gen.build_jobs()
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names))
    for j in jobs:
        cfg = resolve_config(j["cfg"])
        assert cfg["pipeline_version"] == 5
        assert cfg["data"]["inductive_holdout_seed"] == 42, "every run must share one holdout manifest"
        assert cfg["wandb"]["project"] == "resilient-iot-v5"
    groups = {j["group"] for j in jobs}
    assert {"thesis_v4", "p1_hygiene", "p2_masking", "p3_inputs", "p4_likelihood", "p5_magnitude",
            "p6_mtl", "p7_v5_suite", "loo"} <= groups
    # Thesis ch.7 requirements that need dedicated runs.
    for required in ("v4r_total_only", "v4r_night_zero", "v4r_identity_graph", "v4r_lstm_rich",
                     "v4r_no_masks_no_impute", "v4r_bs128_steps4x", "v4r_bs32_lr_div4"):
        assert sum(j["experiment"] == required for j in jobs) == 3, f"{required} needs 3 seeds"


def test_generated_files_match_generator():
    manifest = ROOT / "configs/train/v5/suite_manifest.yaml"
    if not manifest.exists():
        pytest.skip("configs not generated")
    with open(manifest) as f:
        on_disk = yaml.safe_load(f)["jobs"]
    assert [j["name"] for j in on_disk] == [j["name"] for j in gen.build_jobs()], \
        "configs/train/v5 is stale — re-run scripts/riotv5/generate_v5_configs.py"


def test_invalid_combinations_crash():
    cfg = copy.deepcopy(gen.V5F)
    cfg["data"]["ablate_masks"] = True  # unified encoding would re-raise the zeroed masks
    with pytest.raises(AssertionError):
        resolve_config(cfg)
    cfg = copy.deepcopy(gen.V5F)
    cfg["loss"]["target_mask"] = "all"  # mixture routes missing targets itself
    with pytest.raises(AssertionError):
        resolve_config(cfg)
    cfg = copy.deepcopy(gen.V5F)
    del cfg["loss"]["w_missing_lik"]
    with pytest.raises(KeyError):
        resolve_config(cfg)


# --------------------------------------------------------------------------- #
# Likelihood / loss arithmetic                                                  #
# --------------------------------------------------------------------------- #
def test_masked_mean_ignores_infinite_excluded_slots():
    x = torch.tensor([1.0, 3.0, float("inf")], requires_grad=True)
    m = torch.tensor([True, True, False])
    out = masked_mean(x, m)
    assert out.item() == pytest.approx(2.0)
    out.backward()
    assert torch.isfinite(x.grad).all()


def test_ztnb_is_finite_on_observed_and_normalised():
    mu, theta = torch.tensor(3.0), torch.tensor(2.0)
    ys = torch.arange(1, 400, dtype=torch.float32)
    total = torch.exp(ztnb_logpmf(ys, mu, theta)).sum()
    assert total.item() == pytest.approx(1.0, abs=1e-4)
    assert ztnb_logpmf(torch.tensor(0.0), mu, theta).item() == -math.inf


def test_mixture_hand_check():
    """Observed: -log(1-q) - log NB(y); missing: -log(q + (1-q) NB(0))."""
    y = torch.tensor([5.0, 0.0])
    mu, theta, q = torch.tensor([4.0, 4.0]), torch.tensor([2.0, 2.0]), torch.tensor([0.2, 0.2])
    obs = torch.tensor([True, False])
    cfg = {"w_artificial_outage": 0.0, "w_missing_lik": 1.0}
    elem, _ = mixture_terms(y, mu, theta, q, obs, torch.zeros(2, dtype=torch.bool), cfg)
    nb5 = math.exp(nb_logpmf(torch.tensor(5.0), torch.tensor(4.0), torch.tensor(2.0)).item())
    p0 = math.exp(nb_log_p0(torch.tensor(4.0), torch.tensor(2.0)).item())
    assert elem[0].item() == pytest.approx(-math.log(0.8) - math.log(nb5), rel=1e-4)
    assert elem[1].item() == pytest.approx(-math.log(0.2 + 0.8 * p0), rel=1e-4)


# --------------------------------------------------------------------------- #
# Batch preparation / hygiene                                                   #
# --------------------------------------------------------------------------- #
def test_synthetic_feed_never_reports_zero(dataset):
    """Like the real API: observed counts are >= 1, zeros only at missing slots."""
    sample = dataset[100]
    y, obs = sample["target_forecast"], sample["target_observed"]
    assert (y[obs] >= 1).all()


def test_legacy_v5_loss_equals_v4_tuple_loss(v4r_cfg, dataset):
    """With every v5 key at its legacy value the v5 loss path reproduces the v4
    pipeline loss on the same prepared batch."""
    torch.manual_seed(0)
    model = build_model(v4r_cfg, dataset, torch.device("cpu"), covariate_dim=0, edge_features_cache_path=None)
    model.eval()
    batch, _, ind = prepared_batch(v4r_cfg, dataset)
    with torch.no_grad():
        legacy, _ = build_adapter("gnn", model).step(batch, WEIGHTS, loss_node_mask=~ind)
        v5, _ = build_adapter("gnn", model, v4r_cfg).step(batch, WEIGHTS, loss_node_mask=~ind)
    for name in ("forecast", "impute", "kriging", "total"):
        assert getattr(v5, name).item() == pytest.approx(getattr(legacy, name).item(), rel=1e-4, abs=1e-5), name


def test_observed_only_loss_ignores_synthetic_zeros(dataset):
    cfg = small(gen.P2B)
    torch.manual_seed(0)
    model = build_model(cfg, dataset, torch.device("cpu"), covariate_dim=0, edge_features_cache_path=None)
    model.eval()
    batch, _, ind = prepared_batch(cfg, dataset)
    adapter = build_adapter("gnn", model, cfg)
    with torch.no_grad():
        a, _ = adapter.step(batch, WEIGHTS, loss_node_mask=~ind)
        poked = dict(batch)
        poked["target_forecast"] = torch.where(batch["target_observed"], batch["target_forecast"],
                                               torch.full_like(batch["target_forecast"], 999.0))
        b, _ = adapter.step(poked, WEIGHTS, loss_node_mask=~ind)
    assert a.forecast.item() == pytest.approx(b.forecast.item(), rel=1e-6)


def test_ablate_masks_never_changes_routing(v4r_cfg):
    cfg = copy.deepcopy(v4r_cfg)
    cfg["data"]["ablate_masks"] = True
    ablated = build_dataset(cfg, use_mock_data=True)
    plain = build_dataset(v4r_cfg, use_mock_data=True)
    a, b = ablated[300], plain[300]
    assert torch.equal(a["target_observed"], b["target_observed"])
    assert torch.equal(a["input_observed"], b["input_observed"])
    assert a["structural_masks"].abs().sum() == 0


def test_unified_encoding_marks_artificial_gaps(dataset):
    cfg = small(gen.P3B)
    batch, _, _ = prepared_batch(cfg, dataset, train=True)
    art = batch["art"]
    assert art.any()
    assert (batch["structural_masks"][..., 0][art] == 1).all(), "artificial gaps must raise mask channel 0"
    # Blanked stations are zeroed wholesale by apply_spatial_holdout (v4 semantics).
    visible = ~batch["blank_mask"].view(1, -1, 1)
    assert torch.isnan(batch["dynamic_features"][..., 0][art & visible]).all()
    real_gap = ~batch["input_observed"] & visible
    assert torch.isnan(batch["dynamic_features"][..., 0][real_gap]).all(), "real gaps: same NaN encoding"
    assert (batch["dynamic_features"][~visible.expand_as(art)] == 0).all(), "blanked stations fully zeroed"
    assert (art <= batch["input_observed"]).all(), "structured gaps only over observed traffic"


def test_synthetic_outage_only_hides_observed_slots(dataset):
    cfg = small(gen.V5F)
    batch, _, ind = prepared_batch(cfg, dataset, so_length=4)
    sel = batch["so_selected"]
    assert sel.any()
    assert (sel <= batch["target_observed"]).all(), "SO targets must have ground truth"
    assert not sel[:, ind].any(), "SO never selects held-out stations"


# --------------------------------------------------------------------------- #
# Adapters                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["v5_full", "v5_lstm", "naive_profile", "naive_last_obs", "e6_2_pcgrad"])
def test_v5_adapters_step_and_predict(name, dataset):
    exp = next(e for e in gen.EXPERIMENTS if e["name"] == name)
    cfg = small(gen.derive(exp["base"], exp["overrides"]))
    batch, preparer, ind = prepared_batch(cfg, dataset, train=True)
    torch.manual_seed(0)
    model = build_model(cfg, dataset, torch.device("cpu"), covariate_dim=preparer.covariates.dim,
                        edge_features_cache_path=None)
    adapter = build_adapter(cfg["model_type"], model, cfg, scale=traffic_scale(dataset, cfg),
                            covariates=preparer.covariates)
    losses, preds = adapter.step(batch, WEIGHTS, loss_node_mask=~ind)
    for field in ("total", "impute", "forecast", "kriging"):
        assert torch.is_tensor(getattr(losses, field)) and torch.isfinite(getattr(losses, field))
    assert preds.point.shape == batch["target_forecast"].shape
    assert (preds.point >= 0).all()
    if losses.total.requires_grad:
        losses.total.backward()


def traffic_scale(dataset, cfg):
    return TrafficScale(torch.tensor(dataset.traffic_mean), torch.tensor(dataset.traffic_std),
                        cfg["data"]["log1p_inputs"])


# --------------------------------------------------------------------------- #
# Sensor-density subsampling (data.train_node_count)                            #
# --------------------------------------------------------------------------- #
def test_network_subsets_are_nested_and_keep_the_holdout():
    locs = [f"loc{i:03d}" for i in range(81)]
    held = inductive_holdout_ids(locs, 0.2, 42)
    prev = set()
    for n in (15, 25, 35, 45, 57):
        keep = select_network_locations(locs, 0.2, 42, n, 0)
        assert set(held) <= set(keep), "the inductive holdout never changes with density"
        train = set(keep) - set(held)
        assert len(train) == n and prev <= train, "subsets must be nested"
        prev = train
    assert len(held) + 57 == 81
    with pytest.raises(AssertionError):
        select_network_locations(locs, 0.2, 42, 58, 0)


def test_late_v5_keys_backfill_and_validate():
    cfg = copy.deepcopy(gen.V5F)
    del cfg["data"]["train_node_count"], cfg["data"]["train_node_seed"]
    resolved = resolve_config(cfg)  # a config saved by the original 209-run suite
    assert resolved["data"]["train_node_count"] is None
    cfg = copy.deepcopy(gen.V5F)
    cfg["eval_mode"], cfg["data"]["train_node_count"] = "temporal", 20
    with pytest.raises(AssertionError):
        resolve_config(cfg)


def test_station_filter_keeps_time_axis_and_is_identity_at_full_count(v4r_cfg, dataset):
    cfg = copy.deepcopy(v4r_cfg)
    pool = NODES - len(inductive_holdout_ids(dataset.all_locations, 0.2, 42))
    cfg["data"]["train_node_count"] = pool
    full = build_dataset(cfg, use_mock_data=True)
    assert torch.equal(full.dynamic_data, dataset.dynamic_data)
    assert torch.equal(full.profile_log1p, dataset.profile_log1p)
    cfg["data"]["train_node_count"] = pool - 2
    sub = build_dataset(cfg, use_mock_data=True)
    assert sub.num_nodes == NODES - 2 and sub.all_locations == dataset.all_locations
    assert sub.times == dataset.times, "dropping stations must not shift the chronological splits"
    cols = [dataset.locations.index(loc) for loc in sub.locations]
    assert torch.equal(sub.observed, dataset.observed[:, cols])


def test_scaling_sweep_replicates_the_v5_runs_except_for_density():
    import generate_scaling_configs as scaling

    jobs = scaling.build_jobs(scaling.N_TRAIN, [42], [0])
    assert len(jobs) == 2 * len(scaling.N_TRAIN)
    for j in jobs:
        source = scaling.experiment_cfg(j["source_experiment"])
        cfg = resolve_config(copy.deepcopy(j["cfg"]))
        assert cfg["data"]["train_node_count"] == j["n_train"]
        cfg["data"]["train_node_count"], cfg["data"]["train_node_seed"] = None, 0
        cfg["wandb"] = source["wandb"]
        assert cfg == resolve_config(copy.deepcopy(source)), f"{j['name']} drifted from {j['source_experiment']}"
