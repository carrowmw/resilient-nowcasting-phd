"""
Harness contract tests.

Covers the generic evaluation harness (src/harness/):
  - Adapter contract: both model families return Predictions/Losses with
    device tensors (never python floats), via one shared fixture.
  - Capability gating: null (not 0.0) metrics for incapable models; tripwires
    fire only for claimed capabilities.
  - Inductive zero-shot guarantee: holdout-node targets produce NO loss signal.
  - Holdout manifest round-trip + sha stability (cross-model comparability key).
"""
import json

import pytest
import torch

from baselines import LSTMForecaster, GRUForecaster
from decoders import compute_loss
from harness import (
    MetricAccumulator,
    evaluate_batch,
    load_holdout_manifest,
    make_fixed_holdout_mask,
    manifest_sha,
    mask_from_manifest,
    save_holdout_manifest,
)
from harness.adapters import RIoTAdapter, RecurrentBaselineAdapter, build_adapter
from harness.contract import Capabilities, Losses, Predictions
from harness.evaluate import validate_predictions
from pipeline import ResilientIoTPipeline

B, N, T, D, S = 2, 10, 6, 3, 5
WEIGHTS = {"w_impute": 1.0, "w_forecast": 1.0, "w_kriging": 0.5}


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    dyn = torch.randn(B, N, T, D, generator=g)
    return {
        "static_features": torch.randn(N, S, generator=g),
        "dynamic_features": dyn,
        "target_impute": dyn.clone(),
        "target_forecast": torch.randint(0, 30, (B, N), generator=g).float(),
        "node_mask": torch.rand(B, N, T, D, generator=g) < 0.15,
        "location_ids": [f"loc_{i}" for i in range(N)],
    }


@pytest.fixture(scope="module")
def gnn_adapter():
    torch.manual_seed(0)
    model = ResilientIoTPipeline(
        num_nodes=N, static_feat_dim=S, time_series_dim=D, seq_len=T,
        embedding_dim=8, hidden_dim=16, sparsity_threshold=0.5,
        num_layers=1, num_heads=2,
    )
    return build_adapter("gnn", model)


@pytest.fixture(scope="module")
def lstm_adapter():
    torch.manual_seed(0)
    return build_adapter("lstm", LSTMForecaster(time_series_dim=D, hidden_dim=16))


# --------------------------------------------------------------------------- #
# Adapter contract — shared across model families                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("adapter_fixture", ["gnn_adapter", "lstm_adapter"])
def test_adapter_contract(adapter_fixture, request):
    adapter = request.getfixturevalue(adapter_fixture)
    batch = make_batch()
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[1, 4]] = True
    batch["spatial_holdout_mask"] = mask

    losses, preds = adapter.step(batch, WEIGHTS)

    assert isinstance(losses, Losses) and isinstance(preds, Predictions)
    # Every loss is a scalar tensor — .item() and .backward() must both work.
    for component in losses:
        assert torch.is_tensor(component), f"loss component is {type(component)}, not a tensor"
        component.item()
    losses.total.backward()

    assert preds.mu.shape == (B, N)
    if adapter.capabilities.imputation:
        assert preds.imputed.shape == (B, N, T, D)
    if adapter.capabilities.kriging:
        assert preds.kriging_pred.shape == (B, 2, T, D)
    validate_predictions(adapter.capabilities, preds)


def test_gru_adapter_also_satisfies_contract():
    adapter = RecurrentBaselineAdapter(GRUForecaster(time_series_dim=D, hidden_dim=16))
    losses, preds = adapter.step(make_batch(), WEIGHTS)
    losses.total.item()
    assert preds.theta is None and preds.pi is None


# --------------------------------------------------------------------------- #
# Capability gating                                                            #
# --------------------------------------------------------------------------- #
def test_baseline_metrics_are_null_not_zero(lstm_adapter):
    batch = make_batch()
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[2, 7]] = True
    batch["spatial_holdout_mask"] = mask

    with torch.no_grad():
        _, preds = lstm_adapter.step(batch, WEIGHTS)
    m = evaluate_batch(lstm_adapter.capabilities, preds, batch, holdout_mask=mask)

    # No zero-inflation head -> null, never a fabricated 0%.
    assert m["zero_true_count"] is None and m["zero_correct_count"] is None
    # Genuine (naive) attempts -> real numbers.
    assert m["rmse"] is not None and m["impute_mse"] is not None
    assert m["kriging_rmse"] is not None and m["kriging_rmse"] > 0.0


def test_gnn_metrics_are_complete(gnn_adapter):
    batch = make_batch()
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[2, 7]] = True
    batch["spatial_holdout_mask"] = mask

    with torch.no_grad():
        _, preds = gnn_adapter.step(batch, WEIGHTS)
    m = evaluate_batch(gnn_adapter.capabilities, preds, batch, holdout_mask=mask)
    assert all(v is not None for v in m.values()), f"unexpected null in GNN metrics: {m}"


def test_tripwires_fire_only_for_claimed_capabilities():
    mu = torch.rand(B, N) + 0.1
    # Baseline predictions (no pi/theta) pass under baseline capabilities...
    validate_predictions(
        RecurrentBaselineAdapter.capabilities,
        Predictions(mu=mu, imputed=torch.randn(B, N, T, D)),
    )
    # ...but a dummy pi=0 under full ZINB capabilities is exactly the fabricated
    # value the harness must reject.
    with pytest.raises(AssertionError, match="pi"):
        validate_predictions(
            RIoTAdapter.capabilities,
            Predictions(mu=mu, theta=torch.ones(B, N), pi=torch.zeros(B, N),
                        imputed=torch.randn(B, N, T, D)),
        )
    # Supplying pi without the capability is also a contract violation.
    with pytest.raises(AssertionError, match="capability"):
        validate_predictions(
            RecurrentBaselineAdapter.capabilities,
            Predictions(mu=mu, pi=torch.full((B, N), 0.5), imputed=torch.randn(B, N, T, D)),
        )


def test_metric_accumulator_preserves_nulls_and_folds_counts():
    acc = MetricAccumulator()
    acc.update({"rmse": 2.0, "impute_mse": None, "zero_true_count": 4, "zero_correct_count": 1})
    acc.update({"rmse": 4.0, "impute_mse": None, "zero_true_count": 6, "zero_correct_count": 4})
    s = acc.summary()
    assert s["rmse"] == pytest.approx(3.0)
    assert s["impute_mse"] is None
    assert s["zero_acc"] == pytest.approx(0.5)  # (1+4)/(4+6), not mean of ratios


# --------------------------------------------------------------------------- #
# Inductive zero-shot guarantee                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("adapter_fixture", ["gnn_adapter", "lstm_adapter"])
def test_holdout_targets_produce_no_loss_signal(adapter_fixture, request):
    """Randomizing the held-out nodes' targets must not move any loss component:
    their ground truth is unreachable by training."""
    adapter = request.getfixturevalue(adapter_fixture)
    holdout = torch.zeros(N, dtype=torch.bool)
    holdout[[0, 5]] = True
    loss_node_mask = ~holdout

    def run(target_seed):
        torch.manual_seed(0)  # freeze dropout etc. so only targets differ
        batch = make_batch()
        g = torch.Generator().manual_seed(target_seed)
        batch["target_impute"][:, holdout] = torch.randn(B, 2, T, D, generator=g)
        batch["target_forecast"][:, holdout] = torch.randint(0, 30, (B, 2), generator=g).float()
        # kriging training signal drawn from visible nodes only (as in train.py)
        reroll = torch.zeros(N, dtype=torch.bool)
        reroll[[3, 8]] = True
        batch["spatial_holdout_mask"] = reroll
        losses, _ = adapter.step(batch, WEIGHTS, loss_node_mask=loss_node_mask)
        return losses

    a, b = run(1), run(2)
    for got, other, name in zip(a, b, Losses._fields):
        assert got.item() == pytest.approx(other.item(), abs=1e-9), \
            f"{name} moved when holdout targets changed — target leakage"


def test_compute_loss_rejects_leaky_kriging_holdout_under_grad():
    """A training-time (grad-enabled) kriging holdout containing loss-excluded
    nodes is target leakage and must be rejected; the same call under no_grad
    is validation scoring and must be allowed."""
    imputed = torch.randn(B, N, T, D, requires_grad=True)
    target = torch.randn(B, N, T, D)
    mu = torch.rand(B, N) + 0.1
    theta = torch.rand(B, N) + 0.1
    pi = torch.rand(B, N) * 0.5
    tf = torch.randint(0, 30, (B, N)).float()
    holdout = torch.zeros(N, dtype=torch.bool)
    holdout[[0, 5]] = True
    kwargs = dict(
        imputed_features=imputed, target_impute=target, mu=mu, theta=theta, pi=pi,
        target_forecast=tf, traffic_mean=torch.tensor(0.0), traffic_std=torch.tensor(1.0),
        kriging_pred=torch.randn(B, 2, T, D), spatial_holdout_mask=holdout,
        w_kriging=0.5, loss_node_mask=~holdout,
    )
    with pytest.raises(AssertionError, match="leakage"):
        compute_loss(**kwargs)
    with torch.no_grad():
        total, _, _, l_kr = compute_loss(**kwargs)
        assert torch.isfinite(total) and l_kr.item() > 0.0


# --------------------------------------------------------------------------- #
# Holdout manifest round-trip                                                  #
# --------------------------------------------------------------------------- #
def test_holdout_manifest_round_trip(tmp_path):
    location_ids = [f"sensor_{i}" for i in range(N)]
    mask = make_fixed_holdout_mask(num_nodes=N, holdout_frac=0.3, seed=7)

    save_holdout_manifest(str(tmp_path), location_ids, mask, seed=7, holdout_frac=0.3)
    manifest = load_holdout_manifest(str(tmp_path))
    assert manifest["seed"] == 7

    rebuilt = mask_from_manifest(manifest, location_ids)
    assert torch.equal(rebuilt, mask)

    # sha is order-insensitive (comparability key), and changes with the split.
    shuffled = dict(manifest, holdout_location_ids=list(reversed(manifest["holdout_location_ids"])))
    assert manifest_sha(manifest) == manifest_sha(shuffled)
    other = dict(manifest, holdout_location_ids=manifest["holdout_location_ids"][:1])
    assert manifest_sha(manifest) != manifest_sha(other)
    assert manifest_sha(None) is None


def test_mask_from_manifest_rejects_unknown_nodes(tmp_path):
    manifest = {"holdout_location_ids": ["ghost_sensor"], "num_nodes": 3}
    with pytest.raises(AssertionError, match="absent"):
        mask_from_manifest(manifest, ["a", "b", "c"])


def test_fixed_holdout_requires_seed_and_nontrivial_split():
    with pytest.raises(AssertionError):
        make_fixed_holdout_mask(num_nodes=N, holdout_frac=0.2, seed=None)
    with pytest.raises(AssertionError):
        make_fixed_holdout_mask(num_nodes=N, holdout_frac=0.0, seed=1)
