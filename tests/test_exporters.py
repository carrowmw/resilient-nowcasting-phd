"""Adapter-served exporter tests.

The payload exporters (export_ui_payload / export_paginated_payload) build the
served model through the shared build_model_and_adapter() helper and then drive
inference exclusively through adapter.predict() + adapter.capabilities. These
tests lock in that seam so a baseline checkpoint yields null (not fabricated)
uncertainty/graph payload sections, and a GNN yields the full contract.
"""
import torch

from baselines import LSTMForecaster
from export_ui_payload import build_model_and_adapter
from harness.adapters import RIoTAdapter, RecurrentBaselineAdapter
from harness.evaluate import validate_predictions
from pipeline import ResilientIoTPipeline

N, T, D, S = 6, 6, 3, 5
BASELINE_CFG = {"hidden_dim": 8, "num_layers": 1, "dropout": 0.1}
GNN_CFG = {
    "embedding_dim": 8, "hidden_dim": 16, "sparsity_threshold": 0.5,
    "num_layers": 1, "num_heads": 2, "dropout": 0.1, "dim_feedforward": None,
    "zinb_eps": 1e-6, "edge_features_cache_path": None,  # learned-only
}


def _predict_batch():
    dyn = torch.randn(1, N, T, D)
    return {
        "static_features": torch.randn(N, S),
        "dynamic_features": dyn,
        "structural_masks": torch.zeros(1, N, T, 2),
        "simulated_nodes_mask": torch.zeros(N, dtype=torch.bool),
        "location_ids": [f"loc_{i}" for i in range(N)],
    }


def test_builder_returns_baseline_adapter_for_lstm(tmp_path):
    ckpt = tmp_path / "best_model.pt"
    torch.save(LSTMForecaster(time_series_dim=D, hidden_dim=8, num_layers=1).state_dict(), ckpt)

    adapter = build_model_and_adapter(
        "lstm", str(ckpt), dict(BASELINE_CFG),
        num_nodes_total=N, static_feat_dim=S, time_series_dim=D,
        seq_len=T, device=torch.device("cpu"),
    )
    caps = adapter.capabilities
    assert isinstance(adapter, RecurrentBaselineAdapter)
    # A baseline claims no uncertainty / zero-inflation / graph — the exporter
    # reads exactly these flags to null out those payload sections.
    assert caps.uncertainty is False and caps.zero_inflation is False and caps.graph is False
    assert caps.imputation is True and caps.kriging is True

    with torch.no_grad():
        preds = adapter.predict(_predict_batch())
    validate_predictions(caps, preds)  # capability-gated tripwires must pass
    # These Nones are what become null uncertainty/graph sections downstream.
    assert preds.theta is None and preds.pi is None
    assert preds.edge_index is None and preds.attention is None
    assert preds.imputed is not None and preds.mu.shape == (1, N)


def test_builder_returns_gnn_adapter_with_full_contract(tmp_path):
    ckpt = tmp_path / "best_model.pt"
    torch.save(
        ResilientIoTPipeline(
            num_nodes=N, static_feat_dim=S, time_series_dim=D, seq_len=T,
            embedding_dim=8, hidden_dim=16, sparsity_threshold=0.5,
            num_layers=1, num_heads=2,
        ).state_dict(),
        ckpt,
    )
    adapter = build_model_and_adapter(
        "gnn", str(ckpt), dict(GNN_CFG),
        num_nodes_total=N, static_feat_dim=S, time_series_dim=D,
        seq_len=T, device=torch.device("cpu"),
    )
    caps = adapter.capabilities
    assert isinstance(adapter, RIoTAdapter)
    assert all([caps.uncertainty, caps.zero_inflation, caps.imputation, caps.kriging, caps.graph])

    with torch.no_grad():
        preds = adapter.predict(_predict_batch())
    validate_predictions(caps, preds)
    # Full ZINB + graph — the sections a GNN payload genuinely carries.
    assert preds.theta is not None and preds.pi is not None
    assert preds.edge_index is not None and preds.attention is not None


def test_builder_rejects_unknown_model_type(tmp_path):
    ckpt = tmp_path / "best_model.pt"
    torch.save(LSTMForecaster(time_series_dim=D, hidden_dim=8, num_layers=1).state_dict(), ckpt)
    try:
        build_model_and_adapter(
            "transformer", str(ckpt), dict(BASELINE_CFG),
            num_nodes_total=N, static_feat_dim=S, time_series_dim=D,
            seq_len=T, device=torch.device("cpu"),
        )
        assert False, "expected ValueError for unknown model_type"
    except ValueError as exc:
        assert "model_type" in str(exc)
