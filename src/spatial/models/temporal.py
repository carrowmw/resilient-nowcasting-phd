"""
Temporal-only forecasters — the baselines the spatial models must beat.

Non-trainable baselines are pure NumPy and operate directly on the shared
scaled grid (scaling is affine per node, so they are unaffected by it).
Fallbacks are never silent: every fallback event is counted in
`self.diagnostics`, which the runner persists into the job manifest.

Trainable baselines are weight-shared per-node networks (one network, applied
to every node independently, no cross-node information). Weight sharing is
deliberate: it matches the GAT models' parameterisation, so the spatial delta
is not confounded with per-node capacity.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from spatial.contract import (
    FitReport,
    GraphForecaster,
    GraphSpec,
    GraphSplit,
    SpatialCapabilities,
)
from spatial.models.base import TorchGraphForecaster
from spatial.models.encoders import build_temporal_encoder


def _last_observed_scaled(split: GraphSplit) -> np.ndarray:
    """
    [n, N] last observed scaled value inside each window, NaN where the whole
    window is unobserved for that node.
    """
    ctx = split.context
    scaled = ctx.values_scaled
    observed = ctx.observed_mask
    n, N, w = len(split), ctx.n_nodes, ctx.window
    out = np.full((n, N), np.nan, dtype=np.float32)
    for i, start in enumerate(split.window_start_idx):
        sl = slice(int(start), int(start) + w)
        obs = observed[sl]                       # [w, N]
        vals = scaled[sl]
        any_obs = obs.any(axis=0)
        # Index of the last True per column (argmax on the reversed axis).
        last = w - 1 - np.argmax(obs[::-1], axis=0)
        out[i, any_obs] = vals[last[any_obs], np.arange(N)[any_obs]]
    return out


class PersistenceForecaster(GraphForecaster):
    """Last observed value in the window, held flat across horizons.
    All-missing windows fall back to the node's scaled train mean (≈ 0 by
    construction of the train-only scaler) — counted, never silent."""

    capabilities = SpatialCapabilities()
    name = "persistence"

    def __init__(self):
        self.diagnostics: Dict = {}

    def fit(self, train: GraphSplit, val: GraphSplit, graph: GraphSpec) -> FitReport:
        return FitReport(fitted=True)

    def predict(self, split: GraphSplit, graph: GraphSpec) -> np.ndarray:
        last = _last_observed_scaled(split)                    # [n, N]
        n_fallback = int(np.isnan(last).sum())
        self.diagnostics["all_missing_window_fallbacks"] = n_fallback
        last = np.nan_to_num(last, nan=0.0)
        H = len(split.context.horizons)
        return np.repeat(last[:, :, None], H, axis=2)


class SeasonalNaiveForecaster(GraphForecaster):
    """Value one season before the target step; persistence fallback when the
    seasonal reference is unobserved (fraction recorded in diagnostics)."""

    capabilities = SpatialCapabilities()

    def __init__(self, seasonality_days: float):
        self.seasonality_days = float(seasonality_days)
        self.name = f"snaive_{seasonality_days:g}d"
        self.diagnostics: Dict = {}

    def fit(self, train: GraphSplit, val: GraphSplit, graph: GraphSpec) -> FitReport:
        return FitReport(fitted=True)

    def predict(self, split: GraphSplit, graph: GraphSpec) -> np.ndarray:
        ctx = split.context
        season = int(round(self.seasonality_days * ctx.steps_per_day()))
        assert season > ctx.max_horizon, (
            f"CRITICAL: seasonal period {season} steps <= max horizon "
            f"{ctx.max_horizon} — the reference would be after the origin (future leakage)"
        )
        target_idx = split.target_idx                          # [n, H]
        ref_idx = target_idx - season
        n, H = ref_idx.shape
        N = ctx.n_nodes

        preds = np.empty((n, N, H), dtype=np.float32)
        valid_ref = ref_idx >= 0
        fallback = np.zeros((n, N, H), dtype=bool)
        for k in range(H):
            idx = np.clip(ref_idx[:, k], 0, None)
            vals = ctx.values_scaled[idx]                      # [n, N]
            obs = ctx.observed_mask[idx] & valid_ref[:, k][:, None]
            preds[:, :, k] = np.where(obs, vals, np.nan)
            fallback[:, :, k] = ~obs

        last = np.nan_to_num(_last_observed_scaled(split), nan=0.0)  # [n, N]
        for k in range(H):
            miss = np.isnan(preds[:, :, k])
            preds[miss, k] = last[miss]
        self.diagnostics["seasonal_fallback_fraction"] = float(fallback.mean())
        assert np.isfinite(preds).all()
        return preds


class HistoricalAverageForecaster(GraphForecaster):
    """Per-(node, hour-of-week) mean of observed scaled train values — the
    'dumb baseline' required by the NSTAR ablation strategy."""

    capabilities = SpatialCapabilities(trainable=True)
    name = "hist_avg"

    def __init__(self):
        self._table = None       # [168 buckets at hourly res × N] — see fit
        self._node_mean = None
        self.diagnostics: Dict = {}

    @staticmethod
    def _buckets(ctx, grid_idx: np.ndarray) -> np.ndarray:
        dt = ctx.grid_index[grid_idx]
        return (dt.dayofweek * 24 + dt.hour).to_numpy()

    def fit(self, train: GraphSplit, val: GraphSplit, graph: GraphSpec) -> FitReport:
        ctx = train.context
        # Train-touched steps: everything up to the last train target.
        last_touched = int(train.window_start_idx.max()) + ctx.window - 1 + ctx.max_horizon
        steps = np.arange(last_touched + 1)
        buckets = self._buckets(ctx, steps)
        vals = ctx.values_scaled[steps]                        # [t, N]
        obs = ctx.observed_mask[steps]

        table = np.full((168, ctx.n_nodes), np.nan, dtype=np.float64)
        for b in range(168):
            rows = buckets == b
            if not rows.any():
                continue
            v, o = vals[rows], obs[rows]
            counts = o.sum(axis=0)
            with np.errstate(invalid="ignore"):
                means = np.where(counts > 0, np.nansum(np.where(o, v, 0.0), axis=0)
                                 / np.maximum(counts, 1), np.nan)
            table[b] = means
        self._table = table
        node_mean = np.nanmean(np.where(obs, vals, np.nan), axis=0)
        self._node_mean = np.nan_to_num(node_mean, nan=0.0)
        empty = int(np.isnan(table).sum())
        self.diagnostics["empty_buckets"] = empty
        return FitReport(fitted=True, notes={"empty_buckets": empty})

    def predict(self, split: GraphSplit, graph: GraphSpec) -> np.ndarray:
        assert self._table is not None, "CRITICAL: hist_avg.predict() before fit()"
        ctx = split.context
        n, H, N = len(split), len(ctx.horizons), ctx.n_nodes
        preds = np.empty((n, N, H), dtype=np.float32)
        for k in range(H):
            buckets = self._buckets(ctx, split.target_idx[:, k])
            vals = self._table[buckets]                        # [n, N]
            miss = np.isnan(vals)
            vals = np.where(miss, self._node_mean[None, :], vals)
            preds[:, :, k] = vals
        assert np.isfinite(preds).all()
        return preds


class PerNodeRNNForecaster(TorchGraphForecaster):
    """Weight-shared per-node LSTM. No cross-node information flow — the
    perturbation test in tests/test_spatial_models.py proves it."""

    capabilities = SpatialCapabilities(trainable=True)
    name = "lstm"

    def _build_module(self, in_dim, n_nodes, window, n_horizons, graph):
        return _PerNodeNet(build_temporal_encoder(
            "lstm", in_dim, window, int(self.spec["hidden_dim"]), self.spec
        ), int(self.spec["hidden_dim"]), n_horizons)


class PerNodeTransformerForecaster(TorchGraphForecaster):
    """Weight-shared per-node Transformer encoder."""

    capabilities = SpatialCapabilities(trainable=True)
    name = "transformer"

    def _build_module(self, in_dim, n_nodes, window, n_horizons, graph):
        return _PerNodeNet(build_temporal_encoder(
            "transformer", in_dim, window, int(self.spec["d_model"]), self.spec
        ), int(self.spec["d_model"]), n_horizons)


class _PerNodeNet(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, n_horizons: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, n_horizons)

    def forward(self, x, edge_index=None, edge_attr=None, static=None):
        B, N, T, D = x.shape
        h = self.encoder(x.reshape(B * N, T, D))               # [B*N, F]
        return self.head(h).reshape(B, N, -1)                  # [B, N, H]
