"""
The graph-forecasting contract (spatial track).

Research unit: the whole sensor network. A forecaster receives GraphSplits
(window start indices over a shared [T_total, N] grid) plus a static GraphSpec
and returns scaled point forecasts [n_windows, N, n_horizons]. Every model —
persistence through learned-topology GAT — predicts the identical window set,
which is what makes per-node Diebold–Mariano tests paired and the cross-model
tables directly comparable.

Contract semantics:
  - Window i covers grid steps [start, start + window); the forecast origin is
    the last input step; targets sit at origin + h for each h in horizons.
  - Predictions are in scaled space; evaluation inverse-transforms per node.
  - Targets are never imputed. Unobserved targets are NaN in y_scaled/y_raw
    and False in target_observed; losses and metrics must mask to observed
    entries. This keeps low-quality nodes in the study rather than selecting
    them out (complete-case windows across the full network would leave
    almost no data).
  - node_ids is sorted and is THE order contract for every [N, ...] axis.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STATIC_FEATURE_DIM = 15  # src/features.py STATIC_FEATURE_COLS


@dataclass(frozen=True)
class SpatialCapabilities:
    """What a graph forecaster genuinely is/does. The runner gates on these
    flags — never on model-name string prefixes."""
    trainable: bool = False        # has a real fit stage (epochs)
    uses_graph: bool = False       # consumes edge_index (False ⇒ provably per-node)
    uses_edge_attr: bool = False   # consumes edge_attr
    learns_topology: bool = False  # owns an internal graph learner
    emits_attention: bool = False  # can return GATv2 attention weights


@dataclass
class FitReport:
    """Returned by fit(); persisted into the run manifest."""
    fitted: bool
    epochs_run: int = 0
    best_val_loss: Optional[float] = None
    notes: Dict = field(default_factory=dict)


@dataclass
class GraphContext:
    """
    The shared multi-node grid plus everything derived from it that models
    and evaluation share. values_scaled uses per-node train-only standard
    scalers (ported leakage fix from the quality track).
    """
    node_ids: List[str]            # sorted; order contract for every [N, ...] axis
    grid_index: pd.DatetimeIndex   # [T_total], continuous at freq
    values_raw: np.ndarray         # [T_total, N] float32, NaN at gaps
    values_scaled: np.ndarray      # [T_total, N] float32, NaN at gaps
    observed_mask: np.ndarray      # [T_total, N] bool
    static_features: np.ndarray    # [N, STATIC_FEATURE_DIM] float32, standardised
    scaler_mean: np.ndarray        # [N] float64 — per-node, train-window fit
    scaler_std: np.ndarray         # [N] float64
    window: int
    horizons: Tuple[int, ...]      # ascending, all >= 1
    freq: str

    def __post_init__(self):
        n, t = len(self.node_ids), len(self.grid_index)
        assert list(self.node_ids) == sorted(self.node_ids), \
            "CRITICAL: node_ids must be sorted — it is the [N] order contract"
        assert self.values_raw.shape == (t, n), \
            f"CRITICAL: values_raw shape {self.values_raw.shape} != ({t}, {n})"
        assert self.values_scaled.shape == (t, n)
        assert self.observed_mask.shape == (t, n)
        assert self.static_features.shape == (n, STATIC_FEATURE_DIM), \
            f"CRITICAL: static_features shape {self.static_features.shape}"
        assert self.scaler_mean.shape == (n,) and self.scaler_std.shape == (n,)
        assert (self.scaler_std > 0).all(), "CRITICAL: non-positive scaler std"
        hs = tuple(int(h) for h in self.horizons)
        assert len(hs) >= 1 and all(h >= 1 for h in hs), \
            f"CRITICAL: horizons must be >= 1, got {hs}"
        assert hs == tuple(sorted(hs)) and len(set(hs)) == len(hs), \
            f"CRITICAL: horizons must be strictly ascending, got {hs}"
        assert not np.isfinite(self.values_raw[~self.observed_mask]).any(), \
            "CRITICAL: unobserved grid cells must be NaN in values_raw"
        assert self.window >= 1

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def max_horizon(self) -> int:
        return int(self.horizons[-1])

    def steps_per_day(self) -> int:
        assert len(self.grid_index) >= 2, "CRITICAL: grid too short to infer step size"
        step = self.grid_index[1] - self.grid_index[0]
        per_day = pd.Timedelta(days=1) / step
        assert per_day == int(per_day), \
            f"CRITICAL: grid step {step} does not divide a day evenly"
        return int(per_day)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Scaled space → raw counts, clamped to the count domain. The node axis
        must be axis -2 of a [..., N, H] array or axis -1 of a [..., N] array.
        """
        if y_scaled.ndim >= 2 and y_scaled.shape[-2] == self.n_nodes:
            mean = self.scaler_mean[:, None]
            std = self.scaler_std[:, None]
        else:
            assert y_scaled.shape[-1] == self.n_nodes, \
                f"CRITICAL: no node axis of size {self.n_nodes} in {y_scaled.shape}"
            mean, std = self.scaler_mean, self.scaler_std
        return np.clip(y_scaled * std + mean, 0.0, None)


@dataclass
class GraphSplit:
    """
    One split's windows, stored as start indices into the shared grid
    (windows are gathered lazily; materializing [n, N, T] would be huge).
    """
    context: GraphContext
    split: str                    # 'train' | 'val' | 'test'
    window_start_idx: np.ndarray  # [n] int

    def __post_init__(self):
        assert self.split in ("train", "val", "test"), \
            f"CRITICAL: bad split name {self.split!r}"
        assert self.window_start_idx.ndim == 1, "CRITICAL: window_start_idx must be 1-D"
        if len(self.window_start_idx):
            last = int(self.window_start_idx.max())
            limit = len(self.context.grid_index) - self.context.window - self.context.max_horizon
            assert last <= limit, \
                f"CRITICAL: window start {last} overruns the grid (limit {limit})"

    def __len__(self) -> int:
        return len(self.window_start_idx)

    @property
    def origin_idx(self) -> np.ndarray:
        """[n] — grid index of each window's forecast origin."""
        return self.window_start_idx + self.context.window - 1

    @property
    def target_idx(self) -> np.ndarray:
        """[n, H] — grid index of each (window, horizon) target."""
        return self.origin_idx[:, None] + np.asarray(self.context.horizons)[None, :]

    @property
    def y_scaled(self) -> np.ndarray:
        """[n, N, H] scaled targets, NaN where unobserved."""
        return self.context.values_scaled[self.target_idx].transpose(0, 2, 1)

    @property
    def y_raw(self) -> np.ndarray:
        """[n, N, H] raw targets, NaN where unobserved."""
        return self.context.values_raw[self.target_idx].transpose(0, 2, 1)

    @property
    def target_observed(self) -> np.ndarray:
        """[n, N, H] bool — where a real observation exists at the target step."""
        return self.context.observed_mask[self.target_idx].transpose(0, 2, 1)

    @property
    def origin_dt(self) -> pd.DatetimeIndex:
        return self.context.grid_index[self.origin_idx]

    @property
    def target_dt(self) -> np.ndarray:
        """[n, H] datetime64 target timestamps."""
        return self.context.grid_index.to_numpy()[self.target_idx]


@dataclass(frozen=True)
class GraphSpec:
    """
    A static graph over the context's node set. edge_index includes self-loops
    (models must not add their own — add_self_loops=False everywhere).
    Learned-topology models receive the identity spec and ignore it.
    """
    method: str                       # 'identity' | 'gabriel' | 'knn' | 'fully_connected'
    edge_index: np.ndarray            # [2, E] int64
    edge_attr: Optional[np.ndarray]   # [E, edge_dim] float32, or None
    edge_dim: int                     # 0 iff edge_attr is None
    params: Dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.edge_index.ndim == 2 and self.edge_index.shape[0] == 2, \
            f"CRITICAL: edge_index shape {self.edge_index.shape}"
        if self.edge_attr is None:
            assert self.edge_dim == 0, "CRITICAL: edge_dim must be 0 without edge_attr"
        else:
            assert self.edge_attr.shape == (self.edge_index.shape[1], self.edge_dim), \
                f"CRITICAL: edge_attr shape {self.edge_attr.shape} != " \
                f"({self.edge_index.shape[1]}, {self.edge_dim})"
            assert np.isfinite(self.edge_attr).all(), "CRITICAL: non-finite edge_attr"

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


class GraphForecaster(ABC):
    """
    Graph-level forecaster contract. All predictions are in scaled space
    (affine per node, so persistence / seasonal naive are unaffected);
    evaluation inverse-transforms to raw counts.
    """
    capabilities: SpatialCapabilities = SpatialCapabilities()
    name: str = "abstract"

    @abstractmethod
    def fit(self, train: GraphSplit, val: GraphSplit, graph: GraphSpec) -> FitReport:
        """Fit on train (val for early stopping where relevant)."""

    @abstractmethod
    def predict(self, split: GraphSplit, graph: GraphSpec) -> np.ndarray:
        """[len(split), N, H] scaled point forecasts, no NaN."""

    def predict_attention(self, split: GraphSplit, graph: GraphSpec,
                          max_windows: int = 256):
        """AttentionRecord over a seeded window subsample (emits_attention only)."""
        raise NotImplementedError(
            f"CRITICAL: {self.name} does not emit attention "
            f"(capabilities.emits_attention={self.capabilities.emits_attention})"
        )

    def get_params(self) -> Dict:
        """Hyperparameters for the run manifest."""
        return {}


def assert_predictions_valid(split: GraphSplit, preds: np.ndarray, name: str) -> None:
    """Shared offensive check applied by the runner to every predict() output."""
    expected = (len(split), split.context.n_nodes, len(split.context.horizons))
    assert preds.shape == expected, \
        f"CRITICAL: {name} predictions shape {preds.shape} != {expected}"
    assert np.isfinite(preds).all(), \
        f"CRITICAL: {name} produced non-finite predictions"
