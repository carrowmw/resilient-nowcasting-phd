"""
The per-sensor forecasting contract (quality track).

Mirrors the spirit of src/harness/contract.py — capability gating, honest
numbers only — but for the per-sensor research unit: forecasters receive a
PerSensorSplit (window indices over one sensor's gridded series) and return a
scaled point forecast per window. Predictions are always made for the exact
same window set across models, which is what makes Diebold–Mariano tests
paired and the cross-model tables directly comparable.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Feature groups available to the multivariate models; the ablation matrix is
# expressed as subsets of these. 'count' is always required.
FEATURE_GROUP_DIMS = {
    "count": 1,
    "half_day": 2,   # sin/cos, 12 h cycle
    "day": 2,        # sin/cos, 24 h cycle
    "week": 2,       # sin/cos, weekly cycle
    "quarter": 2,    # sin/cos, ~91.25 d cycle
    "year": 2,       # sin/cos, annual cycle
    "timestamp": 1,  # normalized linear time
}
ALL_FEATURE_GROUPS: Tuple[str, ...] = tuple(FEATURE_GROUP_DIMS.keys())


@dataclass(frozen=True)
class QualityCapabilities:
    """What a per-sensor forecaster genuinely is/does."""
    trainable: bool = False      # has a real fit stage (epochs / MLE)
    uses_history: bool = False   # needs series history beyond the input window
    multivariate: bool = False   # consumes engineered temporal features


@dataclass
class FitReport:
    """Returned by fit(); persisted into the run manifest."""
    fitted: bool
    epochs_run: int = 0
    best_val_loss: Optional[float] = None
    notes: Dict = field(default_factory=dict)


@dataclass
class SensorContext:
    """
    One sensor's gridded series plus everything derived from it that
    forecasters and evaluation share. values_scaled uses the train-only
    standard scaler (leakage fix vs the original paper, which fit the scaler
    on the full series).
    """
    sensor_id: str
    grid_index: pd.DatetimeIndex
    values_raw: np.ndarray          # [T] float, NaN at gaps
    values_scaled: np.ndarray       # [T] float, NaN at gaps
    interpolated_mask: np.ndarray   # [T] bool — True where value is synthetic
    scaler_mean: float
    scaler_std: float
    window: int
    horizon: int
    freq: str
    _feature_cache: Dict[Tuple[str, ...], np.ndarray] = field(default_factory=dict, repr=False)

    def steps_per_day(self) -> int:
        assert len(self.grid_index) >= 2, "CRITICAL: grid too short to infer step size"
        step = self.grid_index[1] - self.grid_index[0]
        per_day = pd.Timedelta(days=1) / step
        assert per_day == int(per_day), \
            f"CRITICAL: grid step {step} does not divide a day evenly"
        return int(per_day)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Scaled space → raw counts (clamped to the count domain)."""
        raw = y_scaled * self.scaler_std + self.scaler_mean
        return np.clip(raw, 0.0, None)


@dataclass
class PerSensorSplit:
    """
    One split's windows, stored as indices into the sensor grid (windows are
    gathered lazily — materializing ~1e5 MV windows per sensor is >100 MB).

    Window i: input steps [start, start + window), origin = last input step,
    scalar target at origin + horizon.
    """
    context: SensorContext
    split: str                       # 'train' | 'val' | 'test'
    window_start_idx: np.ndarray     # [n] int
    seq_id: np.ndarray               # [n] int — which consecutive run each window came from

    def __post_init__(self):
        assert self.split in ("train", "val", "test"), \
            f"CRITICAL: bad split name {self.split!r}"
        assert self.window_start_idx.ndim == 1, "CRITICAL: window_start_idx must be 1-D"
        assert len(self.window_start_idx) == len(self.seq_id), \
            "CRITICAL: window/seq_id length mismatch"

    def __len__(self) -> int:
        return len(self.window_start_idx)

    @property
    def origin_idx(self) -> np.ndarray:
        return self.window_start_idx + self.context.window - 1

    @property
    def target_idx(self) -> np.ndarray:
        return self.origin_idx + self.context.horizon

    @property
    def y_scaled(self) -> np.ndarray:
        y = self.context.values_scaled[self.target_idx]
        assert not np.isnan(y).any(), \
            f"CRITICAL: NaN targets in {self.split} split of {self.context.sensor_id}"
        return y

    @property
    def y_raw(self) -> np.ndarray:
        y = self.context.values_raw[self.target_idx]
        assert not np.isnan(y).any(), \
            f"CRITICAL: NaN raw targets in {self.split} split of {self.context.sensor_id}"
        return y

    @property
    def target_interpolated(self) -> np.ndarray:
        return self.context.interpolated_mask[self.target_idx]

    @property
    def origin_dt(self) -> pd.DatetimeIndex:
        return self.context.grid_index[self.origin_idx]

    @property
    def target_dt(self) -> pd.DatetimeIndex:
        return self.context.grid_index[self.target_idx]


class SensorForecaster(ABC):
    """
    Per-sensor forecaster contract. All predictions are in *scaled* space;
    evaluation inverse-transforms to raw counts. Scaling is an affine map, so
    persistence / seasonal naive / ARIMA are unaffected by operating on the
    scaled series.
    """
    capabilities: QualityCapabilities = QualityCapabilities()
    name: str = "abstract"

    @abstractmethod
    def fit(self, train: PerSensorSplit, val: PerSensorSplit) -> FitReport:
        """Fit on the train split (val used for early stopping where relevant)."""

    @abstractmethod
    def predict(self, split: PerSensorSplit) -> np.ndarray:
        """Point forecast per window, [len(split)] in scaled space, no NaN."""

    def get_params(self) -> Dict:
        """Hyperparameters for the run manifest."""
        return {}
