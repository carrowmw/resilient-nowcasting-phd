"""
Per-sensor forecasters: persistence, seasonal naive, ARIMA, and the paper's
bidirectional LSTM (univariate / multivariate via feature groups).

All models operate in scaled space (affine, so the naive baselines and ARIMA
are unaffected); evaluation inverse-transforms to raw counts.
"""
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quality.contract import (
    FEATURE_GROUP_DIMS,
    FitReport,
    PerSensorSplit,
    QualityCapabilities,
    SensorForecaster,
)
from quality.per_sensor_dataset import WindowDataset, train_segments

logger = logging.getLogger(__name__)


class PersistenceForecaster(SensorForecaster):
    """y_hat(t + h) = y(t): the last observed value of the input window."""
    capabilities = QualityCapabilities(trainable=False, uses_history=False, multivariate=False)
    name = "persistence"

    def fit(self, train: PerSensorSplit, val: PerSensorSplit) -> FitReport:
        return FitReport(fitted=True)

    def predict(self, split: PerSensorSplit) -> np.ndarray:
        preds = split.context.values_scaled[split.origin_idx]
        assert not np.isnan(preds).any(), \
            f"CRITICAL: NaN at window origins of {split.context.sensor_id}"
        return preds.astype(np.float64)


class SeasonalNaiveForecaster(SensorForecaster):
    """
    y_hat(t + h) = y(t + h - s) with s the seasonal period in steps
    (daily: 96, weekly: 672 on a 15-min grid). The seasonal lag always lies at
    or before the forecast origin because s > horizon — asserted, never
    assumed. When the seasonal reference is missing (a gap), the forecast
    falls back to persistence; the fallback fraction is reported through
    self.diagnostics, never silently absorbed.
    """
    capabilities = QualityCapabilities(trainable=False, uses_history=True, multivariate=False)

    def __init__(self, seasonality_days: float):
        assert seasonality_days > 0, \
            f"CRITICAL: seasonality_days must be positive, got {seasonality_days}"
        self.seasonality_days = float(seasonality_days)
        self.name = f"snaive_{seasonality_days:g}d"
        self.diagnostics: Dict[str, float] = {}

    def fit(self, train: PerSensorSplit, val: PerSensorSplit) -> FitReport:
        return FitReport(fitted=True)

    def predict(self, split: PerSensorSplit) -> np.ndarray:
        ctx = split.context
        s = int(round(self.seasonality_days * ctx.steps_per_day()))
        assert s > ctx.horizon, (
            f"CRITICAL: seasonal period {s} steps must exceed horizon {ctx.horizon} — "
            "otherwise the seasonal reference lies in the future"
        )
        lag_idx = split.target_idx - s
        origin = split.origin_idx
        assert (lag_idx <= origin).all(), \
            "CRITICAL: seasonal reference after forecast origin (future leakage)"

        valid = lag_idx >= 0
        preds = np.empty(len(split), dtype=np.float64)
        lag_vals = np.full(len(split), np.nan)
        lag_vals[valid] = ctx.values_scaled[lag_idx[valid]]
        have_ref = ~np.isnan(lag_vals)
        preds[have_ref] = lag_vals[have_ref]
        # Fallback: persistence for windows without a seasonal reference.
        fallback = ~have_ref
        preds[fallback] = ctx.values_scaled[origin[fallback]]
        assert not np.isnan(preds).any(), "CRITICAL: NaN in seasonal-naive predictions"

        self.diagnostics = {
            f"{split.split}_fallback_frac": float(fallback.mean()),
            "seasonality_steps": s,
        }
        if fallback.any():
            logger.info(
                f"{ctx.sensor_id} [{self.name}/{split.split}]: persistence fallback on "
                f"{fallback.sum()}/{len(split)} windows ({fallback.mean():.1%})"
            )
        return preds

    def get_params(self) -> Dict:
        return {"seasonality_days": self.seasonality_days}


class ARIMAForecaster(SensorForecaster):
    """
    statsmodels SARIMAX wrapper.

    Fit: maximum-likelihood once per sensor on the longest contiguous train
    segment (rolling-origin *refitting* across ~1e5 windows is computationally
    infeasible and documented as such in the paper).

    Predict: honest walk-forward — for each consecutive run, the fitted
    parameters are applied to the history up to the first forecast origin,
    then the Kalman filter is extended one observation at a time; each origin
    forecasts horizon steps ahead. No future data enters any forecast.
    """
    capabilities = QualityCapabilities(trainable=True, uses_history=True, multivariate=False)
    name = "arima"

    def __init__(
        self,
        order: Sequence[int] = (2, 0, 2),
        seasonal_order: Sequence[int] = (0, 0, 0, 0),
        trend: str = "c",
        min_fit_steps: int = 192,
        max_eval_history: int = 2688,
    ):
        assert len(order) == 3, f"CRITICAL: ARIMA order must have 3 terms, got {order}"
        assert len(seasonal_order) == 4, \
            f"CRITICAL: seasonal_order must have 4 terms, got {seasonal_order}"
        self.order = tuple(int(x) for x in order)
        self.seasonal_order = tuple(int(x) for x in seasonal_order)
        self.trend = trend
        self.min_fit_steps = int(min_fit_steps)
        self.max_eval_history = int(max_eval_history)
        self._results = None
        self.diagnostics: Dict[str, float] = {}

    def fit(self, train: PerSensorSplit, val: PerSensorSplit) -> FitReport:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        ctx = train.context
        segments = train_segments(train)
        assert segments, f"CRITICAL: no train segments for {ctx.sensor_id}"
        start, end = max(segments, key=lambda se: se[1] - se[0])
        fit_series = ctx.values_scaled[start: end + 1]
        assert not np.isnan(fit_series).any(), \
            "CRITICAL: NaN inside ARIMA fit segment — span extraction is broken"
        assert len(fit_series) >= self.min_fit_steps, (
            f"CRITICAL: longest train segment of {ctx.sensor_id} has "
            f"{len(fit_series)} steps < min_fit_steps={self.min_fit_steps}"
        )
        model = SARIMAX(
            fit_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )
        self._results = model.fit(disp=False)
        self.diagnostics = {
            "fit_steps": int(len(fit_series)),
            "aic": float(self._results.aic),
            "converged": bool(self._results.mle_retvals.get("converged", False)),
        }
        return FitReport(fitted=True, notes=dict(self.diagnostics))

    def predict(self, split: PerSensorSplit) -> np.ndarray:
        assert self._results is not None, "CRITICAL: predict() before fit()"
        ctx = split.context
        h = ctx.horizon
        preds = np.full(len(split), np.nan)

        for seq in np.unique(split.seq_id):
            sel = np.flatnonzero(split.seq_id == seq)
            sel = sel[np.argsort(split.window_start_idx[sel], kind="stable")]
            origins = split.origin_idx[sel]
            # History start: bounded lookback, never crossing a NaN gap.
            first_origin = int(origins[0])
            hist_start = max(0, first_origin + 1 - self.max_eval_history)
            chunk = ctx.values_scaled[hist_start: first_origin + 1]
            nan_pos = np.flatnonzero(np.isnan(chunk))
            if len(nan_pos):
                hist_start = hist_start + int(nan_pos[-1]) + 1
            history = ctx.values_scaled[hist_start: first_origin + 1]
            assert len(history) > 0 and not np.isnan(history).any(), \
                f"CRITICAL: bad ARIMA eval history for {ctx.sensor_id} seq {seq}"

            res = self._results.apply(history, refit=False)
            prev_origin = first_origin
            for k, w in enumerate(sel):
                origin = int(origins[k])
                if k > 0:
                    new_obs = ctx.values_scaled[prev_origin + 1: origin + 1]
                    assert len(new_obs) > 0 and not np.isnan(new_obs).any(), \
                        "CRITICAL: gap between consecutive origins within a run"
                    res = res.extend(new_obs)
                prev_origin = origin
                preds[w] = float(res.forecast(steps=h)[-1])

        assert not np.isnan(preds).any(), "CRITICAL: NaN in ARIMA predictions"
        return preds

    def get_params(self) -> Dict:
        return {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "trend": self.trend,
            "max_eval_history": self.max_eval_history,
        }


class PerSensorLSTM(nn.Module):
    """
    The paper's architecture: bidirectional LSTM, last-timestep hidden state
    of both directions concatenated into a linear head → scalar forecast.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        assert input_dim >= 1 and hidden_dim >= 1 and num_layers >= 1, \
            f"CRITICAL: bad LSTM dims ({input_dim}, {hidden_dim}, {num_layers})"
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"CRITICAL: LSTM input must be [B, T, D], got {tuple(x.shape)}"
        out, _ = self.lstm(x)              # [B, T, 2H]
        last = self.dropout(out[:, -1, :])  # [B, 2H]
        y = self.head(last).squeeze(-1)     # [B]
        assert torch.isfinite(y).all(), "CRITICAL: non-finite LSTM output"
        return y


class LSTMSensorForecaster(SensorForecaster):
    """Training wrapper: Adam + MSE + StepLR + early stopping (paper §3.5)."""
    capabilities = QualityCapabilities(trainable=True, uses_history=False, multivariate=True)

    def __init__(
        self,
        feature_groups: Sequence[str],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 30,
        patience: int = 5,
        scheduler_step_size: int = 5,
        scheduler_gamma: float = 0.7,
        fractional_time: bool = True,
        seed: int = 42,
        device: Optional[str] = None,
        num_workers: int = 0,
    ):
        self.feature_groups = tuple(feature_groups)
        self.name = "lstm_uv" if self.feature_groups == ("count",) else "lstm_mv"
        self.capabilities = QualityCapabilities(
            trainable=True, uses_history=False,
            multivariate=len(self.feature_groups) > 1,
        )
        self.input_dim = sum(FEATURE_GROUP_DIMS[g] for g in self.feature_groups)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.scheduler_step_size = int(scheduler_step_size)
        self.scheduler_gamma = float(scheduler_gamma)
        self.fractional_time = bool(fractional_time)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: Optional[PerSensorLSTM] = None

    def _loader(self, split: PerSensorSplit, shuffle: bool) -> DataLoader:
        ds = WindowDataset(split, self.feature_groups, self.fractional_time)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=False,
        )

    def fit(self, train: PerSensorSplit, val: PerSensorSplit) -> FitReport:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = PerSensorLSTM(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)
        model.to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )
        criterion = nn.MSELoss()
        train_loader = self._loader(train, shuffle=True)
        val_loader = self._loader(val, shuffle=False)

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0
        epochs_run = 0
        for epoch in range(self.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(model(xb), yb)
                assert torch.isfinite(loss), \
                    f"CRITICAL: non-finite train loss at epoch {epoch}"
                loss.backward()
                optimiser.step()
            scheduler.step()
            epochs_run = epoch + 1

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_losses.append(criterion(model(xb), yb).item() * len(yb))
            val_loss = sum(val_losses) / len(val_loader.dataset)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        assert best_state is not None, "CRITICAL: LSTM training produced no best state"
        model.load_state_dict(best_state)
        self._model = model
        return FitReport(fitted=True, epochs_run=epochs_run, best_val_loss=best_val)

    def predict(self, split: PerSensorSplit) -> np.ndarray:
        assert self._model is not None, "CRITICAL: predict() before fit()"
        loader = self._loader(split, shuffle=False)
        self._model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self._model(xb.to(self.device)).cpu().numpy())
        out = np.concatenate(preds).astype(np.float64)
        assert len(out) == len(split), \
            f"CRITICAL: prediction count {len(out)} != window count {len(split)}"
        return out

    def get_params(self) -> Dict:
        return {
            "feature_groups": list(self.feature_groups),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "fractional_time": self.fractional_time,
            "seed": self.seed,
        }
