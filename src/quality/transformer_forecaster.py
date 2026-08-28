"""
Per-sensor Transformer forecasters for the quality track.

Two architectures:
  1. PerSensorTransformer — standard Transformer encoder with learned positional
     encoding. Same WindowDataset as the LSTM; the comparison isolates the
     architectural difference (self-attention vs recurrence).
  2. PerSensorMaskedTransformer — missingness-aware Transformer that receives
     windows *with* NaN gaps via MaskedWindowDataset. Uses src_key_padding_mask
     to let attention skip truly missing positions, and augments the input with
     a binary missingness indicator and time-delta encoding.

Both share the same training wrapper pattern as LSTMSensorForecaster:
Adam + MSE + StepLR + early stopping.
"""
import logging
import math
from typing import Dict, List, Optional, Sequence

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
from quality.per_sensor_dataset import WindowDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option A: Standard Transformer
# ---------------------------------------------------------------------------

class PerSensorTransformer(nn.Module):
    """
    Standard Transformer encoder for per-sensor forecasting.

    Architecture:
        input [B, T, D] → Linear(D, d_model) → + learned pos → TransformerEncoder
        → mean pool [B, d_model] → Linear → scalar [B]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        assert input_dim >= 1, f"CRITICAL: input_dim must be >= 1, got {input_dim}"
        assert d_model % nhead == 0, \
            f"CRITICAL: d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input features (contiguous, no NaNs).
        Returns:
            [B] scalar forecasts.
        """
        assert x.dim() == 3, f"CRITICAL: input must be [B, T, D], got {tuple(x.shape)}"
        B, T, D = x.shape

        # Project input features to d_model
        h = self.input_projection(x)  # [B, T, d_model]

        # Add learned positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        h = h + self.pos_embedding(positions)  # [B, T, d_model]

        # Transformer encoder
        h = self.encoder(h)  # [B, T, d_model]

        # Mean pool over time dimension → [B, d_model]
        h = h.mean(dim=1)
        h = self.dropout(h)

        # Scalar head
        y = self.head(h).squeeze(-1)  # [B]
        assert torch.isfinite(y).all(), "CRITICAL: non-finite Transformer output"
        return y


# ---------------------------------------------------------------------------
# Option B: Missingness-Aware Transformer
# ---------------------------------------------------------------------------

class PerSensorMaskedTransformer(nn.Module):
    """
    Missingness-aware Transformer encoder.

    Key differences from the standard version:
      - Input features include a binary missingness indicator (+1 dim)
      - Uses src_key_padding_mask to let attention skip missing positions
      - Time-delta encoding: the gap (in steps) since the last observed
        timestep is encoded as an additional feature (+1 dim)

    Architecture:
        input [B, T, D+2] → Linear(D+2, d_model) → + learned pos
        → TransformerEncoder(src_key_padding_mask=mask)
        → mean pool over observed positions → Linear → scalar [B]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        # +2 for missingness indicator and time-delta
        augmented_dim = input_dim + 2
        assert input_dim >= 1, f"CRITICAL: input_dim must be >= 1, got {input_dim}"
        assert d_model % nhead == 0, \
            f"CRITICAL: d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_projection = nn.Linear(augmented_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D+2] input features with missingness indicator and
               time-delta already concatenated by MaskedWindowDataset.
            mask: [B, T] bool — True where the timestep is missing.
                  Used as src_key_padding_mask.
        Returns:
            [B] scalar forecasts.
        """
        assert x.dim() == 3, f"CRITICAL: input must be [B, T, D], got {tuple(x.shape)}"
        assert mask.dim() == 2, f"CRITICAL: mask must be [B, T], got {tuple(mask.shape)}"
        B, T, _ = x.shape

        # Project augmented input to d_model
        h = self.input_projection(x)  # [B, T, d_model]

        # Add learned positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = h + self.pos_embedding(positions)

        # Transformer encoder with padding mask (True = ignore)
        h = self.encoder(h, src_key_padding_mask=mask)  # [B, T, d_model]

        # Mean pool over *observed* positions only
        observed = (~mask).unsqueeze(-1).float()  # [B, T, 1]
        n_observed = observed.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
        h = (h * observed).sum(dim=1) / n_observed.squeeze(-1)  # [B, d_model]

        h = self.dropout(h)
        y = self.head(h).squeeze(-1)  # [B]
        assert torch.isfinite(y).all(), "CRITICAL: non-finite MaskedTransformer output"
        return y


# ---------------------------------------------------------------------------
# Option A: Standard Transformer Forecaster Wrapper
# ---------------------------------------------------------------------------

class TransformerSensorForecaster(SensorForecaster):
    """Training wrapper for the standard Transformer (Option A).

    Identical interface to LSTMSensorForecaster — same WindowDataset, same
    Adam + MSE + StepLR + early stopping.
    """
    capabilities = QualityCapabilities(trainable=True, uses_history=False, multivariate=True)

    def __init__(
        self,
        feature_groups: Sequence[str],
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
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
        self.name = "transformer_uv" if self.feature_groups == ("count",) else "transformer_mv"
        self.capabilities = QualityCapabilities(
            trainable=True, uses_history=False,
            multivariate=len(self.feature_groups) > 1,
        )
        self.input_dim = sum(FEATURE_GROUP_DIMS[g] for g in self.feature_groups)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.dim_feedforward = int(dim_feedforward)
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
        self._model: Optional[PerSensorTransformer] = None

    def _loader(self, split: PerSensorSplit, shuffle: bool) -> DataLoader:
        ds = WindowDataset(split, self.feature_groups, self.fractional_time)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=False,
        )

    def fit(self, train: PerSensorSplit, val: PerSensorSplit) -> FitReport:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = PerSensorTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=train.context.window + 16,  # small headroom
        )
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

        assert best_state is not None, "CRITICAL: Transformer training produced no best state"
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
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
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


# ---------------------------------------------------------------------------
# Option B: Missingness-Aware Transformer Forecaster Wrapper
# ---------------------------------------------------------------------------

class MaskedTransformerSensorForecaster(SensorForecaster):
    """Training wrapper for the missingness-aware Transformer (Option B).

    Uses MaskedWindowDataset which returns (features, mask, target) tuples.
    The mask indicates which timesteps are missing (NaN in the original grid).
    """
    capabilities = QualityCapabilities(trainable=True, uses_history=False, multivariate=True)

    def __init__(
        self,
        feature_groups: Sequence[str],
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
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
        self.name = (
            "transformer_masked_uv" if self.feature_groups == ("count",)
            else "transformer_masked_mv"
        )
        self.capabilities = QualityCapabilities(
            trainable=True, uses_history=False,
            multivariate=len(self.feature_groups) > 1,
        )
        self.input_dim = sum(FEATURE_GROUP_DIMS[g] for g in self.feature_groups)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.dim_feedforward = int(dim_feedforward)
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
        self._model: Optional[PerSensorMaskedTransformer] = None

    def _loader(self, split: PerSensorSplit, shuffle: bool) -> DataLoader:
        from quality.masked_dataset import MaskedWindowDataset
        ds = MaskedWindowDataset(split, self.feature_groups, self.fractional_time)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=False,
        )

    def fit(self, train: PerSensorSplit, val: PerSensorSplit) -> FitReport:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = PerSensorMaskedTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=train.context.window + 16,
        )
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
            for xb, mb, yb in train_loader:
                xb = xb.to(self.device)
                mb = mb.to(self.device)
                yb = yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(model(xb, mb), yb)
                assert torch.isfinite(loss), \
                    f"CRITICAL: non-finite train loss at epoch {epoch}"
                loss.backward()
                optimiser.step()
            scheduler.step()
            epochs_run = epoch + 1

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, mb, yb in val_loader:
                    xb = xb.to(self.device)
                    mb = mb.to(self.device)
                    yb = yb.to(self.device)
                    val_losses.append(criterion(model(xb, mb), yb).item() * len(yb))
            val_loss = sum(val_losses) / len(val_loader.dataset)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        assert best_state is not None, \
            "CRITICAL: MaskedTransformer training produced no best state"
        model.load_state_dict(best_state)
        self._model = model
        return FitReport(fitted=True, epochs_run=epochs_run, best_val_loss=best_val)

    def predict(self, split: PerSensorSplit) -> np.ndarray:
        assert self._model is not None, "CRITICAL: predict() before fit()"
        loader = self._loader(split, shuffle=False)
        self._model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for xb, mb, _ in loader:
                xb = xb.to(self.device)
                mb = mb.to(self.device)
                preds.append(self._model(xb, mb).cpu().numpy())
        out = np.concatenate(preds).astype(np.float64)
        assert len(out) == len(split), \
            f"CRITICAL: prediction count {len(out)} != window count {len(split)}"
        return out

    def get_params(self) -> Dict:
        return {
            "feature_groups": list(self.feature_groups),
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
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
