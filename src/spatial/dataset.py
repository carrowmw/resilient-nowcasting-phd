"""
PyTorch dataset over GraphSplits, plus the masked loss every trainable model
shares.

Input tensor layout per window: x [N, T, D] with channels
  0                : scaled count, NaN → 0 (dead values are flagged by the mask)
  1 (optional)     : missingness mask channel (1.0 where the count was observed)
  2.. (optional)   : cyclical time encodings (sin/cos pairs), identical across
                     nodes — computed once over the grid and sliced per window.

D is derived from the config, never hardcoded:
  D = 1 + int(mask_channel) + 2 * len(time_features).
"""
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from spatial.contract import GraphSplit

TIME_FEATURE_PERIODS = {
    "half_day": 12.0,          # hours
    "day": 24.0,               # hours
    "week": 24.0 * 7.0,        # hours
}


def input_dim(mask_channel: bool, time_features: Sequence[str]) -> int:
    unknown = set(time_features) - set(TIME_FEATURE_PERIODS)
    assert not unknown, f"CRITICAL: unknown time features {unknown}"
    return 1 + int(mask_channel) + 2 * len(time_features)


def build_time_features(grid_index: pd.DatetimeIndex,
                        time_features: Sequence[str]) -> np.ndarray:
    """[T_total, 2*len(groups)] float32 sin/cos encodings (fractional hour)."""
    unknown = set(time_features) - set(TIME_FEATURE_PERIODS)
    assert not unknown, f"CRITICAL: unknown time features {unknown}"
    hours = (
        grid_index.hour.to_numpy(dtype=np.float64)
        + grid_index.minute.to_numpy(dtype=np.float64) / 60.0
        + grid_index.dayofweek.to_numpy(dtype=np.float64) * 24.0
    )
    cols: List[np.ndarray] = []
    for group in time_features:
        period = TIME_FEATURE_PERIODS[group]
        phase = 2.0 * np.pi * (hours % period) / period
        cols.append(np.sin(phase))
        cols.append(np.cos(phase))
    if not cols:
        return np.zeros((len(grid_index), 0), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)


class GraphWindowDataset(Dataset):
    """Yields dict(x [N,T,D], y [N,H], y_mask [N,H]) per window."""

    def __init__(self, split: GraphSplit, mask_channel: bool = True,
                 time_features: Sequence[str] = ("day", "week")):
        self.split = split
        self.mask_channel = bool(mask_channel)
        self.time_features = tuple(time_features)
        ctx = split.context
        # [T_total, N] tensors shared across windows; NaN → 0 with mask kept.
        scaled = ctx.values_scaled.copy()
        self._observed = ctx.observed_mask
        scaled[~self._observed] = 0.0
        assert np.isfinite(scaled).all(), "CRITICAL: non-finite scaled inputs"
        self._scaled = scaled
        self._time = build_time_features(ctx.grid_index, self.time_features)
        self.n_channels = input_dim(self.mask_channel, self.time_features)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ctx = self.split.context
        start = int(self.split.window_start_idx[i])
        sl = slice(start, start + ctx.window)

        count = self._scaled[sl].T                       # [N, T]
        chans = [count[:, :, None]]                      # [N, T, 1]
        if self.mask_channel:
            chans.append(self._observed[sl].T[:, :, None].astype(np.float32))
        if self._time.shape[1]:
            t = self._time[sl]                           # [T, k]
            chans.append(np.broadcast_to(t[None], (ctx.n_nodes, *t.shape)))
        x = np.concatenate(chans, axis=2).astype(np.float32)

        target_idx = start + ctx.window - 1 + np.asarray(ctx.horizons)
        y = ctx.values_scaled[target_idx].T.copy()       # [N, H]
        y_mask = ctx.observed_mask[target_idx].T.copy()  # [N, H]
        y[~y_mask] = 0.0                                 # dead values; see y_mask

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y.astype(np.float32)),
            "y_mask": torch.from_numpy(y_mask),
        }


def collate_graph(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stacks to x [B,N,T,D], y [B,N,H], y_mask [B,N,H]."""
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def masked_mse(pred: torch.Tensor, y: torch.Tensor,
               y_mask: torch.Tensor) -> torch.Tensor:
    """
    MSE over observed target entries only. An all-masked batch is a data bug
    upstream (the split reported usable targets), so it asserts.
    """
    assert pred.shape == y.shape == y_mask.shape, \
        f"CRITICAL: masked_mse shape mismatch {pred.shape}/{y.shape}/{y_mask.shape}"
    n_obs = y_mask.sum()
    assert int(n_obs) > 0, "CRITICAL: masked_mse got a batch with zero observed targets"
    sq = (pred - y) ** 2
    loss = (sq * y_mask).sum() / n_obs
    assert torch.isfinite(loss), "CRITICAL: non-finite masked_mse loss"
    return loss
