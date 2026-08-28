"""
MaskedWindowDataset for the missingness-aware Transformer (Option B).

Unlike the standard WindowDataset which only extracts windows from contiguous
sequences (no NaNs), this dataset extracts fixed-length windows from the full
grid *including* NaN gaps. It augments the features with:
  1. A binary missingness indicator (1.0 where the value is NaN, 0.0 otherwise)
  2. A time-delta feature (steps since last observed value, normalized)

NaN values in the feature matrix are replaced with 0.0 so the tensor is valid,
and the missingness mask is provided separately for the Transformer's
src_key_padding_mask.

The target (scalar at origin + horizon) must still be non-NaN — windows where
the target is missing are excluded during split construction.
"""
import logging
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from quality.contract import PerSensorSplit
from quality.per_sensor_dataset import build_feature_matrix

logger = logging.getLogger(__name__)


class MaskedWindowDataset(Dataset):
    """
    Lazy torch dataset for the missingness-aware Transformer.

    Returns (features, mask, target) tuples where:
      - features: [T, D+2] float32 — original features (NaN→0) + missingness
        indicator + time-delta
      - mask: [T] bool — True where the timestep is missing
      - target: float32 scalar
    """

    def __init__(
        self,
        split: PerSensorSplit,
        feature_groups: Sequence[str],
        fractional_time: bool = True,
    ):
        assert len(split) > 0, \
            f"CRITICAL: empty {split.split} split for {split.context.sensor_id}"

        # Build the standard feature matrix [T_total, D]
        self._features = build_feature_matrix(
            split.context, feature_groups, fractional_time
        )
        self._starts = split.window_start_idx
        self._window = split.context.window
        self._targets = split.y_scaled.astype(np.float32)

        # Build the missingness mask over the full grid from the raw values
        # NaN in values_raw indicates a missing timestep
        self._is_missing = np.isnan(split.context.values_raw)  # [T_total] bool

        # Pre-compute time-delta array: steps since last observed value
        self._time_delta = self._compute_time_delta(self._is_missing)

    @staticmethod
    def _compute_time_delta(is_missing: np.ndarray) -> np.ndarray:
        """Compute normalized time-delta (steps since last observation).

        For observed timesteps, delta = 0.
        For missing timesteps, delta = number of steps since last observation,
        normalized by dividing by 24 (one full window length) for scale.
        """
        T = len(is_missing)
        delta = np.zeros(T, dtype=np.float32)
        last_observed = -1
        for t in range(T):
            if not is_missing[t]:
                last_observed = t
                delta[t] = 0.0
            else:
                if last_observed >= 0:
                    delta[t] = float(t - last_observed) / 24.0
                else:
                    # No prior observation — use a large sentinel
                    delta[t] = 1.0
        return delta

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self._starts[i]
        end = s + self._window

        # Extract window features and replace NaN with 0.0
        feats = self._features[s:end].copy()  # [T, D]
        nan_mask = np.isnan(feats)
        feats[nan_mask] = 0.0

        # Missingness indicator for this window [T, 1]
        missing = self._is_missing[s:end].astype(np.float32).reshape(-1, 1)

        # Time-delta for this window [T, 1]
        td = self._time_delta[s:end].reshape(-1, 1)

        # Concatenate: [T, D+2]
        augmented = np.concatenate([feats, missing, td], axis=1).astype(np.float32)

        # Boolean mask for src_key_padding_mask [T]
        mask = self._is_missing[s:end]

        return (
            torch.from_numpy(augmented),
            torch.from_numpy(mask.copy()),
            torch.tensor(self._targets[i]),
        )
