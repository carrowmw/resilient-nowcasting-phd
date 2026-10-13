"""
Split arithmetic for the spatial track — the single source of truth.

Chronological 70/15/15 on the global time axis with boundary purging: the
first (window + max_horizon - 1) windows of val and of test are dropped so
that no grid step touched by a val/test window (inputs or targets) was ever
touched by a train window. This is strictly stronger than the usual
"val targets after train targets" rule and is asserted, not assumed.
"""
from typing import Dict

import numpy as np
import pandas as pd

from spatial.contract import GraphContext, GraphSplit

SPLIT_NAMES = ("train", "val", "test")


def assign_split_windows(
    T_total: int,
    window: int,
    max_horizon: int,
    stride: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_boundaries: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Window start indices per split. Window i covers inputs
    [start, start + window) and targets up to start + window - 1 + max_horizon.
    """
    assert window >= 1 and max_horizon >= 1 and stride >= 1
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1, \
        f"CRITICAL: bad split fractions train={train_frac}, val={val_frac}"
    last_start = T_total - window - max_horizon
    assert last_start >= 0, \
        f"CRITICAL: grid of {T_total} steps too short for window {window} + horizon {max_horizon}"

    starts = np.arange(0, last_start + 1, stride, dtype=np.int64)
    n = len(starts)
    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * val_frac))
    splits = {
        "train": starts[:n_train],
        "val": starts[n_train:n_train + n_val],
        "test": starts[n_train + n_val:],
    }

    if purge_boundaries:
        # Drop enough leading windows that the purged split shares no grid
        # step with the preceding split (inputs or targets).
        purge_k = int(np.ceil((window + max_horizon - 1) / stride))
        splits["val"] = splits["val"][purge_k:]
        splits["test"] = splits["test"][purge_k:]

    for name in SPLIT_NAMES:
        assert len(splits[name]) > 0, (
            f"CRITICAL: split '{name}' is empty after purge — grid too short "
            f"(T={T_total}, window={window}, max_horizon={max_horizon})"
        )

    # Offensive proof of the no-shared-steps guarantee.
    if purge_boundaries:
        for earlier, later in (("train", "val"), ("val", "test")):
            last_touched = int(splits[earlier].max()) + window - 1 + max_horizon
            first_touched = int(splits[later].min())
            assert first_touched > last_touched, (
                f"CRITICAL: purge failed — {later} first input step {first_touched} "
                f"<= {earlier} last touched step {last_touched}"
            )
    return splits


def build_graph_splits(
    context: GraphContext,
    stride: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_boundaries: bool = True,
) -> Dict[str, GraphSplit]:
    idx = assign_split_windows(
        T_total=len(context.grid_index),
        window=context.window,
        max_horizon=context.max_horizon,
        stride=stride,
        train_frac=train_frac,
        val_frac=val_frac,
        purge_boundaries=purge_boundaries,
    )
    return {name: GraphSplit(context, name, idx[name]) for name in SPLIT_NAMES}


def count_usable_targets(split: GraphSplit) -> pd.DataFrame:
    """Per (node, horizon) observed-target counts — the split's honest size."""
    obs = split.target_observed  # [n, N, H]
    rows = []
    for j, node in enumerate(split.context.node_ids):
        for k, h in enumerate(split.context.horizons):
            rows.append({
                "node_id": node,
                "horizon": int(h),
                "split": split.split,
                "n_windows": len(split),
                "n_targets_observed": int(obs[:, j, k].sum()),
            })
    return pd.DataFrame(rows)
