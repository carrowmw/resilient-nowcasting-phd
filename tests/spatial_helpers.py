"""Shared toy fixtures for the spatial-track tests (imported by test modules)."""
import numpy as np
import pandas as pd

from spatial.contract import GraphContext, STATIC_FEATURE_DIM


def make_context(
    n_nodes: int = 6,
    T_total: int = 400,
    window: int = 8,
    horizons=(1, 2),
    missing_frac: float = 0.1,
    seed: int = 0,
) -> GraphContext:
    """Small synthetic GraphContext: sinusoidal daily signal + node offsets + noise."""
    rng = np.random.default_rng(seed)
    grid = pd.date_range("2024-01-01", periods=T_total, freq="15min", name="dt")
    hours = grid.hour.to_numpy() + grid.minute.to_numpy() / 60.0
    base = 10.0 + 8.0 * np.sin(2 * np.pi * hours / 24.0)             # [T]
    offsets = rng.uniform(0.5, 3.0, n_nodes)                          # per-node scale
    raw = base[:, None] * offsets[None, :] + rng.normal(0, 1.0, (T_total, n_nodes))
    raw = np.clip(raw, 0.0, None).astype(np.float32)

    observed = rng.random((T_total, n_nodes)) >= missing_frac
    raw[~observed] = np.nan

    # Train-only scaler over the leading 70% of steps (close enough for tests;
    # loader tests verify the real train-window logic separately).
    cut = int(T_total * 0.7)
    mean = np.nanmean(raw[:cut], axis=0).astype(np.float64)
    std = np.nanstd(raw[:cut], axis=0).astype(np.float64)
    std[std < 1e-8] = 1.0
    scaled = ((raw - mean[None, :]) / std[None, :]).astype(np.float32)

    node_ids = sorted(f"node_{chr(97 + i)}" for i in range(n_nodes))
    static = rng.normal(0, 1, (n_nodes, STATIC_FEATURE_DIM)).astype(np.float32)

    return GraphContext(
        node_ids=node_ids,
        grid_index=pd.DatetimeIndex(grid),
        values_raw=raw,
        values_scaled=scaled,
        observed_mask=observed,
        static_features=static,
        scaler_mean=mean,
        scaler_std=std,
        window=window,
        horizons=tuple(horizons),
        freq="15min",
    )
