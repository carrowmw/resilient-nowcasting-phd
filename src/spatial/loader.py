"""
Graph-structured data loading for the spatial track.

GraphSeriesStore loads the master parquet (or a DataConnector), pivots to a
[T_total, N] grid, selects the node set (completeness threshold ∩ edge-cache
coverage — dropped nodes are recorded loudly, never silently), fits per-node
train-only scalers, and produces the GraphContext every model shares.

Node = location; directions are summed (matches the legacy GAT work and
RealIoTDataset(target_dir='all')). Real sensors only — no simulated nodes
(those exist for the Exp 3 kriging question).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataset import STATIC_FEATURE_COLS  # src/dataset.py — the 15-dim order contract
from spatial.contract import GraphContext, STATIC_FEATURE_DIM
from spatial.splits import assign_split_windows

logger = logging.getLogger(__name__)

assert len(STATIC_FEATURE_COLS) == STATIC_FEATURE_DIM, \
    "CRITICAL: STATIC_FEATURE_COLS drifted from the contract's 15-dim promise"


class GraphSeriesStore:
    """
    Loads observations once, builds the GraphContext once. Config keys:

    data_source: 'parquet' | 'database'
    data: {parquet_path, start_date, end_date, veh_class, direction, freq}
    nodes: {min_completeness, edge_features_cache, static_features_cache, min_nodes}
    temporal: {window, horizons, stride, train_frac, val_frac, purge_boundaries}
    """

    def __init__(self, config: dict, connector=None):
        data = config["data"]
        self.start_dt = pd.Timestamp(data["start_date"])
        self.end_dt = pd.Timestamp(data["end_date"])
        assert self.start_dt < self.end_dt, "CRITICAL: start_date >= end_date"
        self.veh_class = data["veh_class"]
        self.freq = data["freq"]
        assert data["direction"] == "all", \
            "CRITICAL: only direction='all' (summed per location) is supported"

        self.nodes_cfg = config["nodes"]
        self.temporal_cfg = config["temporal"]

        source = config["data_source"]
        if source == "parquet":
            frame = self._load_parquet(data["parquet_path"])
        elif source == "database":
            assert connector is not None, \
                "CRITICAL: data_source='database' requires a DataConnector"
            frame = connector.fetch_traffic_data(
                start_date=str(self.start_dt), end_date=str(self.end_dt)
            )
            frame = frame[frame["veh_class"] == self.veh_class]
            frame = frame.dropna(subset=["value"])[["dt", "location", "value"]]
        else:
            raise ValueError(f"CRITICAL: unknown data_source {source!r}")

        frame["dt"] = pd.to_datetime(frame["dt"])
        negatives = int((frame["value"] < 0).sum())
        assert negatives == 0, \
            f"CRITICAL: {negatives} negative counts reached GraphSeriesStore"

        # Sum directions per (location, dt). Duplicates after this are a bug.
        frame = frame.groupby(["location", "dt"], as_index=False)["value"].sum()
        dup = frame.duplicated(subset=["location", "dt"]).sum()
        assert dup == 0, f"CRITICAL: {dup} duplicate (location, dt) after direction sum"
        self._frame = frame
        self._node_report: Optional[pd.DataFrame] = None
        logger.info(
            f"GraphSeriesStore: {len(frame):,} location-summed observations, "
            f"{frame['location'].nunique()} candidate locations, "
            f"{self.start_dt.date()} → {self.end_dt.date()}"
        )

    def _load_parquet(self, parquet_path: str) -> pd.DataFrame:
        assert parquet_path, "CRITICAL: data.parquet_path missing for parquet source"
        df = pd.read_parquet(
            parquet_path,
            columns=["dt", "location", "value", "veh_class"],
            filters=[("dt", ">=", self.start_dt), ("dt", "<", self.end_dt)],
        )
        df = df[df["veh_class"] == self.veh_class]
        # NaN value rows are grid artefacts of resample == "no observation".
        return df.dropna(subset=["value"]).drop(columns=["veh_class"])

    # ------------------------------------------------------------- node set

    def _select_nodes(self, pivot: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """Completeness threshold ∩ edge-cache coverage, loudly reported."""
        min_completeness = float(self.nodes_cfg["min_completeness"])
        cache_path = Path(self.nodes_cfg["edge_features_cache"])
        assert cache_path.exists(), \
            f"CRITICAL: edge features cache not found: {cache_path}"
        with open(cache_path) as f:
            cache = json.load(f)

        completeness = pivot.notna().mean(axis=0)
        rows = []
        selected = []
        for node in pivot.columns:
            comp = float(completeness[node])
            in_cache = node in cache
            if comp < min_completeness:
                reason = f"completeness {comp:.3f} < {min_completeness}"
            elif not in_cache:
                reason = "absent from edge_features_cache"
            else:
                reason = ""
                selected.append(node)
            rows.append({
                "node_id": node, "completeness": comp, "in_edge_cache": in_cache,
                "included": reason == "", "exclusion_reason": reason,
            })
        report = pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)

        # Among the selected set, every directed pair must exist in the cache.
        # A missing pair here is cache corruption, not a selection issue → assert.
        for src in selected:
            for dst in selected:
                if src == dst:
                    continue
                assert dst in cache[src], (
                    f"CRITICAL: edge cache missing directed pair ({src} → {dst}) "
                    f"— regenerate the cache, do not weaken this assert"
                )

        min_nodes = int(self.nodes_cfg["min_nodes"])
        assert len(selected) >= min_nodes, (
            f"CRITICAL: only {len(selected)} nodes survive selection "
            f"(min_nodes={min_nodes}) — see node report"
        )
        return sorted(selected), report

    # ------------------------------------------------------------ context

    def build_context(self) -> GraphContext:
        grid = pd.date_range(self.start_dt, self.end_dt, freq=self.freq,
                             inclusive="left", name="dt")
        pivot = self._frame.pivot(index="dt", columns="location", values="value")
        pivot = pivot.reindex(grid)
        n_lost = len(self._frame) - int(pivot.notna().sum().sum())
        assert n_lost == 0, (
            f"CRITICAL: {n_lost} observations do not align to the {self.freq} grid"
        )

        node_ids, report = self._select_nodes(pivot)
        self._node_report = report
        pivot = pivot[node_ids]

        values_raw = pivot.to_numpy(dtype=np.float32)          # [T, N]
        observed = np.isfinite(values_raw)

        # Per-node scaler fit ONLY on grid steps touched by train windows
        # (quality-track leakage fix, ported to the shared time axis).
        t_cfg = self.temporal_cfg
        window = int(t_cfg["window"])
        horizons = tuple(int(h) for h in t_cfg["horizons"])
        split_idx = assign_split_windows(
            T_total=len(grid), window=window, max_horizon=max(horizons),
            stride=int(t_cfg.get("stride", 1)),
            train_frac=float(t_cfg["train_frac"]),
            val_frac=float(t_cfg["val_frac"]),
            purge_boundaries=bool(t_cfg["purge_boundaries"]),
        )
        train_last_touched = int(split_idx["train"].max()) + window - 1 + max(horizons)
        train_step_mask = np.zeros(len(grid), dtype=bool)
        train_step_mask[: train_last_touched + 1] = True

        scaler_mean = np.empty(len(node_ids), dtype=np.float64)
        scaler_std = np.empty(len(node_ids), dtype=np.float64)
        degenerate = []
        for j, node in enumerate(node_ids):
            vals = values_raw[train_step_mask, j]
            vals = vals[np.isfinite(vals)]
            assert len(vals) > 0, \
                f"CRITICAL: node {node} has zero train-period observations"
            scaler_mean[j] = float(vals.mean())
            std = float(vals.std())
            if std < 1e-8:
                degenerate.append(node)
                std = 1.0
            scaler_std[j] = std
        if degenerate:
            logger.warning(
                f"{len(degenerate)} nodes with ~zero train variance use std=1.0: "
                f"{degenerate}"
            )

        values_scaled = (values_raw - scaler_mean[None, :].astype(np.float32)) \
            / scaler_std[None, :].astype(np.float32)

        static = self._load_static_features(node_ids)

        return GraphContext(
            node_ids=node_ids,
            grid_index=pd.DatetimeIndex(grid),
            values_raw=values_raw,
            values_scaled=values_scaled.astype(np.float32),
            observed_mask=observed,
            static_features=static,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            window=window,
            horizons=horizons,
            freq=self.freq,
        )

    def _load_static_features(self, node_ids: List[str]) -> np.ndarray:
        """
        [N, 15] static features, standardised ACROSS NODES (they are constant
        in time; cross-node standardisation is the legacy Test-6 gradient-
        explosion fix, institutionalised).
        """
        path = self.nodes_cfg["static_features_cache"]
        cols = ["location"] + list(STATIC_FEATURE_COLS)
        df = pd.read_parquet(path, columns=cols)
        static = df.groupby("location").first()
        missing = [n for n in node_ids if n not in static.index]
        assert not missing, \
            f"CRITICAL: static feature cache missing nodes: {missing}"
        mat = static.loc[node_ids, list(STATIC_FEATURE_COLS)].to_numpy(dtype=np.float64)
        assert np.isfinite(mat).all(), "CRITICAL: non-finite static features"
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std[std < 1e-8] = 1.0  # constant columns (e.g. all-zero POI class) pass through
        return ((mat - mean) / std).astype(np.float32)

    def node_report(self) -> pd.DataFrame:
        assert self._node_report is not None, \
            "CRITICAL: node_report requested before build_context()"
        return self._node_report.copy()
