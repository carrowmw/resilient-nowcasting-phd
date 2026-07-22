"""
Per-sensor series loading for the quality track.

The analysis unit is a *directional* pedestrian series — one (location, dir)
pair — matching the original paper's sensor naming
(e.g. GREYSTTHEATRESOUTH_NORTHWEST_TO_SOUTHWEST). Each series is served on a
continuous 15-min grid over its own active span, with NaN marking gaps.

Note on connectors: ParquetConnector.load_traffic asserts value has no NaN,
which is the correct contract for the GNN track (pre-cleaned input) but wrong
for a missingness study. Here a NaN row in the master parquet simply means
"no observation at this grid point", so we read the parquet directly and drop
NaN-value rows before re-gridding; the database backend already returns only
recorded readings. Both paths converge on the same NaN-gapped grid.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SERIES_ID_SEP = "__"


def sensor_series_id(location: str, direction: str) -> str:
    """Canonical series identifier used in every table, manifest and figure."""
    assert location and direction, \
        f"CRITICAL: empty location/direction ({location!r}, {direction!r})"
    assert SERIES_ID_SEP not in location, \
        f"CRITICAL: location {location!r} contains reserved separator {SERIES_ID_SEP!r}"
    return f"{location}{SERIES_ID_SEP}{direction}"


def split_series_id(series_id: str) -> tuple:
    location, _, direction = series_id.partition(SERIES_ID_SEP)
    assert direction, f"CRITICAL: malformed series_id {series_id!r}"
    return location, direction


class SensorSeriesStore:
    """
    Loads the person-class traffic frame once and serves per-series
    NaN-gapped 15-min grids.

    Args:
        config: dict with keys
            data_source: 'parquet' | 'database'
            data.parquet_path (parquet source)
            data.start_date / data.end_date: analysis window (inclusive start,
                exclusive end)
            data.veh_class: e.g. 'person'
            data.freq: grid frequency, e.g. '15min'
        connector: required for data_source='database'
            (src/db_connector.py DataConnector).
    """

    def __init__(self, config: dict, connector=None):
        data_cfg = config["data"]
        self.freq: str = data_cfg["freq"]
        self.veh_class: str = data_cfg["veh_class"]
        self.start_dt = pd.Timestamp(data_cfg["start_date"])
        self.end_dt = pd.Timestamp(data_cfg["end_date"])
        assert self.start_dt < self.end_dt, \
            f"CRITICAL: start_date {self.start_dt} must precede end_date {self.end_dt}"

        source = config["data_source"]
        if source == "parquet":
            frame = self._load_parquet(data_cfg["parquet_path"])
        elif source == "database":
            assert connector is not None, \
                "CRITICAL: data_source='database' requires a DataConnector"
            frame = connector.load_traffic(self.start_dt, self.end_dt, self.veh_class, "all")
            frame = frame.dropna(subset=["value"])
        else:
            raise ValueError(f"CRITICAL: unknown data_source {source!r}")

        assert not frame.empty, "CRITICAL: no observations in the analysis window"
        frame["dt"] = pd.to_datetime(frame["dt"])
        # Physical impossibility tripwire, mirroring PreprocessingPipeline.
        negatives = int((frame["value"] < 0).sum())
        assert negatives == 0, \
            f"CRITICAL: {negatives} negative counts reached SensorSeriesStore — upstream must NaN them"

        frame["series_id"] = [
            sensor_series_id(loc, d) for loc, d in zip(frame["location"], frame["dir"])
        ]
        self._frame = frame[["series_id", "dt", "value"]]
        dup = self._frame.duplicated(subset=["series_id", "dt"]).sum()
        assert dup == 0, f"CRITICAL: {dup} duplicate (series_id, dt) observations"

        self._series_ids: List[str] = sorted(self._frame["series_id"].unique())
        self._cache: Dict[str, pd.Series] = {}
        logger.info(
            f"SensorSeriesStore: {len(self._frame):,} observations, "
            f"{len(self._series_ids)} directional series, "
            f"window {self.start_dt.date()} → {self.end_dt.date()}"
        )

    def _load_parquet(self, parquet_path: str) -> pd.DataFrame:
        assert parquet_path, "CRITICAL: data.parquet_path missing for parquet source"
        df = pd.read_parquet(
            parquet_path,
            columns=["dt", "location", "value", "veh_class", "dir"],
            filters=[
                ("dt", ">=", self.start_dt),
                ("dt", "<", self.end_dt),
            ],
        )
        df = df[df["veh_class"] == self.veh_class]
        # NaN value rows are grid artefacts of resample_traffic == "no observation".
        return df.dropna(subset=["value"]).drop(columns=["veh_class"])

    def series_ids(self) -> List[str]:
        return list(self._series_ids)

    def get_series(self, series_id: str) -> pd.Series:
        """
        Return the series on a continuous grid over its own active span
        (first to last observation), NaN at gaps. Cached.
        """
        if series_id in self._cache:
            return self._cache[series_id]
        assert series_id in self._series_ids, \
            f"CRITICAL: unknown series_id {series_id!r}"
        obs = self._frame[self._frame["series_id"] == series_id]
        obs = obs.set_index("dt")["value"].sort_index()
        grid = pd.date_range(obs.index.min(), obs.index.max(), freq=self.freq, name="dt")
        aligned = obs.reindex(grid)
        # Every observation must land exactly on the grid — a lost observation
        # here means the source is not on the expected frequency.
        n_lost = len(obs) - aligned.notna().sum()
        assert n_lost == 0, (
            f"CRITICAL: {n_lost} observations of {series_id} do not align to the "
            f"{self.freq} grid — source data is not gridded at this frequency"
        )
        aligned.name = series_id
        self._cache[series_id] = aligned
        return aligned

    def drop_from_cache(self, series_id: str) -> None:
        """Free a cached series (the runner iterates sensors lazily)."""
        self._cache.pop(series_id, None)

    def expected_steps_in_window(self) -> int:
        """Grid steps in the configured analysis window (for window completeness)."""
        return len(pd.date_range(self.start_dt, self.end_dt, freq=self.freq, inclusive="left"))
