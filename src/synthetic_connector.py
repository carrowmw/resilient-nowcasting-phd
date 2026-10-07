"""
SyntheticConnector — a DataConnector that fabricates a small, *structurally*
realistic city for --use_mock_data runs and tests.

It mirrors the properties of the real feed that drive zero-collapse
(docs/design_reports/zero_collapse_investigation.md §1.1):
  - the source never emits zero-count rows (zeros only arise from missingness);
  - heavy-tailed diurnal/weekly counts with node-specific levels;
  - MNAR outages: short blips, multi-day outages, and night-time dropouts;
  - per-sensor missingness ranging from ~10% to ~90%.

Mock runs therefore exercise the full RealIoTDataset path (scalers, v5 tables,
masks, covariates) instead of a random-tensor shortcut.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from db_connector import DataConnector, STATIC_FEATURE_COLS


class SyntheticConnector(DataConnector):
    def __init__(self, num_nodes: int, start_date: str, days: int, seed: int = 0):
        assert num_nodes > 1 and days > 2
        rng = np.random.RandomState(seed)
        self.locations = [f"syn_{i:03d}" for i in range(num_nodes)]
        self.times = pd.date_range(pd.Timestamp(start_date), periods=days * 96, freq="15min")
        t, n = len(self.times), num_nodes

        hour = self.times.hour.values + self.times.minute.values / 60.0
        diurnal = np.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2) + 1.3 * np.exp(-0.5 * ((hour - 13.0) / 3.0) ** 2) \
            + np.exp(-0.5 * ((hour - 17.5) / 1.5) ** 2) + 0.02
        weekend = np.where(self.times.dayofweek.values >= 5, 0.6, 1.0)
        level = np.exp(rng.normal(2.5, 1.0, size=n))
        lam = level[None, :] * (diurnal * weekend)[:, None]
        theta = 2.0
        counts = rng.negative_binomial(theta, theta / (theta + lam)).astype(np.float64)

        observed = np.ones((t, n), dtype=bool)
        for j in range(n):
            reliability = rng.uniform(0.1, 0.9)
            for _ in range(rng.poisson(6 * (1 - reliability) * days / 7)):        # long outages
                a = rng.randint(0, t)
                observed[a:a + int(rng.exponential(96 * 1.5)) + 1, j] = False
            for _ in range(rng.poisson(20 * days / 7)):                            # short blips
                a = rng.randint(0, t)
                observed[a:a + rng.randint(1, 5), j] = False
        night = (self.times.hour.values < 5)[:, None]
        observed &= ~(night & (rng.rand(t, n) < 0.3))                              # night dropouts
        observed &= counts > 0                                                     # feed never emits zeros
        # The real feed is a regular grid (some sensor reports every slot); keep that.
        empty = np.where(~observed.any(axis=1))[0]
        fill_node = rng.randint(0, n, size=len(empty))
        counts[empty, fill_node] = np.maximum(counts[empty, fill_node], 1.0)
        observed[empty, fill_node] = True
        self._counts, self._observed = counts, observed

        self._static = pd.DataFrame(
            np.abs(rng.normal(1.0, 0.5, size=(n, len(STATIC_FEATURE_COLS)))), columns=STATIC_FEATURE_COLS
        )
        self._static["is_school_zone"] = (rng.rand(n) < 0.3).astype(float)
        self._static.insert(0, "location", self.locations)

        humidity = 70 + 15 * np.sin(2 * np.pi * np.arange(t) / (96 * 3))[:, None] + rng.normal(0, 2, (t, n))
        solar = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)[:, None] * 500 + rng.normal(0, 5, (t, n))
        self._humidity, self._solar = humidity, solar

    def _window(self, start_dt, end_dt):
        sel = (self.times >= pd.Timestamp(start_dt)) & (self.times < pd.Timestamp(end_dt))
        return np.where(sel)[0]

    def load_traffic(self, start_dt, end_dt, veh_class: str, direction: str) -> pd.DataFrame:
        idx = self._window(start_dt, end_dt)
        ti, ni = np.nonzero(self._observed[idx])
        df = pd.DataFrame({
            "dt": self.times[idx][ti],
            "location": np.array(self.locations)[ni],
            "value": self._counts[idx][ti, ni],
            "veh_class": veh_class,
            "dir": "in",
        })
        return df.sort_values(["dt", "location"]).reset_index(drop=True)

    def load_weather(self, start_dt, end_dt, sensor_ids: List[str]) -> pd.DataFrame:
        idx = self._window(start_dt, end_dt)
        cols = [self.locations.index(s) for s in sensor_ids]
        grid_t, grid_n = np.meshgrid(idx, cols, indexing="ij")
        return pd.DataFrame({
            "dt": self.times[grid_t.ravel()],
            "location": np.array(self.locations)[grid_n.ravel()],
            "Humidity": self._humidity[grid_t.ravel(), grid_n.ravel()],
            "Solar Radiation": self._solar[grid_t.ravel(), grid_n.ravel()],
        })

    def load_static_features(self, sensor_ids: List[str]) -> pd.DataFrame:
        return self._static[self._static["location"].isin(sensor_ids)].reset_index(drop=True)

    def load_sensor_metadata(self) -> Dict[str, Dict[str, float]]:
        return {loc: {"lat": 54.97 + 0.001 * i, "lon": -1.61 + 0.001 * i} for i, loc in enumerate(self.locations)}

    def close(self):
        pass
