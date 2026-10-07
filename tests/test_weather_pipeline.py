"""
test_weather_pipeline.py

Unit tests for Open-Meteo weather fetch, 15-min interpolation, and ParquetConnector integration.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ingestion"))

from db_connector import ParquetConnector
from fetch_and_ingest_weather import interpolate_to_15min


def test_interpolate_to_15min_expansion_and_bounds():
    # 24 hours of dummy data
    times = [f"2024-05-01T{h:02d}:00" for h in range(24)]
    # Intentionally test boundary conditions: negative values and values > 100
    humidity = [105.0 if h == 12 else 50.0 for h in range(24)]
    solar = [-10.0 if h == 2 else (500.0 if h == 12 else 0.0) for h in range(24)]

    df_15 = interpolate_to_15min(times, humidity, solar, "2024-05-01", "2024-05-01")

    # 24 hours * 4 steps/hour = 96 steps
    assert len(df_15) == 96
    assert not df_15["Humidity"].isna().any()
    assert not df_15["Solar Radiation"].isna().any()

    # Humidity must be strictly clamped to <= 100.0 and >= 0.0
    assert df_15["Humidity"].max() <= 100.0
    assert df_15["Humidity"].min() >= 0.0

    # Solar radiation must be non-negative
    assert df_15["Solar Radiation"].min() >= 0.0

    # Deep night hours (00:00 - 04:00) must have exactly 0.0 solar radiation
    night_steps = df_15[df_15["dt"].dt.hour.isin([0, 1, 2, 3])]
    assert (night_steps["Solar Radiation"] == 0.0).all()


def test_parquet_connector_weather_cache_integration():
    cache_path = PROJECT_ROOT / "data" / "external" / "weather_cache.parquet"
    if not cache_path.exists():
        pytest.skip("weather_cache.parquet not found on disk")

    conn = ParquetConnector(
        parquet_path=str(PROJECT_ROOT / "data" / "processed" / "master_dataset_15T.parquet"),
        static_cache_dir=str(PROJECT_ROOT / "data" / "external"),
    )

    test_sensors = ["NclShieldsRdCliffordStNorth", "NclPercyStEldonSqANPR"]
    start_dt = "2024-05-01"
    end_dt = "2024-05-03"

    df_w = conn.load_weather(start_dt, end_dt, test_sensors)
    assert not df_w.empty
    assert set(["dt", "location", "Humidity", "Solar Radiation"]).issubset(df_w.columns)
    assert not df_w["Humidity"].isna().any()
    assert not df_w["Solar Radiation"].isna().any()
    assert df_w["location"].nunique() == len(test_sensors)
    assert (df_w["Humidity"] > 0).any()
    assert (df_w["Solar Radiation"] >= 0).all()
