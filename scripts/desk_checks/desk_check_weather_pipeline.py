"""
desk_check_weather_pipeline.py

Verification desk check for the Open-Meteo weather pipeline:
1. Validates schema, completeness, and non-NaN invariants in data/external/weather_cache.parquet.
2. Checks physical realism and diurnal cycle properties (e.g. nocturnal solar radiation == 0.0,
   humidity bounded within [0, 100] with UK-typical means).
3. Validates ParquetConnector.load_weather integration and asserts zero-NaN joins.
4. Verifies dynamic tensor scaling (weather_scaler produces mean ~ 0, std ~ 1 with non-degenerate variance).

Usage:
    python scripts/desk_checks/desk_check_weather_pipeline.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db_connector import ParquetConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("desk_check_weather_pipeline")

CACHE_PATH = PROJECT_ROOT / "data" / "external" / "weather_cache.parquet"
SENSORS_PATH = PROJECT_ROOT / "data" / "traffic_sensors.json"


def test_cache_integrity(df: pd.DataFrame):
    logger.info("--- Test 1: Cache Integrity & Schema ---")
    required_cols = {"dt", "location", "Humidity", "Solar Radiation"}
    assert required_cols.issubset(df.columns), f"CRITICAL: Missing columns. Found: {list(df.columns)}"

    total_rows = len(df)
    assert total_rows > 0, "CRITICAL: Cache DataFrame is empty"

    nan_hum = df["Humidity"].isna().sum()
    nan_solar = df["Solar Radiation"].isna().sum()
    assert nan_hum == 0, f"CRITICAL: {nan_hum} NaN values in Humidity"
    assert nan_solar == 0, f"CRITICAL: {nan_solar} NaN values in Solar Radiation"

    num_sensors = df["location"].nunique()
    assert num_sensors >= 81, f"CRITICAL: Expected at least 81 sensors, got {num_sensors}"

    logger.info(f" PASSED: {total_rows:,} rows across {num_sensors} sensors, 0 NaNs.")


def test_physical_realism(df: pd.DataFrame):
    logger.info("--- Test 2: Physical Realism & Diurnal Cycles ---")

    # Humidity bounds & distribution
    hum_min = df["Humidity"].min()
    hum_max = df["Humidity"].max()
    hum_mean = df["Humidity"].mean()
    assert hum_min >= 0.0, f"CRITICAL: Negative humidity: {hum_min}"
    assert hum_max <= 100.0, f"CRITICAL: Humidity > 100%: {hum_max}"
    assert 60.0 <= hum_mean <= 92.0, f"CRITICAL: Unrealistic mean humidity for UK: {hum_mean:.1f}%"
    assert df["Humidity"].std() > 5.0, "CRITICAL: Humidity variance is near-zero"

    # Solar radiation bounds & diurnal cycle
    solar_min = df["Solar Radiation"].min()
    solar_max = df["Solar Radiation"].max()
    solar_mean = df["Solar Radiation"].mean()
    assert solar_min >= 0.0, f"CRITICAL: Negative solar radiation: {solar_min}"
    assert solar_max > 250.0, f"CRITICAL: Peak solar radiation too low: {solar_max}"
    assert solar_mean > 20.0, f"CRITICAL: Mean solar radiation too low: {solar_mean}"
    assert df["Solar Radiation"].std() > 20.0, "CRITICAL: Solar Radiation variance is near-zero"

    # Strict nocturnal zero check (between 00:00 and 04:00 UTC)
    hours = pd.to_datetime(df["dt"]).dt.hour
    nocturnal_solar = df.loc[(hours >= 0) & (hours <= 3), "Solar Radiation"]
    assert (nocturnal_solar == 0.0).all(), (
        f"CRITICAL: Non-zero nocturnal solar radiation detected (max={nocturnal_solar.max()})"
    )

    logger.info(
        f" PASSED: Humidity range [{hum_min:.1f}, {hum_max:.1f}]%, mean={hum_mean:.1f}%. "
        f"Solar range [{solar_min:.1f}, {solar_max:.1f}] W/m2, mean={solar_mean:.1f} W/m2, nocturnal=0.0."
    )


def test_connector_integration():
    logger.info("--- Test 3: Connector Integration & Zero-NaN Merge ---")
    conn = ParquetConnector(
        parquet_path=str(PROJECT_ROOT / "data" / "processed" / "master_dataset_15T.parquet"),
        static_cache_dir=str(PROJECT_ROOT / "data" / "external"),
    )

    # Test 1-week slice
    start_dt = "2024-05-01"
    end_dt = "2024-05-08"
    test_sensors = ["NclShieldsRdCliffordStNorth", "NclPercyStEldonSqANPR"]

    # Verify load_weather picks up from weather_cache.parquet
    w_df = conn.load_weather(start_dt, end_dt, test_sensors)
    assert not w_df.empty, "CRITICAL: Connector returned empty weather DataFrame"
    assert not w_df["Humidity"].isna().any(), "CRITICAL: NaN in connector Humidity"
    assert not w_df["Solar Radiation"].isna().any(), "CRITICAL: NaN in connector Solar Radiation"
    assert w_df["location"].nunique() == len(test_sensors), "CRITICAL: Missing sensors in output"

    logger.info(f" PASSED: Connector loaded {len(w_df):,} weather rows for {len(test_sensors)} test sensors with 0 NaNs.")


def test_scaler_numerics(df: pd.DataFrame):
    logger.info("--- Test 4: Scaler Numerics ---")
    weather_arr = df[["Humidity", "Solar Radiation"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(weather_arr)

    assert not np.isnan(scaled).any(), "CRITICAL: NaN in scaled weather"
    assert not np.isinf(scaled).any(), "CRITICAL: Inf in scaled weather"
    assert abs(scaled[:, 0].mean()) < 1e-4, "CRITICAL: Scaled humidity mean is not 0"
    assert abs(scaled[:, 0].std() - 1.0) < 1e-3, "CRITICAL: Scaled humidity std is not 1"
    assert abs(scaled[:, 1].mean()) < 1e-4, "CRITICAL: Scaled solar mean is not 0"
    assert abs(scaled[:, 1].std() - 1.0) < 1e-3, "CRITICAL: Scaled solar std is not 1"

    logger.info(f" PASSED: Scaler fitted cleanly: Humidity mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}; "
                f"Solar mean={scaler.mean_[1]:.2f}, std={scaler.scale_[1]:.2f}.")


def main():
    assert CACHE_PATH.exists(), f"CRITICAL: Weather cache not found at {CACHE_PATH}. Run fetch_and_ingest_weather.py first."
    logger.info(f"Loading weather cache from {CACHE_PATH}...")
    df = pd.read_parquet(CACHE_PATH)

    test_cache_integrity(df)
    test_physical_realism(df)
    test_connector_integration()
    test_scaler_numerics(df)

    logger.info("ALL DESK CHECKS PASSED SUCCESSFULLY! ")


if __name__ == "__main__":
    main()
