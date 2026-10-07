"""
fetch_and_ingest_weather.py

Fetches historical reanalysis weather data (relative humidity and shortwave solar radiation)
from the Open-Meteo Historical Weather API for sensor coordinates in data/traffic_sensors.json,
resamples and interpolates hourly readings to a regular 15-minute grid, enforces physical bounds,
persists the resulting dataset to a local Parquet cache (data/external/weather_cache.parquet),
and optionally ingests into the TimescaleDB weather_readings hypertable.

Features:
- Atomic per-coordinate staging cache: resumes seamlessly if interrupted.
- 1km coordinate clustering: aligns with ERA5 reanalysis resolution while eliminating redundant API calls.
- Polite rate-limit backoff: handles HTTP 429 gracefully with 45s cooldowns.

Usage:
    python scripts/ingestion/fetch_and_ingest_weather.py [--start-date 2023-01-08] [--end-date 2026-06-08] [--to-database] [--batch-size 50000]
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_and_ingest_weather")

DEFAULT_START_DATE = "2023-01-08"
DEFAULT_END_DATE = "2026-06-08"
DEFAULT_SENSORS_PATH = "data/traffic_sensors.json"
DEFAULT_CACHE_PATH = "data/external/weather_cache.parquet"
STAGING_DIR = "data/external/.weather_cache_coords"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_args():
    p = argparse.ArgumentParser(description="Fetch historical weather from Open-Meteo and cache/ingest")
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date (YYYY-MM-DD)")
    p.add_argument("--sensors-path", default=DEFAULT_SENSORS_PATH, help="Path to traffic_sensors.json")
    p.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Path to output weather_cache.parquet")
    p.add_argument("--staging-dir", default=STAGING_DIR, help="Directory for intermediate coordinate caches")
    p.add_argument("--to-database", action="store_true", help="Ingest records into TimescaleDB weather_readings")
    p.add_argument("--batch-size", type=int, default=50_000, help="DB insert batch size")
    p.add_argument("--coord-batch-size", type=int, default=8, help="Number of coordinates per API query")
    return p.parse_args()


def load_sensor_coordinates(sensors_path: str) -> List[Dict]:
    path = Path(sensors_path)
    assert path.exists(), f"CRITICAL: Sensors file not found: {path}"
    with open(path, "r") as f:
        sensors = json.load(f)
    assert isinstance(sensors, list) and len(sensors) > 0, "CRITICAL: traffic_sensors.json is empty or invalid"
    for s in sensors:
        assert "location" in s and "lat" in s and "lon" in s, f"CRITICAL: Sensor missing required keys: {s}"
    logger.info(f"Loaded {len(sensors)} sensor locations from {path}")
    return sensors


def fetch_open_meteo_batch(
    coords: List[Tuple[float, float]],
    start_date: str,
    end_date: str,
    max_retries: int = 6,
) -> List[Dict]:
    """
    Queries Open-Meteo Historical API for a batch of (lat, lon) coordinates with robust
    HTTP 429 rate limit backoff.
    """
    lats_str = ",".join(f"{lat:.4f}" for lat, _ in coords)
    lons_str = ",".join(f"{lon:.4f}" for _, lon in coords)

    url = (
        f"{OPEN_METEO_ARCHIVE_URL}?"
        f"latitude={lats_str}&"
        f"longitude={lons_str}&"
        f"start_date={start_date}&"
        f"end_date={end_date}&"
        f"hourly=relative_humidity_2m,shortwave_radiation&"
        f"timezone=UTC"
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ml-agent-workspace-weather/1.0 (academic research)"},
    )

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                return data
            else:
                return [data]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                cooldown = 45.0 + (attempt * 15.0)
                logger.warning(f"HTTP 429 Too Many Requests (attempt {attempt}/{max_retries}). Cooldown for {cooldown:.0f}s...")
                time.sleep(cooldown)
            else:
                logger.warning(f"HTTP error {e.code} (attempt {attempt}/{max_retries}): {e}. Retrying in 10s...")
                time.sleep(10.0)
        except Exception as e:
            logger.warning(f"Network error (attempt {attempt}/{max_retries}): {e}. Retrying in 10s...")
            time.sleep(10.0)

    raise RuntimeError(f"CRITICAL: Failed to fetch weather batch after {max_retries} attempts: {coords}")


def interpolate_to_15min(
    times: List[str],
    humidity: List[float],
    solar: List[float],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Resamples hourly weather time series to a regular 15-minute grid and clamps
    to physical boundaries.
    """
    df_hourly = pd.DataFrame({
        "dt": pd.to_datetime(times, utc=True),
        "Humidity": np.array(humidity, dtype=np.float32),
        "Solar Radiation": np.array(solar, dtype=np.float32),
    }).set_index("dt")

    # Define strict 15-minute target timeline
    target_dt_index = pd.date_range(
        start=pd.Timestamp(start_date, tz="UTC"),
        end=pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23, minutes=45),
        freq="15min",
    )

    # Reindex & interpolate linearly
    df_15min = df_hourly.reindex(df_hourly.index.union(target_dt_index))
    df_15min = df_15min.interpolate(method="time").reindex(target_dt_index)

    # Physical tripwires and clamping
    # Humidity: strictly bounded in [0.0, 100.0]
    df_15min["Humidity"] = df_15min["Humidity"].clip(lower=0.0, upper=100.0)

    # Solar Radiation: strictly non-negative, strictly 0 at deep night (23:00 - 04:30)
    df_15min["Solar Radiation"] = df_15min["Solar Radiation"].clip(lower=0.0)
    hours = df_15min.index.hour
    minutes = df_15min.index.minute
    night_mask = (hours >= 23) | (hours < 4) | ((hours == 4) & (minutes < 30))
    df_15min.loc[night_mask, "Solar Radiation"] = 0.0

    # Ensure no residual NaNs
    assert not df_15min["Humidity"].isna().any(), "CRITICAL: NaN in interpolated Humidity"
    assert not df_15min["Solar Radiation"].isna().any(), "CRITICAL: NaN in interpolated Solar Radiation"

    return df_15min.reset_index().rename(columns={"index": "dt"})


def coord_file_name(coord: Tuple[float, float]) -> str:
    lat_str = f"{coord[0]:.4f}".replace(".", "p").replace("-", "m")
    lon_str = f"{coord[1]:.4f}".replace(".", "p").replace("-", "m")
    return f"weather_{lat_str}_{lon_str}.parquet"


def build_weather_dataset(
    sensors: List[Dict],
    start_date: str,
    end_date: str,
    staging_dir: str,
    coord_batch_size: int = 8,
) -> pd.DataFrame:
    """
    Fetches and interpolates weather data for all sensors, using an intermediate coordinate cache.
    """
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)

    # Group sensors by ~1km coordinate cells (2 decimal places)
    # ERA5 grid is ~10-25km, so 2 decimal places (~1.1km) provides ample spatial fidelity.
    coord_to_sensors: Dict[Tuple[float, float], List[str]] = {}
    for s in sensors:
        coord = (round(float(s["lat"]), 2), round(float(s["lon"]), 2))
        coord_to_sensors.setdefault(coord, []).append(s["location"])

    unique_coords = sorted(list(coord_to_sensors.keys()))
    logger.info(f"Grouped {len(sensors)} sensors into {len(unique_coords)} ~1km grid cells.")

    # Check staging directory for cached coordinates
    missing_coords: List[Tuple[float, float]] = []
    coord_weather_dfs: Dict[Tuple[float, float], pd.DataFrame] = {}

    for coord in unique_coords:
        f_name = coord_file_name(coord)
        c_path = staging_path / f_name
        if c_path.exists():
            df_c = pd.read_parquet(c_path)
            coord_weather_dfs[coord] = df_c
        else:
            missing_coords.append(coord)

    logger.info(f"Staging cache status: {len(coord_weather_dfs)} already cached, {len(missing_coords)} need fetching.")

    # Fetch missing coordinates in batches
    if missing_coords:
        for i in range(0, len(missing_coords), coord_batch_size):
            batch_coords = missing_coords[i : i + coord_batch_size]
            batch_idx = (i // coord_batch_size) + 1
            num_batches = (len(missing_coords) + coord_batch_size - 1) // coord_batch_size
            logger.info(f"Fetching API batch {batch_idx}/{num_batches} ({len(batch_coords)} coords)...")

            results = fetch_open_meteo_batch(batch_coords, start_date, end_date)
            assert len(results) == len(batch_coords), (
                f"CRITICAL: Batch result size mismatch: expected {len(batch_coords)}, got {len(results)}"
            )

            for coord, res in zip(batch_coords, results):
                hourly_data = res.get("hourly", {})
                times = hourly_data.get("time", [])
                humidity = hourly_data.get("relative_humidity_2m", [])
                solar = hourly_data.get("shortwave_radiation", [])

                assert len(times) > 0, f"CRITICAL: No time readings for coord {coord}"
                assert len(humidity) == len(times), f"CRITICAL: Humidity length mismatch for coord {coord}"
                assert len(solar) == len(times), f"CRITICAL: Solar length mismatch for coord {coord}"

                interp_df = interpolate_to_15min(times, humidity, solar, start_date, end_date)
                coord_weather_dfs[coord] = interp_df

                # Save coordinate cache immediately
                c_path = staging_path / coord_file_name(coord)
                pq.write_table(pa.Table.from_pandas(interp_df), c_path, compression="snappy")

            # Polite pause between batches
            time.sleep(5.0)

    # Replicate interpolated weather series to individual sensors
    logger.info("Replicating interpolated weather series to individual sensors...")
    all_chunks = []
    for coord, sensor_ids in coord_to_sensors.items():
        base_df = coord_weather_dfs[coord]
        for sid in sensor_ids:
            chunk = base_df.copy()
            chunk["location"] = sid
            all_chunks.append(chunk)

    full_df = pd.concat(all_chunks, ignore_index=True)

    # Reorder columns to match standard schema: [dt, location, Humidity, Solar Radiation]
    full_df = full_df[["dt", "location", "Humidity", "Solar Radiation"]]
    full_df["dt"] = pd.to_datetime(full_df["dt"]).dt.tz_localize(None)

    # Offensive assertions
    assert len(full_df) > 0, "CRITICAL: Final weather DataFrame is empty"
    assert not full_df["Humidity"].isna().any(), "CRITICAL: NaN found in final Humidity"
    assert not full_df["Solar Radiation"].isna().any(), "CRITICAL: NaN found in final Solar Radiation"
    assert (full_df["Humidity"] > 0).any(), "CRITICAL: Humidity is degenerate zero"
    assert (full_df["Solar Radiation"] > 0).any(), "CRITICAL: Solar Radiation is degenerate zero"
    assert full_df["Humidity"].std() > 0.1, "CRITICAL: Humidity variance is zero"
    assert full_df["Solar Radiation"].std() > 0.1, "CRITICAL: Solar Radiation variance is zero"

    logger.info(
        f"Weather dataset constructed: {len(full_df):,} rows across {full_df['location'].nunique()} sensors. "
        f"Humidity mean={full_df['Humidity'].mean():.1f}%, Solar mean={full_df['Solar Radiation'].mean():.1f} W/m2."
    )
    return full_df


def save_weather_cache(df: pd.DataFrame, cache_path: str):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved weather cache to {path} ({size_mb:.2f} MB)")


def ingest_to_timescale(df: pd.DataFrame, batch_size: int = 50_000):
    url = os.getenv("DATABASE_URL")
    if not url:
        logger.warning("DATABASE_URL environment variable is not set. Skipping database ingestion.")
        return

    try:
        import psycopg2
        from psycopg2.extras import execute_values
        conn = psycopg2.connect(url)
    except Exception as e:
        logger.warning(f"Could not connect to TimescaleDB ({e}). Skipping database ingestion.")
        return

    logger.info(f"Ingesting {len(df):,} records into TimescaleDB weather_readings...")
    cursor = conn.cursor()
    try:
        # Prepare records: (dt, sensor_id, humidity, solar_radiation)
        records = [
            (row.dt.to_pydatetime(), row.location, float(row.Humidity), float(row["Solar Radiation"]))
            for row in df.itertuples(index=False)
        ]

        total_inserted = 0
        total = len(records)
        for i in range(0, total, batch_size):
            batch = records[i : i + batch_size]
            execute_values(
                cursor,
                """
                INSERT INTO weather_readings (dt, sensor_id, humidity, solar_radiation)
                VALUES %s
                ON CONFLICT (dt, sensor_id) DO UPDATE SET
                    humidity = EXCLUDED.humidity,
                    solar_radiation = EXCLUDED.solar_radiation
                """,
                batch,
                template="(%s, %s, %s, %s)",
            )
            conn.commit()
            total_inserted += len(batch)
            logger.info(f"Ingested batch: {total_inserted:,}/{total:,} records")

        logger.info("TimescaleDB weather_readings ingestion successfully completed.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during TimescaleDB ingestion: {e}")
        raise
    finally:
        conn.close()


def main():
    args = parse_args()
    sensors = load_sensor_coordinates(args.sensors_path)

    # Check if final cache already exists and has full coverage
    cache_path = Path(args.cache_path)
    if cache_path.exists():
        logger.info(f"Found existing weather cache at {cache_path}. Validating...")
        existing_df = pd.read_parquet(cache_path)
        req_cols = {"dt", "location", "Humidity", "Solar Radiation"}
        if req_cols.issubset(existing_df.columns) and not existing_df["Humidity"].isna().any():
            min_dt = pd.to_datetime(existing_df["dt"]).min()
            max_dt = pd.to_datetime(existing_df["dt"]).max()
            req_min = pd.Timestamp(args.start_date, tz="UTC")
            req_max = pd.Timestamp(args.end_date, tz="UTC")
            if min_dt <= req_min and max_dt >= req_max:
                logger.info(f"Cache covers requested range ({min_dt} to {max_dt}). Using cached weather.")
                if args.to_database:
                    ingest_to_timescale(existing_df, batch_size=args.batch_size)
                return

    # Fetch and build
    df = build_weather_dataset(
        sensors,
        start_date=args.start_date,
        end_date=args.end_date,
        staging_dir=args.staging_dir,
        coord_batch_size=args.coord_batch_size,
    )
    save_weather_cache(df, args.cache_path)

    if args.to_database:
        ingest_to_timescale(df, batch_size=args.batch_size)

    logger.info("Weather pipeline execution complete.")


if __name__ == "__main__":
    main()
