"""
ingest_weather.py

Populate the weather_readings hypertable from data/external/static_features_cache.parquet.
Extracts (dt, location, Humidity, Solar Radiation) and inserts per-sensor weather readings.

NOTE: This script is a temporary bridge. In a future task, this will be replaced by a
fetcher that queries the Open-Meteo historical API and interpolates to 15-minute intervals.

Usage:
    python scripts/ingest_weather.py [--batch-size 50000] [--dry-run]
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_weather")

DEFAULT_CACHE_PATH = "data/external/static_features_cache.parquet"
DEFAULT_BATCH_SIZE = 50_000


def parse_args():
    p = argparse.ArgumentParser(description="Ingest weather readings to TimescaleDB")
    p.add_argument("--cache", default=DEFAULT_CACHE_PATH, help="Path to static_features_cache.parquet")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_db_conn():
    url = os.getenv("DATABASE_URL")
    assert url is not None, "DATABASE_URL environment variable is not set in .env"
    return psycopg2.connect(url)


def main():
    args = parse_args()

    cache_path = Path(args.cache)
    assert cache_path.exists(), f"CRITICAL: static features cache not found: {cache_path}"

    logger.info(f"Reading weather columns from {cache_path}")
    df = pd.read_parquet(
        cache_path,
        columns=['dt', 'location', 'Humidity', 'Solar Radiation']
    )

    assert not df.empty, "CRITICAL: empty DataFrame from static_features_cache.parquet"
    assert {'dt', 'location', 'Humidity', 'Solar Radiation'}.issubset(df.columns), \
        f"CRITICAL: missing expected columns. Found: {list(df.columns)}"

    # Deduplicate: each (dt, location) pair should have one weather reading
    df = df.drop_duplicates(subset=['dt', 'location'])
    df = df.dropna(subset=['dt', 'location'])
    # Weather data is placeholder (NaN) — fill with 0.0 to satisfy NOT NULL constraint.
    df['Humidity'] = df['Humidity'].fillna(0.0)
    df['Solar Radiation'] = df['Solar Radiation'].fillna(0.0)

    # Ensure dt is timezone-aware (UTC)
    if df['dt'].dt.tz is None:
        df['dt'] = df['dt'].dt.tz_localize('UTC')

    df = df.rename(columns={
        'location': 'sensor_id',
        'Humidity': 'humidity',
        'Solar Radiation': 'solar_radiation',
    })

    total = len(df)
    logger.info(f"Prepared {total:,} unique (dt, sensor_id) weather records")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would insert {total:,} rows into weather_readings in {args.batch_size:,}-row batches")
        logger.info(f"  Date range: {df['dt'].min()} to {df['dt'].max()}")
        logger.info(f"  Unique sensors: {df['sensor_id'].nunique()}")
        return

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        records = [
            (row.dt.to_pydatetime(), row.sensor_id, float(row.humidity), float(row.solar_radiation))
            for row in df.itertuples(index=False)
        ]

        inserted = 0
        for i in range(0, total, args.batch_size):
            batch = records[i: i + args.batch_size]
            execute_values(
                cursor,
                """
                INSERT INTO weather_readings (dt, sensor_id, humidity, solar_radiation)
                VALUES %s
                ON CONFLICT (dt, sensor_id) DO NOTHING
                """,
                batch,
                template="(%s, %s, %s, %s)",
            )
            conn.commit()
            inserted += len(batch)
            logger.info(f"Committed batch: {inserted:,}/{total:,} rows")

        logger.info(f"Inserted {total:,} rows into weather_readings (ON CONFLICT DO NOTHING)")
    finally:
        conn.close()

    logger.info("DONE")


if __name__ == "__main__":
    main()
