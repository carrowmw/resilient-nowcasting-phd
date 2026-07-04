"""
ingest_sensor_metadata.py

Populate the sensor_metadata table from data/traffic_sensors.json.
Must run before ingest_static_features.py due to the FK constraint.

Usage:
    python scripts/ingest_sensor_metadata.py [--dry-run]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_sensor_metadata")

DEFAULT_SENSORS_PATH = "data/traffic_sensors.json"


def parse_args():
    p = argparse.ArgumentParser(description="Ingest sensor metadata to TimescaleDB")
    p.add_argument("--sensors", default=DEFAULT_SENSORS_PATH, help="Path to traffic_sensors.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_db_conn():
    url = os.getenv("DATABASE_URL")
    assert url is not None, "DATABASE_URL environment variable is not set in .env"
    return psycopg2.connect(url)


def main():
    args = parse_args()

    sensors_path = Path(args.sensors)
    assert sensors_path.exists(), f"CRITICAL: sensors file not found: {sensors_path}"

    with open(sensors_path) as f:
        raw = json.load(f)

    assert isinstance(raw, list) and len(raw) > 0, "CRITICAL: traffic_sensors.json must be a non-empty list"

    records = []
    for entry in raw:
        sensor_id = entry["location"]
        lat = float(entry["lat"])
        lon = float(entry["lon"])
        description = entry.get("description", None)
        records.append((sensor_id, lat, lon, description))

    assert len(records) > 0, "CRITICAL: no sensor records parsed"
    logger.info(f"Parsed {len(records)} sensor records from {sensors_path}")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would insert {len(records)} rows into sensor_metadata")
        for r in records[:3]:
            logger.info(f"  Sample: {r}")
        return

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        execute_values(
            cursor,
            """
            INSERT INTO sensor_metadata (sensor_id, lat, lon, description)
            VALUES %s
            ON CONFLICT (sensor_id) DO NOTHING
            """,
            records,
            template="(%s, %s, %s, %s)",
        )
        conn.commit()
        logger.info(f"Inserted {len(records)} rows into sensor_metadata (ON CONFLICT DO NOTHING)")
    finally:
        conn.close()

    logger.info("DONE")


if __name__ == "__main__":
    main()
