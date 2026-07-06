"""
ingest_static_features.py

Populate the static_features table from:
  - data/external/places_cache.json   (POI counts, landuse, school zone, bus stops, parking, dist_to_city_centre)
  - data/external/osm_features_cache.json  (betweenness, closeness, degree centrality)

Requires sensor_metadata to be populated first (FK constraint).

Usage:
    python scripts/ingestion/ingest_static_features.py [--dry-run] [--allow-duplicates]
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
logger = logging.getLogger("ingest_static_features")

PLACES_PATH = "data/external/places_cache.json"
OSM_PATH = "data/external/osm_features_cache.json"

STATIC_FEATURE_COLS = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking',
]


def parse_args():
    p = argparse.ArgumentParser(description="Ingest static features to TimescaleDB")
    p.add_argument("--places", default=PLACES_PATH, help="Path to places_cache.json")
    p.add_argument("--osm", default=OSM_PATH, help="Path to osm_features_cache.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-duplicates", action="store_true",
                   help="Suppress the assertion that no rows were skipped by ON CONFLICT DO NOTHING.")
    return p.parse_args()


def get_db_conn():
    url = os.getenv("DATABASE_URL")
    assert url is not None, "DATABASE_URL environment variable is not set in .env"
    return psycopg2.connect(url)


def main():
    args = parse_args()

    places_path = Path(args.places)
    osm_path = Path(args.osm)
    assert places_path.exists(), f"CRITICAL: places cache not found: {places_path}"
    assert osm_path.exists(), f"CRITICAL: OSM cache not found: {osm_path}"

    with open("data/traffic_sensors.json") as f:
        valid_sensors = set(item["location"] for item in json.load(f))

    with open(places_path) as f:
        places = json.load(f)
    with open(osm_path) as f:
        osm = json.load(f)

    assert isinstance(places, dict) and len(places) > 0, "CRITICAL: places_cache.json must be a non-empty dict"
    assert isinstance(osm, dict) and len(osm) > 0, "CRITICAL: osm_features_cache.json must be a non-empty dict"

    all_sensor_ids = sorted(set(places.keys()) | set(osm.keys()))
    valid_sensor_ids = [s for s in all_sensor_ids if s in valid_sensors]
    logger.info(f"Found {len(all_sensor_ids)} unique sensor IDs across both caches, {len(valid_sensor_ids)} are valid traffic sensors")

    records = []
    for sensor_id in valid_sensor_ids:
        p = places[sensor_id]
        o = osm[sensor_id]

        row = (
            sensor_id,
            float(p['transit_count']),
            float(p['retail_count']),
            float(p['food_drink_count']),
            float(p['nightlife_count']),
            float(p['education_count']),
            float(p['recreation_count']),
            float(p['total_poi_count']),
            float(p['landuse_entropy']),
            float(1 if p['is_school_zone'] else 0),
            float(o['betweenness_centrality']),
            float(o['closeness_centrality']),
            float(p['dist_to_city_centre']),
            float(o['degree_centrality']),
            float(p['nearby_bus_stops']),
            float(p['nearby_parking']),
        )
        records.append(row)

    assert len(records) > 0, "CRITICAL: no static feature records built"
    logger.info(f"Built {len(records)} static feature records")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would insert {len(records)} rows into static_features")
        for r in records[:2]:
            logger.info(f"  Sample: sensor_id={r[0]}, transit={r[1]}, betweenness={r[10]:.6f}")
        return

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        execute_values(
            cursor,
            """
            INSERT INTO static_features (
                sensor_id,
                transit_count, retail_count, food_drink_count, nightlife_count,
                education_count, recreation_count, total_poi_count, landuse_entropy,
                is_school_zone, betweenness_centrality, closeness_centrality,
                dist_to_city_centre, degree_centrality, nearby_bus_stops, nearby_parking
            )
            VALUES %s
            ON CONFLICT (sensor_id) DO NOTHING
            """,
            records,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        )
        conn.commit()
        inserted = cursor.rowcount
        skipped = len(records) - inserted
        logger.info(f"Inserted {inserted} rows into static_features; {skipped} skipped (ON CONFLICT DO NOTHING)")
        assert args.allow_duplicates or skipped == 0, (
            f"CRITICAL: {skipped} row(s) were skipped by ON CONFLICT DO NOTHING. "
            "Re-run with --allow-duplicates if this is expected (e.g. re-populating an existing table)."
        )
    finally:
        conn.close()

    logger.info("DONE")


if __name__ == "__main__":
    main()
