"""
verify_db_parity.py

Desk-check: verifies that TimescaleConnector and ParquetConnector produce
identical DataFrames for the same 1-week window.

Usage:
    python tests/verify_db_parity.py [--start 2024-05-01] [--end 2024-05-08]

Requires DATABASE_URL in environment and master_dataset_15T.parquet on disk.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from db_connector import TimescaleConnector, ParquetConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_db_parity")

DEFAULT_PARQUET = "data/processed/master_dataset_15T.parquet"
DEFAULT_VEH_CLASS = "person"
DEFAULT_DIRECTION = "all"


def parse_args():
    p = argparse.ArgumentParser(description="Verify DB vs Parquet data parity")
    p.add_argument("--start", default="2024-05-01", help="Window start (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-05-08", help="Window end (YYYY-MM-DD, exclusive)")
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--veh_class", default=DEFAULT_VEH_CLASS)
    p.add_argument("--direction", default=DEFAULT_DIRECTION)
    return p.parse_args()


def normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardise dtypes and sort so frames are comparable."""
    df = df.copy()

    # Ensure dt is tz-naive for comparison
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_convert('UTC').dt.tz_localize(None)

    df['dt'] = pd.to_datetime(df['dt'])
    df = df.sort_values(['dt', 'location', 'dir']).reset_index(drop=True)
    return df


def main():
    args = parse_args()

    db_url = os.getenv("DATABASE_URL")
    assert db_url, "CRITICAL: DATABASE_URL not set — cannot run DB parity check"
    assert Path(args.parquet).exists(), f"CRITICAL: Parquet file not found: {args.parquet}"

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    logger.info(f"Parity check: {args.start} → {args.end}, veh_class={args.veh_class}, dir={args.direction}")

    # --- Load via ParquetConnector ---
    logger.info("Loading via ParquetConnector...")
    parquet_conn = ParquetConnector(args.parquet)
    p_traffic = parquet_conn.load_traffic(start_dt, end_dt, args.veh_class, args.direction)
    p_sensor_ids = p_traffic['location'].unique().tolist()
    p_weather = parquet_conn.load_weather(start_dt, end_dt, p_sensor_ids)
    p_static = parquet_conn.load_static_features(p_sensor_ids)
    p_df = p_traffic.merge(p_weather, on=['dt', 'location'], how='left')
    p_df = p_df.merge(p_static, on='location', how='left')
    parquet_conn.close()

    # --- Load via TimescaleConnector ---
    logger.info("Loading via TimescaleConnector...")
    db_conn = TimescaleConnector(db_url)
    d_traffic = db_conn.load_traffic(start_dt, end_dt, args.veh_class, args.direction)
    d_sensor_ids = d_traffic['location'].unique().tolist()
    d_weather = db_conn.load_weather(start_dt, end_dt, d_sensor_ids)
    d_static = db_conn.load_static_features(d_sensor_ids)
    d_df = d_traffic.merge(d_weather, on=['dt', 'location'], how='left')
    d_df = d_df.merge(d_static, on='location', how='left')
    db_conn.close()

    logger.info(f"ParquetConnector: {len(p_df):,} rows, {p_df['location'].nunique()} sensors")
    logger.info(f"TimescaleConnector: {len(d_df):,} rows, {d_df['location'].nunique()} sensors")

    # --- Normalise and compare ---
    p_norm = normalise(p_df, "parquet")
    d_norm = normalise(d_df, "database")

    # Check shape
    assert p_norm.shape == d_norm.shape, (
        f"PARITY FAIL: shape mismatch. Parquet={p_norm.shape}, DB={d_norm.shape}"
    )

    # Check location sets
    p_locs = set(p_norm['location'].unique())
    d_locs = set(d_norm['location'].unique())
    assert p_locs == d_locs, (
        f"PARITY FAIL: sensor sets differ. "
        f"Only in parquet: {p_locs - d_locs}. Only in DB: {d_locs - p_locs}"
    )

    # Compare column by column
    common_cols = ['dt', 'location', 'value', 'Humidity', 'Solar Radiation'] + [
        'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
        'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
        'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
        'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking',
    ]
    p_cmp = p_norm[common_cols]
    d_cmp = d_norm[common_cols]

    pd.testing.assert_frame_equal(
        p_cmp, d_cmp,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        obj="Parquet vs DB parity",
    )

    logger.info("PARITY CHECK PASSED: ParquetConnector and TimescaleConnector produce identical data.")


if __name__ == "__main__":
    main()
