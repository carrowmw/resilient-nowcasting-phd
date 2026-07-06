"""
ingest_parquet.py

Bulk-load master_dataset_15T.parquet into the traffic_readings hypertable.

Usage:
    python scripts/ingestion/ingest_parquet.py \
        --parquet data/processed/master_dataset_15T.parquet \
        [--batch-size 100000] [--dry-run] [--allow-drops] [--allow-duplicates]
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import psycopg2
import yaml
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_parquet")

DEFAULT_BATCH_SIZE = 100_000
DEFAULT_CHUNK_SIZE = 1_000_000  # Stream parquet file in chunks of 1M rows

_TRAIN_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "train" / "train.yaml"


def _load_train_config() -> dict:
    assert _TRAIN_CONFIG_PATH.exists(), f"CRITICAL: train config not found: {_TRAIN_CONFIG_PATH}"
    return yaml.safe_load(_TRAIN_CONFIG_PATH.read_text())


def parse_args():
    p = argparse.ArgumentParser(description="Ingest Parquet to TimescaleDB")
    p.add_argument("--parquet", required=True, help="Path to master_dataset_15T.parquet")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--create-dataset-version",
        action="store_true",
        help="Also create a dataset_versions record",
    )
    p.add_argument("--allow-drops", action="store_true",
                   help="Suppress the assertion that NaN-row drop rate is < 1%%.")
    p.add_argument("--allow-duplicates", action="store_true",
                   help="Suppress the assertion that no rows were skipped by ON CONFLICT DO NOTHING.")
    return p.parse_args()


def get_db_conn():
    url = os.getenv("DATABASE_URL")
    assert url is not None, "DATABASE_URL environment variable is not set in .env"
    return psycopg2.connect(url)


def compute_parquet_sha256(path: str) -> str:
    """Compute SHA256 of parquet file for versioning."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_metadata():
    """Extract current git branch and commit."""
    try:
        # Avoid "dubious ownership" fatal error when running inside container
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", "/workspace"],
            capture_output=True,
        )
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        return branch, commit
    except subprocess.CalledProcessError:
        return "unknown", "unknown"


def ingest_traffic_readings_chunk(
    df: pd.DataFrame,
    conn,
    batch_size: int,
    dry_run: bool,
    chunk_idx: int,
    allow_drops: bool,
) -> tuple:
    """
    Bulk-insert a single chunk of traffic readings.

    Returns (rows_inserted, rows_attempted, nan_dropped) where rows_inserted is
    based on cursor.rowcount (actual DB inserts, excluding ON CONFLICT DO NOTHING
    conflicts) and rows_attempted is the count after NaN rows are dropped.
    """
    assert set(["dt", "location", "veh_class", "dir", "value"]).issubset(df.columns), \
        f"Missing columns in dataframe chunk. Found: {list(df.columns)}"

    # Filter to core traffic columns
    df_traffic = df[["dt", "location", "veh_class", "dir", "value"]].copy()

    original_len = len(df_traffic)
    df_traffic = df_traffic.dropna(subset=["dt", "location", "veh_class", "dir", "value"])
    nan_dropped = original_len - len(df_traffic)

    if nan_dropped > 0:
        drop_rate = nan_dropped / original_len
        logger.warning(
            f"[Chunk {chunk_idx}] Dropped {nan_dropped:,} NaN rows ({drop_rate:.1%} of {original_len:,})"
        )
        assert allow_drops or drop_rate < 0.01, (
            f"CRITICAL: NaN drop rate {drop_rate:.1%} >= 1% in chunk {chunk_idx}. "
            "Pass --allow-drops to override."
        )

    # Rename to match schema
    df_traffic.columns = ["dt", "sensor_id", "veh_class", "direction", "count"]

    # Convert dt to python datetime objects so to_records does not cast to bigint,
    # using map to preserve the index alignment after dropna.
    df_traffic['dt'] = df_traffic['dt'].map(lambda x: x.to_pydatetime())

    # Convert to records
    records = df_traffic.to_records(index=False).tolist()
    attempted = len(records)

    if dry_run:
        logger.info(f"[DRY RUN] [Chunk {chunk_idx}] Would insert {attempted:,} rows in {batch_size:,}-row batches")
        return attempted, attempted, nan_dropped

    cursor = conn.cursor()
    inserted = 0
    for i in range(0, attempted, batch_size):
        batch = records[i : i + batch_size]
        execute_values(
            cursor,
            """
            INSERT INTO traffic_readings (dt, sensor_id, veh_class, direction, count)
            VALUES %s
            ON CONFLICT (dt, sensor_id, veh_class, direction) DO NOTHING
            """,
            batch,
            template="(%s, %s, %s, %s, %s)",
        )
        inserted += cursor.rowcount
    conn.commit()

    skipped_chunk = attempted - inserted
    logger.info(
        f"[Chunk {chunk_idx}] {attempted:,} attempted → {inserted:,} inserted, "
        f"{skipped_chunk:,} skipped (ON CONFLICT DO NOTHING)"
    )
    return inserted, attempted, nan_dropped


def create_dataset_version(conn, parquet_path: str, row_count: int):
    """Insert a dataset_versions record, reading seq_len and mask_prob from train config."""
    git_branch, git_commit = get_git_metadata()
    sha256 = compute_parquet_sha256(parquet_path)

    train_cfg = _load_train_config()
    seq_len = train_cfg["data"]["seq_len"]
    mask_prob = train_cfg["data"]["mask_prob"]

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO dataset_versions
            (git_branch, git_commit, description, preprocessing, parquet_sha256, row_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (git_commit, parquet_sha256) DO NOTHING
        RETURNING version_id
        """,
        (
            git_branch,
            git_commit,
            f"Bulk ingest from {Path(parquet_path).name}",
            json.dumps({"source": "master_dataset_15T.parquet", "seq_len": seq_len, "mask_prob": mask_prob}),
            sha256,
            row_count,
        ),
    )
    result = cursor.fetchone()
    conn.commit()
    if result:
        logger.info(f"Created dataset_versions record: version_id={result[0]}")
    else:
        logger.info("Dataset version already exists (conflict skipped)")


def main():
    args = parse_args()

    logger.info(f"Opening parquet file: {args.parquet}")
    parquet_file = pq.ParquetFile(args.parquet)

    conn = get_db_conn()
    total_processed_rows = 0
    total_nan_dropped = 0
    total_attempted = 0
    total_inserted = 0
    chunk_idx = 1

    try:
        for batch in parquet_file.iter_batches(
            batch_size=args.chunk_size,
            columns=["dt", "location", "veh_class", "dir", "value"],
        ):
            df_chunk = batch.to_pandas()
            chunk_len = len(df_chunk)
            total_processed_rows += chunk_len

            logger.info(
                f"Processing chunk {chunk_idx} ({chunk_len:,} rows). "
                f"Total so far: {total_processed_rows:,}"
            )
            inserted, attempted, nan_dropped = ingest_traffic_readings_chunk(
                df_chunk, conn, args.batch_size, args.dry_run, chunk_idx, args.allow_drops
            )
            total_inserted += inserted
            total_attempted += attempted
            total_nan_dropped += nan_dropped
            chunk_idx += 1

        total_skipped = total_attempted - total_inserted
        logger.info(
            f"All chunks done. Processed: {total_processed_rows:,} rows | "
            f"NaN-dropped: {total_nan_dropped:,} | "
            f"Attempted: {total_attempted:,} | "
            f"Inserted: {total_inserted:,} | "
            f"Skipped (conflict): {total_skipped:,}"
        )
        assert args.allow_duplicates or total_skipped == 0, (
            f"CRITICAL: {total_skipped} row(s) were skipped by ON CONFLICT DO NOTHING. "
            "Re-run with --allow-duplicates if this is expected (e.g. re-ingesting existing data)."
        )

        if args.create_dataset_version and not args.dry_run:
            create_dataset_version(conn, args.parquet, total_inserted)
    finally:
        conn.close()

    logger.info("DONE")


if __name__ == "__main__":
    main()
