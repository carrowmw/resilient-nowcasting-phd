# Phase 2 & 3 Architecture Plan: Database Integration & CDN Migration

> **Author**: Architect Agent
> **Date**: 2026-06-29
> **Scope**: Full implementation of Phase 2 (Training Database) and remaining Phase 3 (CDN Migration) components
> **Reference Docs**: [implementation_phases.md](file:///Users/administrator/Code/ml-agent-workspace/docs/database/implementation_phases.md)

---

## Executive Summary

This plan provides detailed architecture and implementation roadmaps for:

1. **Phase 2 (Training Database)**: Migrate raw data ingestion from Parquet files to PostgreSQL + TimescaleDB with Alembic migrations, dockerized persistence, and dataset query refactoring.
2. **Phase 3 (CDN Migration - Remaining)**: Complete the R2-backed CDN infrastructure with Next.js proxy rewrites, edge caching rules, and automated publishing pipelines using self-hosted GitHub Actions runners.

Both phases are designed for **standalone value delivery** — Phase 2 enhances training reproducibility and dataset versioning; Phase 3 eliminates Vercel static payload bloat and enables zero-egress CDN delivery.

---

## Phase 2: Training Database (PostgreSQL + TimescaleDB)

### Objectives

1. **Dockerize TimescaleDB** as a persistent service accessible to both the `gpu-runner` and `agent-workspace` containers.
2. **Design & migrate schema** for time-series hypertables (`traffic_readings`), model run metadata (`model_runs`), and dataset versioning (`dataset_versions`).
3. **Implement Alembic migration framework** for schema evolution tracking.
4. **Build ingestion script** to parse and load the existing Parquet file into the database.
5. **Refactor `dataset.py`** to optionally query the database with Parquet fallback.

---

### Architecture Components

#### 2.1 Docker Compose Service

**File**: `docker-compose.yml`

Add a new `timescaledb` service:

```yaml
timescaledb:
  image: timescale/timescaledb:latest-pg16
  container_name: ml-timescaledb
  environment:
    POSTGRES_USER: ${POSTGRES_USER:-mluser}
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mlpassword}
    POSTGRES_DB: ${POSTGRES_DB:-ml_traffic}
  volumes:
    - timescaledb_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
  networks:
    - ml-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-mluser}"]
    interval: 10s
    timeout: 5s
    retries: 5

volumes:
  timescaledb_data:
    driver: local

networks:
  ml-network:
    driver: bridge
```

**Design Rationale**:
- Uses TimescaleDB's official Docker image (PostgreSQL 16 + TimescaleDB extension pre-installed).
- Persistent volume `timescaledb_data` ensures data survives container restarts.
- Healthcheck validates database readiness before dependent services start.
- Environment variables for credentials (`.env` driven).
- Exposes port 5432 to host for external clients (pgAdmin, DBeaver).
- Shares `ml-network` with `gpu-runner` and `agent-workspace` for inter-container connectivity.

**Environment Variables** (add to `.env`):
```bash
POSTGRES_USER=riotuser
POSTGRES_PASSWORD=riotpassword
POSTGRES_DB=riot_traffic
DATABASE_URL=postgresql://riotuser:riotpassword@timescaledb:5432/riot_traffic
```

---

#### 2.2 Database Schema Design

**TimescaleDB Hypertables** (time-series optimized):

##### **Table: `traffic_readings`**
Core time-series hypertable partitioned by `dt` (timestamp).

```sql
CREATE TABLE traffic_readings (
    dt              TIMESTAMPTZ NOT NULL,
    sensor_id       TEXT NOT NULL,
    veh_class       TEXT NOT NULL,
    direction       TEXT NOT NULL,
    count           REAL NOT NULL,
    -- Constraints
    PRIMARY KEY (dt, sensor_id, veh_class, direction)
);

-- Convert to hypertable (partitioned by 'dt' with 7-day chunks)
SELECT create_hypertable(
    'traffic_readings',
    'dt',
    chunk_time_interval => INTERVAL '7 days'
);

-- Indexes for common query patterns
CREATE INDEX idx_traffic_sensor_dt ON traffic_readings (sensor_id, dt DESC);
CREATE INDEX idx_traffic_vehclass ON traffic_readings (veh_class, dt DESC);
CREATE INDEX idx_traffic_direction ON traffic_readings (direction, dt DESC);
```

**Design Rationale**:
- **Composite PK** `(dt, sensor_id, veh_class, direction)` ensures unique time-series points and enables efficient range scans.
- **7-day chunks** balance query performance (typical training windows are 3-30 days) with chunk management overhead.
- **Descending indexes** optimize recent-data queries (most training runs use recent months).
- `REAL` type for `count` matches the Parquet schema and PyTorch float32 tensors.

---

##### **Table: `model_runs`**
Locally queryable training metadata mirroring W&B metrics.

```sql
CREATE TABLE model_runs (
    run_id          TEXT PRIMARY KEY,
    wandb_run_id    TEXT UNIQUE,
    git_commit      TEXT NOT NULL,
    git_branch      TEXT NOT NULL,
    config_yaml     JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    -- Validation metrics (lifted from W&B summary)
    val_rmse        REAL,
    val_mae         REAL,
    val_zero_acc    REAL,
    val_loss        REAL,
    -- Training hyperparameters (denormalized for fast filtering)
    batch_size      INTEGER,
    learning_rate   REAL,
    num_epochs      INTEGER,
    -- Dataset version FK
    dataset_version_id INTEGER REFERENCES dataset_versions(version_id)
);

CREATE INDEX idx_runs_wandb ON model_runs (wandb_run_id);
CREATE INDEX idx_runs_branch ON model_runs (git_branch, created_at DESC);
CREATE INDEX idx_runs_val_loss ON model_runs (val_loss ASC NULLS LAST);
```

**Design Rationale**:
- `run_id` is the human-readable W&B run name (e.g., `feasible-capybara-22`), used as the primary key for `export_paginated_payload.py` output directories.
- `wandb_run_id` is the stable W&B UUID for API lookups.
- `config_yaml` stores the full hyperparameter JSON for reproducibility without hitting W&B API.
- **Denormalized metrics** (val_rmse, val_mae, etc.) enable fast leaderboard queries without JSON traversal.
- `dataset_version_id` FK links runs to their preprocessing snapshots.

---

##### **Table: `dataset_versions`**
Branch-aware dataset versioning for experimental runs.

```sql
CREATE TABLE dataset_versions (
    version_id      SERIAL PRIMARY KEY,
    git_branch      TEXT NOT NULL,
    git_commit      TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    description     TEXT,
    -- Preprocessing parameters (JSONB for schema flexibility)
    preprocessing   JSONB NOT NULL,
    -- Checksums for reproducibility
    parquet_sha256  TEXT,
    row_count       BIGINT,
    UNIQUE (git_commit, parquet_sha256)
);

CREATE INDEX idx_dataset_branch ON dataset_versions (git_branch, created_at DESC);
```

**Design Rationale**:
- `UNIQUE (git_commit, parquet_sha256)` prevents duplicate ingestion while allowing branch-specific experiments to share the same underlying data snapshot.
- `preprocessing` JSONB stores scaling parameters, mask_prob, seq_len, veh_class, dir filters — critical for reproducibility.
- `row_count` provides a fast sanity check without querying `traffic_readings`.

---

#### 2.3 Alembic Migration Framework

**Directory Structure**:
```
alembic/
├── env.py                  # Alembic environment config
├── script.py.mako          # Migration template
└── versions/
    ├── 001_initial_schema.py
    ├── 002_add_run_metadata.py
    └── ...
```

**Setup Steps**:

1. **Install Dependencies**:
```bash
pip install alembic psycopg2-binary
```

2. **Initialize Alembic** (run once):
```bash
alembic init alembic
```

3. **Configure `alembic.ini`**:
```ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql://mluser:mlpassword@localhost:5432/ml_traffic

# Use async driver for TimescaleDB compatibility
[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 100
```

4. **Update `alembic/env.py`** to load `.env`:
```python
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL"))
```

**Migration Files**:

##### **Migration 001: Initial Schema** (`alembic/versions/001_initial_schema.py`)

```python
"""Initial schema: traffic_readings hypertable, model_runs, dataset_versions

Revision ID: 001
Revises:
Create Date: 2026-06-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # Traffic readings hypertable
    op.create_table(
        'traffic_readings',
        sa.Column('dt', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('sensor_id', sa.Text(), nullable=False),
        sa.Column('veh_class', sa.Text(), nullable=False),
        sa.Column('direction', sa.Text(), nullable=False),
        sa.Column('count', sa.Real(), nullable=False),
        sa.PrimaryKeyConstraint('dt', 'sensor_id', 'veh_class', 'direction')
    )
    op.execute("SELECT create_hypertable('traffic_readings', 'dt', chunk_time_interval => INTERVAL '7 days');")
    op.create_index('idx_traffic_sensor_dt', 'traffic_readings', ['sensor_id', sa.text('dt DESC')])
    op.create_index('idx_traffic_vehclass', 'traffic_readings', ['veh_class', sa.text('dt DESC')])
    op.create_index('idx_traffic_direction', 'traffic_readings', ['direction', sa.text('dt DESC')])

    # Dataset versions
    op.create_table(
        'dataset_versions',
        sa.Column('version_id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('git_branch', sa.Text(), nullable=False),
        sa.Column('git_commit', sa.Text(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('description', sa.Text()),
        sa.Column('preprocessing', JSONB, nullable=False),
        sa.Column('parquet_sha256', sa.Text()),
        sa.Column('row_count', sa.BigInteger()),
        sa.UniqueConstraint('git_commit', 'parquet_sha256')
    )
    op.create_index('idx_dataset_branch', 'dataset_versions', ['git_branch', sa.text('created_at DESC')])

    # Model runs
    op.create_table(
        'model_runs',
        sa.Column('run_id', sa.Text(), primary_key=True),
        sa.Column('wandb_run_id', sa.Text(), unique=True),
        sa.Column('git_commit', sa.Text(), nullable=False),
        sa.Column('git_branch', sa.Text(), nullable=False),
        sa.Column('config_yaml', JSONB, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('val_rmse', sa.Real()),
        sa.Column('val_mae', sa.Real()),
        sa.Column('val_zero_acc', sa.Real()),
        sa.Column('val_loss', sa.Real()),
        sa.Column('batch_size', sa.Integer()),
        sa.Column('learning_rate', sa.Real()),
        sa.Column('num_epochs', sa.Integer()),
        sa.Column('dataset_version_id', sa.Integer(), sa.ForeignKey('dataset_versions.version_id'))
    )
    op.create_index('idx_runs_wandb', 'model_runs', ['wandb_run_id'])
    op.create_index('idx_runs_branch', 'model_runs', ['git_branch', sa.text('created_at DESC')])
    op.create_index('idx_runs_val_loss', 'model_runs', [sa.text('val_loss ASC NULLS LAST')])

def downgrade():
    op.drop_table('model_runs')
    op.drop_table('dataset_versions')
    op.execute("SELECT drop_chunks('traffic_readings', older_than => INTERVAL '-1 day');")
    op.drop_table('traffic_readings')
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE;")
```

**Usage**:
```bash
# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1

# Generate new migration
alembic revision -m "add_weather_hypertable"
```

---

#### 2.4 Ingestion Script: `scripts/ingest_parquet.py`

**Purpose**: One-time bulk load of `master_dataset_15T.parquet` into the `traffic_readings` hypertable.

**Design**:

```python
"""
ingest_parquet.py

Bulk-load master_dataset_15T.parquet into the traffic_readings hypertable.

Usage:
    python scripts/ingest_parquet.py \
        --parquet data/processed/master_dataset_15T.parquet \
        [--batch-size 100000] [--dry-run]
"""

import argparse
import hashlib
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_parquet")

DEFAULT_BATCH_SIZE = 100_000

def parse_args():
    p = argparse.ArgumentParser(description="Ingest Parquet to TimescaleDB")
    p.add_argument("--parquet", required=True, help="Path to master_dataset_15T.parquet")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--create-dataset-version", action="store_true",
                   help="Also create a dataset_versions record")
    return p.parse_args()

def get_db_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL not set in .env")
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
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"

def ingest_traffic_readings(df: pd.DataFrame, conn, batch_size: int, dry_run: bool):
    """
    Bulk insert traffic readings using psycopg2 execute_values.

    Schema: (dt, sensor_id, veh_class, direction, count)
    Parquet columns: dt, location, veh_class, dir, value
    """
    # Filter to core traffic columns
    df_traffic = df[["dt", "location", "veh_class", "dir", "value"]].copy()
    df_traffic = df_traffic.dropna(subset=["dt", "location", "veh_class", "dir", "value"])

    # Rename to match schema
    df_traffic.columns = ["dt", "sensor_id", "veh_class", "direction", "count"]

    # Convert to records
    records = df_traffic.to_records(index=False).tolist()
    total = len(records)

    if dry_run:
        logger.info(f"[DRY RUN] Would insert {total:,} rows in {batch_size:,}-row batches")
        return total

    cursor = conn.cursor()
    inserted = 0
    for i in range(0, total, batch_size):
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
        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total:
            logger.info(f"  Inserted {inserted:,} / {total:,} rows ({100*inserted/total:.1f}%)")

    conn.commit()
    logger.info(f"Committed {inserted:,} rows (duplicates skipped via ON CONFLICT)")
    return inserted

def create_dataset_version(conn, parquet_path: str, row_count: int):
    """Insert a dataset_versions record."""
    git_branch, git_commit = get_git_metadata()
    sha256 = compute_parquet_sha256(parquet_path)

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
            {"source": "master_dataset_15T.parquet", "seq_len": 12, "mask_prob": 0.15},
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

    logger.info(f"Loading parquet from {args.parquet}")
    df = pd.read_parquet(args.parquet, columns=["dt", "location", "veh_class", "dir", "value"])
    logger.info(f"Loaded {len(df):,} rows")

    conn = get_db_conn()
    try:
        inserted = ingest_traffic_readings(df, conn, args.batch_size, args.dry_run)

        if args.create_dataset_version and not args.dry_run:
            create_dataset_version(conn, args.parquet, inserted)
    finally:
        conn.close()

    logger.info("DONE")

if __name__ == "__main__":
    main()
```

**Execution**:
```bash
# Dry run
python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --dry-run

# Actual ingestion
python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --create-dataset-version
```

**Performance Estimate**:
- 121M rows ÷ 100k batch = ~1,210 batches
- ~5-10 seconds per batch (execute_values bulk insert)
- **Total: ~2-3 hours** for full ingestion

---

#### 2.5 Dataset Refactor: `src/dataset.py`

**Goal**: Add optional database backend with Parquet fallback.

**Modified `RealIoTDataset.__init__`**:

```python
class RealIoTDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        start_date: str,
        end_date: str,
        seq_len: int = 12,
        mask_prob: float = 0.15,
        target_veh_class: str = 'person',
        target_dir: str = 'all',
        use_database: bool = False,  # NEW: opt-in database mode
    ):
        logger.info(f"Loading data from {parquet_path if not use_database else 'DATABASE'} "
                    f"between {start_date} and {end_date}")

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        if use_database:
            df = self._load_from_database(start_dt, end_dt, target_veh_class, target_dir)
        else:
            df = self._load_from_parquet(parquet_path, start_dt, end_dt, target_veh_class, target_dir)

        # ... rest of __init__ unchanged ...

    def _load_from_parquet(self, parquet_path, start_dt, end_dt, veh_class, direction):
        """Existing Parquet loading logic (unchanged)."""
        df = pd.read_parquet(
            parquet_path,
            filters=[('dt', '>=', start_dt), ('dt', '<', end_dt)],
            columns=[
                'dt', 'location', 'value', 'Humidity', 'Solar Radiation',
                'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
                'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
                'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
                'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking',
                'veh_class', 'dir'
            ]
        )
        return df

    def _load_from_database(self, start_dt, end_dt, veh_class, direction):
        """
        Load traffic readings from TimescaleDB and join with static features cache.

        Returns a DataFrame matching the Parquet schema.
        """
        import psycopg2
        import os
        from dotenv import load_dotenv

        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not set — cannot use database mode")

        try:
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()

            # Query traffic readings with filters
            query = """
                SELECT dt, sensor_id, veh_class, direction, count
                FROM traffic_readings
                WHERE dt >= %s AND dt < %s
                  AND veh_class = %s
                ORDER BY dt, sensor_id
            """
            params = [start_dt, end_dt, veh_class]

            if direction != 'all':
                query = query.replace("ORDER BY", "AND direction = %s ORDER BY")
                params.append(direction)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                raise ValueError(f"No database rows for {start_dt} to {end_dt}, veh_class={veh_class}")

            df = pd.DataFrame(rows, columns=["dt", "location", "veh_class", "dir", "value"])

            # Join with static features cache (weather + spatial metadata)
            # This requires a separate table or JSON cache — for MVP, fall back to Parquet
            logger.warning("Database mode: static features still loaded from Parquet cache")
            static_cache_path = "data/external/static_features_cache.parquet"
            if os.path.exists(static_cache_path):
                static_df = pd.read_parquet(static_cache_path)
                df = df.merge(static_df, on=["dt", "location"], how="left")
            else:
                raise FileNotFoundError(f"Static features cache not found: {static_cache_path}")

            return df

        except Exception as exc:
            logger.error(f"Database load failed: {exc}. Falling back to Parquet.")
            raise  # Force fallback in caller
```

**Fallback Logic in `train.py`**:

```python
def build_dataset(args, split="train"):
    try:
        ds = RealIoTDataset(
            parquet_path=args.parquet_path,
            start_date=args.start_date,
            end_date=args.end_date,
            seq_len=args.seq_len,
            mask_prob=args.mask_prob,
            target_veh_class=args.veh_class,
            target_dir=args.dir,
            use_database=args.use_database,  # CLI flag: --use-database
        )
    except Exception as exc:
        logger.warning(f"Database backend failed ({exc}), falling back to Parquet")
        ds = RealIoTDataset(
            parquet_path=args.parquet_path,
            start_date=args.start_date,
            end_date=args.end_date,
            seq_len=args.seq_len,
            mask_prob=args.mask_prob,
            target_veh_class=args.veh_class,
            target_dir=args.dir,
            use_database=False,
        )
    return ds
```

**CLI Flag**:
```bash
python src/train.py --use-database --start-date 2026-06-01 --end-date 2026-06-15
```

---

### Phase 2 Implementation Checklist

| Task | File(s) | Estimated Time |
|------|---------|----------------|
| Add `timescaledb` service to docker-compose | `docker-compose.yml`, `.env` | 30 min |
| Install Alembic + create initial migration | `alembic/versions/001_initial_schema.py` | 1 hour |
| Test migration: `alembic upgrade head` | CLI | 15 min |
| Implement `scripts/ingest_parquet.py` | `scripts/ingest_parquet.py` | 2 hours |
| Run ingestion (121M rows, ~2-3 hours) | CLI | 3 hours |
| Verify hypertable: `SELECT count(*) FROM traffic_readings;` | SQL | 15 min |
| Refactor `RealIoTDataset` with `use_database` flag | `src/dataset.py` | 2 hours |
| Add `--use-database` CLI flag to `train.py` | `src/train.py` | 30 min |
| Test database-backed training run (1 epoch) | CLI | 1 hour |
| Update `EXPERIMENTS.md` with Phase 2 completion entry | `EXPERIMENTS.md` | 15 min |
| **Total Estimated Effort** | | **~11 hours** (excluding 3-hour ingestion wait time) |

---

## Phase 3: CDN Migration (Cloudflare R2 Completion)

### Objectives

1. **Proxy `/data/runs/*` through Next.js** to R2 (`R2_PUBLIC_URL`) via rewrite rules.
2. **Refactor `payloadSource.ts`** to resolve URLs relatively (no hardcoded `.r2.dev` domain).
3. **Define edge caching headers** (immutable for payloads, short-lived for registry).
4. **Automate publishing pipeline** using self-hosted GitHub Actions runner on GPU server.

---

### Architecture Components

#### 3.1 Next.js Rewrite Proxy

**File**: `frontend/next.config.ts`

**Current State**: Empty config object.

**Updated Config**:

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const R2_PUBLIC_URL = process.env.R2_PUBLIC_URL ||
                          "https://pub-b102fa91750348f78a6385248654688a.r2.dev";

    return [
      {
        source: "/data/runs/:path*",
        destination: `${R2_PUBLIC_URL}/runs/:path*`,
      },
    ];
  },

  async headers() {
    return [
      {
        // Immutable cache for historical payloads (spatial.json, page.json, node_*.json)
        source: "/data/runs/:runId/:date*/(.*).(json)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        // Short-lived cache for run_registry.json
        source: "/data/runs/run_registry.json",
        headers: [
          {
            key: "Cache-Control",
            value: "public, s-maxage=60, stale-while-revalidate=30",
          },
        ],
      },
      {
        // Immutable cache for spatial.json (run-level, no date)
        source: "/data/runs/:runId/spatial.json",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
```

**Design Rationale**:
- **Rewrites** transparently proxy `/data/runs/*` requests to R2, avoiding CORS preflight (browser sees same-origin request).
- **Immutable cache** for historical payloads: once a run's date is published, it never changes → cache indefinitely.
- **`s-maxage=60, stale-while-revalidate=30`** for `run_registry.json`: CDN edge caches for 60s, serves stale for 30s while revalidating in background → low latency for new runs without overwhelming R2.
- **No CORS config needed** (rewrites handle it).

**Verification**:
```bash
curl -I http://localhost:3000/data/runs/run_registry.json
# Expected: Cache-Control: public, s-maxage=60, stale-while-revalidate=30

curl -I http://localhost:3000/data/runs/feasible-capybara-22/spatial.json
# Expected: Cache-Control: public, max-age=31536000, immutable
```

---

#### 3.2 Refactor `payloadSource.ts`

**File**: `frontend/src/lib/payloadSource.ts`

**Current State**: Hardcoded R2 public URL (`https://pub-...r2.dev`).

**Updated (Relative URLs)**:

```typescript
// Central resolver for paginated-payload URLs.
//
// Phase 3 (CDN migration): payloads are proxied through Next.js rewrites
// (/data/runs/* → R2_PUBLIC_URL/runs/*), so all URLs are relative to the
// frontend origin. This avoids CORS, enables Vercel edge caching, and
// centralizes cache-control headers in next.config.ts.

/** Key prefix under the bucket where the run tree lives. */
const PREFIX = "data/runs";

export const registryUrl = (): string =>
  `/${PREFIX}/run_registry.json`;

export const spatialUrl = (runId: string): string =>
  `/${PREFIX}/${runId}/spatial.json`;

export const pageUrl = (runId: string, date: string): string =>
  `/${PREFIX}/${runId}/${date}/page.json`;

export const nodeUrl = (runId: string, date: string, nodeId: number): string =>
  `/${PREFIX}/${runId}/${date}/nodes/node_${nodeId}.json`;
```

**Breaking Change**: This removes the `PAYLOAD_BASE` and `RAW_BASE` env vars — no longer needed since Next.js handles the proxy.

**Verification**:
```typescript
// In browser console:
fetch(registryUrl()).then(r => r.json()).then(console.log)
// Should return the registry without CORS errors
```

---

#### 3.3 Automated Publishing Pipeline (Self-Hosted Runner)

**Challenge**:
- `master_dataset_15T.parquet` (5.8 GB) is `.gitignored` and lives only on the GPU server.
- GitHub-hosted runners cannot access this file.
- **Solution**: Use a **self-hosted GitHub Actions runner** on the GPU server.

---

##### **Self-Hosted Runner Setup** (One-Time)

**On GPU Server**:

1. **Install GitHub Actions Runner**:
```bash
# Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download runner (Linux x64)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
```

2. **Configure Runner** (requires GitHub PAT with `repo` and `admin:org` scopes):
```bash
./config.sh --url https://github.com/YOUR_USERNAME/ml-agent-workspace \
            --token YOUR_RUNNER_TOKEN \
            --name gpu-server-runner \
            --labels self-hosted,gpu,linux,payload-builder \
            --work _work
```

3. **Install as systemd service**:
```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

4. **Verify in GitHub**:
   - Go to **Settings → Actions → Runners**
   - Should see `gpu-server-runner` with status "Idle"

---

##### **GitHub Actions Workflow**: `.github/workflows/publish_payloads.yml`

```yaml
name: Build and Publish Payloads to R2

on:
  workflow_dispatch:  # Manual trigger
    inputs:
      artifact_ref:
        description: 'W&B artifact ref (e.g., entity/project/name:v0)'
        required: false
      all_artifacts:
        description: 'Process all artifacts in collection'
        type: boolean
        default: false
      start_date:
        description: 'Analysis start date (YYYY-MM-DD)'
        required: false
        default: '2026-06-01'
      end_date:
        description: 'Analysis end date (YYYY-MM-DD)'
        required: false
        default: '2026-07-01'

jobs:
  build-and-upload:
    runs-on: [self-hosted, gpu, payload-builder]  # Target GPU server runner

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install torch numpy pandas wandb pyyaml boto3 scikit-learn pyarrow

      - name: Authenticate W&B
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb login --relogin

      - name: Export paginated payload
        run: |
          ARGS="--config configs/analysis/analysis.yaml \
                --output_dir frontend/public/data/runs \
                --start_date ${{ github.event.inputs.start_date }} \
                --end_date ${{ github.event.inputs.end_date }}"

          if [ "${{ github.event.inputs.all_artifacts }}" == "true" ]; then
            ARGS="$ARGS --all_artifacts"
          elif [ -n "${{ github.event.inputs.artifact_ref }}" ]; then
            ARGS="$ARGS --artifact ${{ github.event.inputs.artifact_ref }}"
          fi

          python src/export_paginated_payload.py $ARGS

      - name: Build run registry
        run: |
          python scripts/build_run_registry.py frontend/public/data/runs

      - name: Upload to R2
        env:
          R2_ACCOUNT_ID: ${{ secrets.R2_ACCOUNT_ID }}
          R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          R2_BUCKET_NAME: ${{ secrets.R2_BUCKET_NAME }}
          R2_PUBLIC_URL: ${{ secrets.R2_PUBLIC_URL }}
        run: |
          python scripts/upload_to_r2.py \
            --runs-dir frontend/public/data/runs \
            --prefix runs

      - name: Clean up local payloads
        run: |
          rm -rf frontend/public/data/runs/*
          echo "Payloads published to R2 and local artifacts cleaned up"
```

**GitHub Secrets** (add via Settings → Secrets → Actions):
- `WANDB_API_KEY`
- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET_NAME`
- `R2_PUBLIC_URL`

**Manual Trigger**:
1. Go to **Actions → Build and Publish Payloads to R2**
2. Click **Run workflow**
3. Fill in parameters (artifact ref, date range)
4. Monitor logs in real-time

---

##### **Alternative: CRON Script** (Non-GitHub Approach)

For users preferring local automation without GitHub Actions:

**File**: `scripts/publish_payloads_cron.sh`

```bash
#!/bin/bash
set -euo pipefail

REPO_ROOT="/path/to/ml-agent-workspace"
cd "$REPO_ROOT"

# Load environment
source .env

# Pull latest code
git pull origin main

# Export payloads
python src/export_paginated_payload.py \
    --all_artifacts \
    --start_date 2026-06-01 \
    --end_date 2026-07-01 \
    --output_dir frontend/public/data/runs

# Build registry
python scripts/build_run_registry.py frontend/public/data/runs

# Upload to R2
python scripts/upload_to_r2.py \
    --runs-dir frontend/public/data/runs \
    --prefix runs

# Clean up
rm -rf frontend/public/data/runs/*

echo "Payload publish complete at $(date)" >> logs/publish.log
```

**Crontab Entry** (Daily at 2 AM):
```cron
0 2 * * * /path/to/ml-agent-workspace/scripts/publish_payloads_cron.sh
```

---

### Phase 3 Implementation Checklist

| Task | File(s) | Estimated Time |
|------|---------|----------------|
| Add rewrites to `next.config.ts` | `frontend/next.config.ts` | 30 min |
| Add headers to `next.config.ts` | `frontend/next.config.ts` | 30 min |
| Refactor `payloadSource.ts` to relative URLs | `frontend/src/lib/payloadSource.ts` | 15 min |
| Test proxy locally: `npm run dev` + curl | CLI | 30 min |
| Set up self-hosted GitHub runner on GPU server | GPU server | 1 hour |
| Create `.github/workflows/publish_payloads.yml` | `.github/workflows/publish_payloads.yml` | 1 hour |
| Add GitHub Secrets (R2 credentials, W&B API key) | GitHub UI | 15 min |
| Test workflow: manual trigger with 1 artifact | GitHub Actions | 1 hour |
| Verify CDN cache headers in production | Browser DevTools | 30 min |
| Update `EXPERIMENTS.md` with Phase 3 completion entry | `EXPERIMENTS.md` | 15 min |
| **Total Estimated Effort** | | **~6 hours** |

---

## Critical Paths and Dependencies

```mermaid
graph TD
    P2A[Phase 2: Docker Compose Service] --> P2B[Phase 2: Alembic Migrations]
    P2B --> P2C[Phase 2: Ingestion Script]
    P2C --> P2D[Phase 2: Dataset Refactor]

    P3A[Phase 3: Next.js Rewrites] --> P3B[Phase 3: payloadSource.ts Refactor]
    P3B --> P3C[Phase 3: Self-Hosted Runner Setup]
    P3C --> P3D[Phase 3: GitHub Actions Workflow]

    P2D -.-> P3D[Optional: Database-sourced payloads]

    style P2A fill:#f59e0b
    style P2B fill:#f59e0b
    style P2C fill:#f59e0b
    style P2D fill:#f59e0b
    style P3A fill:#22c55e
    style P3B fill:#22c55e
    style P3C fill:#22c55e
    style P3D fill:#22c55e
```

**Phase 2 Critical Path**: Docker → Migrations → Ingestion → Dataset Refactor (sequential, ~11 hours excluding ingestion wait time).

**Phase 3 Critical Path**: Rewrites → payloadSource → Runner → Workflow (sequential, ~6 hours).

**Cross-Phase Dependency** (optional): Phase 2's database backend *could* eventually source `export_paginated_payload.py`, but is NOT required for Phase 3 completion.

---

## Success Criteria

### Phase 2 (Training Database)

| Criterion | Verification Method |
|-----------|---------------------|
| TimescaleDB service healthy | `docker exec ml-timescaledb pg_isready` returns success |
| Schema migrated | `alembic current` shows migration `001` applied |
| Data ingested | `SELECT COUNT(*) FROM traffic_readings;` returns ~121M rows |
| Dataset queries DB | `python src/train.py --use-database --start-date 2026-06-01 --end-date 2026-06-02` completes 1 epoch without errors |
| Parquet fallback works | Rename `.env` → `.env.bak`, verify train.py falls back to Parquet |
| Experiment logged | `EXPERIMENTS.md` contains Phase 2 completion entry with branch name and date |

### Phase 3 (CDN Migration)

| Criterion | Verification Method |
|-----------|---------------------|
| Proxy rewrites work | `curl -I http://localhost:3000/data/runs/run_registry.json` returns 200 from R2 |
| Cache headers correct | Registry has `s-maxage=60`, spatial.json has `max-age=31536000, immutable` |
| Relative URLs resolve | Browser DevTools → Network → all `/data/runs/*` requests succeed |
| GitHub runner active | **Settings → Actions → Runners** shows `gpu-server-runner` status "Idle" |
| Workflow completes | Manual trigger of `publish_payloads.yml` uploads payloads to R2 successfully |
| Vercel deploy lightweight | `vercel deploy` completes in <60s (no 500MB static payload bloat) |
| Experiment logged | `EXPERIMENTS.md` contains Phase 3 completion entry with branch name and date |

---

## Rollback Strategy

### Phase 2 Rollback

If database integration introduces regressions:

1. **Disable database mode**: Remove `--use-database` flag from training commands.
2. **Revert dataset.py**: `git checkout main -- src/dataset.py`
3. **Stop TimescaleDB** (optional): `docker-compose stop timescaledb`
4. **Drop hypertables** (nuclear option): `alembic downgrade base`

**Data Loss Risk**: Low — Parquet files remain untouched; database is additive-only.

### Phase 3 Rollback

If CDN proxy breaks frontend:

1. **Revert next.config.ts**: `git checkout main -- frontend/next.config.ts`
2. **Revert payloadSource.ts**: `git checkout main -- frontend/src/lib/payloadSource.ts`
3. **Redeploy**: `vercel deploy --prod`
4. **Temporary static hosting**: Copy payloads back to `frontend/public/data/runs/` and commit (bloated but functional).

**Data Loss Risk**: None — R2 bucket retains all payloads.

---

## Post-Implementation Tasks

### Phase 2
1. **Monitor database performance**: Log query times in `dataset.py` and compare to Parquet baseline.
2. **Add weather hypertable** (future): Migrate `weather_20230108_20260108.parquet` to a `weather_readings` hypertable.
3. **Backfill model_runs**: Write a script to scrape W&B runs and populate `model_runs` table retroactively.

### Phase 3
1. **Monitor R2 bandwidth**: Cloudflare dashboard → R2 Analytics → track egress (should remain $0).
2. **Set up Vercel Analytics**: Track `/data/runs/*` cache hit rates.
3. **Document payload versioning**: Add a `PAYLOAD_CHANGELOG.md` to track breaking schema changes.

---

## Experiment Registry Requirement

**MANDATORY**: Before marking either phase as complete, the implementing agent MUST update `EXPERIMENTS.md` with:

```markdown
| 2026-06-XX | feat/phase2-database | Phase 2 (Training Database): Dockerized TimescaleDB, Alembic migrations, Parquet ingestion, dataset refactor with fallback. | **[STATUS].** [DETAILS] |
| 2026-06-XX | feat/phase3-cdn | Phase 3 (CDN Migration): Next.js rewrites, relative URLs, edge caching, self-hosted GitHub Actions runner for R2 publishing. | **[STATUS].** [DETAILS] |
```

Replace `[STATUS]` with one of: **Implemented**, **Fixed**, **In Progress**, **Blocked**.

Replace `[DETAILS]` with a 2-3 sentence summary of outcome, blockers, or verification results.

---

## References

- [Implementation Phases](file:///Users/administrator/Code/ml-agent-workspace/docs/database/implementation_phases.md)
- [Database Serving Strategy](file:///Users/administrator/Code/ml-agent-workspace/docs/database/database_serving_strategy.md)
- [TimescaleDB Hypertables Docs](https://docs.timescale.com/use-timescale/latest/hypertables/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [Next.js Rewrites](https://nextjs.org/docs/app/api-reference/next-config-js/rewrites)
- [GitHub Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners)
- [Cloudflare R2 Cache-Control](https://developers.cloudflare.com/r2/buckets/public-buckets/#cache-control)

---

**END OF PLAN**
