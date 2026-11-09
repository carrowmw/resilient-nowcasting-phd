# Container Usage Guide for Database & Pipeline Operations

> **Quick Answer**: Run Alembic migrations, data ingestion scripts, and CPU evaluations from the **`dev-workspace`** (or **`cpu-runner`**) container, and GPU training from **`gpu-runner`**. Never run Python scripts inside the `timescaledb` container.

```bash
docker compose build dev-workspace
```

---

## Container Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                         Docker Host                               │
│                                                                   │
│  ┌──────────────────────────┐      ┌──────────────────────────┐ │
│  │  dev-workspace           │      │   timescaledb            │ │
│  │  (Python environment)    │──────│   (Database server)      │ │
│  │                          │      │                          │ │
│  │  ✓ Python 3.11           │      │  ✓ PostgreSQL 16         │ │
│  │  ✓ Alembic               │      │  ✓ TimescaleDB extension │ │
│  │  ✓ psycopg2              │      │  ✓ Port: 5432            │ │
│  │  ✓ Training scripts      │      │  ✗ No Python             │ │
│  │  ✓ Ingestion scripts     │      │  ✗ No Alembic            │ │
│  │  ✓ PyTorch               │      │  ✗ No scripts            │ │
│  └──────────────────────────┘      └──────────────────────────┘ │
│             │                                  │                 │
│             └──────────ml-network──────────────┘                 │
│                                                                   │
│  DATABASE_URL: postgresql://riotuser:riotpassword@               │
│                timescaledb:5432/riot_traffic                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Command Reference

### 1. Build & Start Containers

```bash
# Build development workspace
docker compose build dev-workspace

# Start TimescaleDB and dev-workspace
docker compose up -d timescaledb dev-workspace

# Verify both are running
docker compose ps
```

### 2. Apply Database Migrations

```bash
# Run Alembic from dev-workspace
docker compose exec dev-workspace alembic upgrade head

# Verify tables in TimescaleDB
docker exec ml-timescaledb psql -U riotuser -d riot_traffic -c "\dt"
```

### 3. Ingest Data

```bash
# Ingest sensor metadata, static features, weather cache, and traffic readings
docker compose exec dev-workspace python scripts/ingestion/ingest_sensor_metadata.py
docker compose exec dev-workspace python scripts/ingestion/ingest_static_features.py
docker compose exec dev-workspace python scripts/ingestion/ingest_weather.py
docker compose exec dev-workspace python scripts/ingestion/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --create-dataset-version
```

### 4. Run Training

```bash
# CPU smoke test
docker compose exec dev-workspace python src/train.py --use_mock_data

# Full GPU training run
docker compose run --rm gpu-runner python src/train.py \
    --config configs/train/v7/g5_epochs/v7_profile_static_e150__s47.yaml
```

---

## Summary

| Task | Container | Command |
| :--- | :--- | :--- |
| **Apply migrations** | `dev-workspace` | `docker compose exec dev-workspace alembic upgrade head` |
| **Ingest data** | `dev-workspace` | `docker compose exec dev-workspace python scripts/ingestion/ingest_parquet.py ...` |
| **Train model (GPU)** | `gpu-runner` | `docker compose run --rm gpu-runner python src/train.py --config ...` |
| **Query database** | `timescaledb` | `docker exec ml-timescaledb psql -U riotuser -d riot_traffic` |
| **Interactive shell** | `dev-workspace` | `docker compose exec dev-workspace bash` |
