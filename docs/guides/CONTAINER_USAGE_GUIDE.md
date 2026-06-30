# Container Usage Guide for Phase 2 Database Operations

> **Quick Answer**: Run Alembic, ingestion scripts, and training from the **agent-workspace** container. All dependencies are installed in the container. Never run them from inside the TimescaleDB container.

> **IMPORTANT**: After pulling Phase 2 changes, you must rebuild the agent-workspace container to install new dependencies (alembic, psycopg2-binary, python-dotenv):
> ```bash
> docker-compose build agent-workspace
> ```

---

## Container Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Docker Host                               │
│                                                                   │
│  ┌──────────────────────────┐      ┌──────────────────────────┐ │
│  │  agent-workspace         │      │   timescaledb            │ │
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
│                     ↑                                             │
│                hostname (Docker DNS)                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why Not Run Scripts in TimescaleDB Container?

**TimescaleDB container** is a pure database server:
- ❌ No Python installed
- ❌ No pip packages (Alembic, psycopg2, PyTorch, etc.)
- ❌ No access to repository files (`/workspace`)
- ✅ Only PostgreSQL + TimescaleDB extension

**agent-workspace container** has everything needed:
- ✅ Python 3.11 + all dependencies
- ✅ Repository mounted at `/workspace`
- ✅ Can connect to TimescaleDB via `DATABASE_URL`

---

## Prerequisites

Before running any Phase 2 commands, rebuild the agent-workspace container with Phase 2 dependencies:

```bash
# Rebuild agent-workspace (includes alembic, psycopg2-binary, python-dotenv)
docker-compose build agent-workspace

# Verify dependencies installed
docker-compose run --rm agent-workspace pip list | grep -E "alembic|psycopg2|dotenv"
# Should show:
#   alembic           X.X.X
#   psycopg2-binary   X.X.X
#   python-dotenv     X.X.X
```

---

## Command Reference

All commands run from **Docker host** (GPU server), executing inside containers:

### Start Containers

```bash
# Start TimescaleDB
docker-compose up -d timescaledb

# Start agent-workspace (if not already running)
docker-compose up -d agent-workspace

# Verify both running
docker-compose ps
# Should show: ml-timescaledb (Up), my-ml-agent (Up)
```

### Apply Migrations

```bash
# Run Alembic from agent-workspace
docker-compose exec agent-workspace alembic upgrade head

# Verify tables
docker exec ml-timescaledb psql -U riotuser -d riot_traffic -c "\dt"
```

### Ingest Data

```bash
# Create logs directory (on host, visible in container)
mkdir -p logs

# Dry run
docker-compose exec agent-workspace python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --dry-run

# Full ingestion (background in container)
docker-compose exec -d agent-workspace sh -c \
  "python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --create-dataset-version > logs/ingestion.log 2>&1"

# Monitor from host
tail -f logs/ingestion.log
```

### Test Training

```bash
# Run training with database mode
docker-compose exec agent-workspace python src/train.py \
    --use-database \
    --config configs/train/train.yaml
```

### Interactive Shell

```bash
# Enter agent-workspace container
docker-compose exec agent-workspace bash

# Now you're inside the container, run commands normally:
# alembic upgrade head
# python scripts/ingest_parquet.py --dry-run
# python src/train.py --use-database
# exit
```

---

## How Database Connection Works

When you run `docker-compose exec agent-workspace alembic upgrade head`:

1. Alembic reads `DATABASE_URL` from `.env`:
   ```
   postgresql://riotuser:riotpassword@timescaledb:5432/riot_traffic
                                       ↑
                                  hostname
   ```

2. Docker's internal DNS resolves `timescaledb` to the container IP

3. Connection is made over the `ml-network` bridge network

4. No ports need to be exposed (but we expose 5432 for external tools like pgAdmin)

---

## Complete Workflow

```bash
# 1. Rebuild agent-workspace with Phase 2 dependencies (one-time)
docker-compose build agent-workspace

# 2. Start containers
docker-compose up -d timescaledb agent-workspace

# 3. Apply migrations
docker-compose exec agent-workspace alembic upgrade head

# 4. Ingest data (background job)
mkdir -p logs
docker-compose exec -d agent-workspace sh -c \
  "python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --create-dataset-version > logs/ingestion.log 2>&1"

# 5. Monitor progress
tail -f logs/ingestion.log

# 6. Test training
docker-compose exec agent-workspace python src/train.py \
    --use-database --config configs/train/train.yaml
```

---

## Troubleshooting

### Error: "No module named 'alembic'" or "No module named 'psycopg2'"

**Cause**: agent-workspace container not rebuilt after pulling Phase 2 changes.

**Solution**: Rebuild the container:
```bash
docker-compose build agent-workspace
docker-compose up -d agent-workspace
```

---


### Error: "Connection refused" (port 5432)

**Cause**: TimescaleDB not running or not healthy.

**Check**:
```bash
docker-compose ps timescaledb
docker exec ml-timescaledb pg_isready
```

**Solution**:
```bash
docker-compose up -d timescaledb
docker logs -f ml-timescaledb  # Watch startup
```

---

### Error: "Failed to load parquet" when ingesting

**Cause**: Parquet file not accessible from container.

**Check**: Repository is mounted correctly:
```bash
docker-compose exec agent-workspace ls -lh data/processed/master_dataset_15T.parquet
```

**Solution**: Ensure volume mount in `docker-compose.yml`:
```yaml
agent-workspace:
  volumes:
    - .:/workspace  # ← Repository mounted
```

---

## Summary

| Task | Container | Command |
|------|-----------|---------|
| **Apply migrations** | agent-workspace | `docker-compose exec agent-workspace alembic upgrade head` |
| **Ingest data** | agent-workspace | `docker-compose exec agent-workspace python scripts/ingest_parquet.py ...` |
| **Train model** | agent-workspace or gpu-runner | `docker-compose exec agent-workspace python src/train.py --use-database` |
| **Query database** | timescaledb | `docker exec ml-timescaledb psql -U riotuser -d riot_traffic` |
| **Interactive shell** | agent-workspace | `docker-compose exec agent-workspace bash` |

**Golden Rule**: 
- Python scripts (Alembic, ingestion, training) → **agent-workspace** or **local**
- SQL queries → **timescaledb** (via `psql`)
- Database server runs in → **timescaledb** (don't enter this container for Python work)

---

## References

- **Quick Start Guide**: [PHASE2_PHASE3_QUICKSTART.md](../PHASE2_PHASE3_QUICKSTART.md)
- **Implementation Details**: [.agents/IMPLEMENTATION_SUMMARY.md](../.agents/IMPLEMENTATION_SUMMARY.md)
- **docker-compose.yml**: [../docker-compose.yml](../docker-compose.yml)
