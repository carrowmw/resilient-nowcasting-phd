# Phase 2 Migration Notes

> **For existing users pulling Phase 2 changes**

---

## What Changed

Phase 2 adds database infrastructure with the following new dependencies:
- `alembic` - Database migration framework
- `psycopg2-binary` - PostgreSQL adapter for Python
- `python-dotenv` - Environment variable loader
- `postgresql-client` - System package for pg_isready command

These are now included in:
- `requirements.txt` (Python packages)
- `Dockerfile.agent` (system packages + Python packages)

---

## Required Action: Rebuild Container

**You MUST rebuild the agent-workspace container** to install these dependencies:

```bash
# From repository root
docker-compose build agent-workspace

# This will take a few minutes
# Expected output: "Successfully built..." and "Successfully tagged..."
```

---

## Verification

After rebuilding, verify dependencies are installed:

```bash
# Check Python packages
docker-compose run --rm agent-workspace pip list | grep -E "alembic|psycopg2|dotenv"

# Expected output:
#   alembic           1.x.x
#   psycopg2-binary   2.x.x
#   python-dotenv     1.x.x

# Check system packages
docker-compose run --rm agent-workspace which pg_isready

# Expected output: /usr/bin/pg_isready
```

---

## What If I Don't Rebuild?

If you skip the rebuild, you'll see errors like:

```
ModuleNotFoundError: No module named 'alembic'
ModuleNotFoundError: No module named 'psycopg2'
ModuleNotFoundError: No module named 'dotenv'
```

**Solution**: Run `docker-compose build agent-workspace`

---

## Existing Workflows Still Work

Phase 2 is **additive only** - existing training workflows continue to work unchanged:

```bash
# Standard training (no database) - still works
docker-compose exec agent-workspace python src/train.py \
    --config configs/train/train.yaml

# GPU training - still works
docker-compose exec gpu-runner python src/train.py \
    --config configs/train/train.yaml
```

The database is **opt-in** via the `--use-database` flag.

---

## New Workflows Available

After rebuilding, you can use the new database features:

```bash
# 1. Start database
docker-compose up -d timescaledb

# 2. Apply migrations
docker-compose exec agent-workspace alembic upgrade head

# 3. Ingest data
docker-compose exec agent-workspace python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet

# 4. Train with database
docker-compose exec agent-workspace python src/train.py \
    --use-database --config configs/train/train.yaml
```

---

## Files Changed

| File | Change | Impact |
|------|--------|--------|
| `requirements.txt` | Added alembic, psycopg2-binary, python-dotenv | Requires container rebuild |
| `Dockerfile.agent` | Added postgresql-client system package | Requires container rebuild |
| `docker-compose.yml` | Added timescaledb service + networks | No rebuild needed |
| `.env` | Added DATABASE_URL variables | Auto-loaded by containers |
| `src/dataset.py` | Added use_database parameter | Backwards compatible |
| `src/train.py` | Added --use-database flag | Opt-in, default unchanged |

---

## Timeline

```
Before Rebuild                After Rebuild
──────────────               ──────────────
✓ Training works             ✓ Training works
✗ Alembic fails              ✓ Alembic works
✗ Database mode fails        ✓ Database mode works
```

---

## Quick Reference

**One-time setup (required)**:
```bash
docker-compose build agent-workspace
```

**Phase 2 startup** (optional, only if using database):
```bash
docker-compose up -d timescaledb agent-workspace
docker-compose exec agent-workspace alembic upgrade head
```

**Continue with**: [PHASE2_PHASE3_QUICKSTART.md](PHASE2_PHASE3_QUICKSTART.md)

---

## Questions?

- **Container guide**: [docs/CONTAINER_USAGE_GUIDE.md](docs/CONTAINER_USAGE_GUIDE.md)
- **Full implementation**: [.agents/IMPLEMENTATION_SUMMARY.md](.agents/IMPLEMENTATION_SUMMARY.md)
- **Experiment ledger**: See `EXPERIMENTS.md` entry for 2026-06-29
