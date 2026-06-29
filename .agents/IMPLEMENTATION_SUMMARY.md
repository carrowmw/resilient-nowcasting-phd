# Phase 2 & 3 Implementation Summary

> **Implementer**: Implementation Agent  
> **Date**: 2026-06-29  
> **Status**: ✅ Complete (Pending Verification)  
> **Plan Reference**: [.agents/plans/architect_plan.md](file:///Users/administrator/Code/ml-agent-workspace/.agents/plans/architect_plan.md)

---

## Executive Summary

Successfully implemented all components for:
- **Phase 2 (Training Database)**: Complete database infrastructure with TimescaleDB, Alembic migrations, ingestion tooling, and dataset refactoring
- **Phase 3 (CDN Migration)**: Full Next.js proxy configuration, payload URL refactoring, and automated publishing pipeline

**Total Implementation Time**: ~3 hours  
**Files Changed**: 11 files created/modified  
**Lines Added**: ~800 lines

---

## Phase 2: Training Database Implementation

### ✅ Task 1: TimescaleDB Docker Service

**Files Modified**:
- `docker-compose.yml` - Added `timescaledb` service
- `.env` - Added database credentials

**Changes**:
```yaml
# New service in docker-compose.yml
timescaledb:
  image: timescale/timescaledb:latest-pg16
  container_name: ml-timescaledb
  environment:
    POSTGRES_USER: ${POSTGRES_USER:-riotuser}
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-riotpassword}
    POSTGRES_DB: ${POSTGRES_DB:-riot_traffic}
  volumes:
    - timescaledb_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
  networks:
    - ml-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-riotuser}"]
    interval: 10s
    timeout: 5s
    retries: 5
```

**Environment Variables**:
```bash
POSTGRES_USER=riotuser
POSTGRES_PASSWORD=riotpassword
POSTGRES_DB=riot_traffic
DATABASE_URL=postgresql://riotuser:riotpassword@timescaledb:5432/riot_traffic
```

**Verification Pending**:
```bash
# Start database
docker-compose up -d timescaledb

# Verify health
docker exec ml-timescaledb pg_isready
```

---

### ✅ Task 2: Alembic Migration Framework

**Files Created**:
- `alembic.ini` - Alembic configuration
- `alembic/env.py` - Environment loader with .env support
- `alembic/versions/001_initial_schema.py` - Initial schema migration

**Key Features**:
- **Dynamic URL loading**: Reads `DATABASE_URL` from `.env` via `python-dotenv`
- **Three core tables**:
  - `traffic_readings` - TimescaleDB hypertable (7-day chunks, 3 indexes)
  - `model_runs` - W&B metadata mirror with denormalized metrics
  - `dataset_versions` - Branch-aware preprocessing versioning
- **Automatic TimescaleDB extension** installation

**Schema Highlights**:
```sql
-- Hypertable with 7-day chunks for efficient time-range queries
CREATE TABLE traffic_readings (
    dt TIMESTAMPTZ NOT NULL,
    sensor_id TEXT NOT NULL,
    veh_class TEXT NOT NULL,
    direction TEXT NOT NULL,
    count REAL NOT NULL,
    PRIMARY KEY (dt, sensor_id, veh_class, direction)
);
SELECT create_hypertable('traffic_readings', 'dt', 
                         chunk_time_interval => INTERVAL '7 days');

-- Indexes for common query patterns
CREATE INDEX idx_traffic_sensor_dt ON traffic_readings (sensor_id, dt DESC);
CREATE INDEX idx_traffic_vehclass ON traffic_readings (veh_class, dt DESC);
CREATE INDEX idx_traffic_direction ON traffic_readings (direction, dt DESC);
```

**Verification Pending**:
```bash
# Apply migrations
alembic upgrade head

# Verify schema
docker exec ml-timescaledb psql -U riotuser -d riot_traffic -c "\dt"
```

---

### ✅ Task 3: Parquet Ingestion Script

**Files Created**:
- `scripts/ingest_parquet.py` - Bulk ingestion tool

**Features**:
- **Batch processing**: 100k rows per batch using `psycopg2.extras.execute_values`
- **Conflict handling**: `ON CONFLICT DO NOTHING` for idempotent ingestion
- **Dataset versioning**: Optional `--create-dataset-version` flag
- **SHA256 checksumming**: Reproducibility via parquet file hashing
- **Git metadata capture**: Branch + commit tracking
- **Dry-run mode**: Preview without database writes

**Performance Estimate**:
- 121M rows ÷ 100k batch = ~1,210 batches
- ~5-10 seconds per batch
- **Total: ~2-3 hours** for full ingestion

**Verification Pending**:
```bash
# Dry run
python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --dry-run

# Full ingestion
python scripts/ingest_parquet.py \
    --parquet data/processed/master_dataset_15T.parquet \
    --create-dataset-version

# Verify count
docker exec ml-timescaledb psql -U riotuser -d riot_traffic \
    -c "SELECT COUNT(*) FROM traffic_readings;"
```

---

### ✅ Task 4: Dataset Refactor for Database Queries

**Files Modified**:
- `src/dataset.py` - Added `use_database` parameter and helper methods
- `src/train.py` - Added `--use-database` CLI flag with fallback logic

**Changes**:

**dataset.py**:
```python
def __init__(self, 
             parquet_path: str, 
             start_date: str, 
             end_date: str, 
             seq_len: int = 12, 
             mask_prob: float = 0.15,
             target_veh_class: str = 'person',
             target_dir: str = 'all',
             use_database: bool = False):  # NEW
    
    if use_database:
        df = self._load_from_database(start_dt, end_dt, target_veh_class, target_dir)
    else:
        df = self._load_from_parquet(parquet_path, start_dt, end_dt, 
                                      target_veh_class, target_dir)

def _load_from_database(self, start_dt, end_dt, veh_class, direction):
    """Query TimescaleDB with filters, join with static features cache."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = conn.cursor()
    
    query = """
        SELECT dt, sensor_id, veh_class, direction, count
        FROM traffic_readings
        WHERE dt >= %s AND dt < %s AND veh_class = %s
        ORDER BY dt, sensor_id
    """
    cursor.execute(query, [start_dt, end_dt, veh_class])
    # ... merge with static features cache ...
```

**train.py**:
```python
parser.add_argument(
    "--use-database",
    action="store_true",
    help="Load data from TimescaleDB instead of Parquet (Phase 2)",
)

# Fallback logic in dataset instantiation
try:
    full_dataset = RealIoTDataset(..., use_database=use_db)
except Exception as db_exc:
    if use_db:
        logger.warning(f"Database failed ({db_exc}), falling back to Parquet")
        full_dataset = RealIoTDataset(..., use_database=False)
    else:
        raise
```

**Verification Pending**:
```bash
# Test database mode (requires TimescaleDB running + data ingested)
python src/train.py --use-database \
    --config configs/train/train.yaml \
    --start-date 2026-06-01 --end-date 2026-06-02

# Verify fallback
mv .env .env.bak
python src/train.py --use-database  # Should fall back to Parquet
mv .env.bak .env
```

---

## Phase 3: CDN Migration Implementation

### ✅ Task 5: Next.js Proxy Rewrites and Caching

**Files Modified**:
- `frontend/next.config.ts` - Complete rewrite

**Changes**:
```typescript
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
        // Immutable cache for historical payloads
        source: "/data/runs/:runId/:date*/(.*).(json)",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
      {
        // Short-lived cache for run_registry.json
        source: "/data/runs/run_registry.json",
        headers: [
          { key: "Cache-Control", value: "public, s-maxage=60, stale-while-revalidate=30" },
        ],
      },
      {
        // Immutable cache for spatial.json
        source: "/data/runs/:runId/spatial.json",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
    ];
  },
};
```

**Benefits**:
- ✅ **No CORS preflight** (same-origin proxy)
- ✅ **Vercel edge caching** enabled
- ✅ **Immutable payloads** cached indefinitely (1 year)
- ✅ **Registry** cached for 60s with stale-while-revalidate

**Verification Pending**:
```bash
# Start dev server
cd frontend && npm run dev

# Test proxy
curl -I http://localhost:3000/data/runs/run_registry.json
# Expected: Cache-Control: public, s-maxage=60, stale-while-revalidate=30

curl -I http://localhost:3000/data/runs/feasible-capybara-22/spatial.json
# Expected: Cache-Control: public, max-age=31536000, immutable
```

---

### ✅ Task 6: Refactor payloadSource.ts to Relative URLs

**Files Modified**:
- `frontend/src/lib/payloadSource.ts` - Complete rewrite

**Before**:
```typescript
const RAW_BASE = process.env.NEXT_PUBLIC_PAYLOAD_BASE_URL ||
  "https://pub-b102fa91750348f78a6385248654688a.r2.dev";
export const PAYLOAD_BASE = RAW_BASE.replace(/\/+$/, "");
const PREFIX = "runs";

export const registryUrl = (): string =>
  `${PAYLOAD_BASE}/${PREFIX}/run_registry.json`;
```

**After**:
```typescript
const PREFIX = "data/runs";

export const registryUrl = (): string => `/${PREFIX}/run_registry.json`;
export const spatialUrl = (runId: string): string => `/${PREFIX}/${runId}/spatial.json`;
export const pageUrl = (runId: string, date: string): string => 
  `/${PREFIX}/${runId}/${date}/page.json`;
export const nodeUrl = (runId: string, date: string, nodeId: number): string =>
  `/${PREFIX}/${runId}/${date}/nodes/node_${nodeId}.json`;
```

**Breaking Change**: Removed `PAYLOAD_BASE` and `RAW_BASE` env vars (no longer needed).

**Verification Pending**:
```typescript
// In browser console (with dev server running)
fetch(registryUrl()).then(r => r.json()).then(console.log)
// Should return registry without CORS errors
```

---

### ✅ Task 7: GitHub Actions Publishing Workflow

**Files Created**:
- `.github/workflows/publish_payloads.yml` - Automated publishing workflow
- `docs/SELF_HOSTED_RUNNER_SETUP.md` - Runner setup documentation

**Workflow Features**:
- **Manual trigger** via `workflow_dispatch`
- **Self-hosted runner** targeting `[self-hosted, gpu, payload-builder]` labels
- **Input parameters**:
  - `artifact_ref` - Single W&B artifact (optional)
  - `all_artifacts` - Process all artifacts (checkbox)
  - `start_date` / `end_date` - Analysis window
- **Steps**:
  1. Checkout repo
  2. Set up Python 3.11
  3. Install dependencies (torch, wandb, boto3, etc.)
  4. Authenticate W&B
  5. Export paginated payload (`export_paginated_payload.py`)
  6. Build run registry (`build_run_registry.py`)
  7. Upload to R2 (`upload_to_r2.py`)
  8. Clean up local artifacts

**Required GitHub Secrets**:
- `WANDB_API_KEY`
- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET_NAME`
- `R2_PUBLIC_URL`

**Self-Hosted Runner Setup** (documented in `docs/SELF_HOSTED_RUNNER_SETUP.md`):
```bash
# 1. Install runner
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# 2. Configure
./config.sh --url https://github.com/YOUR_USERNAME/ml-agent-workspace \
            --token YOUR_RUNNER_TOKEN \
            --name gpu-server-runner \
            --labels self-hosted,gpu,linux,payload-builder \
            --work _work

# 3. Install as systemd service
sudo ./svc.sh install
sudo ./svc.sh start
```

**Verification Pending**:
1. Set up self-hosted runner on GPU server
2. Add GitHub secrets
3. Go to **Actions → Build and Publish Payloads to R2**
4. Click **Run workflow**
5. Monitor logs for success

---

### ✅ Task 8: Update EXPERIMENTS.md

**Files Modified**:
- `EXPERIMENTS.md` - Added two new entries at top

**Entries Added**:

**Phase 2 Entry**:
```markdown
| 2026-06-29 | main | Phase 2 (Training Database): Dockerized TimescaleDB, 
Alembic migrations, Parquet ingestion script, dataset refactor with database 
fallback. | **Implemented.** Added `timescaledb` service to `docker-compose.yml` 
with persistent volumes, health checks, and `ml-network` bridge. Created Alembic 
migration `001_initial_schema.py` defining `traffic_readings` hypertable (7-day 
chunks, composite PK on dt/sensor_id/veh_class/direction), `model_runs` table 
(W&B metadata mirror with denormalized metrics), and `dataset_versions` table 
(branch-aware preprocessing versioning). Implemented `scripts/ingest_parquet.py` 
for bulk-loading 121M rows from Parquet with execute_values batching and SHA256 
checksumming. Refactored `src/dataset.py` with `use_database` parameter: added 
`_load_from_database()` method querying TimescaleDB via psycopg2, with automatic 
Parquet fallback on errors. Updated `src/train.py` with `--use-database` CLI flag 
and try/except fallback wrapper. Environment variables (`POSTGRES_USER`, 
`POSTGRES_PASSWORD`, `POSTGRES_DB`, `DATABASE_URL`) added to `.env`. **Not yet 
verified** (requires `docker-compose up timescaledb` + `alembic upgrade head` + 
ingestion run — database service currently inactive). **Files changed:** 
`docker-compose.yml`, `.env`, `alembic.ini`, `alembic/env.py`, 
`alembic/versions/001_initial_schema.py`, `scripts/ingest_parquet.py`, 
`src/dataset.py`, `src/train.py`. |
```

**Phase 3 Entry**:
```markdown
| 2026-06-29 | main | Phase 3 (CDN Migration): Next.js rewrites, relative URLs, 
edge caching, GitHub Actions workflow for R2 publishing. | **Implemented.** 
Updated `frontend/next.config.ts` with async `rewrites()` proxying 
`/data/runs/*` to `R2_PUBLIC_URL/runs/*` (avoids CORS, enables Vercel edge 
caching), and async `headers()` defining cache-control: `public, 
max-age=31536000, immutable` for historical payloads (spatial.json, page.json, 
node_*.json), `public, s-maxage=60, stale-while-revalidate=30` for 
`run_registry.json`. Refactored `frontend/src/lib/payloadSource.ts` to use 
relative `/data/runs/*` URLs (removed hardcoded R2 domain and 
`PAYLOAD_BASE`/`RAW_BASE` env vars). Created 
`.github/workflows/publish_payloads.yml` for self-hosted GitHub Actions runner 
(`[self-hosted, gpu, payload-builder]` labels) with manual trigger 
(workflow_dispatch), steps for W&B auth, payload export 
(`export_paginated_payload.py`), registry build (`build_run_registry.py`), R2 
upload (`upload_to_r2.py`), and cleanup. Created 
`docs/SELF_HOSTED_RUNNER_SETUP.md` documenting one-time runner installation, 
systemd service setup, GitHub secrets configuration (WANDB_API_KEY, 
R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, 
R2_PUBLIC_URL), and troubleshooting steps. **Not yet verified** (requires 
self-hosted runner registration on GPU server + GitHub secrets setup + manual 
workflow trigger — runner currently unregistered). **Files changed:** 
`frontend/next.config.ts`, `frontend/src/lib/payloadSource.ts`, 
`.github/workflows/publish_payloads.yml`, `docs/SELF_HOSTED_RUNNER_SETUP.md`. |
```

---

## Summary of Changes

### Files Created (5)
1. `alembic/versions/001_initial_schema.py` - Database schema migration
2. `scripts/ingest_parquet.py` - Parquet ingestion tool
3. `.github/workflows/publish_payloads.yml` - CI/CD workflow
4. `docs/SELF_HOSTED_RUNNER_SETUP.md` - Runner documentation
5. `.agents/IMPLEMENTATION_SUMMARY.md` - This file

### Files Modified (6)
1. `docker-compose.yml` - Added TimescaleDB service
2. `.env` - Added database credentials
3. `alembic.ini` - Configured for .env loading
4. `alembic/env.py` - Added dotenv support
5. `src/dataset.py` - Added database query methods
6. `src/train.py` - Added `--use-database` flag
7. `frontend/next.config.ts` - Complete rewrite for CDN proxy
8. `frontend/src/lib/payloadSource.ts` - Refactored to relative URLs
9. `EXPERIMENTS.md` - Added Phase 2 & 3 entries

---

## Verification Checklist

### Phase 2: Training Database

- [ ] **Docker Service**: `docker-compose up -d timescaledb`
  - [ ] Health check passes: `docker exec ml-timescaledb pg_isready`
  - [ ] Schema verified: `docker exec ml-timescaledb psql -U riotuser -d riot_traffic -c "\dt"`
  
- [ ] **Alembic Migrations**: `alembic upgrade head`
  - [ ] Current migration: `alembic current` shows `001`
  - [ ] Tables exist: `traffic_readings`, `model_runs`, `dataset_versions`
  
- [ ] **Data Ingestion**: `python scripts/ingest_parquet.py --parquet data/processed/master_dataset_15T.parquet --create-dataset-version`
  - [ ] Row count: `SELECT COUNT(*) FROM traffic_readings;` returns ~121M
  - [ ] Dataset version created: `SELECT * FROM dataset_versions;` has 1 row
  
- [ ] **Dataset Query**: `python src/train.py --use-database --start-date 2026-06-01 --end-date 2026-06-02`
  - [ ] Completes 1 epoch without errors
  - [ ] Parquet fallback works: Rename `.env`, verify fallback, restore `.env`

### Phase 3: CDN Migration

- [ ] **Next.js Proxy**: `cd frontend && npm run dev`
  - [ ] Registry cache header: `curl -I http://localhost:3000/data/runs/run_registry.json`
  - [ ] Spatial cache header: `curl -I http://localhost:3000/data/runs/{runId}/spatial.json`
  
- [ ] **Self-Hosted Runner**: On GPU server
  - [ ] Runner installed and configured
  - [ ] Service running: `sudo ~/actions-runner/svc.sh status`
  - [ ] GitHub shows "Idle" status
  
- [ ] **GitHub Secrets**: Settings → Secrets → Actions
  - [ ] All 6 secrets added (WANDB_API_KEY, R2_*, etc.)
  
- [ ] **Workflow Execution**: Actions → Build and Publish Payloads to R2
  - [ ] Manual trigger completes successfully
  - [ ] R2 bucket contains payloads

---

## Next Steps

### Immediate (Required for Verification)

1. **Start TimescaleDB**:
   ```bash
   docker-compose up -d timescaledb
   docker logs -f ml-timescaledb  # Watch startup
   ```

2. **Run Alembic Migrations**:
   ```bash
   alembic upgrade head
   ```

3. **Ingest Parquet Data** (2-3 hour operation):
   ```bash
   # Dry run first
   python scripts/ingest_parquet.py \
       --parquet data/processed/master_dataset_15T.parquet \
       --dry-run
   
   # Full ingestion (run in background)
   nohup python scripts/ingest_parquet.py \
       --parquet data/processed/master_dataset_15T.parquet \
       --create-dataset-version > logs/ingestion.log 2>&1 &
   
   # Monitor progress
   tail -f logs/ingestion.log
   ```

4. **Set Up Self-Hosted Runner** (GPU server):
   - Follow `docs/SELF_HOSTED_RUNNER_SETUP.md`
   - Add GitHub secrets
   - Test workflow trigger

### Future Enhancements (Optional)

1. **Phase 2 Extensions**:
   - Add weather hypertable (`weather_readings`)
   - Backfill `model_runs` from W&B API
   - Create static features cache Parquet (eliminate database dependency)

2. **Phase 3 Extensions**:
   - Monitor R2 bandwidth (should remain $0)
   - Set up Vercel Analytics for cache hit rates
   - Create `PAYLOAD_CHANGELOG.md` for schema versioning

3. **Performance Monitoring**:
   - Log database query times in `dataset.py`
   - Compare to Parquet baseline
   - Optimize indexes if needed

---

## Rollback Procedures

### Phase 2 Rollback

If database introduces issues:

```bash
# 1. Disable database mode
# Remove --use-database from training commands

# 2. Revert dataset.py
git checkout main -- src/dataset.py

# 3. Stop TimescaleDB (optional)
docker-compose stop timescaledb

# 4. Drop hypertables (nuclear option)
alembic downgrade base
```

### Phase 3 Rollback

If CDN proxy breaks:

```bash
# 1. Revert configs
git checkout main -- frontend/next.config.ts frontend/src/lib/payloadSource.ts

# 2. Redeploy
vercel deploy --prod

# 3. Temporary static hosting (if needed)
# Copy payloads back to frontend/public/data/runs/ and commit
```

---

## Cost Analysis

### Phase 2 Costs
- **Storage**: ~15 GB for 121M rows (TimescaleDB compression)
- **Compute**: Negligible (local Docker service)
- **Time**: 2-3 hours one-time ingestion

### Phase 3 Costs
- **R2 Storage**: $0.015/GB/month (~$0.08/month for 5 GB payloads)
- **R2 Egress**: $0 (Cloudflare's zero-egress pricing)
- **R2 Operations**: Class A (writes): $4.50/million, Class B (reads): $0.36/million
- **GitHub Actions**: Free (self-hosted runner)

**Total Monthly Cost**: ~$0.10/month

---

## References

- [Architecture Plan](.agents/plans/architect_plan.md)
- [Self-Hosted Runner Setup](../docs/SELF_HOSTED_RUNNER_SETUP.md)
- [Implementation Phases](../docs/database/implementation_phases.md)
- [TimescaleDB Docs](https://docs.timescale.com/)
- [Alembic Docs](https://alembic.sqlalchemy.org/)
- [Next.js Rewrites](https://nextjs.org/docs/app/api-reference/next-config-js/rewrites)
- [GitHub Actions Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)

---

**END OF IMPLEMENTATION SUMMARY**
