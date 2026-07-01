# Architect Plan: Unified Database Connector Layer

> **Branch:** `feat/database`
> **Date:** 2026-07-01
> **Phase Constraint:** Phase 2 (Backend Ablation Studies & Architecture Stabilization)
> **Prerequisite Context:** [context_for_architect.md](file:///Users/administrator/Code/ml-agent-workspace/.agents/plans/context_for_architect.md)

---

## Goal

Eliminate all local parquet file dependencies from the training, tuning, and analysis/inference pipelines by:

1. Extending the TimescaleDB schema to store weather readings, static features, and sensor metadata alongside traffic readings.
2. Building a unified `DataConnector` abstraction with `TimescaleConnector` (production) and `ParquetConnector` (smoke tests only) backends.
3. Rewiring all consumers (`train.py`, `tune.py`, `export_ui_payload.py`, `export_paginated_payload.py`) through the connector layer.
4. Removing all defensive fallback patterns that violate the Offensive Programming paradigm.

---

## Architect's Decisions on Open Questions

The Interrogator raised 6 open questions. Here are my rulings:

### Q1: Edge Features Cache → **Keep as file**
The `edge_features_cache.json` (4.3MB) is a precomputed, static artifact that changes only when the OSM road network or sensor set changes. It's loaded once at model init via `pipeline.py` and is not queried per-sample. Moving it to the database adds complexity for no gain. **Keep as `data/external/edge_features_cache.json`** referenced via the existing config path `model.edge_features_cache_path`.

### Q2: Sensor Metadata → **New DB table**
`traffic_sensors.json` is the canonical source of sensor coordinates. Multiple modules need it. It should live in the database as a `sensor_metadata` table so all consumers can query coordinates without assuming file paths. The JSON file remains as a seeding source.

### Q3: Weather Table Design → **Store pre-joined per traffic sensor (Build now, swap source later)**
The master parquet currently contains placeholder weather data. The target state is to fetch historical weather from the Open-Meteo API. 
**Decision:** We will build the `weather_readings` hypertable *now* using the placeholder data to complete the database connector migration. The schema will be keyed by `(dt, sensor_id)` where `sensor_id` is the *traffic* sensor ID, storing `humidity` and `solar_radiation`.
When we transition to the Open-Meteo API, we will simply replace the `ingest_weather.py` script with a new fetcher that queries the API for each traffic sensor's coordinates and interpolates the hourly API data to our 15-minute intervals. This design isolates the data source change from the downstream ML pipeline.

### Q4: Static Features Table → **Simple lookup table**
The 15 static features are per-location, time-invariant. A simple `static_features` table keyed by `sensor_id` with 15 float columns. Loaded once per dataset instantiation and broadcast-joined in Python. Tiny table (~81 rows).

### Q5: Connection Pooling → **Yes, with `psycopg2.pool.SimpleConnectionPool`**
The connector must hold a pool. For sweep workloads (`tune.py`), the dataset is created once and reused across trials, so a pool of `minconn=1, maxconn=4` is sufficient. The pool lives on the `TimescaleConnector` instance.

### Q6: Config-Driven Backend → **Explicit `data_source` key, crash on failure**
Use `data_source: "database"` or `data_source: "parquet"` in the YAML config. No auto-detection, no fallback. If `data_source: "database"` is set and the DB is unreachable, the script crashes. This enforces the Offensive Programming paradigm.

---

## Proposed Changes

### Step 1: New Alembic Migration — Schema Extension

#### [NEW] `alembic/versions/002_weather_static_sensors.py`

Create three new tables in TimescaleDB:

**`weather_readings`** (hypertable):
```sql
CREATE TABLE weather_readings (
    dt          TIMESTAMPTZ NOT NULL,
    sensor_id   TEXT        NOT NULL,   -- traffic sensor ID (pre-mapped)
    humidity    FLOAT       NOT NULL,
    solar_radiation FLOAT   NOT NULL,
    PRIMARY KEY (dt, sensor_id)
);
SELECT create_hypertable('weather_readings', 'dt',
    chunk_time_interval => INTERVAL '7 days');
CREATE INDEX idx_weather_sensor_dt ON weather_readings (sensor_id, dt DESC);
```

**`sensor_metadata`**:
```sql
CREATE TABLE sensor_metadata (
    sensor_id   TEXT PRIMARY KEY,
    lat         FLOAT NOT NULL,
    lon         FLOAT NOT NULL,
    description TEXT
);
```

**`static_features`**:
```sql
CREATE TABLE static_features (
    sensor_id              TEXT PRIMARY KEY,
    transit_count          FLOAT NOT NULL DEFAULT 0,
    retail_count           FLOAT NOT NULL DEFAULT 0,
    food_drink_count       FLOAT NOT NULL DEFAULT 0,
    nightlife_count        FLOAT NOT NULL DEFAULT 0,
    education_count        FLOAT NOT NULL DEFAULT 0,
    recreation_count       FLOAT NOT NULL DEFAULT 0,
    total_poi_count        FLOAT NOT NULL DEFAULT 0,
    landuse_entropy        FLOAT NOT NULL DEFAULT 0,
    is_school_zone         FLOAT NOT NULL DEFAULT 0,
    betweenness_centrality FLOAT NOT NULL DEFAULT 0,
    closeness_centrality   FLOAT NOT NULL DEFAULT 0,
    dist_to_city_centre    FLOAT NOT NULL DEFAULT 0,
    degree_centrality      FLOAT NOT NULL DEFAULT 0,
    nearby_bus_stops       FLOAT NOT NULL DEFAULT 0,
    nearby_parking         FLOAT NOT NULL DEFAULT 0,
    FOREIGN KEY (sensor_id) REFERENCES sensor_metadata(sensor_id)
);
```

> [!IMPORTANT]
> `sensor_metadata` must be created before `static_features` (FK dependency).
> `weather_readings` gets the same 7-day chunk interval as `traffic_readings`.

---

### Step 2: Ingestion Scripts — Seed the New Tables

#### [NEW] `scripts/ingest_weather.py`

Reads the `static_features_cache.parquet` (which contains pre-joined placeholder weather per traffic sensor per timestep), extracts the `(dt, location, Humidity, Solar Radiation)` columns, and bulk-inserts into `weather_readings`. Uses the same `execute_values` + batching pattern as `ingest_parquet.py`.
> [!NOTE]
> This script is a temporary bridge. In a future task, this script will be replaced by a fetcher that pulls data directly from the Open-Meteo historical API and interpolates it to match our 15-minute intervals.

#### [NEW] `scripts/ingest_static_features.py`

Reads `data/external/places_cache.json` and `data/external/osm_features_cache.json`, merges them into the 15-column schema, and inserts one row per sensor into `static_features`.

#### [NEW] `scripts/ingest_sensor_metadata.py`

Reads `data/traffic_sensors.json` and inserts `(sensor_id, lat, lon, description)` into `sensor_metadata`. This must run **before** `ingest_static_features.py` due to the FK constraint.

> [!TIP]
> All three scripts should follow the same structure as the existing `ingest_parquet.py`:
> argparse, `get_db_conn()` from `.env`, `--dry-run` flag, batch processing, and
> `ON CONFLICT DO NOTHING` for idempotency.

---

### Step 3: The Connector Module

#### [NEW] `src/db_connector.py`

This is the centrepiece of the change. The module defines:

```
DataConnector (ABC)
├── load_traffic(start_dt, end_dt, veh_class, direction) → pd.DataFrame
├── load_weather(start_dt, end_dt, sensor_ids) → pd.DataFrame
├── load_static_features(sensor_ids) → pd.DataFrame
├── load_sensor_metadata() → Dict[str, Tuple[float, float]]
└── close()  # cleanup / pool teardown

TimescaleConnector(DataConnector)
├── __init__(database_url: str)
│   └── Creates SimpleConnectionPool(minconn=1, maxconn=4)
├── load_traffic → SQL query on traffic_readings
├── load_weather → SQL query on weather_readings
├── load_static_features → SQL query on static_features
├── load_sensor_metadata → SQL query on sensor_metadata
└── close() → pool.closeall()

ParquetConnector(DataConnector)
├── __init__(parquet_path: str, static_cache_dir: str = "data/external")
├── load_traffic → pd.read_parquet with filters (existing logic)
├── load_weather → extracted from master parquet (Humidity, Solar Radiation cols)
├── load_static_features → read places_cache.json + osm_features_cache.json
├── load_sensor_metadata → read traffic_sensors.json
└── close() → no-op

create_connector(config: dict) → DataConnector
    Factory function:
    - if config.get("data_source") == "database":
        return TimescaleConnector(os.getenv("DATABASE_URL"))
    - elif config.get("data_source") == "parquet":
        return ParquetConnector(config.get("parquet_path", DEFAULT_PARQUET_PATH))
    - else:
        raise ValueError("CRITICAL: data_source must be 'database' or 'parquet'")
```

**Contract:** `load_traffic` returns a DataFrame with columns `[dt, location, value, veh_class, dir]`. `load_weather` returns `[dt, location, Humidity, Solar Radiation]`. `load_static_features` returns `[location, <15 feature cols>]`. The joining logic stays in `RealIoTDataset`.

**Offensive assertions in every method:**
- Assert returned DataFrames are non-empty (crash if no data)
- Assert expected columns are present
- Assert no NaN in primary key columns
- `TimescaleConnector`: assert `DATABASE_URL` is set, assert connection succeeds

**Connection handling:** `TimescaleConnector` uses `pool.getconn()` / `pool.putconn()` with a context manager pattern to guarantee connections are returned to the pool even on exception.

---

### Step 4: Refactor `RealIoTDataset`

#### [MODIFY] `src/dataset.py`

**Before (current signature):**
```python
class RealIoTDataset(Dataset):
    def __init__(self, parquet_path, start_date, end_date, ..., use_database=False):
```

**After (new signature):**
```python
class RealIoTDataset(Dataset):
    def __init__(self, connector: DataConnector, start_date, end_date, ...,):
```

**Changes:**
1. Replace `parquet_path` and `use_database` params with a single `connector: DataConnector` param.
2. Replace the `if use_database: ... else: ...` branch (lines 38-41) with:
   ```python
   traffic_df = connector.load_traffic(start_dt, end_dt, target_veh_class, target_dir)
   weather_df = connector.load_weather(start_dt, end_dt, traffic_df['location'].unique().tolist())
   static_df = connector.load_static_features(traffic_df['location'].unique().tolist())
   df = traffic_df.merge(weather_df, on=['dt', 'location'], how='left')
   df = df.merge(static_df, on='location', how='left')
   ```
3. **Delete** `_load_from_parquet()` method entirely (lines 180-197).
4. **Delete** `_load_from_database()` method entirely (lines 199-255).
5. Add tripwire assertions after the merge:
   ```python
   assert df['Humidity'].notna().all(), "CRITICAL: NaN Humidity after weather join"
   assert df['Solar Radiation'].notna().all(), "CRITICAL: NaN Solar Radiation after weather join"
   for col in STATIC_FEATURE_COLS:
       assert col in df.columns, f"CRITICAL: Missing static feature column '{col}'"
   ```

> [!IMPORTANT]
> The `STATIC_FEATURE_COLS` list is already defined inline in `dataset.py` (lines 160-165).
> Extract it as a module-level constant for reuse in assertions and in the connector.

---

### Step 5: Rewire All Consumers

#### [MODIFY] `src/train.py`

**Changes:**
1. Add import: `from db_connector import create_connector`
2. Replace lines 239-267 (the hardcoded `parquet_path`, `use_db`, and defensive try/except) with:
   ```python
   connector = create_connector(config)
   full_dataset = RealIoTDataset(
       connector,
       data_cfg["start_date"],
       data_cfg["end_date"],
       seq_len=data_cfg["seq_len"],
       mask_prob=data_cfg.get("mask_prob", 0.15),
       target_veh_class=data_cfg["target_veh_class"],
       target_dir=data_cfg["target_dir"],
   )
   ```
3. **Delete** the `--use-database` CLI flag (lines 43-47) — it's now config-driven.
4. **Delete** the defensive fallback block (lines 253-265). If the connector fails, let it crash.
5. Add connector cleanup in the finally block or at script exit.

#### [MODIFY] `src/tune.py`

**Changes:**
1. Add import: `from db_connector import create_connector`
2. Replace lines 431-443 (hardcoded `parquet_path` and `RealIoTDataset` instantiation) with:
   ```python
   connector = create_connector(base_config)
   full_dataset = RealIoTDataset(
       connector,
       data_cfg["start_date"],
       data_cfg["end_date"],
       ...
   )
   ```
3. The connector instance should be created **once** and passed through to the objective function, not recreated per trial (connection pool efficiency).

#### [MODIFY] `src/export_ui_payload.py`

**Changes:**
1. Add import: `from db_connector import create_connector`
2. **Delete** the `PARQUET_PATH` module-level constant (line 70).
3. Add `--config` CLI arg (defaulting to `configs/analysis/analysis.yaml`) to drive the connector.
4. Replace lines 718-726 with connector-based dataset instantiation.
5. The `load_sensor_coords()` function (currently reads `SENSORS_PATH = "data/traffic_sensors.json"`) should use `connector.load_sensor_metadata()` when in database mode, keeping the JSON path as fallback for the `ParquetConnector`.

#### [MODIFY] `src/export_paginated_payload.py`

**Changes:**
1. **Remove** the import of `PARQUET_PATH` from `export_ui_payload` (line 78).
2. Add import: `from db_connector import create_connector`
3. Replace lines 800-806 with connector-based dataset instantiation.
4. The connector is created once in the outer function and passed into `generate_for_artifact()`.

---

### Step 6: Configuration Updates

#### [MODIFY] `configs/train/train.yaml`

Replace `use_database: true` (line 12) with:
```yaml
data_source: "database"  # "database" | "parquet"
```

Remove any residual `parquet_path` references.

#### [MODIFY] `configs/train/train_test.yaml`

Same change: replace `use_database: true` with `data_source: "database"`.

#### [MODIFY] `configs/analysis/analysis.yaml`

Extend from 2 lines to include data source config:
```yaml
data_source: "database"
holdout_start: 2026-05-01
holdout_end: 2026-06-01
veh_class: "person"
dir: "all"
seq_len: 12
```

---

### Step 7: Preserve Smoke Tests & Utilities

The following files are **explicitly excluded** from changes:

| File | Reason |
|------|--------|
| `src/smoke_test.py` | User requested parquet usage preserved for quick smoke tests |
| `src/check_gaps.py` | Utility script, reads raw parquet directly |
| `src/process_all.py` | ETL pipeline, produces the master parquet that feeds `ingest_parquet.py` |
| `scripts/ingest_parquet.py` | Bridge script, reads parquet to populate DB — its purpose is the transition |

---

## Files Not Changed

| File | Reason |
|------|--------|
| `src/pipeline.py` | No data loading, only model forward pass. Edge features loaded via config path — untouched. |
| `src/encoder.py` | Pure model architecture, no data access |
| `src/decoders.py` | Pure model architecture, no data access |
| `src/graph_learner.py` | Pure model architecture, no data access |
| `src/metrics.py` | Pure computation, no data access |
| `src/features.py` | Only used by `process_all.py` (ETL) and `export_ui_payload.py`. The export scripts will use `connector.load_sensor_metadata()` instead of reading `traffic_sensors.json` directly. `features.py` itself is only called during the one-time ETL flow and is **not** on the hot path. |
| `src/preprocess.py` | Only used by `process_all.py` (ETL), not on hot path |

---

## Execution Order

The implementer MUST follow this exact order to avoid broken intermediate states:

```
1. git add . && git commit -m "Auto-commit before unified-db-connector"

2. Create alembic migration (Step 1)
   └── Apply: docker compose exec agent-workspace alembic upgrade head

3. Create ingestion scripts (Step 2)
   └── Run in order:
       a. docker compose exec agent-workspace python scripts/ingest_sensor_metadata.py
       b. docker compose exec agent-workspace python scripts/ingest_static_features.py
       c. docker compose exec agent-workspace python scripts/ingest_weather.py

4. Create src/db_connector.py (Step 3)

5. Refactor src/dataset.py (Step 4)

6. Rewire consumers in order (Step 5):
   a. src/train.py
   b. src/tune.py
   c. src/export_ui_payload.py
   d. src/export_paginated_payload.py

7. Update configs (Step 6)

8. Verify (see Verification Plan below)

9. Update EXPERIMENTS.md
```

> [!CAUTION]
> Steps 4 and 5 are a breaking change together — once `RealIoTDataset` drops
> `parquet_path`, all consumers must be updated in the same commit. The implementer
> should make Steps 4-6 as a single atomic commit.

---

## Verification Plan

### Automated Tests

```bash
# 1. Verify migration applies cleanly
docker compose exec agent-workspace alembic upgrade head

# 2. Verify ingestion scripts (dry run first)
docker compose exec agent-workspace python scripts/ingest_sensor_metadata.py --dry-run
docker compose exec agent-workspace python scripts/ingest_static_features.py --dry-run
docker compose exec agent-workspace python scripts/ingest_weather.py --dry-run

# 3. Run actual ingestion
docker compose exec agent-workspace python scripts/ingest_sensor_metadata.py
docker compose exec agent-workspace python scripts/ingest_static_features.py
docker compose exec agent-workspace python scripts/ingest_weather.py

# 4. Verify training with database connector
docker compose exec agent-workspace python src/train.py \
    --config configs/train/train_test.yaml --use_mock_data

# 5. Verify training with REAL data from database (1 epoch smoke test)
docker compose exec agent-workspace python src/train.py \
    --config configs/train/train_test.yaml

# 6. Run existing test suite
docker compose exec agent-workspace python -m pytest tests/ -v
```

### Manual Verification

- Confirm that `data/processed/master_dataset_15T.parquet` is **NOT** read during database-mode training (add a temporary `assert False` at the top of `ParquetConnector.load_traffic` and verify database-mode training still works).
- Confirm that `smoke_test.py` and `check_gaps.py` still work unmodified.

### Desk Check: Data Parity

The implementer MUST create a one-off verification script (`tests/verify_db_parity.py`) that:
1. Loads 1 week of data via `ParquetConnector`
2. Loads the same week via `TimescaleConnector`
3. Asserts that the resulting DataFrames are identical (column-by-column, `pd.testing.assert_frame_equal` with `check_dtype=False`)

This is the mathematical proof that the connector abstraction preserves data integrity.

---

## Experiment Ledger Mandate

Upon completion, the implementer **MUST** append a row to `EXPERIMENTS.md`:

| Date | Branch | Objective | Outcome |
|------|--------|-----------|---------|
| 2026-07-XX | `feat/database` | Unified Database Connector Layer: eliminate parquet dependencies for train/tune/export | (to be filled by implementer) |

The task is NOT complete until this row exists.
