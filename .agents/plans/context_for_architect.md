# Context for Architect: Unified Database Connector Layer

> **Interrogator Investigation — 2026-07-01**
>
> **Objective:** Map all data access patterns to identify what must change to
> eliminate local parquet file dependencies (except quick smoke tests) and build
> a unified database connector layer for training, tuning, analysis, and
> inference.

---

## 1. Current State: Two Worlds

The codebase currently has a **split personality** for data access:

| Layer | Data Source | Status |
|-------|-----------|--------|
| **Training** (`train.py`) | TimescaleDB via `dataset.py._load_from_database()` | ✅ Implemented (Phase 2) |
| **Tuning** (`tune.py`) | Parquet only | ❌ No DB support |
| **UI Export** (`export_ui_payload.py`) | Parquet only | ❌ No DB support |
| **Paginated Export** (`export_paginated_payload.py`) | Parquet only | ❌ No DB support |
| **Preprocessing** (`process_all.py`) | Raw parquet files → writes master parquet | ⚪ ETL script (separate concern) |
| **Smoke Test** (`smoke_test.py`) | Raw parquet files | ⚪ Keep as-is (per user request) |
| **Gap Checker** (`check_gaps.py`) | Raw parquet files | ⚪ Utility / keep as-is |
| **Ingestion** (`scripts/ingest_parquet.py`) | Reads parquet → writes TimescaleDB | ✅ Bridge script |

---

## 2. AST / High-Level Logic: The Core Data Path

### 2.1 `src/dataset.py` — `RealIoTDataset` (lines 15–288)

This is the **sole entry point** for all ML data loading. Every consumer
(`train.py`, `tune.py`, `export_ui_payload.py`, `export_paginated_payload.py`)
instantiates `RealIoTDataset`.

```
RealIoTDataset.__init__(parquet_path, start_date, end_date, ..., use_database=False)
│
├── if use_database:
│   └── _load_from_database(start_dt, end_dt, veh_class, direction)
│       ├── import psycopg2 (lazy)
│       ├── read DATABASE_URL from .env
│       ├── Query: SELECT dt, sensor_id, veh_class, direction, count
│       │          FROM traffic_readings WHERE dt >= %s AND dt < %s ...
│       ├── ⚠️ CRITICAL GAP: Weather data NOT in database
│       │   └── Falls back to Parquet: data/external/static_features_cache.parquet (42MB)
│       │       (line 242-253: hardcoded path, logger.warning about Parquet fallback)
│       └── Returns DataFrame with columns matching Parquet schema
│
└── else:
    └── _load_from_parquet(parquet_path, start_dt, end_dt, ...)
        └── pd.read_parquet(parquet_path, filters=[...], columns=[...])
            └── Reads: data/processed/master_dataset_15T.parquet
```

**Key Observation:** `RealIoTDataset` is the **choke point**. A unified connector
layer should sit *below* this class, abstracting the backend.

### 2.2 `src/train.py` — Data Loading (lines 237–267)

```python
parquet_path = "data/processed/master_dataset_15T.parquet"  # L239: HARDCODED
use_db = config.get("use_database", False)                   # L241
try:
    full_dataset = RealIoTDataset(parquet_path, ..., use_database=use_db)
except Exception as db_exc:
    if use_db:
        logger.warning(f"Database backend failed ({db_exc}), falling back to Parquet")  # ⚠️ DEFENSIVE
        full_dataset = RealIoTDataset(parquet_path, ..., use_database=False)
    else:
        raise
```

> [!CAUTION]
> **DEFENSIVE CODE ALERT (lines 253–265):** Silent fallback from DB to Parquet on
> failure. This violates the Offensive Programming paradigm — a DB failure should
> crash hard, not silently degrade to a different data source.

### 2.3 `src/tune.py` — Data Loading (lines 431–443)

```python
parquet_path = "data/processed/master_dataset_15T.parquet"  # L433: HARDCODED
full_dataset = RealIoTDataset(
    parquet_path, data_cfg["start_date"], data_cfg["end_date"], ...
)  # No use_database parameter passed → always False
```

**No database support at all.** Always reads from Parquet.

### 2.4 `src/export_ui_payload.py` — Data Loading (lines 69–70, 718–726)

```python
PARQUET_PATH = "data/processed/master_dataset_15T.parquet"  # L70: MODULE-LEVEL CONSTANT

dataset = RealIoTDataset(
    parquet_path=PARQUET_PATH, ...  # L719: No use_database param
)
```

**No database support at all.**

### 2.5 `src/export_paginated_payload.py` — Data Loading (lines 78, 800–806)

```python
# Imports PARQUET_PATH from export_ui_payload.py (L78)
from export_ui_payload import PARQUET_PATH

full_ds = RealIoTDataset(
    parquet_path=PARQUET_PATH, ...  # L801: No use_database param
)
```

**No database support at all.** Inherits the hardcoded constant from `export_ui_payload`.

---

## 3. Database Schema (Current)

From [001_initial_schema.py](file:///Users/administrator/Code/ml-agent-workspace/alembic/versions/001_initial_schema.py):

### Tables in TimescaleDB:

| Table | Columns | Purpose |
|-------|---------|---------|
| `traffic_readings` (hypertable) | `dt`, `sensor_id`, `veh_class`, `direction`, `count` | Traffic time-series only |
| `dataset_versions` | `version_id`, `git_branch`, `git_commit`, `created_at`, `description`, `preprocessing`, `parquet_sha256`, `row_count` | Dataset lineage |
| `model_runs` | `run_id`, `wandb_run_id`, `git_commit`, `git_branch`, `config_yaml`, metrics... | Training run tracking |

### What's MISSING from the database:

1. **Weather readings** — Humidity and Solar Radiation are only in the parquet
2. **Static spatial features** (15 features) — Only in `data/external/places_cache.json` and `data/external/osm_features_cache.json`
3. **Sensor metadata** — Coordinates, names, etc. — Only in `data/traffic_sensors.json` and `data/weather_sensors.json`
4. **Edge features** — Only in `data/external/edge_features_cache.json`

The `_load_from_database()` method in `dataset.py` (line 241-253) explicitly
warns about this and falls back to reading `static_features_cache.parquet`.

---

## 4. Static / Spatial Data Files (Non-Timeseries)

These JSON/parquet files contain data that changes infrequently or never:

| File | Size | Used By | Nature |
|------|------|---------|--------|
| `data/traffic_sensors.json` | 10KB | `features.py`, `export_ui_payload.py`, `process_all.py`, `smoke_test.py` | Sensor coordinates |
| `data/weather_sensors.json` | 869KB | `process_all.py`, `smoke_test.py` | Weather station metadata |
| `data/external/places_cache.json` | 37KB | `features.py`, `export_ui_payload.py` | Google Places POI counts |
| `data/external/osm_features_cache.json` | 20KB | `features.py` | OSMnx centrality metrics |
| `data/external/edge_features_cache.json` | 4.3MB | `pipeline.py` (via config), `features.py` | Pairwise edge features |
| `data/external/static_features_cache.parquet` | 43MB | `dataset.py` (DB mode fallback) | Weather+spatial pre-joined |

---

## 5. Configuration Surface

### 5.1 [configs/train/train.yaml](file:///Users/administrator/Code/ml-agent-workspace/configs/train/train.yaml)

- `use_database: true` (line 12) — Controls train.py's DB toggle
- `model.edge_features_cache_path: "data/external/edge_features_cache.json"` — Hardcoded file path

### 5.2 [configs/analysis/analysis.yaml](file:///Users/administrator/Code/ml-agent-workspace/configs/analysis/analysis.yaml)

- Only 2 lines: `holdout_start` and `holdout_end` — No data source config at all

### 5.3 [.env](file:///Users/administrator/Code/ml-agent-workspace/.env)

- `DATABASE_URL=postgresql://riotuser:riotpassword@timescaledb:5432/riot_traffic` (line 20)

---

## 6. Files That Would Need Editing

### Tier 1 — Core (Must Change)

| File | Lines | What Changes |
|------|-------|-------------|
| [src/dataset.py](file:///Users/administrator/Code/ml-agent-workspace/src/dataset.py) | 356 | Extract data loading into a `DataConnector` abstraction. `_load_from_parquet` and `_load_from_database` become backends behind a unified interface. Eliminate the `static_features_cache.parquet` fallback — weather/static data must come from DB too. |
| [src/train.py](file:///Users/administrator/Code/ml-agent-workspace/src/train.py) | 643 | Remove hardcoded `parquet_path`. Remove defensive DB→Parquet fallback (lines 253-265). Use config-driven connector. |
| [src/tune.py](file:///Users/administrator/Code/ml-agent-workspace/src/tune.py) | 462 | Add `use_database` support (currently absent). Remove hardcoded `parquet_path` on L433. |
| [src/export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py) | 1369 | Remove `PARQUET_PATH` constant (L70). Wire through connector config. Add `--use-database` flag or use config. |
| [src/export_paginated_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_paginated_payload.py) | 939 | Stops importing `PARQUET_PATH` from export_ui_payload. Uses unified connector. |

### Tier 2 — Schema Extension (New DB Tables)

| File | What Changes |
|------|-------------|
| [alembic/versions/](file:///Users/administrator/Code/ml-agent-workspace/alembic/versions) | New migration: add `weather_readings` hypertable, `sensor_metadata` table, and `static_features` table to TimescaleDB |
| [scripts/ingest_parquet.py](file:///Users/administrator/Code/ml-agent-workspace/scripts/ingest_parquet.py) | Extend to also ingest weather data and static features, or create companion scripts |

### Tier 3 — Configuration

| File | What Changes |
|------|-------------|
| [configs/train/train.yaml](file:///Users/administrator/Code/ml-agent-workspace/configs/train/train.yaml) | Add `data_source: "database"` or `"parquet"` key. Potentially move `edge_features_cache_path` to DB. |
| [configs/train/train_test.yaml](file:///Users/administrator/Code/ml-agent-workspace/configs/train/train_test.yaml) | Mirror data_source config |
| [configs/analysis/analysis.yaml](file:///Users/administrator/Code/ml-agent-workspace/configs/analysis/analysis.yaml) | Add data_source and connection config for export scripts |

### Tier 4 — NEW file (Connector Module)

| File | Purpose |
|------|---------|
| `src/db_connector.py` (NEW) | Unified `DataConnector` ABC with `ParquetConnector` and `TimescaleConnector` implementations. Factory function driven by config. Handles: traffic, weather, static features, sensor metadata. |

### Tier 5 — Preserve for Smoke Tests Only

| File | Status |
|------|--------|
| [src/smoke_test.py](file:///Users/administrator/Code/ml-agent-workspace/src/smoke_test.py) | Keep Parquet usage (per user request) |
| [src/check_gaps.py](file:///Users/administrator/Code/ml-agent-workspace/src/check_gaps.py) | Keep Parquet usage (utility) |
| [src/process_all.py](file:///Users/administrator/Code/ml-agent-workspace/src/process_all.py) | Keep as ETL pipeline (reads raw → writes master parquet for ingestion) |

---

## 7. Tripwires & Issues Identified

> [!CAUTION]
> **DEFENSIVE CODE — `train.py` lines 253-265:** Silent fallback from database to
> Parquet on exception. Violates Offensive Programming. Must be removed — DB
> failure should crash hard.

> [!WARNING]
> **INCOMPLETE DB MODE — `dataset.py` lines 241-253:** Database mode still reads
> `static_features_cache.parquet` for weather + spatial features. This is the
> core gap that the unified connector must close.

> [!WARNING]
> **HARDCODED PATHS everywhere:**
> - `"data/processed/master_dataset_15T.parquet"` in `train.py:239`, `tune.py:433`
> - `PARQUET_PATH` constant in `export_ui_payload.py:70`
> - `"data/external/static_features_cache.parquet"` in `dataset.py:243`
> - `"data/traffic_sensors.json"` in `features.py:348`, `export_ui_payload.py:69`

> [!NOTE]
> **Raw DB connector is inline in `dataset.py`:** `psycopg2` is imported lazily
> inside `_load_from_database()`. There is no connection pooling, no context
> managers, and the connection is opened/closed per call. This needs to be
> properly abstracted.

---

## 8. Data Flow Diagram

```
                     ┌─────────────────────┐
                     │  Raw Parquet Files   │
                     │  (traffic, weather)  │
                     └──────────┬──────────┘
                                │
                    ┌───────────▼───────────┐
                    │   process_all.py      │
                    │   (ETL pipeline)      │
                    └───────────┬───────────┘
                                │
               ┌────────────────▼────────────────┐
               │  master_dataset_15T.parquet      │
               └────────┬───────────────┬────────┘
                        │               │
           ┌────────────▼──────┐  ┌─────▼──────────────┐
           │ ingest_parquet.py │  │  Direct Parquet     │
           │ (one-time load)   │  │  Readers            │
           └────────┬──────────┘  │  ┌─ tune.py         │
                    │             │  ├─ export_ui.py     │
           ┌────────▼──────────┐  │  └─ export_pag.py   │
           │   TimescaleDB     │  └────────────────────┘
           │ (traffic_readings │
           │  ONLY)            │       ← NO WEATHER
           └────────┬──────────┘       ← NO STATIC FEATURES
                    │
           ┌────────▼──────────┐
           │  dataset.py       │
           │  (DB mode:        │
           │   traffic from DB │
           │   static from     │
           │   Parquet cache!) │◄──── The gap
           └───────────────────┘
```

### Target Architecture (Post-Connector Layer):

```
           ┌────────────────────────┐
           │      TimescaleDB       │
           │ ┌────────────────────┐ │
           │ │ traffic_readings   │ │  (existing)
           │ ├────────────────────┤ │
           │ │ weather_readings   │ │  (NEW)
           │ ├────────────────────┤ │
           │ │ sensor_metadata    │ │  (NEW)
           │ ├────────────────────┤ │
           │ │ static_features    │ │  (NEW)
           │ └────────────────────┘ │
           └──────────┬─────────────┘
                      │
           ┌──────────▼─────────────┐
           │   db_connector.py      │
           │   (DataConnector ABC)  │
           │  ┌──────────────────┐  │
           │  │TimescaleConnector│  │  ← Production
           │  ├──────────────────┤  │
           │  │ParquetConnector  │  │  ← Smoke tests / fallback
           │  └──────────────────┘  │
           └──────────┬─────────────┘
                      │
           ┌──────────▼─────────────┐
           │  RealIoTDataset        │
           │  (consumer, unchanged  │
           │   interface)           │
           └──────────┬─────────────┘
                      │
        ┌─────────────┼─────────────────┐
        │             │                 │
   train.py      tune.py      export_*.py
```

---

## 9. Column Schema Contract

The unified connector must return a DataFrame matching this exact schema
(currently the implicit contract between `_load_from_parquet` and the rest of
`RealIoTDataset.__init__`):

| Column | Type | Source |
|--------|------|--------|
| `dt` | datetime64 | Traffic readings |
| `location` | str | Traffic readings |
| `value` | float | Traffic readings (count) |
| `veh_class` | str | Traffic readings |
| `dir` | str | Traffic readings |
| `Humidity` | float | Weather readings |
| `Solar Radiation` | float | Weather readings |
| `transit_count` | float | Static features (Google Places) |
| `retail_count` | float | Static features (Google Places) |
| `food_drink_count` | float | Static features (Google Places) |
| `nightlife_count` | float | Static features (Google Places) |
| `education_count` | float | Static features (Google Places) |
| `recreation_count` | float | Static features (Google Places) |
| `total_poi_count` | float | Static features (Google Places) |
| `landuse_entropy` | float | Static features (OSMnx) |
| `is_school_zone` | float | Static features (Google Places) |
| `betweenness_centrality` | float | Static features (OSMnx) |
| `closeness_centrality` | float | Static features (OSMnx) |
| `dist_to_city_centre` | float | Static features (OSMnx) |
| `degree_centrality` | float | Static features (OSMnx) |
| `nearby_bus_stops` | float | Static features (Google Places) |
| `nearby_parking` | float | Static features (Google Places) |

**Total: 21 columns (5 traffic + 2 weather + 15 static)**

---

## 10. Open Questions for Architect

1. **Edge Features Cache:** `edge_features_cache.json` (4.3MB) is loaded by
   `pipeline.py` via config path `model.edge_features_cache_path`. Should this
   also move to the database, or remain as a file cache (it's computed once and
   rarely changes)?

2. **Sensor Metadata:** `traffic_sensors.json` contains lat/lon coordinates used
   by `features.py` and `export_ui_payload.py`. This is effectively static
   reference data. Database table or keep as file?

3. **Weather Table Design:** Weather readings in the master parquet are already
   joined to traffic by nearest-weather-sensor mapping. Should the DB store raw
   weather readings (like the parquet does) and join at query time, or store the
   pre-joined results?

4. **Static Features Table Design:** The 15 static features are per-location
   (not per-timestep). Should this be a simple lookup table keyed by
   `sensor_id`, loaded once and joined in Python?

5. **Connection Pooling:** The current `_load_from_database` opens/closes a
   connection per call. For sweep workloads (tune.py calls RealIoTDataset
   per trial), this could be expensive. Should the connector use a connection
   pool (e.g., `psycopg2.pool.ThreadedConnectionPool`)?

6. **Config-Driven Backend:** Should `data_source: "database" | "parquet"` be
   a top-level YAML key, or should the connector auto-detect (e.g., try DB
   first, crash if unavailable rather than falling back)?