# Data Store Overview — Resilient IoT Spatio-Temporal Digital Twin

This document catalogues **every data store** in the project, from relational databases to file-based caches and cloud object storage. Each store includes a Mermaid schema diagram.

---

## 1. TimescaleDB (PostgreSQL) — `riot_traffic`

The primary relational database, managed via Alembic migrations ([001_initial_schema.py](../../alembic/versions/001_initial_schema.py)). Runs as a Docker service (`timescale/timescaledb:latest-pg16`).

**Connection:** `postgresql://riotuser:riotpassword@timescaledb:5432/riot_traffic`

```mermaid
erDiagram
    traffic_readings {
        TIMESTAMPTZ dt PK "Hypertable partition key (7-day chunks)"
        TEXT sensor_id PK
        TEXT veh_class PK "e.g. person, car, bus"
        TEXT direction PK
        FLOAT count "Vehicle/pedestrian count"
    }

    dataset_versions {
        INTEGER version_id PK "Auto-increment"
        TEXT git_branch
        TEXT git_commit "UNIQUE with parquet_sha256"
        TIMESTAMPTZ created_at "DEFAULT NOW()"
        TEXT description
        JSONB preprocessing "Full preprocessing config snapshot"
        TEXT parquet_sha256
        BIGINT row_count
    }

    model_runs {
        TEXT run_id PK
        TEXT wandb_run_id UK "Links to W&B"
        TEXT git_commit
        TEXT git_branch
        JSONB config_yaml "Full training config"
        TIMESTAMPTZ created_at "DEFAULT NOW()"
        FLOAT val_rmse
        FLOAT val_mae
        FLOAT val_zero_acc
        FLOAT val_loss
        INTEGER batch_size
        FLOAT learning_rate
        INTEGER num_epochs
        INTEGER dataset_version_id FK
    }

    dataset_versions ||--o{ model_runs : "dataset_version_id"
```

> [!NOTE]
> `traffic_readings` is a TimescaleDB **hypertable** with 7-day chunk intervals. Indices cover `(sensor_id, dt DESC)`, `(veh_class, dt DESC)`, and `(direction, dt DESC)`.

---

## 2. Optuna SQLite — Hyperparameter Tuning

Local SQLite databases (e.g. [optuna.db](optuna.db)) used by [tune.py](src/tune.py) for Bayesian hyperparameter optimization.

**Connection:** `sqlite:///optuna.db`

```mermaid
erDiagram
    studies {
        INTEGER study_id PK
        TEXT study_name UK "Aligned with W&B run name"
        TEXT direction "minimize"
    }

    trials {
        INTEGER trial_id PK
        INTEGER study_id FK
        TEXT state "COMPLETE / PRUNED / FAIL"
        FLOAT value "val_loss_total (objective)"
        TIMESTAMPTZ datetime_start
        TIMESTAMPTZ datetime_complete
    }

    trial_params {
        INTEGER trial_id FK
        TEXT param_name "e.g. lr, hidden_dim, dropout"
        FLOAT param_value
        TEXT distribution "LogUniform, IntUniform, etc."
    }

    trial_values {
        INTEGER trial_id FK
        INTEGER objective "0 = primary"
        FLOAT value
    }

    studies ||--o{ trials : "study_id"
    trials ||--o{ trial_params : "trial_id"
    trials ||--o{ trial_values : "trial_id"
```

> [!TIP]
> Optuna's internal schema is managed by the library itself. The project stores one study per sweep, with the study name aligned to the W&B run name for cross-referencing.

---

## 3. Raw Data Files (Parquet + JSON)

Located in [data/](data). These are the **source-of-truth** inputs to the preprocessing pipeline ([process_all.py](src/process_all.py)).

```mermaid
erDiagram
    traffic_parquet["traffic_*.parquet (62 MB)"] {
        TIMESTAMP dt "15-min intervals"
        TEXT location "Sensor site name"
        TEXT category "Sensor category"
        TEXT veh_class "person, car, bus, etc."
        TEXT dir "N, S, E, W, all"
        FLOAT value "Raw vehicle/pedestrian count"
    }

    weather_parquet["weather_*.parquet (22 MB)"] {
        TEXT sensor_name "Met station name"
        TIMESTAMP timestamp
        TEXT variable "Humidity, Solar Radiation, etc."
        FLOAT value
        TEXT units
    }

    traffic_sensors_json["traffic_sensors.json"] {
        TEXT location PK "Sensor site name"
        FLOAT lat "WGS84 latitude"
        FLOAT lon "WGS84 longitude"
    }

    weather_sensors_json["weather_sensors.json"] {
        TEXT Sensor_Name PK
        FLOAT Sensor_Centroid_Latitude
        FLOAT Sensor_Centroid_Longitude
    }
```

**Archive:** [data/archive/](data/archive) contains historical traffic parquet splits used before dataset consolidation.

---

## 4. Processed / Master Dataset

The output of [process_all.py](src/process_all.py) — a fully joined, resampled, feature-enriched parquet consumed by the PyTorch [RealIoTDataset](src/dataset.py#L15-L30).

```mermaid
erDiagram
    master_dataset["master_dataset_15T.parquet (445 MB)"] {
        TIMESTAMP dt "15-min grid"
        TEXT location "Sensor site"
        TEXT veh_class "person, car, bus, etc."
        TEXT dir "N, S, E, W, all"
        FLOAT value "Resampled + imputed count"
        FLOAT Humidity "Joined weather feature"
        FLOAT Solar_Radiation "Joined weather feature"
        FLOAT transit_count "POI: nearby transit stops"
        FLOAT retail_count "POI: nearby retail"
        FLOAT food_drink_count "POI: nearby food/drink"
        FLOAT nightlife_count "POI: nearby nightlife"
        FLOAT education_count "POI: nearby education"
        FLOAT recreation_count "POI: nearby recreation"
        FLOAT total_poi_count "Total POI density"
        FLOAT landuse_entropy "Shannon entropy of land use"
        FLOAT is_school_zone "Boolean as float"
        FLOAT betweenness_centrality "OSM graph metric"
        FLOAT closeness_centrality "OSM graph metric"
        FLOAT dist_to_city_centre "Haversine distance (m)"
        FLOAT degree_centrality "OSM graph metric"
        FLOAT nearby_bus_stops "Count"
        FLOAT nearby_parking "Count"
        TEXT nearest_weather_sensor "Spatial join result"
    }
```

> [!IMPORTANT]
> The master dataset has **3 dynamic channels** (traffic value, Humidity, Solar Radiation) and **15 static features** per node. This is the contract consumed by `RealIoTDataset.__init__`.

---

## 5. External Feature Caches

Located in [data/external/](data/external). Pre-computed spatial and contextual features fetched from external APIs (Google Places, OSM) and cached locally.

```mermaid
erDiagram
    places_cache["places_cache.json (37 KB)"] {
        TEXT location_key PK "Sensor site name"
        INT transit_count
        INT retail_count
        INT food_drink_count
        INT nightlife_count
        INT education_count
        INT recreation_count
        INT total_poi_count
        FLOAT landuse_entropy
        BOOL is_school_zone
        INT nearby_bus_stops
        INT nearby_parking
    }

    osm_features["osm_features_cache.json (20 KB)"] {
        TEXT location_key PK "Sensor site name"
        FLOAT betweenness_centrality
        FLOAT closeness_centrality
        FLOAT degree_centrality
        FLOAT dist_to_city_centre
    }

    edge_features["edge_features_cache.json (4.2 MB)"] {
        TEXT pair_key PK "sensor_i::sensor_j"
        FLOAT shortest_path_distance "OSM walking network (m)"
        FLOAT euclidean_distance "Haversine (m)"
        FLOAT directness "shortest_path / euclidean ratio"
        FLOAT pedestrian "Road-type fraction"
        FLOAT residential "Road-type fraction"
        FLOAT tertiary "Road-type fraction"
        FLOAT secondary "Road-type fraction"
        FLOAT primary "Road-type fraction"
        FLOAT service "Road-type fraction"
    }

    static_features_cache["static_features_cache.parquet (43 MB)"] {
        TIMESTAMP dt
        TEXT location
        FLOAT Humidity "Pre-joined weather"
        FLOAT Solar_Radiation "Pre-joined weather"
        FLOAT transit_count "All 15 static features"
        TEXT _comment "Used by DB-mode fallback"
    }
```

> [!NOTE]
> The 9 edge feature dimensions from `edge_features_cache.json` are combined with 1 learned similarity dimension at runtime, giving `edge_dim=10` for GATv2.

---

## 6. Weights & Biases (Cloud)

Project: **`resilient-iot`** (training) / **`resilient-iot-sweeps`** (tuning)

W&B is used for experiment tracking and model artifact storage. Not a traditional database, but a critical data store.

```mermaid
erDiagram
    wandb_runs["W&B Runs"] {
        TEXT run_id PK "Auto-generated"
        TEXT run_name "Human-readable"
        TEXT project "resilient-iot or resilient-iot-sweeps"
        JSONB config "Full training config dict"
    }

    wandb_metrics["W&B Logged Metrics (per step)"] {
        FLOAT train_loss_total
        FLOAT val_loss_total
        FLOAT val_rmse
        FLOAT val_mae
        FLOAT val_zero_accuracy
        FLOAT val_impute_mse
        FLOAT train_loss_kriging "Phase 2 only"
        FLOAT val_kriging_rmse "Phase 2 only"
        INT epoch
    }

    wandb_artifacts["W&B Model Artifacts"] {
        TEXT artifact_name "resilient-iot-model"
        TEXT type "model"
        FILE best_model_pt "best_model.pt state dict"
        FILE config_yaml "config.yaml snapshot"
        JSONB metadata "Full config"
    }

    wandb_runs ||--o{ wandb_metrics : "logged per epoch"
    wandb_runs ||--o{ wandb_artifacts : "logged on best"
```

---

## 7. Cloudflare R2 (Object Storage)

Bucket: **`resilient-iot-payloads`** — serves the paginated UI payload for the Next.js frontend via CDN.

Upload script: [upload_to_r2.py](scripts/upload_to_r2.py)

```mermaid
erDiagram
    run_registry["runs/run_registry.json"] {
        TEXT default_run_id "Currently displayed run"
        ARRAY runs "List of RunRegistryEntry"
    }

    spatial_json["runs/{run_id}/spatial.json"] {
        TEXT run_id
        TEXT wandb_run_id
        TEXT git_commit_hash
        ARRAY nodes "id, name, lat, lon, is_simulated"
        ARRAY edge_topology "source, target pairs"
        JSONB metadata "hyperparams, metrics, dates"
    }

    temporal_page["runs/{run_id}/{date}/page.json"] {
        TEXT date "YYYY-MM-DD"
        INT timesteps "96 per day"
        RECORD node_timeseries "per-node prediction + uncertainty"
        ARRAY edge_attention "per-edge attention sequences"
        ARRAY pit_histogram "Calibration bins"
        ARRAY zero_inflation_reliability "ZINB calibration"
    }

    node_detail["runs/{run_id}/{date}/nodes/node_{id}.json"] {
        INT id "Node index"
        ARRAY zinb_series "mu, theta, std_dev, pi per timestep"
        ARRAY ground_truth_series "nullable raw counts"
        ARRAY imputed_series "Model imputed values"
        ARRAY mask_series "Boolean NaN injection mask"
        ARRAY edges_1hop "Ego-graph attention edges"
        RECORD two_extremes "MAE/RMSE drop-NaN vs fill-NaN"
    }

    run_registry ||--o{ spatial_json : "one per run"
    spatial_json ||--o{ temporal_page : "one per date"
    temporal_page ||--o{ node_detail : "one per node per date"
```

> [!TIP]
> The frontend proxies `/data/runs/*` → `R2_PUBLIC_URL/runs/*` via Next.js rewrites in [next.config.ts](frontend/next.config.ts).

---

## 8. Local Model Artifacts

Stored in [models/](models) on disk. Namespaced by project and run.

```mermaid
erDiagram
    model_checkpoint["models/{project}/{run_name}/best_model.pt"] {
        DICT state_dict "PyTorch model weights"
    }

    model_config["models/{project}/{run_name}/config.yaml"] {
        YAML data "start_date, end_date, seq_len, etc."
        YAML model "hidden_dim, num_heads, edge_dim, etc."
        YAML training "lr, batch_size, epochs, etc."
        YAML loss "w_forecast, w_impute, w_kriging"
    }

    model_checkpoint ||--|| model_config : "saved together"
```

**Namespace convention:**
- Standalone training: `models/resilient-iot/{wandb_run_name}/`
- Sweep trials: `models/resilient-iot-sweeps/{study_name}/`

---

## Summary Table

| # | Store | Technology | Location | Purpose |
|---|-------|-----------|----------|---------|
| 1 | `riot_traffic` | TimescaleDB (PG16) | Docker container | Traffic readings hypertable + model/dataset lineage |
| 2 | `optuna.db` | SQLite | Project root | Hyperparameter sweep studies & trials |
| 3 | Raw data | Parquet + JSON | `data/` | Source traffic counts, weather, sensor metadata |
| 4 | Master dataset | Parquet | `data/processed/` | Fully preprocessed, feature-enriched training data |
| 5 | Feature caches | JSON + Parquet | `data/external/` | POI, OSM graph metrics, edge features |
| 6 | W&B | Cloud SaaS | wandb.ai | Experiment tracking, metrics, model artifacts |
| 7 | R2 bucket | Cloudflare R2 | `resilient-iot-payloads` | Paginated UI payload served to frontend |
| 8 | Model artifacts | `.pt` + `.yaml` | `models/` | Trained model checkpoints + configs |
