# Urban Observatory (UO) & Contextual Data Directory

This directory contains the raw and cached urban sensor datasets required to reproduce all experiments in this repository (Track 1: Sensor Quality, Track 2: Spatial Topology & GATv2, Track 3: Resilient IoT Unified Pipeline v4–v7 & Sparsity Scaling).

---

## 1. Important Data Provenance Notice (Private vs. Public UO API)

> [!IMPORTANT]
> **Private Urban Observatory API Snapshot**:
> The pedestrian traffic time-series dataset (`data/traffic_20230108_20260608.parquet`) and sensor metadata (`data/traffic_sensors.json`) committed in this repository were extracted from a **private Newcastle Urban Observatory (UO) API endpoint**.
>
> Specifically, this snapshot includes **restricted and internal sensor streams** across Newcastle upon Tyne that are **not exposed on the public Urban Observatory API**. Because all four raw/external Parquet files are compact enough to fit directly within GitHub's file size limits (21 MB – 60 MB each), they are committed directly to this repository so that **all experiments in this codebase can be reproduced out-of-the-box without external API keys or private network credentials**.
>
> **Future Research & Extensions**:
> - To **reproduce the experiments in this repository**, you **must** use the committed data files in `data/`.
> - For **further research, new time windows, or additional cities**, data can be fetched from the **Public Newcastle Urban Observatory API** (`https://api.urbanobservatory.ac.uk/api/v2.0a/` or `https://newcastle.urbanobservatory.ac.uk/`) combined with the Open-Meteo ERA5 weather ingestion script (`scripts/ingestion/fetch_and_ingest_weather.py`).

---

## 2. Committed Data Inventory

All required raw time-series, meteorological caches, and static urban context files are tracked in Git:

| File Path | Size | Description |
| :--- | :--- | :--- |
| `data/traffic_20230108_20260608.parquet` | 60 MB | Raw 15-minute pedestrian and vehicle count observations across Newcastle upon Tyne (`2023-01-08` to `2026-06-08`, ~24.9M rows). Columns: `[location, category, veh_class, dir, dt, value]`. |
| `data/traffic_sensors.json` | 12 KB | Metadata and WGS-84 coordinates (`lat`, `lon`) for all 104 traffic sensor locations. |
| `data/weather_20230108_20260108.parquet` | 21 MB | Raw Urban Observatory meteorological sensor observations (`Humidity`, `Solar Radiation`). |
| `data/weather_sensors.json` | 852 KB | Metadata and centroid coordinates for Urban Observatory weather stations. |
| `data/external/weather_cache.parquet` | 54 MB | Complete 15-minute ERA5 historical weather reanalysis cache (`2023-01-08` to `2026-06-08`, 12.46M rows across all 104 sensor locations) fetched via Open-Meteo (`scripts/ingestion/fetch_and_ingest_weather.py`). Eliminates weather sensor gaps. |
| `data/external/static_features_cache.parquet` | 41 MB | Pre-joined static urban context feature table across sensor locations. |
| `data/external/places_cache.json` | 45 KB | 12 land-use and Point-of-Interest (POI) density features per sensor (e.g., `transit_count`, `retail_count`, `food_drink_count`, `nightlife_count`, `education_count`, `recreation_count`, `total_poi_count`, `landuse_entropy`, `is_school_zone`, `dist_to_city_centre`, `nearby_bus_stops`, `nearby_parking`). |
| `data/external/osm_features_cache.json` | 14 KB | 3 road/pedestrian network graph centrality features per sensor (`betweenness_centrality`, `closeness_centrality`, `degree_centrality`) computed via OSMnx. |
| `data/external/edge_features_cache.json` | 2.1 MB | Pairwise physical road network edge attributes for all $104 \times 104 = 10,712$ sensor pairs: shortest-path distance, Euclidean distance, route directness, and 6 highway-class fraction channels (`pedestrian`, `residential`, `tertiary`, `secondary`, `primary`, `service`). |
| `data/external/simulated_*.json` | 1.3 MB | Pre-cached coordinates, POI features, and OSM edge features for synthetic/simulated evaluation nodes. |

---

## 3. Preparing the Data for Training & Evaluation

The codebase supports two interchangeable data backends via `src/db_connector.py` (`data_source: "parquet"` or `data_source: "database"`):

### Option A: Local Parquet Mode (Zero-Database Setup)
To generate the unified 15-minute master Parquet file (`data/processed/master_dataset_15T.parquet`, ~421 MB, ignored by Git) directly from the committed raw files:

```bash
docker compose run --rm cpu-runner python src/process_all.py
```

Once generated, any training or evaluation script can run directly against the Parquet file without TimescaleDB by setting `data_source: "parquet"` and `parquet_path: "data/processed/master_dataset_15T.parquet"` in the config YAML. Note that **Track 1 (`src/quality/`)** and **Track 2 (`src/spatial/`)** read `data/traffic_20230108_20260608.parquet` and `data/external/` directly and do not require TimescaleDB.

### Option B: TimescaleDB Production Mode (Used by RIoT v5–v7 Suites)
For high-throughput multi-GPU training and payload export, ingest the committed Parquet/JSON files into the Dockerized TimescaleDB service:

```bash
# 1. Start TimescaleDB
docker compose up -d timescaledb

# 2. Run Alembic schema migrations
docker compose run --rm dev-workspace alembic upgrade head

# 3. Ingest sensor metadata, static features, weather cache, and traffic readings
docker compose run --rm dev-workspace python scripts/ingestion/ingest_sensor_metadata.py
docker compose run --rm dev-workspace python scripts/ingestion/ingest_static_features.py
docker compose run --rm dev-workspace python scripts/ingestion/ingest_weather.py
docker compose run --rm dev-workspace python scripts/ingestion/ingest_parquet.py
```

---

## 4. Fetching New Data (Public UO API & Open-Meteo)

For future research beyond the 2023–2026 experimental window:
1. **Public Urban Observatory API**: Query `https://api.urbanobservatory.ac.uk/api/v2.0a/sensors/data/json` (or the Newcastle UO portal) and format readings into a Parquet table with schema `[location, category, veh_class, dir, dt, value]`.
2. **Open-Meteo Historical Weather API**: Run `scripts/ingestion/fetch_and_ingest_weather.py` to fetch and interpolate hourly ERA5 reanalysis `relative_humidity_2m` and `shortwave_radiation` onto the 15-minute sensor grid:
   ```bash
   docker compose run --rm dev-workspace python scripts/ingestion/fetch_and_ingest_weather.py \
     --sensors-json data/traffic_sensors.json \
     --start-date 2023-01-08 \
     --end-date 2026-06-08 \
     --output-parquet data/external/weather_cache.parquet
   ```
3. **Static & Edge Feature Caches**: Run `scripts/cache_edge_features.py` to compute OSMnx road network centralities and pairwise shortest-path edge features for any new sensor coordinates.
