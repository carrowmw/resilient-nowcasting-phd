# Implementation Plan: Open-Meteo Historical Weather Pipeline & Local Cache

## 1. Problem Statement & Root Cause
Inspection of the RIoT v4 data artifacts confirmed that Channels 1 and 2 (`Humidity` and `Solar Radiation`) in `master_dataset_15T.parquet` and `static_features_cache.parquet` are **100% NaN (0 valid observations out of 144.8M rows)**. As a result, the model was trained on fabricated constant zeros (`fillna(0.0)`), leaving the weather channels devoid of physical signal.

This implementation plan details the selection of a suitable weather API, an offline-first caching mechanism, temporal interpolation to the 15-minute traffic grid, and integration into TimescaleDB and the dataset connectors.

---

## 2. API Selection: Open-Meteo Historical Weather API

### Why Open-Meteo?
1. **Open & Keyless Access:** No API key required for non-commercial and academic research (eliminates secret leakage risks and credential management).
2. **Reanalysis Coverage (ERA5 / ERA5-Land):** Continuous, gap-free, physically grounded historical reanalysis from 1940 to present. Perfectly covers our entire window (`2023-01-08` through `2026-06-08`).
3. **Exact Target Variables:**
   - `relative_humidity_2m` (%): Maps directly to Channel 1 (`Humidity`), physical range $[0, 100]$.
   - `shortwave_radiation` ($W/m^2$): Global horizontal solar irradiance, maps directly to Channel 2 (`Solar Radiation`), physical range $[0, \approx 1200]$.
4. **Endpoint Specification:**
   - Base URL: `https://archive-api.open-meteo.com/v1/archive`
   - Request format:
     ```text
     GET https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&hourly=relative_humidity_2m,shortwave_radiation&timezone=UTC
     ```

---

## 3. Architecture & Caching Strategy

```
                                 +-------------------------------+
                                 |  data/traffic_sensors.json    |
                                 | (sensor_id, lat, lon coords)  |
                                 +---------------+---------------+
                                                 |
                                                 v
+-------------------------------+  Query coords  +-------------------------------+
| Open-Meteo Historical API     |<---------------+ scripts/ingestion/            |
| (Hourly ERA5 Reanalysis)      |--------------->| fetch_and_ingest_weather.py   |
+-------------------------------+  JSON stream   +---------------+---------------+
                                                                 |
                                       +-------------------------+-------------------------+
                                       |                                                   |
                                       v                                                   v
                        +-------------------------------+                   +-------------------------------+
                        | Disk Cache (Parquet):         |                   | TimescaleDB Hypertable:       |
                        | data/external/weather_cache/  |                   | weather_readings              |
                        | weather_cache.parquet         |                   | (dt, sensor_id, humidity,     |
                        | (15-min interpolated, no NaN) |                   |  solar_radiation)             |
                        +---------------+---------------+                   +---------------+---------------+
                                        |                                                   |
                                        v                                                   v
                         ParquetConnector.load_weather()                     TimescaleConnector.load_weather()
                                        \                                                   /
                                         +-------------------------+-----------------------+
                                                                   |
                                                                   v
                                                     RealIoTDataset (src/dataset.py)
                                                     Channel 1: Humidity (scaled)
                                                     Channel 2: Solar Radiation (scaled)
```

### Key Principles:
1. **Offline-First Disk Cache:**
   - Raw hourly responses and final 15-minute interpolated series are saved to `data/external/weather_cache.parquet`.
   - Subsequent runs or retraining check the cache first, ensuring zero redundant API calls, deterministic reproducibility, and offline training support.
2. **Coordinate De-duplication:**
   - Group the ~81–89 sensor locations by unique `(round(lat, 3), round(lon, 3))` coordinate cells (~100m resolution) to minimize API queries while preserving micro-climatic variations.
3. **Temporal Interpolation to 15-Min Grid:**
   - Open-Meteo returns hourly values (`:00`).
   - Resample to 15-minute timestamps (`:00`, `:15`, `:30`, `:45`).
   - `Humidity`: Linear interpolation, strictly bounded to $[0.0, 100.0]$.
   - `Solar Radiation`: PCHIP / piecewise cubic or linear interpolation, clamped to $[0.0, \infty)$. Enforce strict $0.0$ solar radiation between sunset and sunrise (zenith angle / night mask).
4. **Idempotent Database Ingestion:**
   - Insert records in batches of 50,000 into TimescaleDB `weather_readings` using `execute_values` with `ON CONFLICT (dt, sensor_id) DO UPDATE SET humidity = EXCLUDED.humidity, solar_radiation = EXCLUDED.solar_radiation`.

---

## 4. File-by-File Execution Plan

### A. New Script: `scripts/ingestion/fetch_and_ingest_weather.py`
- **Location:** `scripts/ingestion/fetch_and_ingest_weather.py`
- **Responsibilities:**
  1. Load target sensors from `data/traffic_sensors.json` (or database `sensor_metadata`).
  2. Parse date bounds from `configs/train/v4/train_v4_full_inductive.yaml` (`2023-01-08` to `2026-06-08`).
  3. Check `data/external/weather_cache.parquet`. For missing coordinates or date ranges:
     - Query Open-Meteo API with exponential backoff and rate-limit compliance (1-2s delay between requests).
     - Collect hourly `relative_humidity_2m` and `shortwave_radiation`.
  4. Perform 15-min interpolation and physical clamping.
  5. Validate data with offensive assertions (no NaNs, non-zero variance, physical bounds).
  6. Write/update `data/external/weather_cache.parquet`.
  7. Ingest into TimescaleDB `weather_readings` hypertable if `--to-database` is specified (default).

### B. Update Script: `scripts/ingestion/ingest_weather.py`
- Deprecate or re-route `scripts/ingestion/ingest_weather.py` to invoke `fetch_and_ingest_weather.py`, ensuring backwards compatibility for automated setup scripts.

### C. Update Connector: `src/db_connector.py`
- Verify that `ParquetConnector.load_weather` can consume `data/external/weather_cache.parquet` directly:
  ```python
  def load_weather(self, start_dt, end_dt, sensor_ids: List[str]) -> pd.DataFrame:
      # If parquet_path is a traffic file without weather, fall back to weather_cache.parquet
  ```
- Ensure `TimescaleConnector.load_weather` trips clean informative errors if `weather_readings` lacks rows for the requested window.

### D. Desk Check: `scripts/desk_checks/desk_check_weather_pipeline.py`
- A standalone verification script that:
  1. Loads 1 week of weather data for 3 test sensors from both the cache and the database.
  2. Asserts non-zero, realistic distributions:
     - Mean humidity between 60% and 90% (typical UK climate).
     - Solar radiation diurnal cycle: peak $> 200\text{ W/m}^2$ at noon, strictly $0.0\text{ W/m}^2$ between 23:00 and 04:00.
     - Zero NaNs.
  3. Tests `RealIoTDataset` dynamic tensor stacking: asserts `dynamic_data[:, :, 1]` and `dynamic_data[:, :, 2]` have positive standard deviation (no static zeros).

---

## 5. Verification & Defensive Tripwires

Before any retraining is triggered, the following gates must pass inside the container:
1. **Assertion Gate:**
   - `assert (df['Humidity'] > 0).any()`
   - `assert (df['Solar Radiation'] > 0).any()`
   - `assert df['Humidity'].isna().sum() == 0`
   - `assert df['Solar Radiation'].isna().sum() == 0`
   - `assert (df['Humidity'] <= 100.0).all()`
   - `assert (df['Solar Radiation'] >= 0.0).all()`
2. **Dataset Parity Check:**
   - Run `python tests/verify_db_parity.py --start 2024-05-01 --end 2024-05-08` and verify that weather columns match between DB and Parquet cache.
3. **Dry-Run Training Pass:**
   - Execute 1 epoch with `train_v4_full_inductive.yaml` to confirm that the `weather_scaler` computes realistic means/stds (e.g. `mean_hum ~ 75`, `std_hum ~ 15`) and no NaN losses are produced.

---

## 6. Documentation & Experiment Ledger
- Update `docs/database/database_schemas.md` to document the Open-Meteo cache layout and ingestion mechanism.
- Append a new row to `EXPERIMENTS.md` once the pipeline is verified.
- Note in `docs/thesis/chapter7_experiment3_reconstruction.tex` that the meteorological channels are now grounded in physical reanalysis data.
