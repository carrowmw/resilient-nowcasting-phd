# Exploratory Data Analysis (EDA) Summary

## 1. Datasets Overview

### Traffic Data (`traffic_20230108_20260108.parquet`)
- **Size:** 21,144,475 records, 6 columns.
- **Columns:** `category` (str), `dir` (str), `dt` (datetime64), `location` (str), `value` (int64), `veh_class` (str).
- **Date Range:** 2023-03-02 to 2026-01-08.
- **Description:** Contains traffic measurements (e.g., flow) at various sensor locations broken down by vehicle class (e.g., 'person', 'car') and direction.

### Weather Data (`weather_20230108_20260108.parquet`)
- **Size:** 2,715,140 records, 13 columns.
- **Columns:** `sensor_name` (str), `variable` (str - e.g., 'Humidity', 'Solar Radiation'), `timestamp` (datetime64), `value` (float64), `units` (object), `flagged` (bool), `location_wkt` (str), `sensor_centroid_latitude` (float64), `sensor_centroid_longitude` (float64), `sensor_height_above_ground` (float64), `ground_height_above_sea_level` (float64), `broker_name` (str), `raw_id` (str).
- **Date Range:** 2023-01-08 to 2025-10-20.

### Sensor Metadata JSONs
- **`traffic_sensors.json`:** Maps string `location` identifiers to numeric `lat` and `lon` coordinates.
- **`weather_sensors.json`:** Maps `Sensor_Name` identifiers to `Location_WKT`, `Sensor_Centroid_Latitude`, `Sensor_Centroid_Longitude`, and elevation details.

---

## 2. Likely Target Variable
The likely target variable for predictive modeling is **`value`** in the **Traffic dataset**. 
Since `Congestion_Level` is not present, `value` corresponds to the quantitative traffic flow/volume or speed metric at a given location and time. The goal of an ML pipeline here would likely be to predict the future traffic `value` using historical traffic patterns and correlated weather phenomena.

---

## 3. Data Quality Summary

### Missing Values
- **Traffic Data:** While there are **0 missing values** across all columns, this is likely because missing and zero-count rows have been pre-filtered. 
    - **Temporal Gaps:** Analysis reveals significant gaps in the time series. The standard measurement frequency is **15 minutes** (the mode). However, **~23% of gaps** are larger than 15 minutes, with many being multiples (30, 45, 60m, etc.). Some gaps span several days or even over a year (max gap of 588 days).
    - **Zero-Count Deletion:** Only 1,964 records (out of 21.1M) have a `value` of 0. This confirms that rows representing zero traffic or missing sensor readings were largely dropped from the dataset before storage, necessitating a re-indexing/resampling step to restore the true temporal continuity for modeling.
- **Weather Data:** The **`units`** column is entirely empty (2,715,140 missing values) and should be dropped. All other 12 columns contain **0 missing values**.

### Outliers and Anomalies
- **Negative Traffic Values:** The `value` column in the traffic dataset has a minimum value of **-27**. Assuming this represents a physical count of vehicles or traffic flow, negative numbers are physically impossible and represent anomalous readings, sensor errors, or special null-equivalent codes (e.g., similar to a -99 placeholder). These must be filtered or imputed before training.
- **Traffic Maximums:** The maximum `value` is 2,691, with a mean of 13.3 and a 75th percentile of 10. The large max value may represent sudden heavy bursts of traffic or aggregated anomalies.

### Class Imbalances / Distributions
- **Traffic `category`:** Heavily imbalanced towards `'flow'` (20.5M out of 21.1M records). 
- **Traffic `veh_class`:** The most frequent class is `'person'` (6.8M records), though there are 16 unique vehicle classes overall.
- **Traffic `dir` (Direction):** 'east_to_west' is the most heavily represented direction (4.1M records) out of 122 unique directions.
- **Weather `variable`:** Dominated by `'Humidity'` (1.38M out of 2.7M records) and collected predominantly by the `aq_mesh_api` broker.

## 4. Next Steps for Pipeline Design
1. **Data Cleaning:** Drop `units` from weather data. Filter or handle negative `value` anomalies in the traffic dataset.
2. **Temporal Alignment:** Re-index the traffic dataset to a continuous 15-minute time series (resampling and filling gaps with zeros or imputed values) to restore physical continuity.
3. **Data Integration:** Use spatial coordinates (from the JSON files) to map traffic sensors to the nearest weather sensors, subsequently merging the dataframes on datetime (`dt` and `timestamp`) and location identifiers.
4. **Feature Engineering:** Extract temporal features (hour, day of week, month) from the datetimes, pivot weather variables to columns, and aggregate traffic volumes to a consistent time-frequency (e.g., hourly).