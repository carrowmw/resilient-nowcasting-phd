# Data Preprocessing Plan

This plan outlines the required steps to preprocess the combined traffic and weather data into a unified, missingness-aware dataset for downstream machine learning. The pipeline will be constructed as a configurable class.

## 1. Temporal Resampling

The pipeline class will be initialized with a configurable `frequency` parameter (defaulting to `'15T'`, representing 15-minute intervals).

- **Process:** For each unique combination of `location`, `category`, `veh_class`, and `dir`, aggregate the traffic `value` into the specified frequency (`15T`).
- **Reindexing:** Generate a complete and continuous date range spanning the global minimum and maximum timestamps at the target frequency. Reindex the grouped time series to this exact date range.
- **Outcome:** This forces the implicitly missing timestamps to reappear as explicit `NaN` rows across all previously skipped intervals, preserving the true structure of the sensor network's recording behavior.

## 2. Imputation Strategy (Missingness-Aware)

We are building a missingness-aware architecture. General forward-filling or interpolation will **not** be used, as the model must see and learn from the true missingness state.

- **Structural Missingness (Priority 1):** Prioritize "Missing Zero" imputation. If an interval is missing but contextually implies zero traffic (e.g., adjacent zero counts, late-night hours, or specific structural patterns where zero-counts were dropped to save space), fill these specific `NaN`s with `0`.
- **True Gaps (Priority 2):** Any remaining `NaN` values that do not qualify as structural zeros will be left as explicit `NaN`s (or replaced with a specific missingness indicator/flag depending on the downstream model). We want the model to natively handle or recognize these gaps.

## 3. Spatial Join

We must pair each traffic observation with the most relevant local weather conditions, mapping traffic sensors to the nearest weather sensors.

- **Process:**
  1. Parse `traffic_sensors.json` for traffic `lat`/`lon`.
  2. Parse `weather_sensors.json` for weather `Sensor_Centroid_Latitude` and `Sensor_Centroid_Longitude`.
- **Defensive Edge Case:** Implement a check for $N=1$ weather sensors. If only a single weather sensor is available or active, bypass the distance matrix calculation entirely and globally assign this weather sensor's data to all traffic sensors to avoid calculation errors.
- **Distance Calculation:** If $N > 1$, calculate the Haversine distance matrix between all traffic and weather sensors, and assign the nearest weather sensor to each traffic location.

## 4. Feature Engineering

- **Negative Traffic Values:** Identify and handle physical impossibilities (e.g., `-27` in the `value` column). Treat these as sensor errors and convert them to `NaN` before the imputation logic.
- **Cyclical Time Features:** Extract components like `time_of_day` (based on the 15-minute bins), `day_of_week`, and `month`. Apply sine and cosine transformations to preserve cyclical continuity for the models.
- **Weather Pivoting:** Pivot the long-format weather data so that variables (e.g., `Humidity`, `Solar Radiation`) become individual feature columns aligned with the target frequency grid.

## 5. Static Feature Engineering (Enriched Metadata)

To provide the model with contextual geographic intelligence, we will define a schema to integrate external API data (e.g., Google Places API) as static features for each traffic sensor location.

- **Schema Definition:**
  - `highway_proximity`: Float (Distance to the nearest major highway/motorway).
  - `commercial_density`: Float/Int (Number of commercial businesses/POI within a specific radius).
  - `school_zone`: Boolean (Is the sensor located within or immediately adjacent to a school zone).
- **Integration:** These static features will be generated per sensor and joined to the main traffic dataset using the `location` identifier, allowing the model to learn baseline behavioral differences between locations.
