# Implementation Plan: Static Spatial Features Pipeline Expansion (5 → 15 Dimensions)

**Date**: 2026-06-09  
**Based on**: `architect_plan.md` + `context_for_architect.md`  
**Target output**: `data/processed/master_dataset_15T.parquet` with 15 static feature columns, no NaNs

---

## Pre-flight Checks (Before Writing Code)

### ENV-1: osmnx / geopandas not installed
```bash
pip install osmnx geopandas
```
Must complete successfully before any OSMnx code runs.

### ENV-2: GOOGLE_PLACES_API_KEY not set
The environment variable `GOOGLE_PLACES_API_KEY` is currently unset. The standalone pagination test and `process_all.py` will both raise `ValueError` without it. The user must export this key before Phase 5 and Phase 6. **The implementation plan does not block on this** — all code is written first; the API calls happen last.

---

## Phase 1: Rewrite `src/features.py`

**Goal**: Replace the 5-feature, non-paginated loader with a 15-feature, paginated loader.

### 1.1 New constants / imports
- Add `import math`, `import time`
- Add `NEWCASTLE_CENTRE = (54.9738, -1.6131)` for Haversine distance
- Add `OSM_CACHE_PATH = 'data/external/osm_features_cache.json'`
- Keep existing imports (`logging`, `pandas`, `requests`, `json`, `os`, `numpy`, `random`)

### 1.2 New helper: `_haversine_distance(lat1, lon1, lat2, lon2) -> float`
- Returns distance in **metres** between two WGS-84 points.
- Pure Python, no external dependency.

### 1.3 New helper: `_shannon_entropy(counts: list[int]) -> float`
- Normalised Shannon entropy in [0, 1] across N categories.
- `H = -sum(p * log2(p)) / log2(N)` where `N = len(counts)`.
- Returns 0.0 if total is 0 or if only one non-zero category.
- Handles edge: if N == 1, return 0.0 (no diversity possible).

### 1.4 Retained helper: `_nearby_search_count(lat, lon, place_types, radius) -> int`
- Same logic as current `get_count()` but extracted as a method.
- Used for low-count features: nightlife, education, recreation, school_zone, bus_stops, parking.
- Still crashes hard (raises `RuntimeError`) on 403/429.
- Returns raw count (no `min(..., 50)` cap — let the data speak).

### 1.5 New helper: `_text_search_paginated(lat, lon, text_query, included_type, radius) -> int`
- Uses `POST https://places.googleapis.com/v1/places:searchText`
- Headers: `X-Goog-Api-Key`, `X-Goog-FieldMask: places.id`
- Body:
  ```json
  {
    "textQuery": "<text_query>",
    "includedType": "<included_type>",
    "locationBias": { "circle": { "center": { "latitude": ..., "longitude": ... }, "radius": 500.0 } },
    "maxResultCount": 20
  }
  ```
- Follows `nextPageToken` for up to **3 pages** (max 60 results).
- `time.sleep(2.0)` between page requests (API requirement).
- Crashes hard (raises `RuntimeError`) on 403/429 on any page.
- Returns **total accumulated count** across all pages (no artificial cap).

### 1.6 New method: `_build_osm_cache(sensors_data: list) -> dict`
- Checks if `data/external/osm_features_cache.json` already exists and is fully populated for all locations; if so, loads and returns it.
- If not: downloads Newcastle walking network via:
  ```python
  G = ox.graph_from_point((54.978, -1.610), dist=5000, network_type='walk')
  ```
- Computes on the full graph (not per-node):
  - `betweenness = nx.betweenness_centrality(G, normalized=True)`
  - `closeness = nx.closeness_centrality(G)`
  - `degree = nx.degree_centrality(G)`
- For each sensor: find nearest OSM node via `ox.nearest_nodes(G, lon, lat)`, then look up its centrality values.
- Saves result as `{ location_name: { betweenness_centrality, closeness_centrality, degree_centrality } }` to `data/external/osm_features_cache.json`.
- Returns the dict.

**Shape invariant assertion**: After mapping, every sensor in `sensors_data` must have all three OSM features. Raise `RuntimeError` on any missing.

### 1.7 Updated method: `_fetch_location_data(lat, lon) -> dict`

Issues the following API calls in order (with `time.sleep(0.1)` between each):

| Call | Method | Type(s) | Radius | Output key |
|------|--------|---------|--------|------------|
| 1 | Text Search paginated | `transit_station` | 500m | `transit_count` |
| 2 | Text Search paginated | `store` | 500m | `retail_count` |
| 3 | Text Search paginated (restaurant) + Nearby Search (cafe) | mixed | 500m | `food_drink_count` (sum) |
| 4 | Nearby Search | `bar`, `night_club` | 500m | `nightlife_count` |
| 5 | Nearby Search | `school`, `university` | 500m | `education_count` |
| 6 | Nearby Search | `park`, `tourist_attraction`, `museum` | 500m | `recreation_count` |
| 7 | Nearby Search | `school` | 250m | `is_school_zone` (bool: count > 0) |
| 8 | Nearby Search | `bus_station` | 300m | `nearby_bus_stops` |
| 9 | Nearby Search | `parking` | 500m | `nearby_parking` |

Derived (no API call):
- `total_poi_count` = sum of `transit_count` + `retail_count` + `food_drink_count` + `nightlife_count` + `education_count` + `recreation_count`
- `landuse_entropy` = `_shannon_entropy([transit_count, retail_count, food_drink_count, nightlife_count, education_count, recreation_count])`
- `dist_to_city_centre` = `_haversine_distance(lat, lon, *NEWCASTLE_CENTRE)`

Returns 12 Google/derived features (OSM 3 merged in `join_features`).

**Note on `food_drink_count`**: Text Search `includedType` accepts only one type. So we make two calls: Text Search for `restaurant` and Nearby Search for `cafe`, then sum them. The combined result can exceed 20 for busy nodes.

### 1.8 Updated method: `join_features(df) -> pd.DataFrame`

```
required_keys = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking'
]
```

Steps:
1. Build/load OSM cache via `_build_osm_cache(sensors_data)`.
2. For each unique location: if cache entry is missing or `required_keys` not all present → call `_fetch_location_data()` + merge in OSM features → save cache.
3. Build `metadata_df` from merged Google + OSM cache.
4. Left-merge with `df` on `location`.
5. Assert all 15 columns exist and have no NaNs.

**Crash on shape mismatch**: `assert len(df) == original_len` (already present, retained).

---

## Phase 2: Update `src/dataset.py`

### 2.1 Update `pd.read_parquet()` columns list (lines 43–47)
Replace old 5 static feature names with all 15 new names:
```python
columns=[
    'dt', 'location', 'value', 'Humidity', 'Solar Radiation',
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking',
    'veh_class', 'dir'
]
```

### 2.2 Update `agg_funcs` dict (lines 66–75)
Add all 15 new static feature names with `'first'` aggregation; remove old 5 names.

### 2.3 Update `stat_cols` list (lines 130–136)
```python
stat_cols = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking'
]
```

### 2.4 Update boolean cast (line 138)
`is_school_zone` column name is unchanged, so the `.astype(float).fillna(0.0)` line needs no change.

**Result**: `dataset.static_feat_dim` will automatically be 15 — no other downstream changes needed.

---

## Phase 3: Clean up `src/preprocess.py`

### 3.1 Remove `enrich_static_features()` method (lines 171–196)
This method uses an obsolete mock schema (`highway_proximity`, `commercial_density`, `school_zone` as NaN/False) and is never called by `process_all.py`. It is dead code and will be removed entirely.

---

## Phase 4: Cache Migration

### 4.1 Clear `data/external/places_cache.json`
Write an empty JSON object `{}` to the cache file. The new `required_keys` check will trigger a full re-fetch for all 94 locations on next run.

**Do NOT delete the file** — preserve the path so `_load_cache()` doesn't fail.

### 4.2 Create `data/external/osm_features_cache.json`
Write an empty JSON object `{}`. The `_build_osm_cache()` method will populate it on first run.

---

## Phase 5: Standalone Pagination Test (`tests/test_text_search_pagination.py`)

**Purpose**: Verify that `_text_search_paginated()` yields counts > 20 for central nodes *before* running the expensive 89-sensor loop.

**Test structure**:
1. Skip with clear message if `GOOGLE_PLACES_API_KEY` is not set.
2. Instantiate `StaticFeatureLoader` with a temp cache path.
3. For `NclQuaysideUnderHighLevelBridge` (lat=54.9675, lon=-1.6097):
   - Call `_text_search_paginated()` for `transit_station`, `store`, `restaurant`
   - Print page-by-page counts
   - **Assert at least one of transit/retail/restaurant > 20** (verifying pagination is actually fetching pages 2 and 3)
4. Compare vs. Nearby Search count for same types (should be ≤ 20)
5. Print a summary table: feature | nearby_count | text_search_count | pages_used

**Crash contract**: The test raises `AssertionError` (not silently passes) if:
- Any 403/429 is encountered
- All text search counts are ≤ 20 (pagination may have failed silently)

---

## Phase 6: Run `process_all.py` (User-triggered)

```bash
export GOOGLE_PLACES_API_KEY=<your_key>
cd /workspace && python src/process_all.py
```

Expected duration: ~15–20 minutes (89 real sensors × ~9 API calls × up to 3 pages, with 0.1s delays + 2.0s between pages, + OSMnx network download ~1–2 min).

---

## Phase 7: Verification Script (`tests/verify_parquet_15features.py`)

After `process_all.py` completes:

1. Load `data/processed/master_dataset_15T.parquet`.
2. **Assert** all 15 feature columns are present.
3. **Assert** no NaN values in any of the 15 columns.
4. Print `df[stat_cols].describe()`.
5. **Assert** no feature column has > 50% identical values (the saturation check).
6. Spot-check: Print Quayside row vs. a suburban sensor (e.g., `NclSaltersHollywood`).

---

## Phase 8: Update `EXPERIMENTS.md`

Add a new row to the experiment ledger documenting:
- Date, branch, goal, outcome
- Feature expansion: 5 → 15 static features
- API strategy change: Nearby Search (capped at 20) → Text Search paginated (up to 60) + OSMnx
- Expected impact: Reduced RMSE on high-volume nodes (Barras Bridge, Quayside) after retrain

---

## Execution Order

```
Phase 0: pip install osmnx geopandas          [run now, env setup]
Phase 1: Rewrite src/features.py               [code]
Phase 2: Update src/dataset.py                 [code]
Phase 3: Clean src/preprocess.py               [code]
Phase 4: Clear/create cache files              [file ops]
Phase 5: Write tests/test_text_search_pagination.py  [code]
─── AWAIT API KEY ───
Phase 5b: python tests/test_text_search_pagination.py  [requires API key]
Phase 6: python src/process_all.py             [requires API key, ~15 min]
Phase 7: python tests/verify_parquet_15features.py    [post-run verification]
Phase 8: Update EXPERIMENTS.md                 [ledger]
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/features.py` | Full rewrite: 15-feature schema, Text Search pagination, OSMnx, Shannon entropy |
| `src/dataset.py` | Update `columns`, `agg_funcs`, `stat_cols` lists |
| `src/preprocess.py` | Remove dead `enrich_static_features()` method |
| `data/external/places_cache.json` | Reset to `{}` (triggers full re-fetch) |
| `data/external/osm_features_cache.json` | Create as `{}` (new file) |
| `tests/test_text_search_pagination.py` | New pagination test |
| `tests/verify_parquet_15features.py` | New post-run verification script |
| `EXPERIMENTS.md` | Add experiment entry |

## Files NOT Changed (auto-adapt)

| File | Why |
|------|-----|
| `src/pipeline.py` | Reads `static_feat_dim` from dataset — auto-adapts |
| `src/graph_learner.py` | Reads `static_feat_dim` as constructor arg — auto-adapts |
| `src/train.py` | Reads `static_feat_dim` from dataset — auto-adapts |
| `src/tune.py` | Reads `static_feat_dim` from dataset — auto-adapts |
| `src/process_all.py` | `join_features()` interface unchanged — no edit needed |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Text Search `includedType` rejects multi-type queries | Split into separate calls; sum results (as planned for `food_drink_count`) |
| OSMnx download fails (network) | Crash with clear error; retry manually; result cached so only runs once |
| `nextPageToken` missing on page 1 despite >20 results | Log a warning but continue; count will be ≤ 20 for that type only |
| 15-col dataset breaks existing checkpoint | Expected and documented — full retrain required after this change |
| `api_key` is None at runtime | `ValueError` raised in `__init__` (already present); plan notes API key requirement |
