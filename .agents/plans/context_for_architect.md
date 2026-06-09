# Context for Architect: Expanded Static Spatial Features

## 1. Problem Statement

The model's spatial understanding is severely degraded because the current static features are **hitting the Google Places API (New) Nearby Search hard ceiling of 20 results per request**. This means 81 of 94 cached locations show `commercial_density = 20` and 60 show `nearby_poi_count = 20`. The `DynamicGraphLearner` receives these near-identical inputs and cannot differentiate between a quiet suburban street and the city centre. This directly explains why the model underpredicts high-volume nodes (e.g., Barras Bridge Civic Centre, RMSE=36.05 despite r=0.98).

Additionally, `dist_to_nearest_intersection` is hardcoded to `150.0` for all locations — a mock value that provides zero information.

## 2. Current Static Feature Pipeline

### Entry Point: `src/process_all.py`
- `process_full_dataset()` (line 48) instantiates `StaticFeatureLoader()` and calls `feature_loader.join_features(joined_df)` per location chunk (line 103).
- The enriched parquet is written to `data/processed/master_dataset_15T.parquet`.

### Feature Loader: `src/features.py`
- `StaticFeatureLoader` class.
- Cache path: `data/external/places_cache.json` (94 entries, 18KB).
- API: Google Places API (New) — `POST https://places.googleapis.com/v1/places:searchNearby`.
- Auth: `GOOGLE_PLACES_API_KEY` env var. Header `X-Goog-Api-Key`.
- Field mask: `places.id` (minimal — only counts place IDs).

#### Current Features Extracted (line 88–111):
| Feature | API Call | Radius | Types | Issue |
|---------|---------|--------|-------|-------|
| `commercial_density` | Nearby Search | 500m | `['store']` | Capped at 20 (81/94 = 20) |
| `is_school_zone` | Nearby Search | 250m | `['school']` | Boolean, OK |
| `nearby_poi_count` | Nearby Search | 500m | `['restaurant', 'cafe', 'park', 'tourist_attraction']` | Capped at 20 (60/94 = 20) |
| `highway_proximity` | Nearby Search | 500m | `['transit_station']` | Capped at 20 (62/94 = 20) |
| `dist_to_nearest_intersection` | None | N/A | N/A | **Hardcoded 150.0** |

#### `get_count()` helper (line 52–86):
- Single POST request, returns `min(len(places), 50)`.
- **No pagination** — just reads `data.get('places', [])` from first response.
- Nearby Search (New) has a **hard max of 20 results per request, no pagination token support**.

### Cache Structure (`data/external/places_cache.json`):
- Some entries have 3 keys (older, simulated nodes): `commercial_density`, `is_school_zone`, `highway_proximity`.
- Some entries have 5 keys (newer, real nodes): adds `nearby_poi_count`, `dist_to_nearest_intersection`.
- `join_features()` checks for missing keys and re-fetches if incomplete (line 136–144).

## 3. Downstream Consumers

### Dataset: `src/dataset.py`
- `RealIoTDataset.__init__()` reads static feature columns from parquet (line 42–47):
  ```
  'nearby_poi_count', 'dist_to_nearest_intersection', 
  'commercial_density', 'is_school_zone', 'highway_proximity'
  ```
- Static features are extracted per-location (line 130–136), scaled via `StandardScaler` (line 143–144), and stored as `self.static_features` tensor `[num_nodes, static_feat_dim]`.
- Currently `static_feat_dim = 5`.

### Graph Learner: `src/graph_learner.py`
- `DynamicGraphLearner.__init__()` takes `static_feat_dim` as input (line 17).
- Linear projection: `nn.Linear(static_feat_dim, embedding_dim)` where `embedding_dim=32`.
- Pairwise dot-product → ReLU → sparsity threshold → `dense_to_sparse`.
- **Impact**: More informative static features → better adjacency matrix → better message passing.

### Pipeline: `src/pipeline.py`
- `ResilientIoTPipeline.__init__()` passes `static_feat_dim` to `DynamicGraphLearner` (line 41).
- Forward pass feeds `batch['static_features']` to graph learner (line 99).

### Export UI Payload: `src/export_ui_payload.py`
- Constants (line 57): `graph_learner.lin.weight → [32, 5]` ⟹ `static_feat_dim=5`.
- **Changing static_feat_dim requires retraining the model** — the checkpoint weights are locked to 5-dim input.

### Train/Tune: `src/train.py`, `src/tune.py`
- Both read `static_feat_dim` from the dataset: `dataset.static_feat_dim`.
- Passed to `ResilientIoTPipeline(static_feat_dim=dataset.static_feat_dim, ...)`.

## 4. Google Places API (New) Capabilities

### Nearby Search (New)
- **Max 20 results per request. NO pagination support.**
- Good for: "Are there schools within 250m?" (boolean/low-count queries).
- Not suitable for density estimation.

### Text Search (New)
- **Max 60 results across 3 pages** (20 per page, with `nextPageToken`).
- Supports `locationBias` (circle with center + radius) for geographic restriction.
- Supports `includedType` (singular) for type filtering.
- Better for density estimation when combined with category-specific queries.

### Strategy for True Density
To get counts > 20, we must issue **multiple targeted queries per category** and/or use **concentric radius rings** (e.g., 100m, 250m, 500m) to build a density profile.

## 5. Sensor Metadata

- `data/traffic_sensors.json`: 89 real sensors with `lat`, `lon`, `location` fields.
- All sensors are in the Newcastle upon Tyne metropolitan area (lat ≈ 54.97–55.01, lon ≈ -1.57 to -1.66).
- Sensor locations range from city centre (Quayside, Grey St) to suburban (Heaton, Gosforth) to rural periphery (Norham Rd, Castle Farm).

## 6. Tripwires and Constraints

- **Offensive Programming**: `features.py` crashes on 403/429 API errors (line 75–76).
- **Cache-first**: Always checks cache before API call. Saves after each fetch.
- **Assertions in dataset.py**: `assert df[col].notna().all()` for all static feature columns (line 178).
- **Model retraining required**: Any change to `static_feat_dim` invalidates the current `best_pipeline.pt` checkpoint.
- **NSTAR_PHASES.md**: Currently in Phase 2 (Backend Ablation Studies). Static feature enrichment aligns with this phase.
- **NSTAR_BACKEND_ARCHITECTURE.md**: Static node features feed the Dynamic Graph Learner. The architecture doc mentions "coordinates, sensor types" as examples — expanding to richer features is fully aligned.

## 7. Files That Must Change

| File | Change Required |
|------|----------------|
| `src/features.py` | Major rewrite: new API strategy, categorised queries, pagination, new feature schema |
| `data/external/places_cache.json` | Will be regenerated with new schema |
| `src/dataset.py` | Update `stat_cols` list (line 130–136) and parquet column list (line 42–47) |
| `src/pipeline.py` | `static_feat_dim` auto-adapts from dataset — no code change needed |
| `src/graph_learner.py` | `static_feat_dim` auto-adapts — no code change needed |
| `src/process_all.py` | May need updates if `join_features()` signature changes |
| `src/export_ui_payload.py` | Checkpoint constant comments (line 57) — update after retraining |
| `src/train.py` / `src/tune.py` | Auto-adapt via `dataset.static_feat_dim` — no code change needed |
| `data/processed/master_dataset_15T.parquet` | Must be regenerated |
