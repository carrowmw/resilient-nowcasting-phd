# Architect Plan: Expanded Static Spatial Features for DynamicGraphLearner

## Problem

The `DynamicGraphLearner` infers graph topology from static node features via a learned linear projection followed by pairwise dot-product similarity. The quality of the inferred adjacency matrix — and therefore the quality of all downstream message passing, imputation, and forecasting — is **upper-bounded** by the discriminative power of those static features.

Currently, 4 of the 5 static features are saturated at their API ceiling of 20, and the 5th (`dist_to_nearest_intersection`) is hardcoded to 150.0. The graph learner receives near-identical feature vectors for 80%+ of nodes and cannot distinguish a quiet suburban path from the city centre Quayside. This is the primary bottleneck for spatial understanding.

## Design Goals

1. **Break the 20-result ceiling**: Use Text Search (New) with pagination (up to 60 results across 3 pages) and concentric radius rings for true density estimation.
2. **Categorised POI features**: Replace the monolithic `nearby_poi_count` with domain-specific counts that encode *pedestrian behavioral context* (transit, retail, nightlife, education, recreation).
3. **Land-use entropy**: Compute a Shannon entropy score across categories to capture mixed-use vs. single-use zoning.
4. **Street network topology**: Use OSMnx (OpenStreetMap) to extract betweenness centrality and distance-to-city-core for each sensor.
5. **Backward-compatible pipeline**: The existing `StaticFeatureLoader.join_features()` contract is preserved. Downstream consumers (`dataset.py`, `pipeline.py`, `train.py`, `tune.py`) auto-adapt via `dataset.static_feat_dim`.

## Proposed Feature Schema (New)

The new cache will store the following features per location. `static_feat_dim` will increase from **5 → 15**.

| # | Feature Name | Source | Method | Rationale |
|---|-------------|--------|--------|-----------|
| 1 | `transit_count` | Google Places (New) | Text Search, type=`transit_station`, 500m, paginated | Commuting peaks (AM/PM). Replaces `highway_proximity`. |
| 2 | `retail_count` | Google Places (New) | Text Search, type=`store`, 500m, paginated | Sustained daytime traffic. Replaces `commercial_density`. |
| 3 | `food_drink_count` | Google Places (New) | Text Search, types=`restaurant`+`cafe`, 500m, paginated | Midday / lunch peaks. |
| 4 | `nightlife_count` | Google Places (New) | Nearby Search, types=`bar`+`night_club`, 500m | Late-night Fri/Sat peaks. Low counts expected, no pagination needed. |
| 5 | `education_count` | Google Places (New) | Nearby Search, types=`school`+`university`, 500m | School-hour and term-time spikes. |
| 6 | `recreation_count` | Google Places (New) | Nearby Search, types=`park`+`tourist_attraction`+`museum`, 500m | Weekend / tourist traffic. |
| 7 | `total_poi_count` | Derived | Sum of features 1–6 | Absolute density. Replaces `nearby_poi_count`. |
| 8 | `landuse_entropy` | Derived | Shannon entropy over categories 1–6 | Mixed-use (high) vs. single-use (low). |
| 9 | `is_school_zone` | Google Places (New) | Nearby Search, type=`school`, 250m, count > 0 | Boolean. Retained from current schema. |
| 10 | `betweenness_centrality` | OSMnx | `ox.graph_from_point()` → `nx.betweenness_centrality()` | On major walking routes? Replaces `dist_to_nearest_intersection`. |
| 11 | `closeness_centrality` | OSMnx | `nx.closeness_centrality()` | How accessible is this node from the rest of the network? |
| 12 | `dist_to_city_centre` | Haversine | Distance from sensor to Newcastle city centre (54.9738, -1.6131) | Proxy for urban density gradient. |
| 13 | `degree_centrality` | OSMnx | Node degree in the walking network | Intersection complexity. |
| 14 | `nearby_bus_stops` | Google Places (New) | Nearby Search, type=`bus_station`, 300m | Bus access strongly correlates with pedestrian flow. |
| 15 | `nearby_parking` | Google Places (New) | Nearby Search, type=`parking`, 500m | Park-and-walk pattern indicator. |

## Architecture Decisions

### 1. Text Search (New) with Pagination for Density Features

**Why Text Search, not Nearby Search?** Nearby Search (New) has a hard max of 20 results and **no pagination**. Text Search (New) supports up to 60 results across 3 pages via `nextPageToken`. For density-critical features (transit, retail, food_drink), we must use Text Search.

**Implementation pattern:**
```
POST https://places.googleapis.com/v1/places:searchText
Headers: X-Goog-Api-Key, X-Goog-FieldMask: places.id
Body: {
  "textQuery": "stores",
  "includedType": "store",
  "locationBias": { "circle": { "center": { "latitude": ..., "longitude": ... }, "radius": 500.0 } },
  "maxResultCount": 20
}
```
Then follow `nextPageToken` for pages 2 and 3, with a 2-second delay between page requests (API requirement).

**For low-count features** (nightlife, education, recreation, school_zone, bus_stops, parking): Nearby Search (New) is sufficient. Max 20 is unlikely to be hit for `bar`+`night_club` in a 500m radius.

### 2. OSMnx for Street Network Topology

OSMnx is the standard Python library for downloading and analyzing OpenStreetMap street networks. It will be used **offline** (run once per sensor set, cached) to compute:
- Download the Newcastle walking network: `ox.graph_from_point((54.978, -1.610), dist=5000, network_type='walk')`
- For each sensor, find the nearest OSM node: `ox.nearest_nodes(G, lon, lat)`
- Compute betweenness, closeness, and degree centrality on the full graph.

**Caching**: OSMnx results will be stored in a separate `data/external/osm_features_cache.json` to keep the Google API cache separate. Both caches are merged during `join_features()`.

### 3. Cache Migration Strategy

The existing `places_cache.json` will be **cleanly replaced** (not appended to). The `join_features()` method already handles cache invalidation via `required_keys` checking (line 141–143 of current `features.py`). The new required keys list will trigger a full re-fetch.

**Rate limiting**: With 89 real sensors × ~9 Google API calls per sensor × 3 pages max = ~2,400 API calls. At the standard rate limit, this should complete in ~15 minutes with appropriate delays. We must add a `time.sleep(0.1)` between requests and `time.sleep(2.0)` between pagination pages.

### 4. Shannon Entropy Calculation

```python
def shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)
```

Normalised to [0, 1] by dividing by `log2(num_categories)`.

### 5. Backward Compatibility & Retraining

- `dataset.py` reads column names from parquet. We update the `stat_cols` list.
- `pipeline.py`, `graph_learner.py`, `train.py`, `tune.py` all read `static_feat_dim` dynamically from the dataset. **No code changes needed** in these files.
- The current `best_pipeline.pt` checkpoint is **invalidated** by this change (weight shape `[32, 5]` → `[32, 15]`). A full retrain is required. This aligns with Phase 2 (Ablation Studies).

## Proposed Changes

---

### Component 1: Static Feature Loader

#### [MODIFY] [features.py](file:///Users/administrator/Code/ml-agent-workspace/src/features.py)

Major rewrite of `StaticFeatureLoader`:

1. **New helper `_text_search_count()`**: Uses Text Search (New) API with pagination. Follows `nextPageToken` for up to 3 pages. Returns count up to 60.
2. **Updated `_fetch_location_data()`**: Issues categorised queries (transit, retail, food_drink via Text Search; nightlife, education, recreation, school_zone, bus_stops, parking via Nearby Search). Computes `total_poi_count` and `landuse_entropy` as derived features. Computes `dist_to_city_centre` via Haversine.
3. **New helper `_fetch_osm_features()`**: Uses OSMnx to compute betweenness, closeness, and degree centrality. Caches to `data/external/osm_features_cache.json`.
4. **Updated `join_features()`**: Merges both Google and OSM caches. Updated `required_keys` list (15 features).
5. **Rate limiting**: `time.sleep(0.1)` between API calls, `time.sleep(2.0)` between pagination pages.

---

### Component 2: Dataset

#### [MODIFY] [dataset.py](file:///Users/administrator/Code/ml-agent-workspace/src/dataset.py)

1. Update parquet column list (line 42–47) to include all 15 new feature names.
2. Update `stat_cols` list (line 130–136) to match.
3. `static_feat_dim` auto-computes from `self.static_features.shape[-1]`. No other changes needed.

---

### Component 3: Data Processing Pipeline

#### [MODIFY] [process_all.py](file:///Users/administrator/Code/ml-agent-workspace/src/process_all.py)

1. If `join_features()` signature or behaviour changes, update the call site (line 103).
2. No structural changes expected — the interface is preserved.

---

### Component 4: Cache Files

#### [DELETE + REGENERATE] `data/external/places_cache.json`

Will be regenerated with the new 15-feature schema when `process_all.py` is next run.

#### [NEW] `data/external/osm_features_cache.json`

New cache file for OSMnx-derived street network features.

---

### Component 5: Preprocessor (Minor)

#### [MODIFY] [preprocess.py](file:///Users/administrator/Code/ml-agent-workspace/src/preprocess.py)

The `enrich_static_features()` method (line 171–196) uses a mock metadata schema that is now obsolete. It should be updated or removed if no longer used (the real enrichment happens via `StaticFeatureLoader.join_features()`).

---

## Verification Plan

### Automated Tests
1. **Unit test**: Run `python src/features.py` standalone test (existing `if __name__` block) with a small subset of sensors against the live API. Verify the cache file contains all 15 features for each location.
2. **Shape assertion**: After `process_all.py` completes, load the parquet and assert all 15 columns exist and have no NaN values.
3. **Distribution check**: Print `describe()` for each new feature column. Verify no feature has > 50% identical values (the saturation problem we're fixing).

### Manual Verification
1. Spot-check cache entries for known locations (e.g., Quayside should have high retail, transit, and food counts; Norham Rd should have low counts across all categories).
2. After retrain: Compare per-node RMSE distribution with the baseline. The expectation is that high-volume nodes should see reduced RMSE due to better graph topology.

## Desk Check: Entropy Calculation

For a node with counts: transit=15, retail=40, food=30, nightlife=5, education=2, recreation=8:
- total = 100
- probs = [0.15, 0.40, 0.30, 0.05, 0.02, 0.08]
- H = -(0.15·log₂0.15 + 0.40·log₂0.40 + 0.30·log₂0.30 + 0.05·log₂0.05 + 0.02·log₂0.02 + 0.08·log₂0.08)
- H = -(0.15·(-2.737) + 0.40·(-1.322) + 0.30·(-1.737) + 0.05·(-4.322) + 0.02·(-5.644) + 0.08·(-3.644))
- H = -(−0.4105 − 0.5288 − 0.5211 − 0.2161 − 0.1129 − 0.2915)
- H = 2.0809
- H_max = log₂(6) = 2.585
- Normalised entropy = 2.081 / 2.585 = **0.805** (high mixed-use)

For a residential node: transit=0, retail=1, food=0, nightlife=0, education=1, recreation=0:
- total = 2
- probs = [0.5, 0.5]
- H = -(0.5·(-1) + 0.5·(-1)) = 1.0
- Normalised = 1.0 / 2.585 = **0.387** (low mixed-use)

✅ The entropy correctly discriminates between mixed-use and single-use areas.

## Implementation Order

1. **features.py** — New API strategy, OSMnx integration, new feature schema.
2. **dataset.py** — Updated column lists.
3. **preprocess.py** — Clean up obsolete mock schema.
4. **Run `process_all.py`** — Regenerate parquet with new features.
5. **Retrain model** — New `static_feat_dim=15` requires fresh training.
6. **Update EXPERIMENTS.md** — Log the feature expansion as a new experiment.
