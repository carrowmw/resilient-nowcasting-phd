# Architect Plan: Simulated-Node Edge Feature Caching
**Date:** 2026-07-01  
**Branch:** main  
**Author:** @architect  
**Status:** APPROVED FOR IMPLEMENTATION

---

## 1. Problem Statement

During payload generation (`export_paginated_payload.py` / `export_ui_payload.py`), five
synthetic nodes (`Simulated_Node_0` … `Simulated_Node_4`) are injected into every batch.
`DynamicGraphLearner._build_physical_dense` iterates every ordered `(src, dst)` pair in
`location_ids` and looks up features in `self.edge_features_cache`.  Because
`data/external/edge_features_cache.json` covers only the 104 real sensors, any pair
touching a simulated node is absent and the method crashes:

```
ValueError: CRITICAL: Missing cache for edge NclAdelaideTer BeechSt -> Simulated_Node_0
```

---

## 2. Architecture Overview

```
data/external/
  edge_features_cache.json           ← IMMUTABLE: 104 real sensors only
  simulated_edge_features_cache.json ← NEW: all pairs involving ≥1 simulated node
```

At `DynamicGraphLearner` init time, if a `simulated_edge_features_cache_path` is
supplied the two caches are **deep-merged in memory**:

```
merged[real_i] = {**real_cache[real_i], **sim_cache.get(real_i, {})}
merged[sim_i]  = sim_cache[sim_i]
```

`max_distance` is then recomputed across **all** pairs in the merged cache.
`_build_physical_dense`, `_build_edge_attr`, and the memoization logic are **unchanged**.

---

## 3. Simulated Cache Schema

The file `simulated_edge_features_cache.json` uses the **same nested dict format** as
the real cache:

```json
{
  "NclAdelaideTer BeechSt": {
    "Simulated_Node_0": { "shortest_path_distance": ..., "euclidean_distance": ...,
                          "directness": ..., "pedestrian": ..., "residential": ...,
                          "tertiary": ..., "secondary": ..., "primary": ..., "service": ... },
    ...
  },
  "Simulated_Node_0": {
    "NclAdelaideTer BeechSt": { ... },
    "Simulated_Node_1": { ... },
    ...
  },
  ...
}
```

Stored pairs:
- `real_i → sim_j`  (all real sensors, all simulated nodes)
- `sim_i → real_j`  (all simulated nodes, all real sensors)
- `sim_i → sim_j`  (all simulated nodes, all other simulated nodes)

`real_i → real_j` pairs are **not stored** here; they stay in the main cache.

---

## 4. Step-by-Step Implementation Plan

### Step 1 — `scripts/cache_simulated_edge_features.py` (NEW)

**Purpose:** Compute and persist the simulated edge feature cache.

**Algorithm:**
1. Load real sensor coords from `data/traffic_sensors.json` (exclude names starting with
   `Simulated`).
2. Build simulated node coord list from `export_ui_payload.SIMULATED_NODE_COORDS`
   (sliced to `--num_simulated`, default 5). Names: `Simulated_Node_0` … `Simulated_Node_{N-1}`.
3. Download Newcastle walking network: `ox.graph_from_point((54.978, -1.610), dist=5000, network_type='walk')`.
4. **Bounding-box fail-fast:** compute the graph's bbox (`min/max` of node x/y attributes).
   For each simulated coordinate, assert it falls within the bbox:
   ```python
   assert lon_min <= lon <= lon_max and lat_min <= lat <= lat_max, \
       f"CRITICAL: Simulated node coord ({lat}, {lon}) outside OSM graph bbox"
   ```
5. Call `compute_pairwise_edge_features(G, real_coords + sim_coords, cache_path=None)` to
   obtain the full N_real+N_sim pairwise dict in memory.
6. Filter: for each `src`, keep only destination entries where at least one of
   `src` or `dst` is a simulated node name (starts with `"Simulated_Node_"`).
   This removes `real → real` pairs from the output — those already live in the main cache.
7. Save the filtered dict to `--output` (default `data/external/simulated_edge_features_cache.json`).
8. Log: number of sources, total directed pairs, any pairs with no OSM path (omitted by
   `compute_pairwise_edge_features`'s existing warning).

**CLI flags:**
```
--output         path for simulated cache (default data/external/simulated_edge_features_cache.json)
--num_simulated  number of simulated nodes (default 5, must be ≤ len(SIMULATED_NODE_COORDS))
--sensors_path   path to traffic_sensors.json (default data/traffic_sensors.json)
--self-test      run on the existing mock graph from cache_edge_features.py; no OSMnx download
```

**`--self-test` behaviour:**
- Use `build_mock_graph()` (import from `cache_edge_features`) with 5 named `S1`…`S5`.
- Treat `S4` and `S5` as "simulated" (by name convention in the test, not by a `Simulated_` prefix).
- Assert the output contains exactly the expected `S3↔S4`, `S3↔S5`, `S4↔S5`,
  `S1↔S4`, `S2↔S4`, `S1↔S5`, `S2↔S5` directed pairs — and no `S1↔S2`, `S1↔S3`, `S2↔S3`.

---

### Step 2 — `src/graph_learner.py`: Dual-Cache Merge

**Change `DynamicGraphLearner.__init__` signature:**
```python
def __init__(self,
             static_feat_dim: int,
             embedding_dim: int = 32,
             sparsity_threshold: float = 0.5,
             edge_features_cache_path: Optional[str] = None,
             simulated_edge_features_cache_path: Optional[str] = None):
```

**Guard:** if `simulated_edge_features_cache_path` is provided but `edge_features_cache_path`
is `None`, raise:
```python
raise ValueError(
    "CRITICAL: simulated_edge_features_cache_path requires edge_features_cache_path to also be set"
)
```

**Load and merge logic (insert after real cache is loaded, still inside `if edge_features_cache_path` block):**
```python
if simulated_edge_features_cache_path is not None:
    sim_path = Path(simulated_edge_features_cache_path)
    if not sim_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: Simulated edge features cache not found: {sim_path}"
        )
    with open(sim_path, 'r') as f:
        sim_cache = json.load(f)
    for src, dst_dict in sim_cache.items():
        if src in self.edge_features_cache:
            self.edge_features_cache[src].update(dst_dict)
        else:
            self.edge_features_cache[src] = dict(dst_dict)
    logger.info(
        f"Merged simulated cache: {len(sim_cache)} additional sources from {sim_path}"
    )

# Recompute max_distance from the merged cache (must happen AFTER merge).
distances = [
    pair_data['shortest_path_distance']
    for pairs in self.edge_features_cache.values()
    for pair_data in pairs.values()
]
self.max_distance = max(distances)
assert self.max_distance > 0, \
    f"CRITICAL: max_distance must be positive, got {self.max_distance}"
```

**No changes** to `_build_physical_dense`, `_build_edge_attr`, or `forward`.

---

### Step 3 — `src/pipeline.py`: Pass-Through

Add `simulated_edge_features_cache_path: Optional[str] = None` to
`ResilientIoTPipeline.__init__` and forward it to `DynamicGraphLearner`:

```python
self.graph_learner = DynamicGraphLearner(
    static_feat_dim=static_feat_dim,
    embedding_dim=embedding_dim,
    sparsity_threshold=sparsity_threshold,
    edge_features_cache_path=edge_features_cache_path,
    simulated_edge_features_cache_path=simulated_edge_features_cache_path,
)
```

Update the `logger.info` line to include `simulated_edge_features_cache_path`.

`encoder_edge_dim = 10 if edge_features_cache_path else 1` — **unchanged**; the simulated
cache does not change edge_dim.

---

### Step 4 — Export Script Integration

**`src/export_paginated_payload.py` — `build_model()`** (around line 314):
```python
model = ResilientIoTPipeline(
    ...
    edge_features_cache_path=model_cfg.get("edge_features_cache_path"),
    simulated_edge_features_cache_path=model_cfg.get("simulated_edge_features_cache_path"),
).to(device)
```

**`src/export_ui_payload.py`** — model construction (around line 899):
```python
edge_features_cache_path=model_cfg.get("edge_features_cache_path"),
simulated_edge_features_cache_path=model_cfg.get("simulated_edge_features_cache_path"),
```

---

### Step 5 — Configuration Schema Updates

**`configs/train/train.yaml`** (under `model:`):
```yaml
simulated_edge_features_cache_path: "data/external/simulated_edge_features_cache.json"
```

**`configs/train/train_test.yaml`** (under `model:`):
```yaml
simulated_edge_features_cache_path: "data/external/simulated_edge_features_cache.json"
```

The analysis config (`configs/analysis/analysis.yaml`) does **not** need a `model:` block —
the co-located artifact config carries hyperparameters. If an artifact was trained without
this key (legacy checkpoint), `model_cfg.get` returns `None` and the existing
`_build_physical_dense` fail-fast fires — no silent degradation.

---

### Step 6 — Unit Tests

**`tests/test_graph_learner.py`** — add five new test functions:

1. **`test_dual_cache_merge_shape()`**  
   Build a real mini-cache (3 real nodes: `R1`, `R2`, `R3`) and a simulated mini-cache
   (2 sim nodes: `Simulated_Node_0`, `Simulated_Node_1`) containing all
   `R↔Sim` and `Sim↔Sim` directed pairs.  
   Init `DynamicGraphLearner` with both paths.  
   Run forward with `location_ids = ["R1", "R2", "R3", "Simulated_Node_0", "Simulated_Node_1"]`.  
   Assert `edge_attr.shape == (E, 10)`.

2. **`test_dual_cache_max_distance()`**  
   Set one simulated pair's `shortest_path_distance` deliberately larger than any real pair.
   Assert `learner.max_distance` equals that larger value (max recomputed across merged cache).

3. **`test_dual_cache_road_sum_invariant()`**  
   After forward pass with 5 nodes (3 real + 2 sim), assert every edge's road-fraction
   columns (3:9) sum to ~1.0 (connected) or ~0.0 (disconnected).

4. **`test_simulated_cache_required_guard()`**  
   Assert `ValueError("CRITICAL: simulated_edge_features_cache_path requires ...")` is
   raised when `simulated_edge_features_cache_path` is set but `edge_features_cache_path`
   is `None`.

5. **`test_simulated_cache_missing_file_fail_fast()`**  
   Assert `FileNotFoundError("CRITICAL: Simulated edge features cache not found")` is raised
   when `simulated_edge_features_cache_path` points to a nonexistent file (with a valid
   `edge_features_cache_path`).

**`tests/test_cache_simulated_edge_features.py`** (NEW):

1. **`test_self_test_runs()`**  
   Call the script's `run_self_test()` function directly; assert it completes without exception.

2. **`test_output_excludes_real_real_pairs()`**  
   Run the cache generator on the mock graph. Load the output JSON.
   Assert no `output[real_name][real_name]` entry exists (only sim-touching pairs).

3. **`test_bounding_box_fail_fast()`**  
   Pass a coordinate far outside the mock graph's node bbox.
   Assert `ValueError` containing "outside OSM graph bbox" is raised.

---

### Step 7 — Desk Check: `scripts/desk_check_simulated_edge_features.py` (NEW)

Scenario: 3 real sensors (`R1`, `R2`, `R3`) + 2 simulated nodes (`Simulated_Node_0`,
`Simulated_Node_1`). Construct both cache files with hand-chosen values so every
assertion is numerically verifiable without OSMnx.

**Verifications (all must PASS):**

| Check | Assertion |
|---|---|
| `edge_attr.shape == (E, 10)` | Assert after forward pass |
| No NaNs in `edge_attr` | `assert not torch.isnan(edge_attr).any()` |
| Road fractions (cols 3:9) sum ∈ {~1.0, ~0.0} | `connected \| disconnected` mask |
| `max_distance` = global max across merged cache | Assert equals hand-computed value |
| `Real→Sim` pair features match sim cache values | Per-edge reference dict lookup |
| `Sim→Real` pair features match sim cache values | Per-edge reference dict lookup |
| `Sim→Sim` pair features match sim cache values | Per-edge reference dict lookup |
| `Real→Real` pair features match real cache values | Per-edge reference dict lookup |
| 10th column = learned `edge_weight` | `torch.allclose(edge_attr[:, 9], edge_weight)` |

Run via:
```
docker compose exec cpu-runner python scripts/desk_check_simulated_edge_features.py
```

---

## 5. File Inventory

### New Files
| Path | Purpose |
|---|---|
| `scripts/cache_simulated_edge_features.py` | Cache generator for simulated node edge features |
| `scripts/desk_check_simulated_edge_features.py` | Desk check for combined-cache correctness |
| `tests/test_cache_simulated_edge_features.py` | Unit tests for the cache generator |

### Modified Files
| Path | Change |
|---|---|
| `src/graph_learner.py` | Add `simulated_edge_features_cache_path` param + deep-merge logic |
| `src/pipeline.py` | Add param + pass-through to `DynamicGraphLearner` |
| `src/export_ui_payload.py` | Pick up `simulated_edge_features_cache_path` from model_cfg |
| `src/export_paginated_payload.py` | Same in `build_model()` |
| `configs/train/train.yaml` | Add `simulated_edge_features_cache_path` under `model:` |
| `configs/train/train_test.yaml` | Same |
| `tests/test_graph_learner.py` | Add 5 dual-cache unit tests |

---

## 6. Key Invariants (Offensive Assertions Required)

The implementer MUST include these assertions exactly as specified:

1. **`cache_simulated_edge_features.py`**: Raise `ValueError` immediately if any simulated
   coordinate lies outside the OSM graph's bounding box — no silent clipping.
2. **`DynamicGraphLearner.__init__`**: Raise `ValueError("CRITICAL: ...")` if
   `simulated_edge_features_cache_path` is not `None` but `edge_features_cache_path` is `None`.
3. **`DynamicGraphLearner.__init__`**: Raise `FileNotFoundError("CRITICAL: Simulated edge
   features cache not found: ...")` if the simulated cache file does not exist.
4. **`DynamicGraphLearner.__init__`**: After merging, `assert self.max_distance > 0`.
5. **`_build_physical_dense`**: The existing `ValueError("CRITICAL: Missing cache for edge ...")`
   remains unchanged — after correct deployment this is an impossible-to-hit safety net.
6. **Desk check**: Road fractions must satisfy the `connected | disconnected` invariant for
   EVERY edge in the combined graph, not just real-real edges.

---

## 7. What NOT to Change

- `compute_pairwise_edge_features` in `src/features.py` — already correct and reusable as-is.
- `_build_physical_dense`, `_build_edge_attr`, or `forward` in `src/graph_learner.py` —
  they operate on `self.edge_features_cache` without needing to know the merge happened.
- `data/external/edge_features_cache.json` — strictly read-only production artifact.
- Any training loop, loss function, or model weight structure.
- `encoder_edge_dim = 10 if edge_features_cache_path else 1` in `pipeline.py`.

---

## 8. Experiment Ledger Update (MANDATORY)

Upon completing implementation, the implementer must append to `EXPERIMENTS.md`:

| Date | Branch | Task | Outcome |
|---|---|---|---|
| 2026-07-01 | main | Simulated edge feature cache: dual-cache merge in DynamicGraphLearner, new cache_simulated_edge_features.py script, 8 new unit tests, desk check | All tests pass; payload export no longer crashes on Real→Sim edges |

---

## 9. Implementer Prompt

@implementer:

Read this plan from `.agents/plans/architect_plan.md`. Before writing any code:

1. Read `src/graph_learner.py` fully.
2. Read `src/pipeline.py` fully.
3. Read `src/export_paginated_payload.py` lines 260–325 (the `build_model` function).
4. Read `src/export_ui_payload.py` lines 890–910 (model construction block).
5. Read `scripts/cache_edge_features.py` (your new script mirrors its structure).
6. Write `.agents/plans/implementation_plan.md` with your file-by-file execution plan.

**Wait for user approval of `implementation_plan.md` before writing any functional code.**

Implement in this order:
1. `scripts/cache_simulated_edge_features.py`
2. `src/graph_learner.py` changes
3. `src/pipeline.py` changes
4. `src/export_ui_payload.py` + `src/export_paginated_payload.py` changes
5. Config YAML changes
6. Tests (`tests/test_graph_learner.py` additions + `tests/test_cache_simulated_edge_features.py`)
7. `scripts/desk_check_simulated_edge_features.py`

After all code is written and tests pass, run:
```
python scripts/desk_check_simulated_edge_features.py
```
Then update `EXPERIMENTS.md`.
