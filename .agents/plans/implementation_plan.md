# Implementation Plan — Phase 3 (Layer 3: Edge Features)

**Implementer:** Claude (Senior ML Engineer)
**Date:** 2026-06-30
**Source spec:** `.agents/plans/architect_plan.md` §3.3 (Layer 3) + §8.4 checklist
**Target branch:** `main`
**Predecessors:** Phase 1 (Scale-Invariant Loss) ✅ merged; Phase 2 (Kriging) ✅ merged.

---

## 0. Reconnaissance (current state on `main`)

- **`src/decoders.py` is NOT touched in Phase 3.** `compute_loss` already returns the 4-tuple
  and `pipeline.forward` already returns the 10-tuple (done in Phase 2).
- `src/graph_learner.py`: `DynamicGraphLearner.forward(static_features)` returns
  `(edge_index, edge_weight)`. No cache logic.
- `src/encoder.py`: `MaskedSpatiotemporalEncoder` hardcodes `GATv2Conv(edge_dim=1)` and batches a
  1-D `edge_weight` via `.repeat(num_graphs)`.
- `src/pipeline.py`: `graph_learner(static_features)` → optional `simulated_nodes_mask` edge
  filtering → `encoder(dynamic, edge_index, edge_weight, ...)`. No `edge_features_cache_path`.
- `src/dataset.py`: `__getitem__` returns 5 keys (no `location_ids`); `custom_collate_real` collates those 5.
- Cache `data/external/edge_features_cache.json`: **104 sources × 103 dests = 10,712 pairs**.

## ⚠️ Deviations from architect_plan.md §3.3.1 pseudocode (REQUIRE acknowledgement)

The plan's pseudocode does not match the real cache. Three concrete corrections, all verified:

1. **Cache key names.** Real keys are `pedestrian`, `residential`, `tertiary`, `secondary`,
   `primary`, `service` — NOT the `*_fraction` names in the pseudocode. The 9 physical features:
   `shortest_path_distance`, `euclidean_distance`, `directness` (3 distance/geometry) + the 6 road
   fractions above. (10th dim = learned similarity from `edge_weight`.)

2. **Road fractions are NOT always ≈1.0.** Verified across all 10,712 pairs: 10,674 sum to 1.0
   (±1e-3); **38 sum to exactly 0.0** (disconnected pairs — no OSM path). The plan's assertion
   `allclose(road_sums, ones, atol=1e-3)` **would crash on the real cache.**
   → **Corrected assertion:** each edge's road-sum ≈1.0 **OR** ≈0.0:
   `((road_sums - 1).abs() < 1e-3) | (road_sums.abs() < 1e-3)`.

3. **Performance — no per-edge Python loop.** The pseudocode loops over E edges with `.item()` +
   dict lookups *every forward pass* (~10k edges/batch) — a severe training bottleneck. Physical
   features are static; only the 10th dim changes. → **Corrected design:** precompute a dense
   `[N, N, 9]` physical-feature tensor **once** (memoized on first forward, keyed by `location_ids`),
   then vectorized-gather `[E, 9]` via `edge_index` and `torch.cat` the learned `edge_weight` as
   column 9. Pure tensor ops, no Python loop.

## 🔵 Decision point (recommendation — will follow plan literally if you prefer)

- **Distance normalization.** Plan normalizes only `shortest_path_distance` by global `max_distance`
  (=7670.55). `euclidean_distance` is also in metres (up to ~7.7k) and would enter the GAT raw,
  dwarfing the [0,1] fractions. **Recommendation:** normalize BOTH `shortest_path_distance` and
  `euclidean_distance` by `max_distance`; `directness` is already [0,1]. Flag if you want plan-literal
  (shortest-path only).

---

## 1. File-by-file changes

### 1.1 `src/graph_learner.py`
- Imports: add `json`, `from pathlib import Path`, `from typing import Optional, List`.
- `__init__(..., edge_features_cache_path: Optional[str] = None)`:
  - If provided: assert file exists (fail-fast `FileNotFoundError`), `json.load` →
    `self.edge_features_cache`; compute `self.max_distance` = max `shortest_path_distance` over all
    pairs, assert `> 0`. If `None`: `self.edge_features_cache = None` (learned-only fallback).
  - Memoization slots: `self._phys_dense = None`, `self._phys_location_ids = None`.
- New `_build_physical_dense(location_ids, device) -> Tensor [N, N, 9]` (built once, numpy→tensor):
  - For each (i, j), i≠j: fail-fast `ValueError` (`"CRITICAL: Missing cache for source/edge ..."`)
    if `src_loc`/`dst_loc` absent; fill 9 features. Diagonal left zero (self-pairs never become edges).
  - Column order `[sp_norm, eucl_norm, directness, pedestrian, residential, tertiary, secondary,
    primary, service]` (per Decision Point).
- New `_build_edge_attr(edge_index, edge_weight, location_ids) -> Tensor [E, 10]`:
  - Cache `None` → return `edge_weight.unsqueeze(1)` (`[E, 1]` fallback).
  - Else: ensure dense built/memoized (rebuild if `location_ids` or device changed);
    `phys = self._phys_dense[edge_index[0], edge_index[1]]` (`[E, 9]`);
    `torch.cat([phys, edge_weight.unsqueeze(1)], dim=1)` (`[E, 10]`).
  - Offensive asserts: `shape == (E, 10)`; `not isnan().any()`; corrected road-sum invariant on cols 3:9.
- `forward(static_features, location_ids: Optional[List[str]] = None)`:
  - Existing similarity/sparsity → `(edge_index, edge_weight)` unchanged.
  - `edge_attr = _build_edge_attr(...)` if `location_ids and cache`, else `edge_weight.unsqueeze(1)`.
  - **Return `(edge_index, edge_weight, edge_attr)`** (3-tuple — signature change).

### 1.2 `src/encoder.py`
- `__init__(..., edge_dim: int = 1)`; pass `edge_dim=edge_dim` to `GATv2Conv`.
- `forward`: 3rd param `edge_weight` → `edge_attr` (accept `[E]` or `[E, D]`; if 1-D, `unsqueeze(1)`).
  Batch as `edge_attr.repeat(num_graphs, 1)` → `[E*num_graphs, D]`; pass `edge_attr=` to `self.gat`
  in both branches. Assert `batched_edge_attr.shape[0] == batched_edge_index.shape[1]`.
  Backward-compatible: existing `test_encoder.py` passes 1-D `edge_weight` with default `edge_dim=1`.

### 1.3 `src/pipeline.py`
- `__init__(..., edge_features_cache_path: Optional[str] = None)`:
  - Pass to `DynamicGraphLearner(...)`.
  - `encoder_edge_dim = 10 if edge_features_cache_path else 1`; pass `edge_dim=encoder_edge_dim`
    to `MaskedSpatiotemporalEncoder(...)`.
- `forward`:
  - `location_ids = batch.get('location_ids')`.
  - `edge_index, edge_weight, edge_attr = self.graph_learner(static_features, location_ids=location_ids)`.
  - In `simulated_nodes_mask` block: also `edge_attr = edge_attr[valid_edges]` (row-alignment).
  - Pass `edge_attr` (not `edge_weight`) to both `self.encoder(...)` calls (analysis + normal).

### 1.4 `src/dataset.py`
- `__getitem__`: add `'location_ids': self.locations` to the returned dict.
- `custom_collate_real`: add `'location_ids': batch_list[0]['location_ids']` (constant across batch;
  list[str] passes through train.py's `.to(device)` untouched).

### 1.5 `src/train.py`
- Read `edge_features_cache_path = model_cfg.get('edge_features_cache_path')` and pass to
  `ResilientIoTPipeline(...)`. Default `None` → mock path & configs without the key fall back to learned-only.

### 1.6 `configs/train/train.yaml` + `train_test.yaml`
- Add under `model:`: `edge_features_cache_path: "data/external/edge_features_cache.json"`.

---

## 2. New script `scripts/desk_check_edge_features.py` (mirrors existing desk-check style)
Using a small temp mock cache:
1. `edge_attr` shape `[E, 10]`; road fractions (cols 3:9) ≈1.0 per connected edge.
2. **Disconnected-pair tolerance:** all-zero road fractions pass the corrected assertion.
3. Distance normalization in `[0, 1]` (cols 0–1).
4. Fail-fast: missing cache entry raises `ValueError` containing `"CRITICAL: Missing cache"`.
5. Fallback: `cache_path=None` → `edge_attr` shape `[E, 1]` equals `edge_weight`.
6. Vectorized-gather matches a reference per-edge dict lookup.
Run: `python scripts/desk_check_edge_features.py` (all asserts pass before tests).

## 3. Tests to update
- `tests/test_graph_learner.py`: unpack **3 returns**; assert `edge_attr.shape == (E, 1)` no-cache.
  Add `test_edge_attr_with_cache` (temp 3-node cache incl. one all-zero disconnected pair → `[E,10]`,
  road-sum invariant, norm range) and `test_missing_cache_fail_fast` (`ValueError`).
- `tests/test_encoder.py`: keep existing `edge_dim=1` test; add `test_encoder_edge_dim_10`
  (`edge_dim=10`, `edge_attr=[E,10]`, assert output shape + gradient flow).
- `tests/test_pipeline.py`: existing tests use no cache → still pass (fallback); verify green.
Run: `pytest tests/test_graph_learner.py tests/test_encoder.py tests/test_pipeline.py`.

## 4. Verification sequence
0. **Git safety:** `git add . && git commit -m "Auto-commit before Phase 3"` (at start of implementation).
1. `python scripts/desk_check_edge_features.py` → all asserts pass.
2. `pytest tests/test_graph_learner.py tests/test_encoder.py tests/test_pipeline.py` → green.
3. `pytest tests/` → full suite green (no regressions).
4. Smoke test (1 epoch, DB mode, real cache):
   `docker compose exec agent-workspace python src/train.py --use-database --config configs/train/train_test.yaml`
   → finite loss, no NaN, kriging telemetry present, cache loaded (104 sources logged).
5. Update `EXPERIMENTS.md` with the Phase 3 entry + desk-check/verification results.

## 5. Risks / watch-items
- **location_id ↔ cache-key match:** dataset `self.locations` must be a subset of cache keys; fail-fast
  `ValueError` surfaces any mismatch at the first batch (caught in smoke test step 4).
- **edge_dim consistency:** guaranteed — pipeline derives `edge_dim` from the same
  `edge_features_cache_path` condition that drives graph-learner output width.
- **Memoization device:** dense tensor rebuilt if `location_ids`/device changes; safe for single-GPU.

---

**Awaiting your approval before writing any functional code.**
