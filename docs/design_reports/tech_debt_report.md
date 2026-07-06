# Codebase Technical Debt Report

After a thorough examination of the codebase and cross-referencing with the project's North Star architecture documents (`AGENTS.md`, `NSTAR_BACKEND_ARCHITECTURE.md`, `NSTAR_FRONTEND_ARCHITECTURE.md`, `NSTAR_MLOPS.md`), I have identified several areas of potential technical debt, particularly violations of the project's strict design principles.

## 1. Defensive Programming Violations (CRITICAL)

The project's guidelines mandate an **Offensive Programming Paradigm (Fail Fast / Fail Hard)**. Specifically, the guidelines state: *"Never use a bare except: block. If an unexpected error occurs, the script must crash immediately... Do not use .dropna() or .fillna() silently."*

I found numerous violations of these rules:

### Broad Exception Handling (`except Exception:`)
There are **23 instances** of broad exception handling (often paired with `pass` or silent log continuation) across the backend Python scripts. This suppresses errors rather than letting the pipeline crash and expose the traceback.

- [src/train.py](file:///Users/administrator/Code/ml-agent-workspace/src/train.py): Lines 50, 190, 321, 699
- [src/tune.py](file:///Users/administrator/Code/ml-agent-workspace/src/tune.py): Lines 131, 230, 256, 331, 338, 397
- [src/features.py](file:///Users/administrator/Code/ml-agent-workspace/src/features.py): Lines 120, 234, 532, 551
- [src/analysis.py](file:///Users/administrator/Code/ml-agent-workspace/src/analysis.py): Lines 37, 75, 80, 122
- [src/export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py): Lines 154, 905, 906
- [src/export_paginated_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_paginated_payload.py): Lines 244, 299, 300, 991

### Silent Data Dropping and Coercion
Despite explicit warnings (e.g., in `preprocess.py`), there are instances where missing data is silently dropped or filled, which can mask upstream data pipeline issues:

- **Silent Drops:** [src/db_connector.py:L243](file:///Users/administrator/Code/ml-agent-workspace/src/db_connector.py#L243) uses `df = df.dropna(subset=['value'])`.
- **Silent Fills:** [src/dataset.py:L125-L127](file:///Users/administrator/Code/ml-agent-workspace/src/dataset.py#L125-L127) uses `.fillna(0.0)` for continuous variables (value, Humidity, Solar Radiation).

> [!CAUTION]
> These defensive code blocks directly violate the "Alert on Defensive Code" mandate for the Interrogator agent. These should be refactored to either handle the specific expected exceptions or removed to allow the script to fail fast.

## 2. Frontend Strict Typing Violations

The `NSTAR_FRONTEND_ARCHITECTURE.md` dictates that the Next.js application must run in strict mode TypeScript. However, the core data payload interface falls back to `any` types for complex nested objects.

In [frontend/src/types/payload.ts](file:///Users/administrator/Code/ml-agent-workspace/frontend/src/types/payload.ts):
- Line 26: `hyperparameters?: any;`
- Line 84: `tripwire_audit_trail?: Record<string, any>[];`
- Line 85: `alpha_node_leaderboard?: Record<string, any>[];`

> [!WARNING]
> Relying on `any` bypasses TypeScript's compile-time safety checks. These should be strongly typed to ensure the UI components don't crash when rendering the backend payload.

## 3. Hardcoded Mock Parameters in Production Scripts

While the project utilizes a centralized YAML configuration approach (as seen with `configs/train.yaml`), the [src/pipeline.py](file:///Users/administrator/Code/ml-agent-workspace/src/pipeline.py) file contains hardcoded parameters at the bottom of the script (around lines 239-243) for what appears to be a mock or standalone execution block:
```python
    batch_size = 4
    num_nodes = 50
    static_feat_dim = 10
    time_series_dim = 3
    seq_len = 12
```
This clutters the module file with execution logic that should ideally be extracted into a separate `tests/` or `smoke_test.py` script.

## 4. Legacy Phase 1 Static Payload Exports

The project is transitioning to **Phase 3 (CDN Migration)**, where the frontend is served data dynamically from Cloudflare R2 via Next.js proxy rewrites. However, legacy Phase 1 local export scripts still write `ui_payload.json` and paginated runs directly to `frontend/public/data/runs`. 

When these files exist locally, they shadow the Next.js `rewrites` configuration and cause the local frontend development environment to fall out of sync with the Vercel production app.

> [!TIP]
> **Action Item:** Clean up or deprecate legacy export scripts so they no longer write to `frontend/public/data/` locally. We should enforce that the local development environment relies entirely on the R2 proxy to ensure complete symmetry with Vercel.

## Summary

The core logic (NaN tripwires, seeding) is well adhered to, but the proliferation of `except Exception:` blocks is a significant "tech debt" liability that conflicts directly with the project's **Fail Fast** doctrine. Furthermore, completing the transition to Phase 3 requires pruning the Phase 1 local data generation logic.

Let me know if you would like me to draft an implementation plan to systematically refactor and eliminate these tech debt items!

---

# Appendix: Deep-Dive Audit Findings — 2026-07-06

A second-pass audit of `src/`, `scripts/`, `configs/`, `.github/`, and `frontend/` against the North Star documents (`AGENTS.md` / `.agents/agents.md`, `NSTAR_BACKEND_ARCHITECTURE.md`, `NSTAR_FRONTEND_ARCHITECTURE.md`, `NSTAR_MLOPS.md`, `NSTAR_PHASES.md`). All items below are **new** — they do not duplicate Sections 1–4 above. Line numbers were verified against the working tree as of this date.

> [!NOTE]
> **Stale line numbers in the original report.** The files have drifted since Sections 1–4 were written: the three `any` types in `frontend/src/types/payload.ts` are now at lines **37, 98, 99** (not 26/84/85), and the flagged broad-except line numbers in `src/export_ui_payload.py` / `src/export_paginated_payload.py` no longer match current content (e.g., the per-artifact `except Exception: continue` is now at `export_paginated_payload.py:1015`). The findings remain valid; the references need refreshing.

## 5. Silent Failure Paths on the Inference/Export Chain (CRITICAL)

These sit directly on the path that produces the UI payload, so silent failures here ship corrupted artifacts to the frontend without any crash.

- **Partial model load served silently** — `src/export_ui_payload.py:302-306`: `model.load_state_dict(state_dict, strict=False)` followed by only a `logger.warning` on `missing_keys`. Layers absent from the checkpoint remain randomly initialized, and the export proceeds to generate "predictions" from them. This is the single most dangerous Fail-Fast violation found: it must be `strict=True`, or assert `missing_keys == []`.
- **Hardcoded hyperparameter fallback for checkpoint reconstruction** — `src/export_ui_payload.py:233-242` (`DEFAULT_MODEL_CFG`: `embedding_dim: 16, hidden_dim: 32, sparsity_threshold: 0.5, ...`) used at `:963-983` when no co-located `config.yaml` exists, logging *"Using defaults. WARNING: Potential shape mismatch risk."* and continuing. Duplicated in `src/export_paginated_payload.py:281-293`. Violates both Fail-Fast and the "no hyperparameters hardcoded in Python" mandate (`NSTAR_MLOPS.md` §2).
- **ZINB tripwires deliberately defeated by clamping** — `src/export_ui_payload.py:504` (`safe_pi = max(1e-6, min(1.0 - 1e-6, float(pi_val)))`), `:565-566`, `:616`: values are clamped specifically so they don't trip the strictly-positive asserts before calibration metrics are computed. Silent coercion in place of a tripwire.
- **Artifact Twinning silently degraded to `"unknown"`** — git-hash retrieval wrapped in `except Exception: pass` at `src/export_ui_payload.py:994-995`; same pattern in `src/tune.py:40-41` (`get_git_metadata` returns `("unknown", "unknown")` on `CalledProcessError`) and `scripts/ingestion/ingest_parquet.py:78-79`. Downstream, `scripts/build_run_registry.py:50-53` defaults missing `wandb_run_id`/`git_commit_hash` to `"unknown"` rather than crashing. The 1-to-1 traceability protocol (`NSTAR_MLOPS.md` §4.3/§5) is unenforced end-to-end.
- **Semantic tags read non-existent keys** — `src/export_ui_payload.py:452,465,474`: `feats.get("commercial_density", 0)`, `feats.get("highway_proximity", 0)`, `feats.get("nearby_poi_count", 0)` — these keys do not exist in the current features schema (`src/features.py` produces `retail_count`/`transit_count`/`total_poi_count`), so every node silently gets 0 and the semantic tags are silently wrong. A schema mismatch masked by `.get()` defaults.
- **Missing coordinates become NaN, then edges silently vanish** — `src/export_ui_payload.py:900` defaults unknown sensors to `{"lat": NaN, "lon": NaN}`, and `:757-761` then warn-and-skips those edges. Same pattern at `src/export_paginated_payload.py:331`.
- **Missing analysis config → hardcoded date fallback** — `src/export_paginated_payload.py:165-170` warns and continues with argparse defaults `"2026-06-01"`/`"2026-07-01"` (`:183-184`).

## 6. Training-Loop Integrity: Warn-and-Continue Where the Doctrine Demands a Crash

- **Config seed is recorded but never used** — `src/train.py:142-143` hardcodes `torch.manual_seed(42)` / `np.random.seed(42)`, while `configs/train/train.yaml:72` defines `random_seed: 42` which is only read to *stamp the results manifest* (`src/train.py:669`) and the inductive holdout (`:420`). Changing the YAML seed changes the recorded provenance but **not** the actual training seed — reproducibility metadata can silently disagree with reality. Compounding this, `torch.manual_seed(42)` is also called inside module constructors (`src/baselines.py:28`, `src/encoder.py:28`, `src/decoders.py:14`), resetting global RNG state as a side effect of instantiation.
- **Silent early training truncation** — `src/train.py:488-495`: when host memory crosses a threshold (itself hardcoded, `memory_threshold_percent = 10.0` at `:485`), the epoch loop `break`s with only a warning and proceeds to upload the artifact as if training completed.
- **Synthetic node IDs in the holdout manifest** — `src/train.py:434-436`: if the dataset exposes no location IDs, the inductive holdout manifest is filled with `node_0, node_1, ...` and a warning. The manifest is the cross-run comparability key; it becomes silently meaningless instead of crashing.
- **Train/val scaler drift only warns** — `src/train.py:384-389`: a >20% train/val scaler-statistic mismatch — a direct data-integrity signal — triggers `logger.warning`, never a failure.
- **Loss weights default silently** — `src/harness/adapters.py:55-57,125-126` and `src/pipeline.py:218-221`: `weights.get("w_impute", 1.0)` etc. A typo'd loss key in YAML silently trains with defaults, precisely the failure mode the centralized-config rule exists to prevent. The same `.get()`-default pattern covers ~12 model/training keys in `src/train.py:212-215,341-345,420-421,479-481`.
- **Invalid config silently corrected** — `src/pipeline.py:38-39`: a configured `dim_feedforward <= 0` is silently replaced with `hidden_dim * 4` instead of rejected.
- **Silent fallbacks in non-interactive tuning** — `src/tune.py:315-324,418-427`: missing storage/config/sweep arguments silently substitute `sqlite:///optuna.db`, `configs/train/train.yaml`, `configs/sweeps/sweep.yaml`.

## 7. Additional Silent Data Coercion (beyond Section 1's list)

- `src/preprocess.py:55` — `.dropna(subset=groups)` silently excludes rows with NaN group keys from the resampled grid (and is inconsistent with `:85`, which does not drop).
- `src/db_connector.py:273-274` — `ParquetConnector.load_weather` fills Humidity/Solar Radiation NaNs with `0.0`; the comment claims "placeholder for smoke tests" but nothing restricts the connector to smoke tests. `:304-318` — 15 static-feature keys read via `.get(..., 0)`, so a missing cache key silently becomes a zero feature. `:369` — silent default parquet path despite the factory docstring stating "No fallback."
- `src/dataset.py:176-177` — further `fillna(0.0)` beyond the flagged lines 125-127: a location missing **all** static features silently becomes an all-zeros row via `reindex(...).fillna(0.0)`.
- `src/dataset.py:204` — `.clamp(min=0)` on `target_forecast` is commented as a "hard tripwire" but is the opposite: negative targets are silently corrected, not detected. Should be an assert.
- `src/features.py:152-154,203-205` — a failed POI API call (`non-200`) logs a warning and **returns 0**, making an outage indistinguishable from "zero POIs nearby" in the cached static features. `:376-378` skip-and-continues unmapped locations.
- `src/gpu_selector.py:44-45` — `nvidia-smi` failure returns an empty list, silently cascading into CPU training.
- `src/harness/adapters.py:97` — `torch.nan_to_num(..., nan=0.0)` in the recurrent baseline forward pass with no assertion that NaNs originate only from the injected mask.
- `src/metrics.py:5-52` — `calculate_metrics` takes six tensors with documented shapes but asserts none of them (contrast `calculate_kriging_metrics:97`, which does). `src/decoders.py:227ff` — `compute_zinb_nll` has no entry asserts and relies on `safe_log` clamping (`:239-240`) to hide non-positive μ/θ rather than tripping; `compute_loss` (`:271ff`) slices channel dims without asserting rank/channel count, and `:329-330` clamps the *target* silently.

## 8. Phase 3 CDN Migration — Legacy Plumbing Still Load-Bearing (specific code paths)

Section 4 flagged the legacy local-export problem in general terms. The audit found the Phase 3 toolchain itself is still **built around** the legacy directory:

- `scripts/upload_to_r2.py:39,132` — `DEFAULT_RUNS_DIR = "frontend/public/data/runs"`: the R2 uploader's default *source* is the deprecated Phase 1 path.
- `scripts/build_run_registry.py:22,71-74` — same default, and the registry is written back **into** `frontend/public/data/runs/run_registry.json`, re-creating the exact local file that shadows the Next.js rewrite.
- `scripts/generate_all_payloads.sh:2,13` — still titled "Phase 1 batch payload generation (Option B)" with `OUTPUT_DIR="${OUTPUT_DIR:-frontend/public/data/runs}"`.
- `.github/workflows/publish_payloads.yml:48,60,71` — CI hardcodes `frontend/public/data/runs` in three separate steps as the scratch dir (drift risk with the scripts' own defaults), and the cleanup step (`:74-77`) lacks `if: always()`, so any failure strands payloads inside `frontend/public/` on the persistent self-hosted runner.
- `scripts/upload_to_r2.py:24` vs `:131-137` — the docstring advertises a `--delete-extra` reconciliation flag that was never implemented; orphaned runs accumulate in the R2 bucket and drift from the registry.
- `frontend/public/data/master_index.json` — a **4.6 MB** Phase 1 monolithic payload still shipped with every deploy; no source file references it (grep confirms only comments), and it lives one path segment away from the `/data/runs/*` proxied namespace.
- `frontend/src/types/payload.ts:102-103` + duplicated `mergePayload` functions — the Phase 1 monolithic `MasterIndex`/`UIPayload` schema is still the in-memory model; `frontend/src/app/analytics/page.tsx:80-122` carries a copy-paste of `MapVisualizer`'s merge logic with the comment "kept in sync manually" (they have already drifted: `MapVisualizer.tsx:60` has an `as EdgeEntry[]` cast the analytics copy lacks).
- **Committed debug debris from the R2 work** — `r2_temp.json`, `registry_temp.json`, `dummy.txt` (contents: "hello"), and `run.log` (42 KB training log) are all **tracked in git** at the repo root (added in commits `89239f9`/`21834c2`), referenced by no code. `frontend/frontend/package-lock.json` is a stray nested lockfile from an `npm i` in the wrong cwd.
- `scripts/verify_phase2_phase3.sh` — the Phase 2/3 "verification" script (a) references `scripts/ingest_parquet.py` at `:86,140,243` although the scripts moved to `scripts/ingestion/` (the check can never pass), (b) has a `check_fail()` (`:36-38`) that only echoes — the script **always exits 0** even when every check fails, and (c) greps `curl -I` output for `"200 OK"` (`:216`), which HTTP/2 responses (`HTTP/2 200`) never contain. The ingestion scripts' own usage docstrings carry the same stale paths (`scripts/ingestion/ingest_parquet.py:7`, `ingest_sensor_metadata.py:8`, `ingest_static_features.py:11`, `ingest_weather.py:11`).

## 9. Phase 2 Ingestion — Silent Handling and Fabricated Data

- `scripts/ingestion/ingest_weather.py:65-67` — placeholder NaN weather is `fillna(0.0)`-ed *to satisfy the NOT NULL constraint* and ingested as real readings: zero humidity/solar values are fabricated into the training database.
- `scripts/ingestion/ingest_parquet.py:122,128,187` — `ON CONFLICT ... DO NOTHING` while the "Total inserted" log reports **attempted** rows (the `inserted` counter at `:114` never uses `cursor.rowcount`), silently masking duplicates/drops. Same pattern in `ingest_sensor_metadata.py:77,83`, `ingest_static_features.py:124,130`, `ingest_weather.py:104,113`. Also `ingest_parquet.py:95` drops NaN rows with no count or drop-rate assert.
- `scripts/ingestion/ingest_static_features.py:83-97` — every feature read via `.get('...', 0)`: a sensor missing from a cache gets all-zero features in the DB instead of crashing.
- `scripts/ingestion/ingest_parquet.py:151` — `seq_len: 12, mask_prob: 0.15` hardcoded into the `dataset_versions` provenance JSON, duplicating (and free to drift from) `configs/train/train.yaml`.

## 10. Frontend — Core Architecture Rule Violations

`tsconfig.json:7` confirms `"strict": true` and no `any` exists outside the three flagged lines — but the *rendering architecture* rules of `NSTAR_FRONTEND_ARCHITECTURE.md` §3 are violated:

- **`DataFilterExtension` is not used anywhere in the repo** (zero grep hits). Instead, `frontend/src/components/MapVisualizer.tsx:265-277` runs `payload.edges.map(...).filter(...).sort(...).slice(...)` over the **full edge array in JavaScript** inside `buildLayers`, whose effect (`:543-595`) depends on `currentTimestep` — i.e., a full CPU re-filter on every slider tick and every 300 ms during playback. This is the exact pattern the North Star bans ("React components do not filter the massive data payload... use DataFilterExtension on the GPU").
- **Zustand's purpose is defeated on the timestep path** — `MapVisualizer.tsx:384` subscribes the component to `currentTimestep`, so every tick re-renders React and rebuilds all five Deck.gl layers (the hover path correctly bypasses React via refs at `:368,549`; the clock path does not). `NodeInspector.tsx:330-346` similarly re-filters/re-sorts edge lists per tick.
- **Run-switch deadlock (race)** — `MapVisualizer.tsx:498-538`: Stage 3's guard reads `spatialRef` (a ref), which does not re-trigger the effect when the new run's `spatial.json` arrives; if the new run's first date equals the current `activeDate`, Stage 3 never fires and the map shows the old run forever. `analytics/page.tsx:164-165` explicitly fixed this same bug ("Uses setSpatialIndex (state, not a ref) so Stage 3 re-triggers") — the fix was never ported to `MapVisualizer`.
- **Graceful-degradation gaps** — `MapVisualizer.tsx:60`: `edges: page.edge_attention as EdgeEntry[]` with no `?? []` fallback (unlike `node_timeseries` at `:39`); a payload from a graph-ablated run makes `payload.edges.map(...)` at `:265` crash — precisely the "crash on undefined when an ablation removes a component" scenario the North Star forbids. `analytics/page.tsx:728`: `valueFormatter: (value: number) => value.toFixed(3)` over the untyped `alpha_node_leaderboard` crashes on a missing/null field; `:768` fabricates `|| 0` MSE values when `global_healing_quality` is absent instead of hiding the series.
- **Silent async failure handling on the R2 proxy path** — `MapVisualizer.tsx:448-495`: registry/spatial fetch failures produce only `console.warn` — the user gets a permanently blank map with no error state. `NodeInspector.tsx:422,428`: `r.ok ? r.json() : null` plus an empty `.catch` coerce HTTP errors and outages into "no data". `analytics/page.tsx:150,157,170,178,192,198`: `Promise.reject()` with no reason and `.catch(() => setLoading(false))` swallow all failure causes. None of the three fetch chains uses an `AbortController`, so rapid run/date switching downloads and parses every superseded multi-MB page.
- **Hardcoded R2 bucket fallback committed to source** — `frontend/next.config.ts:5-7`: `process.env.R2_PUBLIC_URL || "https://pub-b102fa91750348f78a6385248654688a.r2.dev"`. A missing env var silently proxies to a specific bucket instead of failing fast, and no `.env.example` documents the variable. Cache TTLs are inline magic constants (`:25,35`) that also drift from `scripts/upload_to_r2.py:41` (`max-age=3600`).
- **Debug instrumentation shipping to production** — `TimeSlider.tsx:26-28` logs every timestep change (~3.3×/sec during playback); `MapVisualizer.tsx:330-336` has a `console.log` **inside the `getWidth` accessor** of the ego-graph ArcLayer (fires per edge per rebuild). `app/api/log/route.ts` + `GlobalErrorReporter.tsx` are dev instrumentation with synchronous `fs.appendFileSync` in an async handler (`route.ts:13,23`), no body-size limit, and an unauthenticated DELETE.
- **Dependency hazard** — `package.json:35`: `@types/maplibre-gl ^1.13.2` against `maplibre-gl ^5.23.0`; maplibre ships its own types since v2, and the deprecated v1 stub can shadow the real types under strict mode. Dead Phase 1 state: `useTimeStore.ts:13,47,78` — `startDate`/`setStartDate` are written but read by no component.

## 11. CI / Infrastructure

- **Script injection on the self-hosted GPU runner** — `.github/workflows/publish_payloads.yml:50-53`: `${{ github.event.inputs.artifact_ref }}` is interpolated directly into a `run:` shell block; a crafted input executes arbitrary commands on the GPU server. Should be passed via `env:`.
- **Zero Host Download rule broken by CI itself** — `publish_payloads.yml:38-41` runs bare `pip install -r requirements.txt` into the runner's system Python on every run, outside any of the project's Docker images (`NSTAR_MLOPS.md` §7 prohibits host installs).
- **Full secret set granted to the runner container** — `docker-compose.yml:101-117`: the `github-runner` service gets `env_file: .env` (R2 keys, W&B key, `DATABASE_URL`, `GITHUB_TOKEN`) although `start-runner.sh` needs only two variables; every Actions job on that runner inherits everything. Relatedly, `scripts/start-runner.sh:33-40,65-70` falls back to "assuming GITHUB_TOKEN is a direct registration token" and continues when the token API call fails.
- **Default DB credentials, published port** — `docker-compose.yml:84-90`: `riotuser`/`riotpassword` defaults with `5432:5432` exposed to the host; the same credentials are re-hardcoded in `scripts/verify_phase2_phase3.sh:120,127,134`. Also `:13,16` — the gcloud volume mount is duplicated (copy-paste bug).
- **Silent data-dir fallback in CI** — `publish_payloads.yml:23-33`: if `/data` is absent, the workflow prints a note and exports against whatever stale `./data` exists.

## 12. Config Hygiene

- **Three-way geo-constant duplication** — the OSMnx graph parameters (`center (54.978, -1.610)`, `dist 5000`) are hardcoded in `src/features.py:243`, `scripts/cache_edge_features.py:44-47`, and `scripts/cache_simulated_edge_features.py:40-41`, with a comment admitting they must be manually mirrored. Guaranteed drift; belongs in `configs/`.
- **W&B project names as scattered `.get()` defaults** — `"resilient-iot-v2"` / `"resilient-iot-sweeps-v2"` / `"resilient-iot-model-v2"` hardcoded in `src/train.py:152-154`, `src/tune.py:345,439`, `src/export_ui_payload.py:111-135`, `src/export_paginated_payload.py:94-99`, and repeated across all five `configs/sweeps/*.yaml`.
- **Modelling heuristics hardcoded in Python** — `src/preprocess.py:106-107` (structural-zero night window `hour >= 0 & hour < 5`), `src/train.py:224,286` (0.8 train split), `src/export_ui_payload.py:79-99` (Newcastle bounding box, 5 hardcoded simulated-node coordinates, `NUM_TIMESTEPS_TARGET = 96`, `TOP_K_EGO_EDGES = 10`), `src/export_ui_payload.py:832` / `src/export_paginated_payload.py:869` (`mask_prob=0.15` duplicating `train.yaml`).
- **Type-hint/logging gaps (representative)** — all four `scripts/ingestion/*.py` scripts and `scripts/cache_edge_features.py` have largely unannotated signatures; `src/check_gaps.py` is a print-driven EDA script with a silent `.dropna()` (`:15`) living in `src/`; `Optional` misuse (`seed: int = None`) at `src/dataset.py:225`, `src/train.py:124-133`, `scripts/cache_edge_features.py:51`.

## Priority Ranking of New Findings

1. **`export_ui_payload.py:302` `strict=False` partial model load** — can ship predictions from randomly initialized layers.
2. **Frontend `DataFilterExtension` absence + per-tick CPU filtering + timestep-driven React re-renders** (`MapVisualizer.tsx:265-277,384`) — violates the core 60fps rendering architecture.
3. **Seed provenance lie** (`train.py:142-143` vs `train.yaml:72`) — recorded seed and actual seed can disagree.
4. **CI script injection on the self-hosted GPU runner** (`publish_payloads.yml:50-53`).
5. **Phase 3 toolchain still routed through `frontend/public/data/runs`** (`upload_to_r2.py:39`, `build_run_registry.py:22`, `publish_payloads.yml`, `generate_all_payloads.sh`).
6. **Fabricated zero weather readings ingested as real data** (`ingest_weather.py:65-67`).
7. **MapVisualizer run-switch deadlock** (`MapVisualizer.tsx:498-538`) — fix exists in `analytics/page.tsx` but was never ported.

---

# GPU Performance Optimizations — 2026-07-06 audit

Deep-dive audit of the training pipeline (`src/train.py`, `src/dataset.py`, `src/pipeline.py`, plus the in-loop dependencies `src/harness/adapters.py` and `src/graph_learner.py`) targeting sustained near-100% GPU utilization. Line numbers verified against the working tree as of this date. Findings are ordered by expected impact; each includes a concrete fix that preserves the Fail-Fast / Offensive Programming doctrine (no tripwire is removed — several are relocated to points where a CPU↔GPU synchronization already occurs, so they become free).

The current loop has a classic serialized profile: **single-process data loading → synchronous H2D copy → FP32 forward/backward → 4 forced GPU syncs per batch for loss bookkeeping.** The GPU idles at both ends of every step.

## 13. DataLoader Configuration — Single-Process, Unpinned, No Prefetch (HIGH IMPACT)

**Where:** `src/train.py:241-251` and `src/train.py:310-320` (real-data loaders), `src/train.py:121` (mock loader).

All `DataLoader` constructions specify only `batch_size`, `collate_fn`, and `shuffle`:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=train_cfg["batch_size"],
    collate_fn=custom_collate_real,
    shuffle=True,
)
```

Consequences:

- **`num_workers` defaults to 0** — every `__getitem__` (window slice, float64→float32 conversion, mask draw, two `.clone()`s — see §14) and every `custom_collate_real` `torch.stack` runs **on the main process, serially, between GPU steps**. The GPU is guaranteed idle for the full duration of batch preparation. This is the single largest utilization killer in the pipeline.
- **`pin_memory` defaults to False** — H2D copies go through pageable memory (extra staging copy, and `non_blocking=True` would be a silent no-op; see §15).
- **`prefetch_factor` / `persistent_workers` are unset** — moot at `num_workers=0`, but must be set together with the worker fix: without `persistent_workers=True`, the worker pool is torn down and re-forked at **every epoch boundary**, and with short epochs (early stopping at patience 5, small datasets) fork/teardown becomes a measurable fraction of wall time.
- The collate functions (`src/dataset.py:277-287`, `src/train.py:112-119`) are list-comprehension `torch.stack` calls over per-sample dicts. This is acceptable *inside workers*, but today it executes on the main process. Note also that `custom_collate_real` stacks `node_mask` as `torch.bool` and discards nothing — three full-window tensors per sample cross the worker→main IPC boundary (see §14 for the structural fix that shrinks this 3× payload to 1×).

**Proposed change** (`src/train.py`, both real-data branches; values sourced from config to honor the centralized-YAML mandate — add them to `configs/train/train.yaml` under `training:`):

```python
use_cuda = torch.cuda.is_available()
num_workers = train_cfg["num_workers"]          # no .get() default — fail fast on missing key
assert num_workers >= 0, f"CRITICAL: num_workers must be >= 0, got {num_workers}"

train_loader = DataLoader(
    train_dataset,
    batch_size=train_cfg["batch_size"],
    collate_fn=custom_collate_real,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=num_workers > 0,
    prefetch_factor=4 if num_workers > 0 else None,
    drop_last=True,   # stable batch shapes; also required for CUDA-graph/compile work later
)
```

A starting point of `num_workers: 4` is appropriate (the dataset is fully in-RAM after `__init__`, so workers are CPU-bound on masking/stacking, not I/O-bound; 2–4 is typically enough once §14 lands). `drop_last=True` on the train loader only — the val loader must keep every sample for metric integrity.

> [!CAUTION]
> **Worker-seed tripwire required.** With `num_workers > 0`, each worker inherits forked RNG state; PyTorch reseeds `torch` per worker but **not NumPy**. `RealIoTDataset.__getitem__` currently uses `torch.rand_like` (safe), but `make_spatial_holdout_mask` uses `np.random.RandomState` — if any NumPy randomness ever moves into `__getitem__`, all workers would draw identical masks silently. Add a `worker_init_fn` that seeds NumPy from `torch.initial_seed()` and asserts distinct worker seeds, keeping the failure loud.

## 14. `__getitem__` CPU Overhead — Per-Sample dtype Conversion, Triple Copies, and Constant Payloads (HIGH IMPACT)

**Where:** `src/dataset.py:192-220`, with root causes in `__init__` at `src/dataset.py:143-167,183`.

Per sample, `__getitem__` currently performs:

1. **float64→float32 conversion of the entire window, every call** — `src/dataset.py:198,204`. `self.dynamic_data` is built from `StandardScaler` output (`:143-167`), which is **float64 NumPy**; `self.unscaled_traffic` (`:132`) likewise. So `torch.tensor(dyn_slice, dtype=torch.float32)` re-converts and copies `[N, seq_len, 3]` doubles on every access, and the arrays occupy 2× the RAM they need for their entire lifetime.
2. **Layout transpose per call** — `np.transpose(dyn_slice, (1, 0, 2))` at `:196` produces a non-contiguous view that the `torch.tensor(...)` copy then has to gather.
3. **Two full-window `.clone()`s plus a same-shape bool mask** — `:207-210`. The clean window is cloned into `target_impute`, cloned *again* into `dyn_masked`, then NaN-scattered. Every sample therefore materializes **three window-sized tensors** (2 float + 1 bool ≈ 2.25× the raw window) that all cross collate, worker IPC, and the PCIe bus.
4. **Constant payloads shipped per sample** — `:213,219`. `self.static_features` (full `[N, S]` tensor) and `self.locations` (a Python list of strings) are returned from every `__getitem__`. The collate at `:280,286` keeps only `batch_list[0]`'s copy, but with workers enabled each sample still pickles both through the IPC queue just to be discarded. The string list is particularly wasteful (per-sample pickling of N Python strings).

**Proposed changes:**

- **Pre-convert once in `__init__`** — after `:167`, store `self.dynamic_data = torch.from_numpy(self.dynamic_data.astype(np.float32))` in `[T, N, D]` layout, and `self.unscaled_traffic = torch.from_numpy(self.unscaled_traffic.astype(np.float32))`. Keep the existing offensive assertions on scaler stats; add one at conversion: `assert not torch.isnan(self.dynamic_data).any(), "CRITICAL: NaN in scaled dynamic data"` — this moves NaN detection from N-per-epoch to once-per-run *and makes it stronger* (it covers every element, not just sampled windows). `__getitem__` then becomes `self.dynamic_data[idx : idx + self.seq_len].permute(1, 0, 2)` — a zero-copy view until collate stacks it.
- **Move mask injection to the GPU (recommended)** — the masking at `:207-210` is data augmentation, not data loading. Have `__getitem__` return only the *clean* window; generate the mask and NaN-blank **on-device** in the train loop right after the H2D copy:

  ```python
  # train.py inner loop, after batch is on device
  clean = batch["dynamic_features"]                                # [B, N, T, D], clean
  node_mask = torch.rand(clean.shape, device=device) < mask_prob   # GPU RNG, no H2D
  batch["target_impute"] = clean
  batch["dynamic_features"] = clean.masked_fill(node_mask, float("nan"))
  batch["node_mask"] = node_mask
  ```

  This cuts the CPU→GPU transfer volume for the dominant tensors by ~3× (one float window instead of two floats + one bool), eliminates both `.clone()`s from the hot CPU path, and keeps the doctrine intact — the NaN tripwire semantics downstream (`pipeline.py:145` `nan_to_num`, decoder asserts) are unchanged because the masked tensor is bit-identical to today's. The per-sample RNG stream changes (GPU generator instead of per-worker CPU), which must be noted in the run manifest for reproducibility, and `mask_prob` moves from a dataset arg to a train-loop arg.
- **Stop shipping constants through the loader** — remove `static_features` and `location_ids` from the per-sample dict and collate (`:213,219,280,286`); read them once from the dataset in `run_training` and insert into the batch dict on the training side (they are already treated as batch-invariant by both collates and by `pipeline.py:136,141`). Note `random_split` wraps the dataset in `Subset`, so access them via the propagated attributes already set at `train.py:300-302`.

## 15. Host-to-Device Transfers — Synchronous Copies, and a Per-Batch NumPy→GPU Round Trip (MEDIUM-HIGH IMPACT)

**Where:** `src/train.py:505` (train), `src/train.py:560` (val), `src/train.py:513-518` + `src/dataset.py:248-250` (mask re-roll), `src/dataset.py:272-274` (holdout blanking).

- **`.to(device)` without `non_blocking=True`** — `train.py:505,560`:

  ```python
  batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
  ```

  Each copy blocks the host until the transfer completes. Fix (effective only in combination with `pin_memory=True` from §13 — without pinned staging, `non_blocking=True` silently degrades to a blocking copy, which is why §13 and §15 must land together):

  ```python
  batch = {
      k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
      for k, v in batch.items()
  }
  ```

- **Per-batch CPU RNG + H2D transfer for the kriging re-roll** — `train.py:513-518` calls `make_spatial_holdout_mask(..., seed=None)` every training batch when kriging is active; that function (`dataset.py:248-250`) draws from `np.random.RandomState()` on the CPU and then does `torch.from_numpy(mask).bool().to(device)` — an extra synchronous H2D copy *and* a fresh `RandomState` construction per batch. Fix: draw directly on-device — `torch.rand(num_nodes, device=device) < holdout_prob` — for the `seed=None` training path, keeping the seeded NumPy path for the deterministic validation mask (built once, outside the loop, at `train.py:453`). Keep both existing asserts; they check Python floats/ints and cost nothing.
- **No inadvertent GPU→CPU→GPU round trips of batch data were found** in the steady-state loop — good. `apply_spatial_holdout` (`dataset.py:272-274`) runs on-device but does `masked = dynamic_features.clone()` followed by advanced-index assignment; a broadcast `masked_fill` avoids the scatter kernel: `dynamic_features.masked_fill(holdout_mask.view(1, -1, 1, 1), 0.0)` (`masked_fill` is out-of-place, so the "input left untouched" contract and all three entry asserts are preserved verbatim).

## 16. Implicit CPU-GPU Synchronization — Four Forced Syncs per Training Batch (HIGH IMPACT)

**Where:** `src/train.py:542-545` (train loop), `src/train.py:573,578-580` (val loop), `src/train.py:600`, `src/pipeline.py:225`, `src/graph_learner.py:197,201-207,248`.

- **Loss bookkeeping calls `.item()` four times per training batch** — `train.py:542-545`:

  ```python
  epoch_loss_total += losses.total.item()
  epoch_loss_impute += losses.impute.item()
  epoch_loss_forecast += losses.forecast.item()
  epoch_loss_kriging += losses.kriging.item()
  ```

  Each `.item()` stalls the host until the entire forward+backward for that batch has drained, defeating the CUDA stream's ability to overlap the next batch's data prep with the current batch's compute. The same pattern repeats in the val loop (`:573,578-580` — up to three `.item()`s per val batch). Fix: accumulate **as 0-dim GPU tensors** and synchronize once per epoch:

  ```python
  epoch_loss_total += losses.total.detach()      # stays on device, no sync
  ...
  avg_train_loss = (epoch_loss_total / len(train_loader)).item()   # ONE sync per epoch
  ```

  (Initialize the accumulators as `torch.zeros((), device=device)`.) All downstream logging/W&B/early-stopping code is untouched — it already consumes Python floats at epoch granularity.
- **The per-batch NaN tripwire in the pipeline forward** — `pipeline.py:225` `assert not torch.isnan(loss).any()` forces a full sync inside every forward. Under the doctrine this tripwire must stay — so **relocate it, don't delete it**: with the accumulator fix above, assert on the *epoch* aggregate at the point of the already-paid sync (`assert not math.isnan(avg_train_loss), "CRITICAL FAILURE: train loss is NaN"`). A NaN in any batch propagates into the sum, so detection strength is preserved; latency of detection moves from batch-level to epoch-level, which is an acceptable trade the report should record explicitly. If batch-level detection latency is deemed mandatory, the middle ground is `torch.autograd.set_detect_anomaly` in a debug config, not a per-batch sync in production.
- **Graph-learner per-forward syncs** — `graph_learner.py:197` (`assert not torch.isnan(edge_attr).any()`) and `:201-207` (road-fraction sum check via `torch.all(...)`) each force a sync **every forward pass**, yet they validate data that is *static per node-ordering*: `phys` is a gather from the memoized `_phys_dense` tensor built once at `:183`. Move both assertions into `_build_physical_dense` / the memoization branch so they run once per (node-ordering, device) instead of once per batch — the check is then strictly stronger (validates the full dense tensor, not just the edges that survived sparsification) and costs nothing in steady state. The `edge_attr.shape` assert at `:195` is shape metadata, sync-free, and can stay.
- **`dense_to_sparse` at `graph_learner.py:248`** is an unavoidable per-batch sync today (`nonzero()` has a data-dependent output shape, so the host must wait for the count). It is legitimate — the learned adjacency changes every optimizer step — but it caps how far asynchrony can go and blocks future `torch.compile`/CUDA-graph capture of the full model. Worth recording as a known ceiling; a fixed-top-k sparsification would remove it, but that is a modelling change requiring separate sign-off.
- **Minor:** `train.py:600` wraps a Python float in a tensor to check NaN (`torch.isnan(torch.tensor(avg_val_loss))`) — use `math.isnan(avg_val_loss)`; `train.py:658` `int(val_holdout_mask.sum().item())` syncs once per epoch inside the W&B block — compute it once before the epoch loop (the mask is fixed).

## 17. Mixed Precision — Entirely Absent (HIGH IMPACT on tensor-core GPUs)

**Where:** `src/train.py:504-545` (train step), `:558-586` (val step). No `torch.autocast`, no `GradScaler`, no TF32 configuration anywhere in `src/`.

The full forward/backward — GATv2 attention, transformer-style encoder layers with `dim_feedforward` up to 256+, and all decoder matmuls — runs in FP32, leaving tensor cores idle.

**Proposed change, in two stages:**

1. **TF32 (zero-risk, one line, do immediately):** `torch.set_float32_matmul_precision("high")` at startup in `run_training`. On Ampere+ this routes FP32 matmuls through tensor cores with no numerics change relevant to this model, no API changes, no scaler.
2. **bf16 autocast (recommended over fp16):**

   ```python
   amp_enabled = train_cfg["amp"] and device.type == "cuda"   # explicit config key, no silent default
   if amp_enabled:
       assert torch.cuda.is_bf16_supported(), "CRITICAL: amp=true but device lacks bf16 support"

   with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
       losses, _ = adapter.step(batch, loss_weights, loss_node_mask=loss_node_mask)
   losses.total.backward()          # outside autocast, as required
   ```

   bf16 has FP32's exponent range, so **no `GradScaler` is needed** and there is no silent under/overflow failure mode to defend against — the better fit for this codebase's doctrine than fp16+scaler (which introduces `scaler.step()`'s *silent skip-on-inf* behavior, a defensive pattern the project forbids; if fp16 is ever required for pre-Ampere hardware, the skip counter must be asserted on, not ignored).

   **Mathematical-safety carve-out:** the ZINB NLL (`src/decoders.py`, `compute_zinb_nll`) composes `log`/`lgamma`/division near clamp boundaries, and the existing tripwires there are calibrated for FP32. autocast already keeps many reductions in FP32, but the loss should be *explicitly* excluded rather than implicitly trusted: wrap `compute_loss` in `with torch.autocast(device_type="cuda", enabled=False):` inside `pipeline.py:207` and upcast its float inputs (`.float()`) — encoder/decoder matmuls get tensor cores; the numerically delicate NLL stays FP32. Apply the same autocast context to the val loop. Existing NaN tripwires then serve as the burn-test for the mixed-precision rollout: any instability crashes loudly, exactly as intended.

## 18. Secondary Observations (LOW impact individually, cheap to fix)

- **`torch.save` on the main thread at every improvement** — `train.py:633` serializes the checkpoint synchronously between epochs. Acceptable at current model sizes; revisit only if checkpoints grow.
- **Validation runs under `torch.no_grad()` but not `model.eval()`-specific fast paths beyond dropout** — already correct at `train.py:553,558`; no change. (Noted so the audit trail shows it was checked.)
- **`RIoTAdapter.step` dict copies** (`harness/adapters.py:54`) and the memo-key `tuple(location_ids)` comparison (`graph_learner.py:179-182`) are per-batch Python overhead of O(N) — negligible next to §13-§17, not worth churn.
- **Mock loader** (`train.py:121`) generates `torch.randn` per `__getitem__` on the main process — fine for smoke tests; do not propagate its configuration to real runs.

## Recommended Sequencing

| Order | Item | Sections | Risk |
|---|---|---|---|
| 1 | Tensor accumulators + relocated NaN tripwires (one sync/epoch) | §16 | Low — pure loop refactor |
| 2 | `num_workers` / `pin_memory` / `persistent_workers` / `prefetch_factor` + worker-seed tripwire | §13 | Low |
| 3 | `non_blocking=True` transfers + on-device mask re-roll | §15 | Low |
| 4 | TF32 matmul precision | §17.1 | Negligible |
| 5 | Dataset float32 pre-conversion + constant-payload removal | §14 | Low |
| 6 | GPU-side mask injection (3× H2D reduction) | §14 | Medium — RNG stream changes; document in manifest |
| 7 | bf16 autocast with FP32 loss carve-out | §17.2 | Medium — validate against burn test + one full A/B training run |

Items 1–4 are independent and can ship in a single PR; each preserves every existing assertion (some strengthened, some relocated to already-synchronized points). Measure before/after each stage with `torch.profiler` (one schedule-based trace per stage) rather than `nvidia-smi` sampling, so DataLoader wait time vs. kernel time is attributed correctly.
