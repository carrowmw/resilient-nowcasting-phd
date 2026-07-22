# Tech Debt Remediation Plan — Phased Execution Guide

**Source:** `docs/design_reports/tech_debt_report.md` (Sections 1–12: correctness/doctrine audit; Sections 13–18: GPU performance audit).
**Intended executor:** Claude Sonnet (or similar), one phase per session/PR.
**Date:** 2026-07-06.

---

## How to use this document (read first)

Rules for the executing model — these apply to **every** phase:

1. **One phase = one branch = one PR.** Do not mix items across phases. Complete the phase's verification gate before opening the PR.
2. **Line numbers drift.** Every reference below was verified on 2026-07-06. Locate code by the quoted pattern/context, not by line number alone. If a referenced pattern no longer exists, note it in the PR description as "already fixed" — do not improvise a substitute change.
3. **Never delete a tripwire.** This project follows an Offensive Programming / Fail-Fast doctrine. Assertions may be *relocated* or *strengthened* exactly as specified below, never removed. If a fix seems to require weakening an assert, stop and flag it in the PR description instead.
4. **No new `.get()` defaults, no new `try/except`.** When adding a config key, read it with `cfg["key"]` (KeyError is the desired failure) and add the key to the YAML file in the same commit.
5. **Do not refactor beyond the listed items.** Adjacent ugliness is out of scope unless a listed item cannot be completed without touching it.
6. **Scope of "tested":** every phase ends with a Verification Gate — a concrete list of commands that must pass. Paste their output into the PR description.
7. **Baseline before you start (Phases 1 and 5):** run `python src/train.py --use_mock_data` once on the unmodified branch and save the final epoch losses. Performance phases must show loss parity (identical values where the change is mathematically neutral; documented deltas where RNG streams change).
8. **Update the ledger**: after each PR please update the EXPERIMENT.md ledger with the changes made.

Phase dependency graph:

```
Phase 1 (perf, low-risk)  ──►  Phase 5 (perf, medium-risk)
Phase 2 (backend fail-fast) ──► Phase 3 (data/CDN)   [weak dep: shared files]
Phase 4 (frontend)          — independent
Phase 5 also contains CI/infra items — independent of 1 except where noted
```

Recommended order: 1 → 2 → 3 → 4 → 5.

---

## Phase 1 — Training-Loop Performance, Low-Risk (report §13, §14-partial, §15, §16, §17.1, §18)

**Goal:** remove every per-batch CPU–GPU stall and serialization point that does not change model numerics or RNG streams.
**Files:** `src/train.py`, `src/dataset.py`, `src/pipeline.py`, `src/graph_learner.py`, `configs/train/train.yaml`.

### 1.1 One sync per epoch instead of four per batch (§16)
- `src/train.py` train loop (`epoch_loss_total += losses.total.item()` and the three lines after it): change all four accumulators to 0-dim GPU tensors initialized as `torch.zeros((), device=device)`, accumulate with `losses.total.detach()` (and `.impute/.forecast/.kriging`), and call `.item()` once per accumulator **after** the batch loop when computing `avg_train_*`.
- Same pattern in the val loop (`val_loss_total`, `val_loss_kriging_total`; note the inductive-mode arithmetic `losses.total.item() - w_kriging * losses.kriging.item()` must become tensor arithmetic, synced once at the end).
- Relocate the per-forward NaN tripwire: delete `assert not torch.isnan(loss).any()` in `src/pipeline.py` (pattern: `CRITICAL FAILURE: Pipeline total_loss is NaN`) and add, in `src/train.py` immediately after `avg_train_loss` is computed: `assert not math.isnan(avg_train_loss), "CRITICAL FAILURE: train loss is NaN"` (add `import math`). This is a relocation, not a removal — record it in the PR description.
- Replace `torch.isnan(torch.tensor(avg_val_loss))` with `math.isnan(avg_val_loss)`.
- Hoist `int(val_holdout_mask.sum().item())` (inside the W&B block) to a variable computed once before the epoch loop (the mask is fixed for the whole run).

### 1.2 DataLoader configuration (§13)
- Add to `configs/train/train.yaml` under `training:`: `num_workers: 4`.
- In `src/train.py`, for the **two real-data** DataLoader pairs (the `full_dataset is not None` branch and the `RealIoTDataset` branch — NOT the mock loader): add `num_workers=train_cfg["num_workers"]`, `pin_memory=torch.cuda.is_available()`, `persistent_workers=num_workers > 0`, `prefetch_factor=4 if num_workers > 0 else None`. Add `drop_last=True` to the **train** loaders only.
- Read `num_workers` once with `train_cfg["num_workers"]` (no `.get()`), and assert `num_workers >= 0`.
- Add a module-level `worker_init_fn` in `src/train.py` that seeds NumPy from `torch.initial_seed() % 2**32` and pass it to the real-data loaders. Include the comment: `# PyTorch reseeds torch per worker but NOT numpy; identical numpy streams across workers would silently correlate augmentation.`

### 1.3 Non-blocking transfers and on-device mask re-roll (§15)
- Both batch-to-device dict comprehensions in `src/train.py`: `v.to(device)` → `v.to(device, non_blocking=True)`. (Only valid because 1.2 enables `pin_memory` — keep in the same PR.)
- In the training loop's kriging re-roll (call to `make_spatial_holdout_mask(..., seed=None)`), replace with an on-device draw: `reroll_mask = torch.rand(num_nodes, device=device) < spatial_holdout_prob`. Do **not** touch the seeded validation-mask call (`seed=42`) — it must stay deterministic and is built once, outside the loop.
- In `src/dataset.py::apply_spatial_holdout`, replace the `clone()` + advanced-index body with `return dynamic_features.masked_fill(holdout_mask.view(1, -1, 1, 1), 0.0)`. Keep all three entry asserts verbatim (`masked_fill` is out-of-place, so the "input left untouched" contract holds).

### 1.4 TF32 (§17.1)
- In `run_training`, immediately after the device is resolved: `torch.set_float32_matmul_precision("high")`.

### 1.5 Dataset float32 pre-conversion + constant payload removal (§14, low-risk subset)
- `src/dataset.py::__init__`, after `self.dynamic_data = np.stack([...])`: convert once — `self.dynamic_data = torch.from_numpy(self.dynamic_data.astype(np.float32))` and `self.unscaled_traffic = torch.from_numpy(self.unscaled_traffic.astype(np.float32))`. Add: `assert not torch.isnan(self.dynamic_data).any(), "CRITICAL: NaN in scaled dynamic data"` (this strengthens the existing per-sample tripwire to full coverage).
- `__getitem__`: replace the numpy slice/transpose/`torch.tensor` with `dyn_tensor = self.dynamic_data[idx : idx + self.seq_len].permute(1, 0, 2)`; replace the target conversion with `self.unscaled_traffic[idx + self.seq_len].clamp(min=0)`. Note: because `torch.stack` in collate copies, the permuted view is safe to return.
- Remove `'static_features'` and `'location_ids'` from the `__getitem__` return dict and from `custom_collate_real`. In `src/train.py`, after each batch is moved to device, inject them from dataset-level attributes: `batch["static_features"] = static_features_dev` (moved to device once, before the epoch loop) and `batch["location_ids"] = location_ids` (plain list, once-resolved). Remember `random_split` produces `Subset` objects — read from `full_dataset` / the propagated attributes, mirroring how `num_nodes` is already handled. Update the mock collate/dataset only if the mock path breaks (it uses its own collate and does not pass `location_ids`).

### 1.6 Graph-learner assert relocation (§16)
- `src/graph_learner.py::_build_edge_attr`: move the NaN assert (`CRITICAL: NaN in edge_attr`) and the road-fraction-sum assert block into `_build_physical_dense` (run them on the full `[N, N, 9]` tensor once, at build time — check `dense[..., 3:9].sum(-1)` against the same ~1.0/~0.0 tolerance). Keep the sync-free shape assert in `_build_edge_attr`. Add a comment noting the checks are strictly stronger at build time (full dense tensor vs. surviving edges).

### Verification Gate — Phase 1
```bash
python src/pipeline.py                      # burn test must pass
python src/train.py --use_mock_data         # must complete; no NaN tripwire fires
python -m pytest tests/ -x                  # if a tests/ dir exists; otherwise skip and say so
```
- Confirm mock-run losses are **identical** to the saved baseline for 1.1/1.3/1.6 alone; 1.2/1.5 may change batch order composition only if `drop_last` drops a partial batch — if losses differ, explain exactly why in the PR (partial-batch drop is the only acceptable reason).
- `grep -n "\.item()" src/train.py` — the only remaining hits must be per-epoch (outside the batch loops).
- `grep -rn "num_workers" src/train.py configs/train/train.yaml` — key present in both.

---

## Phase 2 — Backend Fail-Fast Integrity (report §1, §5, §6, §7)

**Goal:** eliminate silent failure paths on the training and export chains. This is the doctrine-critical phase.
**Files:** `src/export_ui_payload.py`, `src/export_paginated_payload.py`, `src/train.py`, `src/tune.py`, `src/dataset.py`, `src/db_connector.py`, `src/preprocess.py`, `src/features.py`, `src/gpu_selector.py`, `src/metrics.py`, `src/decoders.py`, `src/pipeline.py`, `src/harness/adapters.py`, `configs/train/train.yaml`.

### 2.1 Export chain (§5) — highest priority in the whole plan
- `export_ui_payload.py`: `load_state_dict(state_dict, strict=False)` → `strict=True`. Delete the `missing_keys` warning path.
- Delete `DEFAULT_MODEL_CFG` fallbacks in **both** export scripts: absence of a co-located `config.yaml` must `raise FileNotFoundError` with a message naming the expected path. Same for the hardcoded date fallback in `export_paginated_payload.py`.
- Replace the `safe_pi` clamps (three sites in `export_ui_payload.py`) with asserts that `pi` is already within `(0, 1)` — the clamp exists to defeat a tripwire; restore the tripwire.
- Git-hash retrieval (`except Exception: pass` sites in `export_ui_payload.py`, `src/tune.py::get_git_metadata`, `scripts/ingestion/ingest_parquet.py`): let `subprocess.CalledProcessError` propagate. In `scripts/build_run_registry.py`, missing `wandb_run_id`/`git_commit_hash` must raise, not default to `"unknown"`.
- Semantic-tag keys in `export_ui_payload.py` (`commercial_density`, `highway_proximity`, `nearby_poi_count`): these keys do not exist in the features schema. Map them to the real columns (`retail_count`, `transit_count`, `total_poi_count` — confirm against `src/features.py` output before choosing) and use `feats["key"]` indexing so a missing key raises.
- NaN-coordinate defaults (`{"lat": NaN, "lon": NaN}` + warn-and-skip edges, both export scripts): unknown sensor → raise `KeyError` naming the sensor.

### 2.2 Training-loop integrity (§6)
- Seed provenance: in `run_training`, replace hardcoded `torch.manual_seed(42)` / `np.random.seed(42)` with the config value `train_cfg["random_seed"]` (add the key requirement; it already exists in `train.yaml`). Remove the `torch.manual_seed(42)` side-effect calls from module constructors in `src/baselines.py`, `src/encoder.py`, `src/decoders.py` (and the module-level seeds at the top of `src/dataset.py`) — seeding happens exactly once, in `run_training`.
- Soft-memory `break`: keep the early exit (OOM protection is legitimate) but make it loud and honest — `raise RuntimeError` OR set a flag that (a) skips the artifact upload and (b) marks the results manifest `"aborted": true`. Choose the raise unless the PR reviewer objects; do not silently upload a truncated model.
- Synthetic holdout-manifest IDs (`node_0, ...` fallback): replace the warning with `raise ValueError` when the dataset exposes no location IDs in inductive mode.
- Train/val scaler drift >20%: `logger.warning` → `raise ValueError`.
- Loss-weight and config `.get()` defaults (`harness/adapters.py`, `pipeline.py`, the ~12 sites in `train.py`): convert to required keys with `[...]` indexing and add every key explicitly to `configs/train/train.yaml`. Do this mechanically: for each `.get("k", default)`, add `k: <default>` to the YAML, then switch to `["k"]`.
- `pipeline.py` silent `dim_feedforward` correction: `<= 0` → `raise ValueError`. `None` may keep the documented `hidden_dim * 4` default (that is an API default, not silent correction) — state this distinction in a comment.
- `src/tune.py` silent storage/config/sweep fallbacks: required arguments; raise when absent.

### 2.3 Silent data coercion (§1, §7)
- `src/db_connector.py`: `dropna(subset=['value'])` → assert no NaNs, or (if upstream legitimately produces them) count and `raise` with the count. Weather `fillna(0.0)` in `ParquetConnector.load_weather` → assert-not-NaN. The 15 `.get(..., 0)` static-feature reads → `[...]` indexing. Silent default parquet path → required argument.
- `src/dataset.py`: the three `fillna(0.0)` calls on pivoted dynamic channels and the static `reindex(...).fillna(0.0)` → replace with asserts that no NaN exists after reindex (if real gaps exist in the data, this phase will crash on them — that is the intended behavior; the gap-handling decision then goes to a human, per doctrine). `.clamp(min=0)` on `target_forecast` → `assert (target_tensor >= -1e-6).all(...)` then clamp only the float-precision dust (keep a tight tolerance; a genuinely negative count must crash).
- `src/preprocess.py` `.dropna(subset=groups)` inconsistency: make both sites assert instead of drop.
- `src/features.py`: non-200 POI response → raise (an outage must not be cached as "zero POIs"); unmapped-location skip-and-continue → raise.
- `src/gpu_selector.py`: `nvidia-smi` failure → raise (CPU fallback must be an explicit user decision, e.g. a `--cpu` flag, not a silent cascade).
- `src/metrics.py::calculate_metrics`: add shape/rank asserts for all six tensors, mirroring `calculate_kriging_metrics`.
- `src/decoders.py`: add entry asserts to `compute_zinb_nll` (μ, θ strictly positive *before* `safe_log`, pi in (0,1)); add rank/channel asserts at the top of `compute_loss`; target clamp → assert-then-clamp as in dataset above.
- Remaining broad `except Exception:` sites from §1's list (`train.py`, `tune.py`, `features.py`, `analysis.py`, both exporters): narrow each to the specific expected exception or delete the handler. If a handler guards W&B/network flakiness, narrow it to the specific exception class and re-raise after logging.

### Verification Gate — Phase 2
```bash
grep -rn "except Exception" src/ | grep -v test        # must return ZERO hits (or listed+justified in PR)
grep -rn "strict=False" src/                           # zero hits
grep -rn "fillna(0" src/dataset.py src/db_connector.py # zero hits
python src/pipeline.py && python src/train.py --use_mock_data
```
- Negative tests (write as one-off scripts or pytest, include in PR): (a) export with a checkpoint missing a key → must raise; (b) export with no co-located config.yaml → must raise; (c) `run_training` with a config missing `loss.w_impute` → must raise KeyError before training starts.
- If the real-data path is available: one short real training run to confirm the new NaN/fill asserts do not fire on clean data. If any fires, **stop** — that is a genuine data-integrity discovery; document it, do not soften the assert.

---

## Phase 3 — Data Ingestion Integrity + Phase-3 CDN Cleanup (report §4, §8, §9)

**Goal:** finish the CDN migration honestly and make ingestion crash on fabricated/dropped data.
**Files:** `scripts/upload_to_r2.py`, `scripts/build_run_registry.py`, `scripts/generate_all_payloads.sh`, `scripts/verify_phase2_phase3.sh`, `scripts/ingestion/*.py`, `.github/workflows/publish_payloads.yml`, `frontend/public/data/`, repo root debris.

### 3.1 Legacy path removal (§4, §8)
- Introduce one canonical scratch dir (suggest `artifacts/payload_staging/`, gitignored). Update the `frontend/public/data/runs` default in: `upload_to_r2.py` (`DEFAULT_RUNS_DIR`), `build_run_registry.py` (default + registry output path), `generate_all_payloads.sh` (`OUTPUT_DIR`), and the three hardcoded steps in `.github/workflows/publish_payloads.yml`. The workflow should reference a single env var, not three copies of the literal.
- Add `if: always()` to the workflow cleanup step.
- Delete `frontend/public/data/master_index.json` (4.6 MB, referenced by nothing — re-verify with `grep -rn "master_index" frontend/src scripts src` before deleting).
- Delete committed debris: `r2_temp.json`, `registry_temp.json`, `dummy.txt`, `run.log`, `frontend/frontend/package-lock.json`. Add matching `.gitignore` entries.
- `upload_to_r2.py`: either implement `--delete-extra` (list bucket, diff against registry, delete orphans behind an explicit flag) or delete the docstring claim. Implementing is preferred; if implemented, require a `--yes` confirmation flag for the destructive path.

### 3.2 verify script (§8)
- `scripts/verify_phase2_phase3.sh`: fix stale `scripts/ingest_parquet.py` paths → `scripts/ingestion/...` (also fix the same stale paths in the four ingestion scripts' usage docstrings); make `check_fail()` set a failure flag and `exit 1` at the end; replace the `"200 OK"` grep with a check that tolerates HTTP/2 (`grep -E "HTTP/(1\.1 200|2 200)"` or use `curl -s -o /dev/null -w "%{http_code}"`).

### 3.3 Ingestion honesty (§9)
- `ingest_weather.py`: remove the `fillna(0.0)`-to-satisfy-NOT-NULL. Placeholder rows must either be skipped **with a logged, asserted count** and a distinct provenance marker, or the ingestion must fail. Zero must never be fabricated as a reading.
- All four ingestion scripts: replace attempted-row counters with `cursor.rowcount` sums so "Total inserted" is truthful; when `ON CONFLICT DO NOTHING` swallows rows, log inserted vs. skipped separately and assert skipped == 0 unless an explicit `--allow-duplicates` flag is passed.
- `ingest_parquet.py`: the NaN-row drop gets a counted, logged, asserted drop-rate (suggest: assert drop rate < 1% unless `--allow-drops`); the hardcoded `seq_len: 12, mask_prob: 0.15` provenance JSON must read from `configs/train/train.yaml` instead.
- `ingest_static_features.py`: `.get('...', 0)` reads → `[...]` indexing (missing feature = crash).

### Verification Gate — Phase 3
```bash
grep -rn "frontend/public/data" scripts/ .github/ | grep -v ".gitignore"   # zero hits
bash scripts/verify_phase2_phase3.sh ; echo "exit=$?"                      # runs; exit code is honest (nonzero on any failed check)
git ls-files | grep -E "r2_temp|registry_temp|dummy.txt|^run.log|frontend/frontend"  # zero hits
```
- Dry-run an ingestion script against a small parquet fixture and confirm the inserted-count now matches `cursor.rowcount` reality (insert the same file twice; second run must report 0 inserted and fail without `--allow-duplicates`).

---

## Phase 4 — Frontend Architecture & Typing (report §2, §10)

**Goal:** restore the 60fps rendering architecture and strict typing; make fetch failures visible.
**Files:** `frontend/src/components/MapVisualizer.tsx`, `NodeInspector.tsx`, `TimeSlider.tsx`, `frontend/src/app/analytics/page.tsx`, `frontend/src/types/payload.ts`, `frontend/src/store/useTimeStore.ts`, `frontend/next.config.ts`, `frontend/package.json`, `app/api/log/route.ts`, `GlobalErrorReporter.tsx`.

Ordered by impact:

1. **Run-switch deadlock** — port the `analytics/page.tsx` fix ("Uses setSpatialIndex (state, not a ref) so Stage 3 re-triggers") into `MapVisualizer.tsx` Stage 3. This is a straight port of an existing, working pattern — read the analytics implementation first and mirror it.
2. **Per-tick CPU filtering → `DataFilterExtension`** — replace the `payload.edges.map(...).filter(...).sort(...).slice(...)` inside `buildLayers` with Deck.gl's `DataFilterExtension` (`getFilterValue` on the timestep, `filterRange` driven by `currentTimestep`) so filtering happens on the GPU and layer data arrays stay referentially stable across ticks. The top-K ego-edge slice can remain CPU-side but must be memoized on `[selectedNode, payload]`, not recomputed per tick.
3. **Timestep re-render path** — move `currentTimestep` consumption off React state for the layer path: subscribe via `useTimeStore.subscribe` into a ref + `layer.setProps`/`updateTriggers` (mirror the existing hover-path pattern in the same file). Same treatment for the per-tick re-filter/re-sort in `NodeInspector.tsx`.
4. **Typing** — replace the three `any`s in `payload.ts` (`hyperparameters`, `tripwire_audit_trail`, `alpha_node_leaderboard`) with concrete interfaces derived from the actual exporter output (read `src/export_paginated_payload.py` to derive fields; do not guess). Fix `valueFormatter` null-crash and the `|| 0` fabricated MSE in `analytics/page.tsx` (hide the series when data is absent). Add `?? []` to the `edge_attention as EdgeEntry[]` cast site.
5. **Fetch failure visibility** — registry/spatial/page fetch chains in all three components: replace `console.warn` / `r.ok ? r.json() : null` / empty `.catch` with an error state rendered to the user (a simple error banner component is sufficient). Add `AbortController` to each chain, aborting superseded requests on run/date switch.
6. **Hygiene** — remove the per-tick `console.log` in `TimeSlider.tsx` and the accessor `console.log` in the ArcLayer `getWidth`; gate or delete `app/api/log/route.ts` + `GlobalErrorReporter.tsx` (if kept: async fs, body-size limit, auth on DELETE, dev-only). Remove `@types/maplibre-gl` from `package.json` (maplibre ≥2 ships its own types). Delete dead `startDate`/`setStartDate` from `useTimeStore.ts`. `next.config.ts`: remove the hardcoded R2 bucket fallback — throw at build time when `R2_PUBLIC_URL` is unset, and add `.env.example` documenting it; lift the two cache TTL constants to named constants with a comment referencing `scripts/upload_to_r2.py`'s `max-age`.

### Verification Gate — Phase 4
```bash
cd frontend && npx tsc --noEmit          # zero errors, strict mode
npm run lint                             # zero errors
npm run build                            # must succeed WITH R2_PUBLIC_URL set; must FAIL clearly without it
grep -rn "console.log" src/components/TimeSlider.tsx src/components/MapVisualizer.tsx   # zero hits
grep -rn ": any" src/types/payload.ts    # zero hits
```
- Manual check (describe in PR if no browser available): map renders, timestep playback works, switching runs updates the map (deadlock fix), killing the network shows the error banner instead of a silent blank map.

---

## Phase 5 — CI/Infra Security, Config Hygiene, and Medium-Risk Performance (report §11, §12, §14-GPU-masking, §17.2)

**Goal:** the items that need elevated care: security-sensitive CI changes and the two performance changes that alter RNG streams / numerics. Do 5.1–5.2 first (independent); 5.3–5.4 require Phase 1 to be merged.

### 5.1 CI / Infra security (§11)
- `publish_payloads.yml`: pass `${{ github.event.inputs.artifact_ref }}` via `env:` and reference it as `"$ARTIFACT_REF"` in the `run:` block (script-injection fix — **do first**). Replace the bare `pip install -r requirements.txt` with execution inside the project's Docker image per `NSTAR_MLOPS.md` §7. Replace the silent `/data`-absent fallback with a hard failure.
- `docker-compose.yml`: give the `github-runner` service an explicit minimal `environment:` list instead of `env_file: .env`; parameterize the Postgres credentials via env vars (no `riotuser`/`riotpassword` literals — also remove the re-hardcoded copies in `verify_phase2_phase3.sh`); bind Postgres to `127.0.0.1:5432:5432` unless external access is documented as required; delete the duplicated gcloud volume mount line.
- `scripts/start-runner.sh`: remove the "assume direct registration token" fallback — API failure must exit nonzero.

### 5.2 Config hygiene (§12)
- Create `configs/geo.yaml` holding the OSMnx constants (`center: [54.978, -1.610]`, `dist: 5000`); read it from `src/features.py`, `scripts/cache_edge_features.py`, `scripts/cache_simulated_edge_features.py`.
- W&B project names: single source in `configs/train/train.yaml` `wandb:` block; remove the scattered `.get()` defaults in `train.py`, `tune.py`, both exporters (required keys per Phase-2 convention) and deduplicate across `configs/sweeps/*.yaml` (sweeps may reference the train config or repeat the value — pick one and state it).
- Move to config: structural-zero night window (`preprocess.py`), 0.8 train split (`train.py`, two sites), exporter constants (`NUM_TIMESTEPS_TARGET`, `TOP_K_EGO_EDGES`, bounding box, simulated-node coordinates), duplicated `mask_prob` in both exporters.
- Fix `Optional` type hints (`seed: int = None` → `seed: Optional[int] = None`) at the listed sites; move `src/check_gaps.py` out of `src/` (to `scripts/eda/`) and fix its silent `.dropna()`.

### 5.3 GPU-side mask injection (§14 medium-risk) — requires Phase 1 merged
- Implement exactly as specified in report §14: `__getitem__` returns the clean window only (no `target_impute`, no `node_mask`, no NaN injection); train/val loops generate `node_mask` on-device with `torch.rand(clean.shape, device=device) < mask_prob` and build `dynamic_features`/`target_impute`/`node_mask` post-transfer. `mask_prob` moves from a dataset constructor arg to `train_cfg["mask_prob"]` (add to YAML; keep the dataset arg removed, not defaulted).
- Record in the results manifest that the augmentation RNG stream changed (add a `"mask_rng": "cuda"` field).
- **A/B gate:** run mock training before/after — losses will differ (different RNG stream); verify they converge to the same ballpark (< 5% relative difference in final val loss on the mock run) and that no NaN tripwire fires. Paste both loss curves into the PR.

### 5.4 bf16 autocast (§17.2) — requires Phase 1 merged; do LAST
- Implement per report §17.2: `amp: true` key in `train.yaml` (required, no default); `assert torch.cuda.is_bf16_supported()` when enabled; `torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled)` around `adapter.step` in both train and val loops; `backward()` outside the context; **no GradScaler**.
- FP32 loss carve-out: in `src/pipeline.py`, wrap the `compute_loss(...)` call in `with torch.autocast(device_type="cuda", enabled=False):` and `.float()` the tensor inputs to it (`imputed_features, target_impute, mu, theta, pi, target_forecast, kriging_pred`). Same guard in `harness/adapters.py::RecurrentBaselineAdapter.step` around its `F.mse_loss` calls.
- **A/B gate (mandatory):** one full training run with `amp: false` and one with `amp: true` on real data (or the longest available mock config if no GPU/data access — state which). Compare final val metrics (< 2% relative difference expected) and confirm every ZINB tripwire stays silent. The existing NaN asserts are the burn test — if one fires under bf16, revert the flag to `false` in YAML, keep the code, and file the failure with the offending tensor stats.

### Verification Gate — Phase 5
```bash
# 5.1
grep -n "github.event.inputs" .github/workflows/publish_payloads.yml   # only inside env: blocks
grep -rn "riotpassword" . --include="*.yml" --include="*.sh"           # zero hits
# 5.2
grep -rn "54.978" src/ scripts/ | grep -v configs                      # zero hits outside config loading
# 5.3 / 5.4
python src/train.py --use_mock_data                                    # passes with amp: false
# plus the A/B evidence described above pasted into the PR
```

---

## Item → Phase index (completeness check)

| Report section | Phase |
|---|---|
| §1 broad excepts / silent drops | 2 |
| §2 frontend `any` types | 4 |
| §3 hardcoded mock block in `pipeline.py` | 2 (move burn test to `smoke_test.py`/tests during 2.2, optional; low priority) |
| §4 legacy Phase-1 exports | 3 |
| §5 export-chain silent failures | 2 |
| §6 training-loop warn-and-continue | 2 |
| §7 silent data coercion | 2 |
| §8 CDN legacy plumbing | 3 |
| §9 ingestion fabrication | 3 |
| §10 frontend architecture | 4 |
| §11 CI/infra | 5 |
| §12 config hygiene | 5 |
| §13 DataLoader config | 1 |
| §14 `__getitem__` overhead | 1 (low-risk subset) + 5.3 (GPU masking) |
| §15 H2D transfers | 1 |
| §16 sync elimination | 1 |
| §17 mixed precision | 1 (TF32) + 5.4 (bf16) |
| §18 secondary observations | 1 (no-ops noted) |
