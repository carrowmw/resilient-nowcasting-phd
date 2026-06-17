# Implementation Plan — Phase 1: Multi-Run Paginated Payload Architecture

**Date**: 2026-06-17
**Branch**: `database` (confirmed current)
**Architect Plan**: `docs/database/phase1_detailed_plan.md`
**Payload generation strategy**: Option B (batch script) — per Architect §6 recommendation
**Status**: ⏸ **AWAITING USER APPROVAL — no execution code will be written until approved.**

---

## 0. Goal

Move the digital twin off the single inlined `master_index.json` and onto a
**multi-run, paginated** layout so the frontend can (a) switch between W&B model
runs and (b) page through analysis days without loading everything at once:

```
frontend/public/data/runs/
├── run_registry.json                    # all runs + default_run_id
└── {run_id}/                            # e.g. feasible-capybara-22
    ├── spatial.json                     # nodes + edge topology + run metadata (NO timeseries)
    └── {YYYY-MM-DD}/                    # one folder per analysis day
        ├── page.json                    # 96-step timeseries + edge attention + day metrics
        └── nodes/node_{id}.json         # per-node deep analytics (day-scoped)
```

Three workstreams: **Backend (B)** → validated independently → **Frontend (F)** → **Cleanup (C)**.

---

## 1. Verified Facts (checked against the repo on 2026-06-17)

| Claim | Status |
|---|---|
| On `database` branch | ✅ `git branch --show-current` = `database` |
| `configs/analysis/analysis.yaml` exists with flat keys `holdout_start: 2026-05-01`, `holdout_end: 2026-06-01` | ✅ read directly |
| All 15 helpers/constants the new script imports exist in `export_ui_payload.py` | ✅ `load_sensor_coords`, `select_artifact_from_wandb`, `load_model_from_wandb`, `build_simulated_static`, `compute_std_dev`, `extract_timestep_attention`, `load_semantic_tags`, `compute_pit_histogram`, `compute_zero_inflation_reliability`, `compute_two_extremes`, `build_ego_graph_edges`, `SIMULATED_NODE_COORDS`, `NUM_TIMESTEPS_TARGET` (=96), `TOP_K_EGO_EDGES` (=10), `SENSORS_PATH`, `PARQUET_PATH` (=`data/processed/master_dataset_15T.parquet`) |
| `export_ui_payload.py` is 1347 lines; `main()` at L672 | ✅ |
| `MapVisualizer.tsx:316` fetches `/data/master_index.json` | ✅ |
| `NodeInspector.tsx:413` fetches `/data/nodes/node_${selectedNodeId}.json` | ✅ |
| `analytics/page.tsx:92` ALSO fetches `/data/master_index.json` | ✅ (drives Cleanup decision C-3) |
| `frontend/public/data/runs/` exists and is empty | ✅ |
| Frontend components present: `MapScene`, `MapVisualizer`, `NodeInspector`, `TimeSlider`, `AttentionControls` | ✅ |
| `frontend/src/store/useTimeStore.ts`, `frontend/src/types/payload.ts` present | ✅ |
| `frontend/AGENTS.md`: "This is NOT the Next.js you know — read `node_modules/next/dist/docs/` before writing code" | ✅ |
| docker-compose services: `agent-workspace` (my-ml-agent), `frontend-dev` (nextjs-frontend), `gpu-runner`, `cpu-runner` | ✅ |

---

## 2. ⚠️ Decisions & Deviations (READ FIRST)

### BLK-1 — Execution context: already inside `agent-workspace` ✅ RESOLVED
The session **runs inside the `agent-workspace` container** (`hostname 914e32a0d702`,
Python 3.11.15, wandb 0.27.2 confirmed). So Python/W&B commands run **directly** — the
`docker compose exec agent-workspace …` prefix in the Architect examples is dropped.
Frontend `tsc`/`vitest` still need the `frontend-dev` container; from inside this
container that means either `docker compose exec frontend-dev …` (if the docker CLI is
reachable here) **or** running them in the `frontend-dev` shell. **PF-1b**: confirm how
to reach `frontend-dev` from here before F-11. Either way, **no host installs.**

### DEC-1 — Phase 3: serve payloads from Cloudflare R2, NOT git ✅ (user, FINAL)
Decision history: git-ignore → commit-to-git → **(FINAL)** skip git entirely and serve
from **Cloudflare R2** (S3-compatible object storage). Driver: the measured full-batch
size is **~404 MB/model pretty / ~215 MB minified / ~62 MB gzipped**, and there are
**10 model artifacts (v0–v9) × 31 days** → ~4.0 GB pretty / 2.1 GB minified in git — past
the <500 MB target and into repo/Vercel-bloat territory (B-4 size projection).
- Generated `runs/**` stay **git-ignored** (`.gitignore` rule re-added). Local generation
  is a staging step only.
- A new **`scripts/upload_to_r2.py`** (boto3, S3-compatible endpoint) uploads `runs/**`
  (incl. `run_registry.json`) to R2. **gzip + `ContentEncoding: gzip` + `ContentType:
  application/json`** so `fetch()` transparently decompresses (no frontend decode code) —
  ~62 MB/model over the wire.
- The **frontend fetches from `R2_PUBLIC_URL`** instead of local `/data/...` static files.
- Env (already in `.env`): `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
  `R2_BUCKET_NAME`, `R2_PUBLIC_URL` (= `https://pub-…r2.dev`). Endpoint:
  `https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`. The browser base URL must be exposed
  to Next.js as **`NEXT_PUBLIC_PAYLOAD_BASE_URL`** (= `R2_PUBLIC_URL`).
- `boto3` added to `requirements.txt` (installed in agent-workspace).
- The generator now **minifies** JSON (compact separators) regardless — smaller disk,
  upload, and fetch.
- W&B `--upload` to `riot-smoke` is retained as optional archival (used by the smoke test).

### DEC-2 — Smoke-test isolation: `riot-smoke` **project** ✅ (user, confirmed)
Smoke-test outputs go to a W&B **project named `riot-smoke`** (not a new entity) under the
production entity. Smoke tests still **read production model artifacts** from the real
entity; only the *generated payload artifacts* / smoke W&B run go to `riot-smoke`.

### DEC-3 — Production W&B entity ✅ (user)
Entity: **`rappel-velvet9310-newcastle-university`**. Source model artifacts live at
`rappel-velvet9310-newcastle-university/resilient-iot/resilient-iot-model`. Smoke uploads:
`rappel-velvet9310-newcastle-university/riot-smoke`.

### D-1 — Config path & schema (per user instruction #2) 🟡
- Architect §2.2 said: create `configs/analysis.yaml`, nested `analysis:` block, dates
  `2026-06-01 → 2026-07-01`.
- **Reality / user instruction**: use the existing **`configs/analysis/analysis.yaml`**
  (note the subdirectory) with **flat keys** and **dates `2026-05-01 → 2026-06-01`**:
  ```yaml
  holdout_start: 2026-05-01
  holdout_end: 2026-06-01
  ```
- **Action**: **Do NOT create `configs/analysis.yaml`. Do NOT overwrite the dates.**
  The export script's config loader (B-1) must:
  1. Read `holdout_start` / `holdout_end` as the primary date source.
  2. Also accept a nested `analysis: {start_date, end_date, …}` block if present (forward-compat).
  3. Fall back to CLI/hardcoded defaults for fields the file omits — the user's file has
     **no** `veh_class`/`dir`/`seq_len`/`num_simulated`, so default to `person`/`all`/`12`/`5`
     (matching `configs/train/train.yaml` + current `export_ui_payload.py`).
  4. CLI `--start_date`/`--end_date` override the file when explicitly passed.
- `generate_all_payloads.sh` points `CONFIG` at `configs/analysis/analysis.yaml`.

### D-2 — Analysis window is 31 days, not 30 🟡
`2026-05-01 → 2026-06-01` (exclusive) = **31 inference days/model**. Per-day inference
loads `[date − 1d, date + 1d)` for seq_len look-back, so the earliest data touched is
**2026-04-30**. **PF-2 must confirm the parquet covers 2026-04-30.** (Parquet is known to
end ~2026-06-08, so the upper bound is safe.)

### D-3 — New file, not in-place edit 🟡
Per Architect §2.1: create **new** `src/export_paginated_payload.py` and **import shared
helpers** from `export_ui_payload.py` (left working & untouched). W&B-cleanup docstring
edits (Architect §7 item 3) are comment-only and deferred to Cleanup C-1.

### D-4 — Pre-existing working-tree changes are not mine 🟡
`git status` already shows unrelated changes (`notes/scaling_verification.md` deleted,
`docs/scaling_verification.md` added, `master_index.json` modified, `__pycache__` churn,
`configs/analysis/` untracked). **Every commit stages only its own files explicitly** —
no `git add -A`.

---

## 3. Pre-flight Checks (run in-container, before writing execution code)

> Running **inside `agent-workspace`** → Python runs directly (no `docker compose exec`).
> Node tooling targets `frontend-dev` (PF-1b). **No host installs.**

| ID | Check | Command (run directly in this container) |
|----|-------|---------|
| PF-1b | Can reach `frontend-dev` for tsc/vitest | `docker compose exec frontend-dev node -v` (or confirm the frontend shell) |
| PF-2 | Parquet covers 2026-04-30 → 2026-06-01 | `python -c "import pandas as pd; df=pd.read_parquet('data/processed/master_dataset_15T.parquet'); print(df.index.min(), df.index.max())"` |
| PF-3 | `WANDB_API_KEY` (+ `GOOGLE_PLACES_API_KEY`) load via `.env` | `python -c "import os;print(bool(os.getenv('WANDB_API_KEY')), bool(os.getenv('GOOGLE_PLACES_API_KEY')))"` (raw shell showed W&B auth not yet loaded — the script auto-loads `.env`; verify there) |
| PF-4 | W&B collection lists expected artifacts | `python src/export_paginated_payload.py --all_artifacts --dry_run` (after B-1) |
| PF-5 | Places caches exist | `ls data/external/places_cache.json data/external/simulated_places_cache.json` |

If PF-2 shows the parquet starting after 2026-04-30, **surface to the user before generating** (first day(s) would lack look-back).

---

## 4. Backend Workstream (B)

### B-1 — `src/export_paginated_payload.py` (NEW)
Multi-run, per-day refactor of `export_ui_payload.py`. **Imports, never duplicates**, shared helpers:
```python
from export_ui_payload import (
    load_sensor_coords, select_artifact_from_wandb, load_model_from_wandb,
    build_simulated_static, compute_std_dev, extract_timestep_attention,
    load_semantic_tags, compute_pit_histogram, compute_zero_inflation_reliability,
    compute_two_extremes, build_ego_graph_edges,
    SIMULATED_NODE_COORDS, NUM_TIMESTEPS_TARGET, TOP_K_EGO_EDGES,
    SENSORS_PATH, PARQUET_PATH,
)
```
(Importing triggers `export_ui_payload`'s module-level env/logging setup — idempotent; we never call its `main()`/`parse_args()`.)

**New CLI args** (Architect §2.1, adapted to D-1):
- `--config` (default `configs/analysis/analysis.yaml`)
- `--start_date` / `--end_date` (override config)
- `--output_dir` (default `frontend/public/data/runs`)
- `--artifact <ref>` | `--all_artifacts`
- `--veh_class` / `--dir` / `--seq_len` / `--num_simulated` (fallback `person/all/12/5`)
- `--dry_run` (resolve artifacts + dates, print, exit — supports PF-4)
- `--upload` / `--wandb_project` / `--wandb_entity` (DEC-1/2/3) — **opt-in archival**:
  when set, also upload the generated per-run payload tree to W&B as an artifact after
  writing it locally. Smoke tests pass `--wandb_entity rappel-velvet9310-newcastle-university
  --wandb_project riot-smoke` to isolate from production. The git-committed local tree
  (not W&B) is what Vercel serves. Reads of the source *model* artifact always use the
  production entity regardless.

**New functions in this file:**
1. `load_analysis_config(path, args) -> dict` — D-1 reconciliation; returns resolved
   `{start_date, end_date, veh_class, dir, seq_len, num_simulated}`.
2. `load_model_and_meta(artifact) -> (ckpt, cfg_path, wandb_run_id, run_name)` — wraps
   `load_model_from_wandb` but also returns the producer run's `.name` for use as the
   filesystem `run_id` (e.g. `feasible-capybara-22`); sanitized fallback if name missing.
3. `resolve_artifacts(args) -> list[str]` — single (`--artifact`) or full collection
   (`--all_artifacts`, reusing collection-listing logic from `select_artifact_from_wandb`).
4. `select_day_windows(dataset, seq_len) -> (skip, num_timesteps)` — generalises the
   existing day-2 `half/skip` selection in `export_ui_payload.main()` for the 2-day window.
5. `run_day_inference(model, dataset, skip, n, …) -> accumulators` — port of the 24-hour
   inference loop in `export_ui_payload.main()`; returns the same per-node/per-edge accumulators.
   **Keep every existing tripwire/assert** from the source loop.
6. Writers:
   - `write_page_json(out, run_id, date, acc, seq_len)` — schema §5.3 (`node_timeseries`,
     `edge_attention`, day-scoped `pit_histogram`, `zero_inflation_reliability`,
     `global_imputation_burden`, `global_healing_quality`, `attention_entropy_series`,
     `tripwire_audit_trail`, `alpha_node_leaderboard`).
   - `write_node_detail_jsons(out, run_id, date, acc, …)` — same `NodeDetailPayload` shape
     as today (zinb_series, ground_truth_series, imputed_series, mask_series, edges_1hop,
     two_extremes, ego_graph_edges). Only the output path changes.
   - `write_spatial_json(out, run_id, nodes_meta, edge_topology, metadata)` — schema §5.2;
     `metadata.available_dates` = sorted generated day folders; `semantic_tags` per node
     (computed once per run via `load_semantic_tags`).

**Main flow:**
```
cfg = load_analysis_config(args.config, args)
for artifact in resolve_artifacts(args):
    try:
        ckpt, cfg_path, wandb_run_id, run_name = load_model_and_meta(artifact)
        model, model_cfg, git_hash = build_model(ckpt, cfg_path)   # port from main()
        available = []
        for date in date_range(cfg.start_date, cfg.end_date):       # exclusive end
            ds  = RealIoTDataset(PARQUET_PATH, date-1d, date+1d, cfg.seq_len, …)
            skip, n = select_day_windows(ds, cfg.seq_len)
            acc = run_day_inference(model, ds, skip, n, …)
            write_page_json(...); write_node_detail_jsons(...)
            available.append(date)
        write_spatial_json(..., metadata={run_id, run_name, wandb_run_id, git_hash,
                                          hyperparameters, metrics_summary, available_dates: available, …})
        if args.upload:                                             # DEC-1: W&B is canonical
            upload_run_artifact(out, run_id, project=args.wandb_project, entity=args.wandb_entity)
    except Exception:
        log.exception("artifact %s failed; continuing", artifact)   # Option-B retry/skip
```
`upload_run_artifact(...)` logs `{output_dir}/{run_id}/` as a W&B artifact
(`type="ui-payload"`, name=`run_id`) under the given entity/project — **only when
`--upload` is passed** (DEC-1: git is the serving path; W&B upload is archival/smoke-only).

### B-2 — `scripts/build_run_registry.py` (NEW)
Direct port of Architect §2.3 `build_registry()`. Scans `output_dir` for `*/spatial.json`,
emits `run_registry.json` with `default_run_id` = first run.
CLI: `python scripts/build_run_registry.py <runs_dir>`.

### B-3 — `scripts/generate_all_payloads.sh` (NEW)
Architect §6 Option B, with D-1 config path (runs directly in-container):
```bash
#!/bin/bash
set -euo pipefail
CONFIG="configs/analysis/analysis.yaml"      # D-1
OUTPUT_DIR="frontend/public/data/runs"
# Git is the serving path (DEC-1) → no --upload here; W&B upload is opt-in/archival only.
python src/export_paginated_payload.py --all_artifacts --config "$CONFIG" --output_dir "$OUTPUT_DIR"
python scripts/build_run_registry.py "$OUTPUT_DIR"
echo "Done. du -sh $OUTPUT_DIR to check total size."
```
`chmod +x`. Invoked as `bash scripts/generate_all_payloads.sh`. Default artifact discovery
uses entity `rappel-velvet9310-newcastle-university` (DEC-3).

### B-4 — Backend smoke test (before full batch)
Smoke output uploaded to the `riot-smoke` project (DEC-2); reads the production model
artifact from the `rappel-velvet9310-newcastle-university` entity (DEC-3):
```bash
python src/export_paginated_payload.py \
  --artifact rappel-velvet9310-newcastle-university/resilient-iot/resilient-iot-model:latest \
  --start_date 2026-05-01 --end_date 2026-05-03 \
  --output_dir frontend/public/data/runs \
  --upload --wandb_entity rappel-velvet9310-newcastle-university --wandb_project riot-smoke
```
Validate (Architect §9): `spatial.json`, `2026-05-01/page.json`, `2026-05-01/nodes/node_0.json`;
`jq .metadata.num_nodes …/spatial.json`, `jq .timesteps …/2026-05-01/page.json`; confirm the
`riot-smoke` artifact appears in W&B.
✅ **Done** — run `dark-brook-37` (v9), 61 nodes (56 real + 5 sim), 1761 edges, schemas valid.

### B-5 — `scripts/upload_to_r2.py` (NEW, boto3 → Cloudflare R2) [DEC-1 Phase 3]
S3-compatible uploader. Walks a local `runs/` tree and uploads each file to R2 under the
same key path (`runs/{run_id}/…`). Per object: gzip the bytes, set
`ContentType=application/json`, `ContentEncoding=gzip`, `CacheControl`. Uses a boto3 client
with `endpoint_url=https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
`region_name="auto"`, sigv4. CLI: `--runs-dir`, `--prefix runs`, `--dry-run`, `--no-gzip`.
Reads R2_* creds from env (`.env` auto-loaded). Invoked after generation; the batch script
calls it.

---

## 5. Frontend Workstream (F)

> ⚠️ Per `frontend/AGENTS.md`: **before writing any frontend code, read the relevant guide
> in `node_modules/next/dist/docs/`** (via `docker compose exec frontend-dev cat …`).

### F-1 — `frontend/src/types/registry.ts` (NEW)
All new pagination types in one module: `RunRegistry`, `RunRegistryEntry`, `SpatialIndex`,
`SpatialNode`, `EdgeTopology`, `TemporalPage` (schemas §5.1–5.3). Re-use the existing
`NodeDetailPayload` from `payload.ts` (unchanged).

### F-2 — `frontend/src/store/useTimeStore.ts` (MODIFY)
Add Architect §4.1 fields + actions: `activeRunId`, `activeDate`, `availableDates`,
`runRegistry`, `isLoadingPage`; `setActiveRunId`, `setActiveDate`, `setAvailableDates`,
`setRunRegistry`, `setIsLoadingPage`, `navigateDay('prev'|'next')`. Keep existing
`startDate` (TimeSlider uses it) and drive it from `activeDate`.

### F-3 — `frontend/src/components/MapVisualizer.tsx` (MODIFY)
Replace the single `/data/master_index.json` fetch (around L316–378) with three-stage loading (Architect §4.2):
- **Stage 1 (mount)**: fetch `run_registry.json` → `setRunRegistry`, `setActiveRunId(default)`.
- **Stage 2 (on `activeRunId`)**: fetch `{runId}/spatial.json` → `spatialRef`, `setAvailableDates`, `setActiveDate(dates[0])`.
- **Stage 3 (on `activeRunId`+`activeDate`)**: fetch `{runId}/{date}/page.json` →
  `mergePayload(spatialRef, page)` → `payloadRef`, `setMaxTimesteps`, `setTimestep(0)`, rebuild layers.
- Add `mergePayload(spatial, page)` producing the **existing `UIPayload` shape** so
  `buildLayers()`/`nodeMap` stay unchanged → minimal refactor surface.

### F-4 — `frontend/src/components/NodeInspector.tsx` (MODIFY)
Change the fetch at **L413** from `/data/nodes/node_${selectedNodeId}.json` to
`/data/runs/${activeRunId}/${activeDate}/nodes/node_${selectedNodeId}.json`; add
`activeRunId`/`activeDate` to the effect deps + null-guard (Architect §4.3).

### F-5 — `frontend/src/components/ModelSelector.tsx` (NEW)
Dropdown bound to `runRegistry`/`activeRunId` (Architect §3.1). On change → `setActiveRunId`
(Stages 2–3 cascade). URL `?run=` sync **deferred** unless trivial.

### F-6 — `frontend/src/components/DatePaginator.tsx` (NEW)
Prev/next over `availableDates` via `navigateDay`; disable at boundaries; spinner on
`isLoadingPage` (Architect §3.3). Mounted inside `TimeSlider`.

### F-7 — `frontend/src/components/ModelInfoOverlay.tsx` (NEW)
Glassmorphic card from `spatial.json` metadata + registry entry (Architect §3.2). Lowest
priority — lands last.

### F-8 — `frontend/src/components/TimeSlider.tsx` (MODIFY)
Mount `DatePaginator` in the header row; replace static `startDate` label with `activeDate`.

### F-9 — `frontend/src/components/MapScene.tsx` (MODIFY)
Add `<ModelSelector />` + `<ModelInfoOverlay />` to the composition (Architect §4.4) as
`dynamic(..., { ssr:false })` imports, matching the existing pattern.

### F-10 — `frontend/src/app/analytics/page.tsx` (DECISION: leave on legacy)
Still fetches `/data/master_index.json` (L92). **Phase-1 decision: leave analytics on the
legacy payload** (small blast radius; run-scoping deferred to Phase 2).
**Consequence: Cleanup C-3 must NOT delete `master_index.json` in Phase 1.**

### F-11 — Type-check + tests (via `frontend-dev`, per PF-1b)
```bash
docker compose exec frontend-dev npx tsc --noEmit
docker compose exec frontend-dev npx vitest run
```

---

## 6. Cleanup Workstream (C) — deferred until F validates
- **C-1**: Comment-only docstring fixes in `export_ui_payload.py` (remove `models/best_pipeline.pt` refs). No behaviour change.
- **C-2**: Update `NSTAR_MLOPS.md` + `.agents/plans/context_for_architect.md` (Architect §7 items 5–6).
- **C-3**: After manual + automated frontend verification, list (do **not** delete in Phase 1)
  `master_index.json` (still used by analytics — F-10), `nodes/*`, `ui_payload.json` for user sign-off.

---

## 7. Git / Commit Strategy (branch `database`)
Already on `database`. Each commit stages **only its own files** (D-4 — no `git add -A`).
Generated `runs/**` **is committed** to git for Vercel static serving (DEC-1).
1. `feat(backend): paginated payload generator + registry builder + batch script` (B-1..B-3)
2. *(B-4 smoke test — no commit; verify + size projection)*
3. `feat(data): generated paginated payloads + run registry` (full batch `runs/**` +
   `run_registry.json`) — **only after the size projection is confirmed acceptable (DEC-1 caveat)**
4. `feat(frontend): registry types + store fields` (F-1, F-2)
5. `feat(frontend): three-stage loading + day-scoped node fetch` (F-3, F-4)
6. `feat(frontend): model selector, date paginator, info overlay` (F-5..F-9)
7. `chore: w&b model cleanup docs` (C-1, C-2)

---

## 8. Execution Order (after approval)
```
PF-1b..PF-5
  → B-1, B-2, B-3 → [commit 1]
  → B-4 smoke (→ riot-smoke) + jq + du -sh size projection
  → full batch (generate_all_payloads.sh) → confirm size acceptable → [commit 3]
  → F-1, F-2 → [commit 4]
  → F-3, F-4 → tsc → [commit 5]
  → F-5..F-9 → tsc + vitest → [commit 6]
  → manual verify → C-1, C-2 → [commit 7] → C-3 sign-off
```

---

## 9. Open Questions — all resolved
- **OQ-2 (prod serving)** → **RESOLVED (DEC-1)**: commit `runs/**` to git; Vercel serves
  them statically. Size projection reported before the full-batch commit.
- **OQ-3 (`riot-smoke`)** → **RESOLVED (DEC-2)**: `riot-smoke` **project** under the
  production entity; no new entity created.
- **Prod entity** → **RESOLVED (DEC-3)**: `rappel-velvet9310-newcastle-university`.

---

---

## 10. Execution Log & Deviations (actual)

**Backend (B) — done.** `export_paginated_payload.py`, `build_run_registry.py`,
`generate_all_payloads.sh`, `upload_to_r2.py` written & committed (`a7d9f1c` for the first
three). Smoke test passed (run `dark-brook-37`/v9). JSON is **minified**.

**Phase-3 pivot (DEC-1):** size projection showed 10 models × 31 days ≈ 4.0 GB pretty /
2.1 GB minified in git → user chose **Cloudflare R2** instead. `runs/**` git-ignored;
`upload_to_r2.py` (boto3, gzip + `ContentEncoding: gzip`) pushes to R2; frontend fetches
from `R2_PUBLIC_URL` via `frontend/src/lib/payloadSource.ts` (`NEXT_PUBLIC_PAYLOAD_BASE_URL`
override). R2 upload verified end-to-end via the S3 API (`head_object` OK).
- ⚠️ **ACTION (user)**: the bucket's **public access (`r2.dev` URL) returns 403** until
  enabled in the Cloudflare dashboard (or a custom domain is attached). Data is uploaded
  and ready; the frontend can't fetch until this is on.

**Two batch failures found & fixed:**
- **D-5 — per-day location instability.** 2026-05-19 has **55** active sensors vs the
  canonical **56**; the original per-day 2-day loads + strict equality assert aborted 8/10
  runs. **Fix:** load **one dataset over `[start−1d, end)`** per run — a single sorted
  location set (union), one scaler, one static-feature matrix, one topology; the dataset's
  `reindex(...).fillna(0.0)` zero-fills absent sensors per day. Windows are grouped to a
  target day by `times[idx+seq_len].date()`. Full 32-day load peak ≈ 3.7 GB (cgroup=max).
- **D-6 — incompatible architectures (v5, v4).** Trained with `edge_dim: 10`
  (`encoder.gat.lin_edge.weight` `[32,10]`), but the current `ResilientIoTPipeline.__init__`
  has **no `edge_dim`** param → `state_dict` shape mismatch. These are an older arch variant
  the current code can't load (legacy script can't either). **Decision:** skip gracefully
  (try/except per artifact) → **8/10 models** generate (v0–v3, v6–v9).

**Frontend (F) — done (pending type-check).** All files written:
`lib/payloadSource.ts`, `types/registry.ts`, `store/useTimeStore.ts` (+pagination state),
`MapVisualizer.tsx` (three-stage R2 loading + `mergePayload`, run-switch race guard),
`NodeInspector.tsx` (day/run-scoped R2 node fetch), `ModelSelector.tsx`, `DatePaginator.tsx`,
`ModelInfoOverlay.tsx`, `TimeSlider.tsx`, `MapScene.tsx`. Stack is **Next 16 / React 19**;
components mirror existing patterns (no new Next API surface). `analytics/page.tsx` left on
legacy local `master_index.json` (F-10).
- ⚠️ **PF-1b unresolved**: `tsc`/`vitest` need the `frontend-dev` container (deps live only
  there; no docker CLI in agent-workspace, `frontend/node_modules` empty). **Type-check +
  tests must be run there**: `docker compose exec frontend-dev npx tsc --noEmit` and
  `… npx vitest run`. Did a careful manual TS review (fixed `React.*` type refs → named
  imports; removed unused `startDate`).

**Open items for user:**
1. Enable R2 public access (403 today). 2. Run frontend `tsc`/`vitest` in `frontend-dev`.
3. `NEXT_PUBLIC_PAYLOAD_BASE_URL` for Vercel (defaults to the `r2.dev` URL in code).
