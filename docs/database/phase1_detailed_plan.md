# Phase 1: Detailed Implementation Plan

> **Scope**: Multi-run paginated payloads, model selector, day pagination  
> **Strategy**: D — Hybrid CDN + Live API (non-realtime mode)  
> **Estimated effort**: 7–10 days  
> **Dependencies**: Existing W&B artifacts, existing `export_ui_payload.py`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Backend: Payload Generation](#2-backend-payload-generation)
3. [Frontend: New Components](#3-frontend-new-components)
4. [Frontend: Refactored Components](#4-frontend-refactored-components)
5. [Data Schemas](#5-data-schemas)
6. [UI Payload Generation Options](#6-ui-payload-generation-options)
7. [W&B Model Cleanup](#7-wb-model-cleanup)
8. [File Manifest](#8-file-manifest)
9. [Verification Plan](#9-verification-plan)

---

## 1. Architecture Overview

### Data Flow

```
                     W&B Artifact Registry
                     (7 model versions)
                            │
                            ▼
              ┌─────────────────────────────┐
              │  export_paginated_payload.py │
              │                             │
              │  For each model artifact:   │
              │  1. Download from W&B       │
              │  2. Run 30 days of          │
              │     inference (96 steps/day)│
              │  3. Write paginated JSON    │
              └──────────┬──────────────────┘
                         │
                         ▼
              frontend/public/data/runs/
              ├── run_registry.json
              ├── feasible-capybara-22/
              │   ├── spatial.json
              │   ├── 2026-06-01/
              │   │   ├── page.json
              │   │   └── nodes/node_{id}.json
              │   ├── 2026-06-02/
              │   │   ├── page.json
              │   │   └── nodes/...
              │   └── ...
              └── sunny-capybara-18/
                  └── ...
```

### Frontend Load Sequence

```
App Mount
  │
  ├──► fetch("/data/runs/run_registry.json")
  │    → Populate model selector dropdown
  │    → Select default run
  │
  ├──► fetch("/data/runs/{run_id}/spatial.json")
  │    → Set node coordinates, edges, semantic tags
  │    → Set available_dates array
  │    → Display model info overlay
  │
  ├──► fetch("/data/runs/{run_id}/{date}/page.json")
  │    → Set timeseries + attention for all nodes (24h)
  │    → Slider becomes interactive (96 in-memory timesteps)
  │
  └──► (on node click) fetch("/data/runs/{run_id}/{date}/nodes/node_{id}.json")
       → Populate NodeInspector with ZINB series, ego-graph, etc.
```

---

## 2. Backend: Payload Generation

### 2.1 Refactor: `export_paginated_payload.py`

> **Source**: Refactor of [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py)  
> **New file**: `src/export_paginated_payload.py`

#### New CLI Arguments

```python
parser.add_argument("--start_date", type=str, default="2026-06-01",
                    help="First day of the analysis window (inclusive)")
parser.add_argument("--end_date", type=str, default="2026-07-01",
                    help="Last day of the analysis window (exclusive)")
parser.add_argument("--output_dir", type=str, default="frontend/public/data/runs",
                    help="Root output directory for paginated payloads")
parser.add_argument("--artifact", type=str, default=None,
                    help="W&B artifact (e.g., entity/project/name:v3). "
                         "If not provided, prompts interactively.")
parser.add_argument("--all_artifacts", action="store_true",
                    help="Generate payloads for ALL W&B artifacts in the collection.")
```

#### Key Changes from Current Export Script

| Current (`export_ui_payload.py`) | New (`export_paginated_payload.py`) |
|---|---|
| Emits a single `master_index.json` containing all nodes + timeseries inline | Emits `spatial.json` (no timeseries) + per-day `page.json` (timeseries only) |
| Single `node_{id}.json` covering entire analysis period | Per-day `nodes/node_{id}.json` scoped to 96 timesteps |
| Fixed output paths (`MASTER_INDEX_PATH`, `NODES_DIR`) | Parameterised output directory: `{output_dir}/{run_id}/{date}/` |
| Default analysis: 2 days (2026-06-01 to 2026-06-03) | Default analysis: 30 days (2026-06-01 to 2026-07-01) |
| Hardcoded date handling (single "target day" logic) | Iterates over each date in the analysis window |
| No run registry output | Appends/updates `run_registry.json` with run metadata |

#### Per-Day Inference Loop

```python
# Pseudocode — iterates over each date in the analysis window
for date in date_range(start_date, end_date):
    # Load dataset for 2-day window: [date - 1 day, date + 1 day)
    # (provides seq_len look-back context for day-boundary predictions)
    dataset = RealIoTDataset(
        start_date=date - timedelta(days=1),
        end_date=date + timedelta(days=1),
        ...
    )
    
    # Run 96 inference steps for the target date
    for t in range(96):
        window_idx = calculate_window_index(t, seq_len, total_times)
        batch = build_batch(dataset[window_idx], simulated_static, ...)
        result = model(batch, return_analysis=True)
        accumulate_results(t, result)
    
    # Write per-day outputs
    write_page_json(output_dir, run_id, date, accumulated_results)
    write_node_detail_jsons(output_dir, run_id, date, accumulated_results)

# After all days: write spatial.json and update run_registry.json
write_spatial_json(output_dir, run_id, nodes_meta, edges_topology, metadata)
update_run_registry(output_dir, run_id, metadata, available_dates)
```

#### Decision: New File vs. In-Place Refactor

> [!IMPORTANT]
> **Recommendation: Create a new file** (`export_paginated_payload.py`) rather than modifying the existing `export_ui_payload.py` in-place.
> 
> Rationale:
> - The existing script works and is used for the current single-day prototype.
> - A new file allows parallel development without breaking the working frontend.
> - Once Phase 1 is validated, the old script can be deprecated.
> - The core inference logic (model loading, ZINB extraction, attention extraction) can be imported as shared functions from a common module.

### 2.2 New File: `configs/analysis.yaml`

Standardises the analysis holdout window across all model runs:

```yaml
analysis:
  start_date: "2026-06-01"
  end_date: "2026-07-01"      # 30 days, exclusive
  veh_class: "person"
  dir: "all"
  seq_len: 12
  num_simulated: 5
```

### 2.3 New Script: `scripts/build_run_registry.py`

Scans the output directory and builds `run_registry.json` from the `spatial.json` metadata in each run subdirectory:

```python
def build_registry(runs_dir: str) -> dict:
    """Scan runs_dir for subdirectories containing spatial.json and build the registry."""
    registry = {"default_run_id": None, "runs": []}
    for run_dir in sorted(Path(runs_dir).iterdir()):
        spatial_path = run_dir / "spatial.json"
        if not spatial_path.exists():
            continue
        spatial = json.loads(spatial_path.read_text())
        meta = spatial["metadata"]
        dates = sorted([d.name for d in run_dir.iterdir() if d.is_dir()])
        registry["runs"].append({
            "run_id": run_dir.name,
            "wandb_run_id": meta.get("wandb_run_id", "unknown"),
            "wandb_run_name": run_dir.name,
            "git_commit": meta.get("git_commit_hash", "unknown"),
            "generated_at": meta.get("generated_at", "unknown"),
            "analysis_period": {"start_date": dates[0], "end_date": dates[-1]},
            "train_config": meta.get("hyperparameters", {}),
            "metrics_summary": meta.get("metrics_summary", {}),
            "available_dates": dates,
            "num_nodes": meta.get("num_nodes", 0),
            "num_simulated_nodes": meta.get("num_simulated_nodes", 0),
        })
    if registry["runs"]:
        registry["default_run_id"] = registry["runs"][0]["run_id"]
    return registry
```

---

## 3. Frontend: New Components

### 3.1 `ModelSelector.tsx`

A dropdown or slide-out panel anchored to the top-left of the map (below the existing analytics button).

**Data source**: `/data/runs/run_registry.json`  
**Zustand state**: `activeRunId`

```
┌──────────────────────────────┐
│  ▼ feasible-capybara-22      │  ← Dropdown trigger
│  ────────────────────────── │
│  • sunny-capybara-18         │
│  • wild-monkey-20            │
│  • sparkling-thunder-16      │
│  • comfy-dew-19              │
│  • hopeful-waterfall-17      │
│  • (model v6)                │
└──────────────────────────────┘
```

**Behaviour**:
- On selection change → fetch new `spatial.json` → fetch first available `page.json`
- Show a brief loading skeleton during transition
- Persist `activeRunId` in URL query param (`?run=feasible-capybara-22`) for shareability

### 3.2 `ModelInfoOverlay.tsx`

A formatted card displayed when the user clicks an "info" icon next to the model selector. Similar aesthetic to `NodeInspector.tsx` (glassmorphic dark panel).

**Data source**: `spatial.json` metadata + `run_registry.json` entry

```
┌────────────────────────────────────────┐
│  feasible-capybara-22           ✕     │
│  ──────────────────────────────────── │
│                                        │
│  TRAINING                              │
│  Period    2024-04-02 → 2025-04-02     │
│  Epochs    100 (early stop @ 47)       │
│  Batch     32                          │
│  LR        3e-4                        │
│                                        │
│  ARCHITECTURE                          │
│  Hidden    32                          │
│  Layers    2                           │
│  Heads     4                           │
│  Embed     16                          │
│  Dropout   0.1                         │
│                                        │
│  VALIDATION                            │
│  RMSE      12.4                        │
│  MAE       8.1                         │
│  Zero Acc  0.83                        │
│                                        │
│  ANALYSIS                              │
│  Period    2026-06-01 → 2026-06-30     │
│  Nodes     71 (66 real + 5 sim)        │
│                                        │
│  [View on W&B ↗]                       │
└────────────────────────────────────────┘
```

### 3.3 `DatePaginator.tsx`

A navigation control integrated into the `TimeSlider` bar at the bottom of the screen.

```
┌─────────────────────────────────────────────────────────────────┐
│  ◀ prev   June 3, 2026   next ▶   │   ▶ Play   ═══●═══   96  │
└─────────────────────────────────────────────────────────────────┘
```

**Behaviour**:
- `← prev` / `→ next` buttons navigate between available dates
- Clicking the date label opens a date picker (optional — can be a simple calendar dropdown)
- On date change → fetch new `page.json` → reset slider to t=0
- Disable prev/next when at boundary dates
- Show loading spinner during fetch

---

## 4. Frontend: Refactored Components

### 4.1 `useTimeStore.ts` — New State Fields

```typescript
interface TimeState {
  // Existing fields
  currentTimestep: number;
  maxTimesteps: number;
  isPlaying: boolean;
  selectedNodeId: number | null;
  globalAttentionThreshold: number;
  egoGraphEdges: AttentionEdge[] | null;
  
  // NEW: Multi-run + pagination state
  activeRunId: string | null;
  activeDate: string | null;           // "2026-06-01"
  availableDates: string[];            // sorted array from spatial.json
  runRegistry: RunRegistryEntry[];     // from run_registry.json
  isLoadingPage: boolean;              // true during page.json fetch
  
  // NEW: Actions
  setActiveRunId: (runId: string) => void;
  setActiveDate: (date: string) => void;
  setAvailableDates: (dates: string[]) => void;
  setRunRegistry: (runs: RunRegistryEntry[]) => void;
  setIsLoadingPage: (loading: boolean) => void;
  navigateDay: (direction: 'prev' | 'next') => void;
}
```

### 4.2 `MapVisualizer.tsx` — Data Loading Refactor

**Current**: Fetches `master_index.json` once on map load, stores in `payloadRef`.

**New**: Three-stage loading:

```typescript
// Stage 1: On mount — fetch run_registry.json
useEffect(() => {
  fetch("/data/runs/run_registry.json")
    .then(r => r.json())
    .then(registry => {
      setRunRegistry(registry.runs);
      setActiveRunId(registry.default_run_id);
    });
}, []);

// Stage 2: On run change — fetch spatial.json
useEffect(() => {
  if (!activeRunId) return;
  fetch(`/data/runs/${activeRunId}/spatial.json`)
    .then(r => r.json())
    .then(spatial => {
      // Store node positions, edges, metadata
      spatialRef.current = spatial;
      setAvailableDates(spatial.metadata.available_dates);
      setActiveDate(spatial.metadata.available_dates[0]);
    });
}, [activeRunId]);

// Stage 3: On date change — fetch page.json
useEffect(() => {
  if (!activeRunId || !activeDate) return;
  setIsLoadingPage(true);
  fetch(`/data/runs/${activeRunId}/${activeDate}/page.json`)
    .then(r => r.json())
    .then(page => {
      // Merge spatial coords with temporal timeseries → build UIPayload
      payloadRef.current = mergePayload(spatialRef.current, page);
      setMaxTimesteps(page.timesteps - 1);
      setTimestep(0);
      setIsLoadingPage(false);
      rebuildLayers();
    });
}, [activeRunId, activeDate]);
```

### 4.3 `NodeInspector.tsx` — Day-Scoped Fetch

**Current**: `fetch(/data/nodes/node_${id}.json)`  
**New**: `fetch(/data/runs/${activeRunId}/${activeDate}/nodes/node_${id}.json)`

```typescript
useEffect(() => {
  if (selectedNodeId === null || !activeRunId || !activeDate) {
    setNodeDetail(null);
    return;
  }
  fetch(`/data/runs/${activeRunId}/${activeDate}/nodes/node_${selectedNodeId}.json`)
    .then(r => r.ok ? r.json() : null)
    .then(d => {
      setNodeDetail(d);
      setEgoGraphEdges(d?.ego_graph_edges ?? null);
    });
}, [selectedNodeId, activeRunId, activeDate]);
```

### 4.4 `MapScene.tsx` — New Component Composition

```tsx
export default function MapScene() {
  return (
    <main className="relative h-screen w-screen overflow-hidden">
      <MapVisualizer />
      <NodeInspector />
      <ModelSelector />         {/* NEW */}
      <ModelInfoOverlay />      {/* NEW */}
      <TimeSlider />            {/* Updated with DatePaginator */}
      <AttentionControls />

      {/* Existing analytics button */}
      <Link href="/analytics" ...>...</Link>
    </main>
  );
}
```

---

## 5. Data Schemas

### 5.1 `run_registry.json`

```typescript
interface RunRegistry {
  default_run_id: string;
  runs: RunRegistryEntry[];
}

interface RunRegistryEntry {
  run_id: string;
  wandb_run_id: string;
  wandb_run_name: string;
  git_commit: string;
  generated_at: string;           // ISO 8601
  analysis_period: {
    start_date: string;           // "2026-06-01"
    end_date: string;             // "2026-06-30"
  };
  train_config: {
    data?: Record<string, unknown>;
    model?: Record<string, unknown>;
    training?: Record<string, unknown>;
  };
  metrics_summary: {
    val_rmse?: number;
    val_mae?: number;
    val_zero_acc?: number;
  };
  available_dates: string[];      // sorted, e.g. ["2026-06-01", "2026-06-02", ...]
  num_nodes: number;
  num_simulated_nodes: number;
}
```

### 5.2 `spatial.json` (per run)

```typescript
interface SpatialIndex {
  metadata: {
    run_id: string;
    wandb_run_id: string;
    git_commit_hash: string;
    generated_at: string;
    hyperparameters: Record<string, unknown>;
    num_nodes: number;
    num_real_nodes: number;
    num_simulated_nodes: number;
    available_dates: string[];
    veh_class: string;
    dir: string;
    metrics_summary?: Record<string, number>;
  };
  nodes: SpatialNode[];
  edge_topology: EdgeTopology[];  // static edge index (source, target) without attention
}

interface SpatialNode {
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  is_simulated: boolean;
  semantic_tags: string[];
}

interface EdgeTopology {
  source: number;
  target: number;
}
```

### 5.3 `page.json` (per day)

```typescript
interface TemporalPage {
  date: string;                   // "2026-06-01"
  timesteps: number;              // 96
  seq_len: number;
  
  // Per-node timeseries: node_timeseries[node_id][t] = { prediction, uncertainty }
  node_timeseries: {
    [node_id: number]: TimestepEntry[];
  };
  
  // Per-edge attention: edge_attention[(src,tgt)] = number[]
  edge_attention: {
    source: number;
    target: number;
    attention_sequence: number[];
  }[];
  
  // Global calibration metrics for this day
  pit_histogram?: PITBin[];
  zero_inflation_reliability?: ZeroInflationBin[];
  global_imputation_burden?: number[];
  global_healing_quality?: number[];
  attention_entropy_series?: number[];
  tripwire_audit_trail?: Record<string, unknown>[];
  alpha_node_leaderboard?: Record<string, unknown>[];
}
```

### 5.4 `node_{id}.json` (per day, per node)

Same as existing `NodeDetailPayload` type — no schema change needed. Only the fetch URL changes.

---

## 6. UI Payload Generation Options

Three options for how to run the export for all 7 models:

### Option A: Manual Sequential (Simplest)

Run `export_paginated_payload.py` manually for each model, one at a time:

```bash
docker compose exec agent-workspace python src/export_paginated_payload.py \
  --artifact entity/resilient-iot/resilient-iot-model:v0 \
  --start_date 2026-06-01 --end_date 2026-07-01

# Repeat for v1, v2, ... v6
```

**Pros**: Maximum control. Easy to debug per-run.  
**Cons**: Manual. ~30 min per model × 7 models = ~3.5 hours of execution.

### Option B: Batch Script (Recommended for Phase 1)

```bash
# scripts/generate_all_payloads.sh
#!/bin/bash
set -euo pipefail

CONFIG="configs/analysis.yaml"
OUTPUT_DIR="frontend/public/data/runs"

# Auto-discover all artifacts from W&B
python src/export_paginated_payload.py \
  --all_artifacts \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR"

# Build the registry
python scripts/build_run_registry.py "$OUTPUT_DIR"

echo "Done. Run 'du -sh $OUTPUT_DIR' to check total size."
```

**Pros**: One command. Automated. Idempotent.  
**Cons**: Long-running. If one model fails, need retry logic.

### Option C: Integrated into `train.py` Post-Training Hook (Future)

After training completes and the W&B artifact is uploaded, automatically generate the paginated payload for that run:

```python
# At the end of train.py, after wandb.log_artifact():
if args.auto_export:
    from export_paginated_payload import generate_for_artifact
    generate_for_artifact(
        artifact_name=f"entity/resilient-iot/resilient-iot-model:{artifact.version}",
        output_dir="frontend/public/data/runs/",
        config_path="configs/analysis.yaml",
    )
```

**Pros**: Zero manual steps. New models appear in the frontend automatically.  
**Cons**: Adds ~30 min to training time. Requires standardised analysis data on GPU server. Only useful once the pipeline is mature.

> [!IMPORTANT]
> **Recommendation for Phase 1: Option B** (batch script). It's automated enough to be practical, while keeping the export decoupled from training. Option C can be adopted in Phase 2+ once the workflow is proven.

---

## 7. W&B Model Cleanup

### Immediate Actions

| # | Task | Effort |
|---|---|---|
| 1 | Verify all 7 W&B artifacts have `best_model.pt` + `config.yaml` | 30 min |
| 2 | Test `select_artifact_from_wandb()` lists all 7 correctly | 15 min |
| 3 | Remove `CHECKPOINT_PATH` constant and update comments in `export_ui_payload.py` lines 17–18, 64–67 | 10 min |
| 4 | Run `rm -rf models/resilient-iot/ models/dummy/ models/local/` on local machines | 5 min |
| 5 | Update `NSTAR_MLOPS.md` section on model checkpointing | 15 min |
| 6 | Update `.agents/plans/context_for_architect.md` to remove local checkpoint reference | 5 min |

### Files Referencing Local Models

Found via `grep -r "models/" src/`:

| File | Line | Reference | Action |
|---|---|---|---|
| `export_ui_payload.py:18` | `--checkpoint models/best_pipeline.pt` | Update docstring |
| `export_ui_payload.py:65` | `Derived from models/best_pipeline.pt weight shapes` | Update comment |
| `train.py:130` | `models/resilient-iot-sweeps/{study_name}/` | Keep — this is the local training write path |
| `train.py:137` | `models/{project}/{run_name}/` | Keep — needed during training |
| `train.py:141` | `models/local/{timestamp}/` | Keep — local fallback |

> [!NOTE]
> The `train.py` references are correct — it *should* write locally during training, then upload to W&B. The cleanup is about *consuming* models for inference/export, not *producing* them.

---

## 8. File Manifest

### New Files

| File | Purpose |
|---|---|
| `src/export_paginated_payload.py` | Paginated payload generator (new script) |
| `configs/analysis.yaml` | Standardised analysis period |
| `scripts/generate_all_payloads.sh` | Batch generation for all W&B artifacts |
| `scripts/build_run_registry.py` | Builds `run_registry.json` from output dirs |
| `frontend/src/components/ModelSelector.tsx` | Run selector dropdown |
| `frontend/src/components/ModelInfoOverlay.tsx` | Run metadata display card |
| `frontend/src/components/DatePaginator.tsx` | Day navigation control |
| `frontend/src/types/registry.ts` | TypeScript types for run registry & spatial index |
| `frontend/public/data/runs/run_registry.json` | Generated: run registry |
| `frontend/public/data/runs/{run_id}/spatial.json` | Generated: per-run spatial data |
| `frontend/public/data/runs/{run_id}/{date}/page.json` | Generated: per-day temporal page |
| `frontend/public/data/runs/{run_id}/{date}/nodes/*.json` | Generated: per-day node details |

### Modified Files

| File | Change |
|---|---|
| `frontend/src/store/useTimeStore.ts` | Add `activeRunId`, `activeDate`, `availableDates`, `runRegistry`, `isLoadingPage` |
| `frontend/src/components/MapVisualizer.tsx` | Three-stage loading (registry → spatial → page) |
| `frontend/src/components/MapScene.tsx` | Add `ModelSelector`, `ModelInfoOverlay` to composition |
| `frontend/src/components/NodeInspector.tsx` | Day-scoped and run-scoped fetch URL |
| `frontend/src/components/TimeSlider.tsx` | Integrate `DatePaginator` |
| `frontend/src/types/payload.ts` | Add `SpatialIndex`, `TemporalPage`, `RunRegistry` types |

### Deprecated (After Validation)

| File | Status |
|---|---|
| `frontend/public/data/master_index.json` | Replaced by `runs/{run_id}/spatial.json` + `page.json` |
| `frontend/public/data/nodes/*.json` | Replaced by `runs/{run_id}/{date}/nodes/*.json` |
| `frontend/public/data/ui_payload.json` | Legacy — already superseded by `master_index.json` |

---

## 9. Verification Plan

### Automated Tests

```bash
# 1. Generate paginated payload for one model (smoke test)
docker compose exec agent-workspace python src/export_paginated_payload.py \
  --artifact entity/resilient-iot/resilient-iot-model:latest \
  --start_date 2026-06-01 --end_date 2026-06-03 \
  --output_dir frontend/public/data/runs/

# 2. Verify file structure
ls frontend/public/data/runs/*/spatial.json
ls frontend/public/data/runs/*/2026-06-01/page.json
ls frontend/public/data/runs/*/2026-06-01/nodes/node_0.json

# 3. Validate JSON schema (via jq)
jq '.metadata.num_nodes' frontend/public/data/runs/*/spatial.json
jq '.timesteps' frontend/public/data/runs/*/2026-06-01/page.json

# 4. Run frontend type-check
docker compose exec frontend-dev npx tsc --noEmit

# 5. Run frontend tests
docker compose exec frontend-dev npx vitest run
```

### Manual Verification

| Check | Expected Result |
|---|---|
| Load the digital twin in browser | Model selector dropdown appears with available runs |
| Select a different model | Map updates with new node positions/predictions |
| Click "Model Info" icon | Overlay card shows training config, metrics, W&B link |
| Click "Next Day" | Page.json loads, slider resets to t=0, timeseries updates |
| Click "Previous Day" | Returns to previous day's data |
| Drag slider | 60fps scrubbing with no network calls (check DevTools Network tab) |
| Click a node | NodeInspector loads day-scoped `node_{id}.json` with ZINB series |
| Navigate to /analytics | Analytics page still functions (may need run-scoping in future) |
| Check Vercel deploy size | Total `runs/` directory < 500 MB |

### Performance Targets

| Metric | Target | How to Measure |
|---|---|---|
| Spatial index load | < 100ms | DevTools Network timing |
| Page turn (day change) | < 600ms total | DevTools Network + Zustand timing |
| Slider scrubbing FPS | ≥ 55fps | Chrome Performance panel |
| Node detail load | < 200ms | DevTools Network timing |
| Time to interactive | < 2s from page load | Lighthouse |

---

> **Next step for @architect**: Review this plan, particularly the schema decisions (§5) and the payload generation approach (§6). Once approved, create the implementation checklist for @implementer.
