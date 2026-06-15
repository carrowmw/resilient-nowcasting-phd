# Database Serving Strategy: Hybrid CDN + Live API (Strategy D)

> **Author**: Interrogator Agent  
> **Date**: 2026-06-12 (v2 — streamlined to approved strategy)  
> **Status**: Approved Strategy — Implementation Planning  
> **Decision**: Strategy D adopted. Strategies A/B/C archived.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture Baseline](#2-current-architecture-baseline)
3. [Architecture: Hybrid CDN + Live API](#3-architecture-hybrid-cdn--live-api)
4. [Latency Budget Analysis](#4-latency-budget-analysis)
5. [Pagination Strategy](#5-pagination-strategy)
6. [Multi-Model Run Architecture](#6-multi-model-run-architecture)
7. [W&B as Model Source of Truth](#7-wb-as-model-source-of-truth)
8. [Analysis Period & Payload Generation](#8-analysis-period--payload-generation)
9. [Near Real-Time Data Flow (Future)](#9-near-real-time-data-flow-future)
10. [CDN & Edge Caching Strategy](#10-cdn--edge-caching-strategy)
11. [Risk Register](#11-risk-register)

---

## 1. Problem Statement

The Geospatial Digital Twin frontend is evolving beyond its Phase 1 static prototype. Three new requirements break the existing `ui_payload.json` static file bridge:

| Requirement | Impact on Current Architecture |
|---|---|
| **A) Multi-Model Run Comparison** | 7 W&B model artifacts exist. Each produces ~29 MB of payload files. Serving N runs from static files means N × 29 MB. Users need to select and compare runs in the UI. |
| **B) Extended Paginated Timeseries** | The current 96-timestep (24h) payload covers a single day. Expanding to 30 days of analysis data inflates `master_index.json` beyond browser memory limits. |
| **C) Near Real-Time Inference** | Live sensor data must eventually be ingested and served. Static files cannot support this, but it is **not** the immediate priority. |

---

## 2. Current Architecture Baseline

```
┌──────────────────┐        git push         ┌──────────────────┐
│   GPU Server     │ ──────────────────────► │   GitHub Repo    │
│   (Training)     │    static JSON files     │                  │
└──────────────────┘                         └────────┬─────────┘
                                                      │
                                                      │ auto-deploy
                                                      ▼
                                             ┌──────────────────┐
                                             │  Vercel (CDN)    │
                                             │  Next.js SSG     │
                                             └────────┬─────────┘
                                                      │
                                                      │ fetch("/data/master_index.json")
                                                      │ fetch("/data/nodes/node_{id}.json")
                                                      ▼
                                             ┌──────────────────┐
                                             │  User Browser    │
                                             │  Deck.gl WebGL   │
                                             └──────────────────┘
```

### Current Payload Sizes (Measured 2026-06-12)

| Asset | Size | Load Pattern |
|---|---|---|
| `master_index.json` | 8.16 MB | Fetched once on page load |
| `node_{id}.json` (×66 files) | ~290 KB each (21 MB total) | Fetched lazily on node click |
| Total per model run | ~29 MB | — |

### Current Performance Characteristics

- **Initial load**: Single 8 MB fetch from Vercel CDN → ~200ms (gzipped to ~1.5 MB).
- **Node inspection**: 290 KB lazy fetch → < 100ms from CDN edge.
- **Slider scrubbing**: Zero network traffic — all 96 timesteps in browser memory. 60fps WebGL.

This architecture is extremely fast because **all data is pre-computed and edge-cached**. Strategy D preserves this for historical data.

---

## 3. Architecture: Hybrid CDN + Live API

### Core Principle

Separate the **immutable historical analysis** path (CDN-cached static JSON) from the **mutable live inference** path (API on GPU server). Historical data is always available even when the GPU server is offline.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           GPU SERVER                                 │
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐ │
│  │  train.py     │    │  PostgreSQL +    │    │  FastAPI Inference │ │
│  │  (W&B logs)   │───►│  TimescaleDB    │◄───│  Server (Phase 3)  │ │
│  └──────┬───────┘    │  (Training DB)   │    └────────────────────┘ │
│         │            └──────────────────┘                            │
│         │                                                            │
│         │  W&B Artifact                                              │
│         ▼                                                            │
│  ┌─────────────────────────────────────────┐                        │
│  │  export_paginated_payload.py            │                        │
│  │  (Downloads model from W&B → generates  │                        │
│  │   paginated JSON per run, per day)      │                        │
│  └──────────────┬──────────────────────────┘                        │
│                 │                                                    │
└─────────────────┼────────────────────────────────────────────────────┘
                  │
                  │  Upload (R2 / Vercel Blob / git push)
                  ▼
        ┌─────────────────────────┐
        │  CDN Static Storage     │
        │  (Cloudflare R2 or      │
        │   Vercel /public/data)  │
        │                         │
        │  runs/                  │
        │  ├── run_registry.json  │
        │  ├── {run_id}/          │
        │  │   ├── spatial.json   │ ← loaded ONCE per run (coords, tags, metadata)
        │  │   ├── {date}/        │
        │  │   │   ├── page.json  │ ← loaded per day-page (96 timesteps of all nodes)
        │  │   │   └── nodes/     │
        │  │   │       └── *.json │ ← loaded lazily on node click
        │  │   └── ...            │
        │  └── ...                │
        └────────┬────────────────┘
                 │
                 │ fetch()
                 ▼
        ┌──────────────────┐
        │  Vercel (Next.js) │
        │  Browser / Deck.gl│
        └──────────────────┘
```

### Why This Strategy

| Property | How Strategy D Delivers |
|---|---|
| **Slider reactivity** | All 96 timesteps for the current day are in browser memory. Zero network calls during scrubbing. 60fps guaranteed. |
| **Page turn latency** | Single ~1 MB (gzipped) fetch from CDN edge → 200–400ms. |
| **Multi-run comparison** | Each run is a namespaced directory. Frontend loads `run_registry.json` to populate a model selector. |
| **Graceful degradation** | CDN serves history even when GPU server is offline. |
| **Future real-time** | A `live/` namespace can be added later for CRON-pushed live inference pages. Same JSON schema. |
| **Cost** | Cloudflare R2: $0.015/GB stored, $0 egress. Or free via Vercel `/public` until sizes warrant R2. |

---

## 4. Latency Budget Analysis

### The Slider Interaction Model

```
User drags slider ──► Browser reads in-memory array[t] ──► Deck.gl re-renders
                      ↑
                      │ This MUST be < 16ms (60fps)
                      │ ALL data for the current 24h page must be in browser memory
```

**Critical constraint**: The slider cannot make network requests per timestep.

### Pagination Network Budget

| Scenario | Raw Size | Gzipped | CDN Delivery |
|---|---|---|---|
| 24h (96 steps), current master_index format | 8 MB | ~1.5 MB | < 200ms |
| 24h (96 steps), restructured per-day page | ~5 MB | ~1 MB | < 200ms |
| 7d (672 steps), single payload | ~56 MB | ~10 MB | Too large for one load |

**Conclusion**: 24 hours is the natural pagination unit. The user's "page turn" mental model naturally accommodates a 300–600ms load:

```
1. Browser shows loading skeleton        0ms
2. Fetch 24h page.json from CDN     200–400ms
3. Parse JSON, update Zustand        50–100ms
4. Deck.gl re-renders first frame      ~16ms
                                  ────────────
                            Total: 300–600ms
```

---

## 5. Pagination Strategy

### Payload Restructuring

The current `master_index.json` embeds full timeseries inline per node. For pagination, we separate **spatial metadata** (loaded once per run) from **temporal data** (loaded per day):

```
┌──────────────────────────────────────┐
│         spatial.json                  │
│  Loaded ONCE per run switch          │
│  (nodes, edges, coordinates,         │
│   semantic_tags, model metadata,     │
│   available_dates[], run config)     │
│  Size: ~200 KB (gzipped ~40 KB)      │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│      {date}/page.json (24h)          │
│  Loaded per day navigation           │
│  { date, timesteps: 96,             │
│    node_timeseries: [...],           │
│    edge_attention: [...],            │
│    pit_histogram, reliability,       │
│    imputation_burden, entropy }      │
│  Size: ~5 MB (gzipped ~1 MB)         │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│   {date}/nodes/node_{id}.json        │
│  Loaded lazily on node click         │
│  { zinb_series, ground_truth,        │
│    imputed, mask, ego_graph }        │
│  Size: ~100 KB (gzipped ~20 KB)      │
└──────────────────────────────────────┘
```

### File Layout on CDN

```
/data/
  └── runs/
      ├── run_registry.json              ← loaded on app mount
      ├── feasible-capybara-22/
      │   ├── spatial.json               ← loaded on run selection
      │   ├── 2026-06-01/
      │   │   ├── page.json              ← loaded on day selection
      │   │   └── nodes/
      │   │       ├── node_0.json        ← loaded on node click
      │   │       ├── node_1.json
      │   │       └── ...
      │   ├── 2026-06-02/
      │   │   ├── page.json
      │   │   └── nodes/...
      │   └── ... (up to 30 days)
      └── sunny-capybara-18/
          └── ...
```

---

## 6. Multi-Model Run Architecture

### Run Registry Schema

A `run_registry.json` file lists all available model runs with summary metadata. This is the first file the frontend loads.

```json
{
  "default_run_id": "feasible-capybara-22",
  "runs": [
    {
      "run_id": "feasible-capybara-22",
      "wandb_run_id": "abc123",
      "wandb_run_name": "feasible-capybara-22",
      "git_commit": "e4f5a6b",
      "generated_at": "2026-06-12T14:00:00Z",
      "analysis_period": {
        "start_date": "2026-06-01",
        "end_date": "2026-06-30"
      },
      "train_config": {
        "data": { "start_date": "2024-04-02", "end_date": "2025-04-02", "seq_len": 12 },
        "model": { "embedding_dim": 16, "hidden_dim": 32, "num_layers": 2, "num_heads": 4 },
        "training": { "epochs": 100, "batch_size": 32, "lr": 0.0003 }
      },
      "metrics_summary": {
        "val_rmse": 12.4,
        "val_mae": 8.1,
        "val_zero_acc": 0.83
      },
      "available_dates": ["2026-06-01", "2026-06-02", "..."],
      "num_nodes": 71,
      "num_simulated_nodes": 5
    }
  ]
}
```

### Frontend Model Info Overlay

When a user selects a model run, a formatted overlay panel displays:

- **Run name** (W&B alias) and generation timestamp
- **Training period** vs. **Analysis period** (critical distinction)
- **Architecture summary**: hidden_dim, num_layers, num_heads, etc.
- **Validation metrics**: RMSE, MAE, Zero Accuracy
- **Available date range** for pagination
- **W&B link** (direct clickable to the run dashboard)

---

## 7. W&B as Model Source of Truth

### Decision: Remove Local `models/` Directory

Currently, `train.py` saves checkpoints locally to `models/{project}/{run_name}/` and then uploads them to W&B as artifacts. This creates two sources of truth.

**Proposal**: Make W&B artifacts the **sole** source of truth for model checkpoints.

| Action | Detail |
|---|---|
| **Keep** | `train.py` still saves locally during training (needed for checkpointing / early stopping). |
| **Keep** | W&B artifact upload at end of training (`wandb.log_artifact`). |
| **Remove** | References to hardcoded `CHECKPOINT_PATH` in `export_ui_payload.py`. ✅ *Already done — checkpoint is now `None` by default.* |
| **Remove** | `models/` directory contents from the repo working tree. The directory is already `.gitignore`d but accumulates locally. |
| **Add** | `select_artifact_from_wandb()` as the default model resolution path. ✅ *Already implemented.* |
| **Clean up** | Comments referencing `models/best_pipeline.pt` in `export_ui_payload.py` (lines 17–18, 64–67). |

### Current W&B Artifact Inventory

Based on `select_artifact_from_wandb()`, the collection `resilient-iot-model` in the `resilient-iot` project contains 7 versioned model artifacts. Each includes:
- `best_model.pt` — the frozen state dict
- `config.yaml` — co-located training hyperparameters

### Local Models Still on Disk

```
models/resilient-iot/sparkling-thunder-16/   (config.yaml + best_pipeline.pt)
models/resilient-iot/feasible-capybara-22/   (config.yaml + best_model.pt)
models/resilient-iot/sunny-capybara-18/      (config.yaml + best_model.pt)
models/resilient-iot/wild-monkey-20/         (config.yaml + best_model.pt)
models/resilient-iot/hopeful-waterfall-17/   (best_model.pt only)
models/resilient-iot/comfy-dew-19/           (config.yaml + best_model.pt)
models/dummy/dummy-gmimgf2e/                 (test artifact)
models/local/*/                              (3 local-only runs)
Total: 2.6 MB — safe to delete once W&B confirmed complete.
```

---

## 8. Analysis Period & Payload Generation

### The Core Question

> *Should the analysis period be a parameter in train.py (so we get train.py → model → export automatically), or should validation be standardised for all models?*

### Recommendation: Standardised Analysis Period, Decoupled from Training

**The analysis period should NOT be a training parameter.** It should be a separate, standardised configuration used exclusively by `export_ui_payload.py`:

| Argument | Rationale |
|---|---|
| **Against coupling to train.py** | Models are trained on different date ranges (e.g., 2024-04-02 to 2025-04-02). The analysis/holdout period must be a **disjoint** future window to demonstrate generalisation, not memorisation. Coupling it to training creates a temptation to visualise training data. |
| **For standardisation** | All 7 models should be evaluated on the **same** analysis period for fair comparison. This makes the run selector meaningful — users compare apples to apples. |
| **For decoupling** | The export script already accepts `--start_date` and `--end_date` independently. The current defaults are `2026-06-01` to `2026-06-03`. Extending to a full month (`2026-06-01` to `2026-07-01`) produces 30 day-pages. |

### Proposed Standard Analysis Period

```yaml
# configs/analysis.yaml (NEW — standardises the holdout window)
analysis:
  start_date: "2026-06-01"
  end_date: "2026-07-01"      # 30 days
  veh_class: "person"
  dir: "all"
  num_simulated: 5
```

### Batch Generation Workflow

A new script `scripts/generate_all_payloads.sh` iterates over all W&B artifacts and generates paginated payloads for each:

```bash
#!/bin/bash
# Generate paginated payloads for all W&B model artifacts
ARTIFACTS=(
  "entity/resilient-iot/resilient-iot-model:v0"
  "entity/resilient-iot/resilient-iot-model:v1"
  "entity/resilient-iot/resilient-iot-model:v2"
  # ... all 7 versions
)

for artifact in "${ARTIFACTS[@]}"; do
  python src/export_paginated_payload.py \
    --artifact "$artifact" \
    --start_date 2026-06-01 \
    --end_date 2026-07-01 \
    --output_dir frontend/public/data/runs/
done

# Generate run_registry.json from the output directories
python scripts/build_run_registry.py frontend/public/data/runs/
```

---

## 9. Near Real-Time Data Flow (Future)

> **Status**: Designed but **not** implemented in Phase 1. The architecture is ready for it.

When ready, a `live/` run namespace is added to the CDN alongside historical runs. A 15-minute CRON job on the GPU server:

1. Fetches latest 12 sensor readings from TimescaleDB
2. Runs a forward pass through the frozen model
3. Formats the result as a standard `page.json` 
4. Uploads to R2, replacing the previous live page

The frontend treats `live` as just another run in the registry, with a special "LIVE" badge and 5-minute polling via SWR. The slider still operates on in-memory data — no WebSocket complexity.

```
/data/runs/live/
  ├── spatial.json         (same nodes/edges as historical)
  └── today/
      ├── page.json        (updated every 15 min)
      └── nodes/...        (optional — only if node detail is needed for live)
```

---

## 10. CDN & Edge Caching Strategy

### Cache-Control Headers

```
# Run registry (changes when new runs are exported): short cache
Cache-Control: public, max-age=300, s-maxage=600, stale-while-revalidate=3600

# Spatial index per run (immutable once generated): cache forever
Cache-Control: public, max-age=31536000, immutable

# Historical day pages (immutable): cache forever
Cache-Control: public, max-age=31536000, immutable

# Live day page (future — updates every 15 min):
Cache-Control: public, max-age=60, s-maxage=300, stale-while-revalidate=900
```

### Compression

JSON payloads compress 6–7× with Brotli:

| Payload | Raw | Brotli | Transfer Time (50 Mbps) |
|---|---|---|---|
| `spatial.json` (~200 KB) | 200 KB | ~35 KB | < 10ms |
| `page.json` (~5 MB) | 5 MB | ~850 KB | ~140ms |
| `node_{id}.json` (~290 KB) | 290 KB | ~45 KB | < 10ms |

### Transition Strategy: Vercel /public First, R2 Later

For immediate implementation, paginated payloads can be served from `frontend/public/data/runs/` — Vercel serves these as static assets with automatic CDN distribution. When the total data size exceeds Vercel's comfortable limits (~500 MB), migrate to Cloudflare R2 with a Next.js rewrite rule.

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 30 days × 7 runs exceeds Vercel size limits | Medium | Medium | Migrate to R2 when total surpasses ~500 MB. R2 has $0 egress. |
| Different models have incompatible node counts | High | Medium | `spatial.json` per run captures the exact node topology. Frontend adapts. |
| `export_ui_payload.py` fails on older model architectures | Medium | Low | `load_state_dict(strict=False)` already handles mismatched keys (implemented). |
| Analysis data unavailable for standardised period | Low | High | Ensure Parquet data covers June 2026. Fall back to latest available data. |
| Browser memory pressure with 96-timestep × 71-node page | Low | Low | Current 8 MB master_index works. Restructured page.json will be smaller (~5 MB). |
| W&B API rate limiting during batch generation | Low | Low | Rate-limit calls in batch script. Cache downloaded artifacts locally. |

---

> **Next Steps**: See [implementation_phases.md](file:///Users/administrator/Code/ml-agent-workspace/docs/database/implementation_phases.md) for the phased roadmap and [phase1_detailed_plan.md](file:///Users/administrator/Code/ml-agent-workspace/docs/database/phase1_detailed_plan.md) for the full Phase 1 implementation specification.
