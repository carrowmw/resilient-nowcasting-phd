# Database Serving Strategy: Training Pipeline & Frontend Digital Twin

> **Author**: Interrogator Agent  
> **Date**: 2026-06-10  
> **Status**: Draft — Awaiting Architect review  
> **Audience**: @architect, project lead

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture Baseline](#2-current-architecture-baseline)
3. [Proposed Dual-Database Architecture](#3-proposed-dual-database-architecture)
4. [Training Database (GPU Server)](#4-training-database-gpu-server)
5. [Frontend Serving Database](#5-frontend-serving-database)
6. [Latency Budget Analysis](#6-latency-budget-analysis)
7. [Pagination Strategy](#7-pagination-strategy)
8. [Near Real-Time Data Flow](#8-near-real-time-data-flow)
9. [CDN & Edge Caching Strategy](#9-cdn--edge-caching-strategy)
10. [Strategy Comparison Matrix](#10-strategy-comparison-matrix)
11. [Recommended Architecture](#11-recommended-architecture)
12. [Phased Implementation Roadmap](#12-phased-implementation-roadmap)
13. [Risk Register](#13-risk-register)

---

## 1. Problem Statement

The Geospatial Digital Twin frontend is evolving beyond its Phase 1 static prototype. Three new requirements break the existing `ui_payload.json` static file bridge:

| Requirement | Impact on Current Architecture |
|---|---|
| **A) Multi-Model Run Comparison** | Each model run produces its own `master_index.json` + 86 `node_{id}.json` files (~29 MB total). Serving N model runs from static files means N × 29 MB committed to the Git repo and loaded by Vercel. |
| **B) Extended Paginated Timeseries** | The current payload covers 96 timesteps (24 hours at 15-min resolution). Expanding to weeks or months multiplies the per-node JSON payload linearly — e.g., 7 days = 672 timesteps, inflating `master_index.json` from 8 MB to ~56 MB. The browser cannot hold this in memory. |
| **C) Near Real-Time Inference** | Live sensor data must be ingested, passed through the frozen PyTorch model, and the resulting ZINB predictions served to the frontend. Static files cannot support this. |

The fundamental question: **Can a PostgreSQL/TimescaleDB database hosted on the GPU training server also serve the Vercel-hosted frontend with acceptable latency?**

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

### Current Payload Sizes (Measured)

| Asset | Size | Load Pattern |
|---|---|---|
| `master_index.json` | 8.16 MB | Fetched once on page load |
| `node_{id}.json` (×86) | ~290 KB each (21 MB total) | Fetched lazily on node click |
| Total per model run | ~29 MB | — |

### Current Performance Characteristics

- **Initial load**: Single 8 MB fetch from Vercel CDN → typically < 500ms on broadband (Vercel edge, gzipped to ~1.5 MB).
- **Node inspection**: 290 KB lazy fetch → < 100ms from CDN edge.
- **Slider scrubbing**: Zero network traffic — all 96 timesteps are held in browser memory. 60fps WebGL rendering via Deck.gl `DataFilterExtension`.

This architecture is extremely fast because **all data is pre-computed and edge-cached**. Any database-backed alternative must match or justify deviating from this baseline.

---

## 3. Proposed Dual-Database Architecture

Rather than forcing one database to serve both the training pipeline (heavy writes, local-only) and the frontend (light reads, global reach), we propose **separating concerns** into two distinct data stores:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU SERVER                                │
│                                                                  │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐  │
│  │  Training     │    │  PostgreSQL +     │    │  FastAPI      │  │
│  │  Pipeline     │───►│  TimescaleDB      │◄───│  Inference    │  │
│  │  (train.py)   │    │  (Training DB)    │    │  Server       │  │
│  └──────────────┘    └───────┬───────────┘    └──────┬───────┘  │
│                              │                        │          │
│                              │  Post-training         │          │
│                              │  export job            │          │
│                              ▼                        │          │
│                    ┌─────────────────┐                │          │
│                    │  Sync Worker    │────────────────┘          │
│                    │  (Materialized  │                            │
│                    │   Views / CRON) │                            │
│                    └────────┬────────┘                            │
│                             │                                    │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                              │  HTTPS API / Webhook push
                              ▼
                    ┌─────────────────────┐
                    │  Frontend Serving   │ ◄── Strategy Options:
                    │  Layer              │     A) Vercel Edge Functions + Supabase
                    │                     │     B) Vercel Edge Functions + Cloudflare D1
                    │                     │     C) Self-hosted FastAPI on GPU server
                    │                     │     D) Hybrid: CDN-cached static + live API
                    └────────┬────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Vercel (Next.js) │
                    │  User Browsers    │
                    └──────────────────┘
```

---

## 4. Training Database (GPU Server)

This database is dedicated to the ML pipeline. It lives on the GPU server in a Docker container with a mapped persistent volume. It is **never directly exposed to the public internet**.

### Purpose
- Store raw traffic and weather time-series (replacing Parquet files).
- Store versioned preprocessed datasets (branch-specific scaling, imputation).
- Store model run metadata (checkpoint paths, W&B run IDs, hyperparameters).
- Provide high-throughput local reads for `dataset.py` data loaders.

### Technology
PostgreSQL 16 + TimescaleDB 2.x + PostGIS 3.x (as documented in the prior context file).

### Key Constraint
**This database must NOT be the primary data source for the Vercel frontend.** The GPU server is a single machine, likely behind a residential or university firewall, without TLS termination, DDoS protection, or geographic distribution. Exposing it directly to end users introduces unacceptable latency variance and availability risk.

---

## 5. Frontend Serving Database

This is the critical design decision. We evaluate four strategies for serving model outputs to the Vercel-hosted Next.js frontend:

### Strategy A: Managed PostgreSQL (Supabase / Neon)

```
GPU Server ──(HTTPS push)──► Supabase (PostgreSQL + PostGIS)
                                        │
                                        │ REST / GraphQL API
                                        ▼
                              Vercel Edge Functions / Browser
```

- **Latency**: Supabase regions (Frankfurt, US-East) → Vercel edge. Sub-100ms for metadata queries. Timeseries queries (96–672 rows × 86 nodes) may take 200–500ms depending on indexing.
- **Cost**: Free tier: 500 MB storage, 2 GB bandwidth. Paid: ~$25/month for Pro.
- **Pros**: Managed backups, Row-Level Security, PostGIS for spatial queries, real-time subscriptions via WebSockets (useful for near real-time updates).
- **Cons**: Another managed service to maintain. Query latency for bulk timeseries data is higher than pre-computed static files.

### Strategy B: Edge-Native Database (Cloudflare D1 / Turso)

```
GPU Server ──(HTTPS push)──► Cloudflare D1 (SQLite at the edge)
                                        │
                                        │ Workers API
                                        ▼
                              Vercel Edge Functions / Browser
```

- **Latency**: D1 replicates reads to 300+ Cloudflare PoPs globally. Sub-50ms read latency worldwide.
- **Cost**: Free tier: 5 GB storage, 5M reads/day. Very cost-effective.
- **Pros**: Exceptional read latency. SQLite compatibility. Data lives at the CDN edge.
- **Cons**: No PostGIS. Limited to 1 GB per database (free). Write propagation delay (eventual consistency, ~60s). Not suitable for complex spatial queries.

### Strategy C: Self-Hosted FastAPI on GPU Server (Direct Serve)

```
GPU Server (FastAPI + PostgreSQL) ◄────── Vercel (CORS fetch)
                    │
                    │ HTTPS (via reverse proxy + Cloudflare Tunnel)
                    ▼
              User Browsers
```

- **Latency**: Depends entirely on the network path between the GPU server and the end user. If the server is on a UK university network, UK users see ~30ms. Global users may see 150–300ms.
- **Cost**: Zero incremental cost (uses existing hardware).
- **Pros**: Maximum control. Database and API are co-located. Zero data synchronisation overhead.
- **Cons**: Single point of failure. No geographic distribution. Requires Cloudflare Tunnel or ngrok for HTTPS. University/ISP firewalls may block incoming connections. GPU training load competes with API request serving.

### Strategy D: Hybrid — CDN-Cached Static JSON + Live API Fallback (Recommended)

```
GPU Server
  │
  ├──(post-training)──► Generate static JSON payloads
  │                     ├── master_index_{run_id}.json
  │                     └── nodes/{run_id}/node_{id}.json
  │                          │
  │                          │ git push / R2 upload
  │                          ▼
  │                   CDN (Vercel Static / Cloudflare R2)
  │                          │
  │                          │ Paginated 24h chunks
  │                          ▼
  │                   Browser (instant slider scrubbing)
  │
  └──(live inference)──► FastAPI on GPU server
                              │
                              │ Cloudflare Tunnel
                              ▼
                        Vercel Edge Function (proxy)
                              │
                              ▼
                        Browser (near real-time updates)
```

- **Latency**: Static payloads served from CDN edge (~50ms globally). Live inference via tunnel (~100–300ms).
- **Cost**: Cloudflare R2: $0.015/GB stored, $0 egress. Cloudflare Tunnel: free.
- **Pros**: Preserves the existing high-performance static serving model for historical data. Only uses the live API path for real-time inference (where slight latency is acceptable). Graceful degradation — if the GPU server is offline, historical data is still fully available from CDN.
- **Cons**: Two serving paths to maintain. Post-training export job required to generate and upload static payloads.

---

## 6. Latency Budget Analysis

Your key insight is correct: **the time slider must remain highly reactive**. Let's quantify the latency budget:

### The Slider Interaction Model

```
User drags slider ──► Browser reads in-memory array[t] ──► Deck.gl re-renders
                      ↑
                      │ This MUST be < 16ms (60fps) for fluid animation
                      │ Therefore: ALL data for the current "page" must be in browser memory
```

**Critical constraint**: The slider cannot make network requests per timestep. The data for the entire visible time window must be pre-loaded into browser memory before the slider becomes interactive.

### Pagination Network Budget

| Scenario | Payload Size (est.) | Gzipped | CDN Delivery | DB Query + API |
|---|---|---|---|---|
| 24h (96 steps), current format | 8 MB (master_index) | ~1.5 MB | < 200ms | N/A |
| 24h (96 steps), per-run payload | ~8 MB | ~1.5 MB | < 200ms | ~300ms |
| 7d (672 steps), per-run | ~56 MB | ~10 MB | < 1.5s | ~800ms |
| 30d (2880 steps), per-run | ~240 MB | ~45 MB | Too large | Too large |

**Conclusion**: A 24-hour page is the natural pagination unit. At ~1.5 MB gzipped, it transfers in well under 1 second on any broadband connection and fits comfortably in browser memory. The slider operates on this in-memory dataset with zero network overhead.

### The "Page Turn" Interaction

When the user navigates to a different 24-hour window (e.g., clicking "Next Day" or selecting a date):

```
1. Browser shows loading skeleton           0ms
2. Fetch 24h payload from CDN/API      200–500ms
3. Parse JSON, update Zustand store     50–100ms
4. Deck.gl re-renders first frame        ~16ms
                                    ─────────────
                              Total: 300–600ms (feels instant)
```

This is well within the perceived-instant threshold (< 1 second). The user's mental model of "turning a page" naturally accommodates a brief load.

---

## 7. Pagination Strategy

### Payload Restructuring

The current `master_index.json` embeds full timeseries arrays inline per node. For pagination, we restructure to separate **spatial metadata** (loaded once) from **temporal data** (loaded per page):

```
                         ┌──────────────────────────────────┐
                         │         Spatial Index             │
  Loaded ONCE            │  (nodes, edges, coordinates,     │
  on app mount ──────►   │   semantic_tags, model metadata)  │
                         │  Size: ~200 KB (gzipped ~40 KB)   │
                         └──────────────────────────────────┘

                         ┌──────────────────────────────────┐
                         │      Temporal Page (24h)          │
  Loaded per page        │  { run_id, date, timesteps: 96,  │
  navigation ────────►   │    node_timeseries: [...],        │
                         │    edge_attention: [...],          │
                         │    calibration_metrics: {...} }    │
                         │  Size: ~5 MB (gzipped ~1 MB)      │
                         └──────────────────────────────────┘

                         ┌──────────────────────────────────┐
                         │      Node Detail (24h)            │
  Loaded on node         │  { zinb_series, ground_truth,     │
  click (lazy) ──────►   │    imputed, mask, ego_graph }     │
                         │  Size: ~100 KB (gzipped ~20 KB)   │
                         └──────────────────────────────────┘
```

### Multi-Model Run Support

For comparing results across model runs, the spatial index includes a **run registry**:

```json
{
  "runs": [
    {
      "run_id": "feasible-capybara-22",
      "wandb_run_id": "abc123",
      "git_commit": "e4f5a6b",
      "train_config": { "hidden_dim": 32, "num_layers": 2 },
      "available_dates": ["2025-05-10", "2025-05-11", "2025-05-12"],
      "metrics_summary": { "val_rmse": 12.4, "val_mae": 8.1 }
    },
    {
      "run_id": "swift-otter-23",
      "wandb_run_id": "def456",
      "git_commit": "7g8h9i0",
      "train_config": { "hidden_dim": 64, "num_layers": 3 },
      "available_dates": ["2025-05-10", "2025-05-11"],
      "metrics_summary": { "val_rmse": 10.8, "val_mae": 7.2 }
    }
  ]
}
```

The frontend fetches temporal pages via a URL pattern:
```
/api/data/{run_id}/{date}/page.json        → 24h temporal page
/api/data/{run_id}/{date}/node_{id}.json   → per-node detail for that day
```

### URL Naming Convention for CDN-Static Payloads

If using Strategy D (Hybrid CDN + Live API), these files are pre-generated and uploaded to Cloudflare R2 or Vercel Blob Storage:

```
r2://iot-twin-data/
  └── runs/
      ├── feasible-capybara-22/
      │   ├── spatial_index.json              (loaded once)
      │   ├── 2025-05-10/
      │   │   ├── page.json                   (24h temporal data)
      │   │   └── nodes/
      │   │       ├── node_0.json
      │   │       ├── node_1.json
      │   │       └── ...
      │   └── 2025-05-11/
      │       ├── page.json
      │       └── nodes/...
      └── swift-otter-23/
          └── ...
```

---

## 8. Near Real-Time Data Flow

### Architecture

```
┌───────────────┐    Webhook / Poll    ┌───────────────────────┐
│ Traffic Sensor │ ─────────────────► │     GPU Server          │
│ Data Source    │    (15-min cadence) │                         │
│ (Urban Obs.)  │                     │  1. Ingest raw reading  │
└───────────────┘                     │  2. Append to rolling   │
                                      │     12-step buffer      │
                                      │  3. Run frozen model    │
                                      │     forward pass        │
                                      │  4. Store predictions   │
                                      │     in TimescaleDB      │
                                      │  5. Push update to      │
                                      │     serving layer       │
                                      └───────────┬─────────────┘
                                                  │
                               ┌──────────────────┼──────────────────┐
                               │                  │                  │
                               ▼                  ▼                  ▼
                        Option A:           Option B:          Option C:
                      Push to Supabase    Push pre-built     Expose FastAPI
                      Realtime channel    JSON to R2/Blob    via Cloudflare
                      (WebSocket to       (CDN invalidation  Tunnel (direct
                       browser)            + poll every 5m)   SSE/WebSocket)
```

### Timeliness Analysis

You noted that near real-time timeliness is not critical. This is important because it opens up simpler architectures:

| Update Cadence | Architecture Complexity | User Experience |
|---|---|---|
| Every 15 min (sensor cadence) | Low: CRON job on GPU server pushes a small JSON update to CDN. Frontend polls every 5 minutes. | "Live" indicator shows data is at most 20 minutes old. Acceptable for urban planning dashboards. |
| Every 1 min | Medium: WebSocket/SSE from FastAPI via tunnel. Frontend subscribes. | Requires persistent server connection. More complex but enables live animation. |
| Sub-second | High: Redis pub/sub, dedicated streaming infra. | Overkill for this use case. |

### Recommended: 15-Minute CRON Push to CDN

```python
# Pseudo-code: runs every 15 minutes on GPU server via CRON
def update_live_page():
    # 1. Fetch latest 12 readings from TimescaleDB
    readings = db.query("""
        SELECT * FROM traffic_readings
        WHERE dt >= NOW() - INTERVAL '3 hours'
        ORDER BY dt DESC
    """)
    
    # 2. Build tensor, run frozen model
    batch = build_inference_batch(readings)
    with torch.no_grad():
        result = model(batch, return_analysis=True)
    
    # 3. Format as temporal page JSON (same schema as historical pages)
    page = format_temporal_page(result, date="live")
    
    # 4. Upload to R2, replacing the "live" page
    r2.put_object("runs/live/today/page.json", json.dumps(page))
```

The frontend checks for a `"live"` run entry in the spatial index and, if present, polls `runs/live/today/page.json` every 5 minutes. The slider operates on the in-memory data between polls — no WebSocket complexity needed.

---

## 9. CDN & Edge Caching Strategy

### Why CDN Caching Matters

The current architecture already achieves sub-200ms global delivery because Vercel serves static files from its CDN edge. Any database-backed alternative must compete with this baseline.

### Cache-Control Headers

```
# Spatial index (rarely changes): cache for 1 hour
Cache-Control: public, max-age=3600, s-maxage=3600, stale-while-revalidate=86400

# Historical temporal pages (immutable after generation): cache forever
Cache-Control: public, max-age=31536000, immutable

# Live temporal page (updates every 15 min): short cache + revalidation
Cache-Control: public, max-age=60, s-maxage=300, stale-while-revalidate=900
```

### Gzip/Brotli Compression

JSON payloads compress exceptionally well. Measured compression ratios from the current payloads:

| Payload | Raw | Gzipped | Brotli | Ratio |
|---|---|---|---|---|
| `master_index.json` (8.16 MB) | 8.16 MB | ~1.5 MB | ~1.2 MB | 6–7× |
| `node_{id}.json` (290 KB) | 290 KB | ~55 KB | ~45 KB | 5–6× |

A 24-hour temporal page (~5 MB raw) will compress to ~1 MB with Brotli. At typical broadband speeds (50 Mbps), this downloads in **160ms**. Well within the 1-second "page turn" budget.

---

## 10. Strategy Comparison Matrix

| Criterion | A: Supabase | B: Cloudflare D1 | C: Self-Hosted API | D: Hybrid CDN + API |
|---|---|---|---|---|
| **Slider latency** | ⚠️ Needs pre-fetch | ✅ Edge-cached | ⚠️ Network per page | ✅ In-memory (0ms) |
| **Page turn latency** | 200–500ms | 50–150ms | 100–400ms | 100–300ms |
| **Live data support** | ✅ WebSocket | ⚠️ Eventual consistency | ✅ Direct SSE | ✅ CDN poll |
| **Multi-run support** | ✅ SQL queries | ✅ Partitioned tables | ✅ Full control | ✅ File namespacing |
| **PostGIS support** | ✅ Native | ❌ No | ✅ Native | ✅ On GPU server |
| **Availability** | ✅ 99.95% SLA | ✅ 99.99% (Cloudflare) | ⚠️ Single server | ✅ CDN (history) + server (live) |
| **Cost (monthly)** | $25+ (Pro) | ~$5 | $0 | ~$5 (R2 storage) |
| **Setup complexity** | Medium | Medium | Low | Medium |
| **Graceful degradation** | ❌ DB down = app down | ❌ DB down = app down | ❌ Server down = app down | ✅ History always available |
| **Training interference** | None | None | ⚠️ CPU/RAM contention | Minimal (export job only) |

---

## 11. Recommended Architecture

### Primary Recommendation: **Strategy D — Hybrid CDN + Live API**

This strategy is recommended because it:

1. **Preserves slider performance**: All temporal data for the current 24-hour page is pre-loaded into browser memory. The Deck.gl `DataFilterExtension` operates at 60fps with zero network calls during scrubbing.

2. **Scales to multi-model runs**: Each model run's output is a set of static files in a namespaced directory. Adding a new run is a file upload, not a schema migration.

3. **Handles pagination naturally**: Navigating between days triggers a single ~1 MB fetch from CDN edge. The "page turn" completes in 200–400ms.

4. **Supports near real-time without complexity**: A 15-minute CRON job on the GPU server generates and pushes a "live" page to CDN. The frontend polls this page at a relaxed cadence. No WebSocket infrastructure needed.

5. **Degrades gracefully**: If the GPU server is offline, all historical model runs remain fully browsable from CDN. Only live inference is unavailable.

### Concrete Component Choices

| Component | Choice | Justification |
|---|---|---|
| **Training DB** | PostgreSQL + TimescaleDB (Docker on GPU server) | Local, high-throughput, versioned |
| **Static payload storage** | Cloudflare R2 | $0 egress, global CDN, S3-compatible API |
| **Live inference API** | FastAPI on GPU server, exposed via Cloudflare Tunnel | Free, secure, no firewall config needed |
| **Frontend data layer** | Vercel Next.js + SWR/React Query | Handles caching, revalidation, loading states |
| **Schema migrations** | Alembic (for training DB) | Version-controlled, automated |

### Data Flow Summary

```
[Historical Data Path — Bulk, Immutable]

  GPU Server: train.py completes
       │
       ▼
  export_paginated_payload.py
       │  Generates: spatial_index.json + per-day page.json + per-node node_{id}.json
       ▼
  Upload to Cloudflare R2
       │
       ▼
  Vercel fetches from R2 (via rewrite rules or API route)
       │
       ▼
  Browser: loads spatial index on mount, fetches temporal page on date select


[Live Data Path — Incremental, Mutable]

  Sensor webhook → GPU Server CRON (every 15 min)
       │
       ▼
  FastAPI inference endpoint
       │  Generates: live/today/page.json
       ▼
  Upload to R2 (replacing previous live page)
       │
       ▼
  Browser: polls live page every 5 minutes via SWR
```

---

## 12. Phased Implementation Roadmap

### Phase 2A: Paginated Static Payloads (No Database Required)

**Goal**: Restructure `export_ui_payload.py` to emit paginated, per-day payloads compatible with multi-run storage. Deploy to Cloudflare R2.

**Tasks**:
1. Refactor `export_ui_payload.py` to split output into `spatial_index.json` + per-day `page.json` + per-day `node_{id}.json`.
2. Add a `run_id` parameter derived from the W&B run name or git branch.
3. Create an R2 upload script (`scripts/upload_to_r2.py`) using `boto3` with S3-compatible endpoint.
4. Update `MapVisualizer.tsx` to fetch spatial index on mount, then lazy-load temporal pages on date navigation.
5. Add a date picker / pagination control to the UI.
6. Update `NodeInspector.tsx` to fetch per-day node detail files.

**Estimated effort**: 2–3 days.
**Dependencies**: Cloudflare R2 account (free tier sufficient).

---

### Phase 2B: Training Database Setup

**Goal**: Stand up PostgreSQL + TimescaleDB on the GPU server. Migrate raw data ingestion.

**Tasks**:
1. Add `timescaledb` service to `docker-compose.yml`.
2. Create Alembic migration scripts for the schema defined in the prior context document.
3. Write a one-time ingestion script to load raw Parquet data into hypertables.
4. Refactor `dataset.py` to optionally read from the database (with Parquet fallback).
5. Implement `dataset_versions` table and row-level version tagging.

**Estimated effort**: 3–5 days.
**Dependencies**: None beyond existing GPU server access.

---

### Phase 2C: Multi-Run Frontend

**Goal**: Enable the UI to switch between different model runs and compare predictions.

**Tasks**:
1. Create a run registry JSON (or API endpoint) listing all available runs.
2. Add a run selector dropdown to the frontend.
3. Implement split-view or overlay comparison mode for two runs.
4. Update `useTimeStore` to support `activeRunId` state.

**Estimated effort**: 2–3 days.

---

### Phase 3: Near Real-Time Inference

**Goal**: Ingest live sensor data, run inference, and serve predictions to the frontend.

**Tasks**:
1. Set up Cloudflare Tunnel on the GPU server (`cloudflared`).
2. Implement FastAPI inference endpoint wrapping the frozen model.
3. Implement a 12-step rolling window cache (Redis or in-memory deque).
4. Create a CRON job that runs every 15 minutes: fetch latest readings → infer → upload `live/today/page.json` to R2.
5. Add a "Live" toggle to the frontend that switches to the live temporal page and enables 5-minute polling via SWR.
6. Implement the sensor data webhook receiver (FastAPI endpoint on GPU server).

**Estimated effort**: 5–7 days.

---

## 13. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU server offline during live inference | Medium | Low (history still available) | Frontend shows "Live data unavailable" banner. Graceful degradation by design. |
| R2 upload fails after training | Low | Medium (new run not visible) | Retry logic in upload script. Alert via W&B webhook. |
| 24h page exceeds 5 MB budget with future features | Medium | Low (still compresses well) | Monitor payload sizes. Consider MessagePack or Protocol Buffers if JSON bloats beyond 10 MB raw. |
| University firewall blocks Cloudflare Tunnel | Low | High (no live data) | Alternative: push live page directly to R2 via API (no tunnel needed). Only direct API queries need the tunnel. |
| Database on GPU server causes training slowdown | Medium | Medium | Set `shared_buffers`, `work_mem` limits in PostgreSQL config. Pin database container to specific CPU cores. Monitor with `pg_stat_activity`. |
| Stale live data confuses users | Low | Low | Display "Last updated: X minutes ago" timestamp prominently. Use `stale-while-revalidate` cache headers. |

---

> **Next Step**: This document should be reviewed by the @architect to formalize the schema changes and create a detailed implementation plan. The critical path is **Phase 2A** (paginated payloads) — this can proceed immediately without any database infrastructure.
