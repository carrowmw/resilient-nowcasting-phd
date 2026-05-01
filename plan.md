# Plan: Phase 1 Analytics Route — Master-Detail Data Selection

**Phase Anchor:** `NSTAR_PHASES.md` → Phase 1 (Static Data Bridge ONLY). No live API.

**North Stars consulted:** `NSTAR_BACKEND_ARCHITECTURE.md` (ZINB tripwires, softplus+eps),
`NSTAR_FRONTEND_ARCHITECTURE.md` (defensive rendering, MUI stack),
`NSTAR_MLOPS.md` (container isolation, anonymous volume),
`NSTAR_PHASES.md` (Phase 1 constraints).

---

## Current State Assessment

| Component | Location | Status |
|:---|:---|:---|
| Monolithic payload | `frontend/public/data/ui_payload.json` | Exists, served at `/data/ui_payload.json` |
| `export_ui_payload.py` | `src/export_ui_payload.py` | Writes `data/processed/` (stale path), needs update |
| `MapVisualizer.tsx` | Fetches `/data/ui_payload.json` | Must migrate to `master_index.json` |
| `NodeInspector.tsx` | Loads monolithic payload | Must migrate to per-node lazy fetch |
| `payload.ts` | Has `ground_truth_series` on `NodeEntry` | Must restructure types |
| Analytics route | Does not exist | New file |

---

## Pre-flight (CLAUDE.md §1 — Safety First)

1. **Branch confirmation:** This task creates significant new files across both frontend and backend. Confirm whether to stay on `main` or cut a branch (Rule 6 Branching Protocol).
2. **Auto-commit rollback point** before any multi-file changes:
   `git add . && git commit -m "Auto-commit before Analytics Route (Phase 1)"`

---

## Step 1 — Backend Data Sharding (`src/export_ui_payload.py`)

### 1a. New Output Schema Design

**`frontend/public/data/master_index.json`** — lightweight; used by the map and analytics sidebar navigation:
```jsonc
{
  "metadata": { /* same 9-field block as before */ },
  "nodes": [
    {
      "id": 0, "location_name": "...", "lat": ..., "lon": ..., "is_simulated": false,
      "timeseries": [{ "t": 0, "prediction": 12.3, "uncertainty": 4.1 }, ...]
      // NO ground_truth_series / imputed_series / mask_series — those are in per-node files
    }
  ],
  "edges": [{ "source": 0, "target": 1, "attention_sequence": [...] }]
}
```

**`frontend/public/data/nodes/node_{id}.json`** — deep per-node analytics data:
```jsonc
{
  "id": 0, "location_name": "...", "lat": ..., "lon": ..., "is_simulated": false,
  "zinb_series": [
    { "t": 0, "mu": 12.3, "theta": 5.1, "std_dev": 4.1 }  // theta is NEW
  ],
  "ground_truth_series": [12.0, 14.5, ..., null],   // null for simulated nodes
  "imputed_series": [11.8, 14.2, ...],
  "mask_series": [false, true, ...],
  "edges_1hop": [
    {
      "neighbor_id": 3,
      "direction": "outgoing",       // this node is the source
      "attention_sequence": [0.12, 0.15, ...]
    }
  ]
}
```

### 1b. Changes to `export_ui_payload.py`

1. **Fix output path**: Change `OUTPUT_PATH` constant → write sharded outputs to
   `frontend/public/data/master_index.json` and `frontend/public/data/nodes/node_{id}.json`.
   Remove `OUTPUT_PATH` global; use two new constants:
   - `MASTER_INDEX_PATH = "frontend/public/data/master_index.json"`
   - `NODES_DIR = "frontend/public/data/nodes"`

2. **Capture `theta` per node per timestep**: Add a new accumulator alongside `timeseries_data`:
   ```python
   theta_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
   ```
   In the inference loop, after `theta: torch.Tensor = result["theta"][0]`, append to `theta_data`.

3. **ZINB non-negative tripwires (Offensive Programming, per task spec)**:
   In the inference loop, immediately after computing `mu` and `theta`:
   ```python
   assert (mu > 0).all(), f"t={t}: ZINB tripwire — mu has non-positive values: {mu.min()}"
   assert (theta > 0).all(), f"t={t}: ZINB tripwire — theta has non-positive values: {theta.min()}"
   ```
   These are guaranteed by the model's `F.softplus + eps` but we assert explicitly before export.

4. **Build 1-hop edge lookup**: After the inference loop, construct a dict mapping each
   `node_id → list of {"neighbor_id", "direction", "attention_sequence"}` using
   `edge_index_cached` and `edge_attention_data`. Direction is `"outgoing"` if
   `edge[0] == node_id`, else `"incoming"`.

5. **NaN assertion on final output values**: Before writing each `node_{id}.json`, assert:
   ```python
   for entry in node_data["zinb_series"]:
       assert not math.isnan(entry["mu"]),    f"NaN in node {node_id} zinb_series mu at t={entry['t']}"
       assert not math.isnan(entry["theta"]), f"NaN in node {node_id} zinb_series theta at t={entry['t']}"
       assert not math.isnan(entry["std_dev"]),f"NaN in node {node_id} zinb_series std_dev at t={entry['t']}"
   for v in node_data["imputed_series"]:
       assert not math.isnan(v), f"NaN in node {node_id} imputed_series"
   ```

6. **Write `master_index.json`** (single file).

7. **Write `node_{id}.json`** in a loop (one file per node).
   `os.makedirs(NODES_DIR, exist_ok=True)` before the loop.

8. **Existing `ui_payload.json`** generation is removed (task spec: "stop generating a single
   monolithic JSON"). The `frontend/public/data/ui_payload.json` file will become stale —
   it remains on disk but is no longer referenced by the updated frontend code.

### 1c. Desk Check — ZINB Tripwire (per CLAUDE.md §8)

| Scenario | Input | Expected | Tripwire fires? |
|:---|:---|:---|:---|
| Model output softplus(x=3.0)+eps | mu = softplus(3.0)+1e-6 = 3.049 | > 0 ✓ | No |
| Model output softplus(x=0)+eps | mu = softplus(0)+1e-6 = 0.693+1e-6 > 0 ✓ | > 0 ✓ | No |
| Hypothetical unclamped 0.0 | mu = 0.0 | FAIL | Yes → crash immediately |
| NaN from degenerate gradient | mu = NaN | FAIL | Yes → existing `assert not torch.isnan(mu).any()` catches first |

Conclusion: the tripwire correctly passes all valid model outputs and crashes on pathological values.

---

## Step 2 — Frontend Types (`frontend/src/types/payload.ts`)

Restructure into two distinct schemas:

```typescript
// --- MAP SCHEMA (master_index.json) ---

export interface TimestepEntry {       // unchanged
  t: number;
  prediction: number;   // μ
  uncertainty: number;  // std_dev
}

export interface NodeEntry {           // simplified — NO imputation fields
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  is_simulated: boolean;
  timeseries: TimestepEntry[];
}

export interface EdgeEntry { /* unchanged */ }

export interface MasterIndex {        // replaces UIPayload
  metadata: PayloadMetadata;
  nodes: NodeEntry[];
  edges: EdgeEntry[];
}

// Keep UIPayload as alias for backward compat in tests
export type UIPayload = MasterIndex;

// --- PER-NODE ANALYTICS SCHEMA (nodes/node_{id}.json) ---

export interface ZINBEntry {           // NEW — has theta
  t: number;
  mu: number;
  theta: number;
  std_dev: number;
}

export interface HopEdge {             // NEW — 1-hop edge record
  neighbor_id: number;
  direction: "outgoing" | "incoming";
  attention_sequence: number[];
}

export interface NodeDetailPayload {   // NEW — deep per-node data
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  is_simulated: boolean;
  zinb_series: ZINBEntry[];
  ground_truth_series: (number | null)[];
  imputed_series: number[];
  mask_series: boolean[];
  edges_1hop: HopEdge[];
}
```

---

## Step 3 — Update Existing Frontend Components

### 3a. `MapVisualizer.tsx`
Single-line change: `fetch("/data/ui_payload.json")` → `fetch("/data/master_index.json")`.
The `UIPayload` type alias (= `MasterIndex`) means no other code changes needed here.

### 3b. `NodeInspector.tsx`
Replace the monolithic-payload fetch with a lazy per-node fetch:

```typescript
// OLD: fetches the full ui_payload.json once on mount, then finds the node
// NEW: fetches /data/nodes/node_{id}.json whenever selectedNodeId changes
const [nodeDetail, setNodeDetail] = useState<NodeDetailPayload | null>(null);
useEffect(() => {
  if (selectedNodeId === null) { setNodeDetail(null); return; }
  fetch(`/data/nodes/node_${selectedNodeId}.json`)
    .then(r => r.ok ? r.json() : null)
    .then((d: NodeDetailPayload | null) => setNodeDetail(d))
    .catch(() => setNodeDetail(null));
}, [selectedNodeId]);
```

Update `NodeChart` to use `nodeDetail.zinb_series[t].mu` as `prediction` and
`nodeDetail.zinb_series[t].std_dev` as `uncertainty` for the stats display row.
The chart itself (`ground_truth_series`, `imputed_series`, `mask_series`) is unchanged.

---

## Step 4 — Package Installation (Manual User Action Required)

**The agent container does not have `docker` available.** The user must run these commands
on the host machine (where `docker compose` is accessible) **before** the frontend TypeScript
compilation step. Service name is `frontend-dev`.

```bash
docker compose exec frontend-dev npm install @mui/material @emotion/react @emotion/styled @mui/icons-material
docker compose exec frontend-dev npm install @mui/x-data-grid
docker compose exec frontend-dev npm install recharts
docker compose exec frontend-dev npm install react-force-graph-2d
```

Verify inside the container (NOT by checking host `node_modules`):
```bash
docker compose exec frontend-dev node -e "require('@mui/x-data-grid'); console.log('OK')"
```

The agent will write `package.json` references for these deps alongside code changes,
to document intent even if the anonymous volume prevents host verification.

---

## Step 5 — New Analytics Route (`frontend/src/app/analytics/nodes/[id]/page.tsx`)

### 5a. File structure
```
frontend/src/app/analytics/
  nodes/
    [id]/
      page.tsx          ← Single new file; all tab content co-located here
```

### 5b. Layout design
Client Component (`"use client"`). Uses `useParams()` from `next/navigation` to read `id`.
On mount: fetches `master_index.json` (for sidebar node list) + `node_{id}.json` (for tabs).

MUI layout:
```
┌─────────────────────────────────────────────────────────┐
│  AppBar: "Analytics — {location_name}"   [☰] [←Back]    │
├──────────────────┬──────────────────────────────────────┤
│  Drawer sidebar  │  Main content                         │
│  (collapsible    │  ┌────────────────────────────────┐  │
│  IconButton)     │  │ Tab 1 │ Tab 2 │ Tab 3          │  │
│                  │  ├────────────────────────────────┤  │
│  - Node list     │  │ DataGrid (top 50%)              │  │
│    from          │  ├────────────────────────────────┤  │
│    master_index  │  │ Chart / Graph (bottom 50%)      │  │
│  - Each item     │  └────────────────────────────────┘  │
│    links to      │                                       │
│    /analytics/   │                                       │
│    nodes/{id}    │                                       │
└──────────────────┴──────────────────────────────────────┘
```

AppBar contains a `←` back link to `/` (the map view) and a hamburger icon to toggle the drawer.
Drawer uses MUI `Drawer` with `variant="permanent"` on >= md breakpoints, `"temporary"` on mobile.

### 5c. Tab 1 — Forecast Variance

**DataGrid columns:** `t` (timestep, filterable), `mu` (4dp), `theta` (4dp), `std_dev` (4dp),
`ci_lower` (mu − 2·std_dev, 4dp), `ci_upper` (mu + 2·std_dev, 4dp).

**State:** `selectedRows: Set<number>` (row indices). Default: all rows selected (show full chart).

**Chart:** Recharts `ComposedChart`:
- `Line` for `mu` (blue, solid)
- `Area` for CI band `[ci_lower, ci_upper]` (blue, 20% opacity)
- X-axis: timestep t
- Y-axis: traffic count units
- When rows selected in DataGrid → X domain restricted to `[min(selected_t), max(selected_t)]`

Defensive: if `zinb_series` is empty, show "No ZINB data available" message.

### 5d. Tab 2 — Network Influence

**DataGrid columns:** `neighbor_id`, `direction` ("out"/"in"), `location_name` (looked up from
master_index nodes), `attention_at_t` (current timestep's α, sortable), `avg_attention` (mean α
over all 96 timesteps), `max_attention`.

Default sort: `attention_at_t` descending.

**State:** `selectedNeighbor: number | null` (neighbor_id).

**Force graph:** `react-force-graph-2d` loaded via `dynamic()` with `ssr: false`.
- Nodes: center node (id = current node, fixed position) + all 1-hop neighbors
- Edges: width and opacity proportional to `attention_at_t` (attention at current timestep from store)
- Selected neighbor from DataGrid → node highlighted (different color / glow ring)
- Hovering graph node → tooltip with `location_name`

Defensive: if `edges_1hop` is empty, show "No 1-hop edges found" message.

Current timestep sync: read `currentTimestep` from Zustand store.

### 5e. Tab 3 — Self-Healing Log

**DataGrid rows:** Only timesteps where `mask_series[t] === true` (these are the "masking events").
**Columns:** `t`, `ground_truth` (null shown as "—"), `imputed`, `delta` (imputed − ground_truth,
null if simulated).

**State:** `focusTStep: number | null` — the selected timestep from the DataGrid.

**Chart:** Recharts `LineChart` with two series:
- `ground_truth_series` (blue, skip null gaps with `connectNulls: false`)
- `imputed_series` (orange)
- When a row is selected → X domain zooms to `[focusTStep − 8, focusTStep + 8]`, clamped to [0, T-1]
- Vertical `ReferenceLine` at `focusTStep` to highlight the selected masked event
- Shaded `ReferenceArea` over all masked timesteps (faint red band)

Defensive: if `mask_series` has no `true` values, show "No masking events in this payload" message.

---

## Step 6 — Validation

### 6a. Backend (Python)
Run in agent container:
```bash
python src/export_ui_payload.py --start_date 2024-05-09 --end_date 2024-05-11
```
Confirm:
- `frontend/public/data/master_index.json` written without assertion failures.
- `frontend/public/data/nodes/` directory created with one file per node.
- Schema assertions: `len(node_0.zinb_series) == 96`, etc.
- ZINB tripwire assertions pass (no positive violations expected from trained model).

### 6b. Frontend TypeScript
Run via host or agent:
```bash
cd frontend && npx tsc --noEmit
```
Must return 0 errors.

### 6c. Visual verification
Navigate to `http://localhost:3000` — existing map view still works.
Navigate to `http://localhost:3000/analytics/nodes/0` — new analytics page renders with 3 tabs.
Verify DataGrid row selection drives chart updates for each tab.

---

## Step 7 — EXPERIMENTS.md Ledger Update

Per CLAUDE.md §6, append a row upon completion of a successful rendering.
**The task is not complete until this row is written.**

---

## Files Created/Modified Summary

| Path | Type | Change |
|:---|:---|:---|
| `src/export_ui_payload.py` | edit | Stop writing monolithic; write `master_index.json` + per-node `node_{id}.json`; add theta capture; ZINB tripwires |
| `frontend/src/types/payload.ts` | edit | Add `ZINBEntry`, `HopEdge`, `NodeDetailPayload`; simplify `NodeEntry`; add `MasterIndex` alias |
| `frontend/src/components/MapVisualizer.tsx` | edit | Fetch `master_index.json` instead of `ui_payload.json` |
| `frontend/src/components/NodeInspector.tsx` | edit | Replace monolithic fetch with lazy per-node fetch of `node_{id}.json` |
| `frontend/src/app/analytics/nodes/[id]/page.tsx` | **new** | Analytics route with MUI Drawer + Tabs + 3 DataGrid/chart panes |
| `EXPERIMENTS.md` | edit | Append ledger row on completion |

---

## Risks & Tripwires

- **`theta` shape**: `result["theta"]` is `[1, num_nodes_total]` (same shape as `mu`). Verify this
  matches before using; the assert on imputed_features shape will catch broader mismatches.
- **`react-force-graph-2d` SSR crash**: Must use `dynamic({ ssr: false })`. Failure to do so will
  cause a Next.js server-side crash on `window` access.
- **MUI emotion cache**: In Next.js App Router with RSC, `@emotion/react` requires a client-side
  registry. Since the analytics page is a full Client Component (`"use client"`), the default
  emotion setup works without a server registry.
- **`useParams` in Next.js 16**: `params` as a `Promise` applies to Server Components only.
  `useParams()` from `next/navigation` in a Client Component returns the values synchronously —
  no await needed.
- **Node file count**: With ~20 real nodes + 5 simulated = ~25 files. File I/O in the export
  script adds negligible overhead vs. 96 forward passes.

---

**Awaiting explicit user approval before implementation per CLAUDE.md §1.**
