# Plan: Analytics Dashboard & Human-in-the-Loop Telemetry (Phase 1)

**Phase Anchor:** `NSTAR_PHASES.md` → Phase 1 (Static Data Bridge ONLY). No live API.
**North Stars consulted:** `NSTAR_FRONTEND_ARCHITECTURE.md`, `NSTAR_BACKEND_ARCHITECTURE.md`, `NSTAR_PHASES.md`
**Date:** 2026-05-01
**Branch:** `main` (auto-commit rollback point already created: `a207d08`)

---

## Current State Assessment

| Component | Location | Status |
|:---|:---|:---|
| Monolithic fetch | `MapVisualizer.tsx` line 207 | Fetches `/data/ui_payload.json` → must migrate to `/data/master_index.json` |
| Monolithic fetch | `NodeInspector.tsx` line 193 | Fetches `/data/ui_payload.json` → must migrate to per-node lazy fetch |
| Sharded data files | `frontend/public/data/` | `master_index.json` + `nodes/node_*.json` already exist (pre-generated) |
| `export_ui_payload.py` | `src/` | Still writes to `data/processed/ui_payload.json`; must be refactored to write sharded output formally |
| `payload.ts` types | `frontend/src/types/` | Has `ground_truth_series` / `imputed_series` on `NodeEntry` → must restructure |
| Analytics route | — | Does not exist |
| Telemetry API | — | Does not exist |
| Floating nav button | `MapScene.tsx` | Does not exist |

---

## Step 1 — Telemetry Pipe (Error Catching)

### 1a. API Route: `frontend/src/app/api/log/route.ts`

A lightweight Next.js App Router `POST` handler. Accepts a JSON body, prepends a UTC timestamp,
and **appends** (never overwrites) to `/workspace/frontend_runtime_errors.log` using Node's `fs.appendFileSync`.

```typescript
// POST /api/log
// Body: { type: string, message: string, source?: string, stack?: string }
// Writes one JSON line per call to /workspace/frontend_runtime_errors.log
```

**Tripwire:** The route must return `405 Method Not Allowed` on non-POST requests (no GET handler,
preventing log-flooding via browser prefetch).

### 1b. Global Error Listener in `layout.tsx`

Since `layout.tsx` is a **Server Component**, the listeners must be injected via a new small
**Client Component** (`GlobalErrorReporter`) declared in the same file or a nearby file:

```typescript
"use client";
export function GlobalErrorReporter() {
  useEffect(() => {
    const post = (payload: object) =>
      fetch("/api/log", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload) }).catch(() => {/* silent — the logger must not itself throw */});

    const onError = (event: ErrorEvent) =>
      post({ type: "onerror", message: event.message, source: event.filename, stack: event.error?.stack });

    const onUnhandled = (event: PromiseRejectionEvent) =>
      post({ type: "unhandledrejection", message: String(event.reason), stack: event.reason?.stack });

    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onUnhandled);
    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onUnhandled);
    };
  }, []);
  return null;
}
```

`layout.tsx` renders `<GlobalErrorReporter />` inside `<body>` above `{children}`.

---

## Step 2 — Backend Data Sharding (`src/export_ui_payload.py`)

The sharded files already exist on disk (generated in a prior session). This step formally
refactors the script to make it the authoritative source.

### 2a. New Output Constants

Remove `OUTPUT_PATH`. Add:

```python
MASTER_INDEX_PATH = "frontend/public/data/master_index.json"
NODES_DIR         = "frontend/public/data/nodes"
```

### 2b. Capture `theta` per timestep

Add a new accumulator alongside `timeseries_data`:

```python
theta_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
```

In the inference loop, after `theta: torch.Tensor = result["theta"][0]`:

```python
# ZINB non-negative tripwires (Offensive Programming)
assert (mu    > 0).all(), f"t={t}: ZINB tripwire — mu has non-positive values: {mu.min()}"
assert (theta > 0).all(), f"t={t}: ZINB tripwire — theta has non-positive values: {theta.min()}"
for node_id in range(num_nodes_total):
    theta_data[node_id].append(round(float(theta[node_id]), 4))
```

### 2c. Build 1-hop edge lookup (post-loop)

After the inference loop, construct `node_edges` dict mapping:
`node_id → list[{"neighbor_id", "direction", "attention_sequence"}]`

Direction is `"outgoing"` if `edge[0] == node_id`, else `"incoming"`.

### 2d. Per-node JSON schema (`nodes/node_{id}.json`)

```jsonc
{
  "id": 0, "location_name": "...", "lat": ..., "lon": ..., "is_simulated": false,
  "zinb_series": [{ "t": 0, "mu": 12.3, "theta": 5.1, "std_dev": 4.1 }],
  "ground_truth_series": [12.0, 14.5, ..., null],
  "imputed_series": [11.8, 14.2, ...],
  "mask_series": [false, true, ...],
  "edges_1hop": [{ "neighbor_id": 3, "direction": "outgoing", "attention_sequence": [...] }]
}
```

### 2e. Master index JSON schema (`master_index.json`)

```jsonc
{
  "metadata": { /* same 9-field block */ },
  "nodes": [{ "id": 0, "location_name": "...", "lat": ..., "lon": ..., "is_simulated": false,
               "timeseries": [{ "t": 0, "prediction": 12.3, "uncertainty": 4.1 }] }],
  "edges": [{ "source": 0, "target": 1, "attention_sequence": [...] }]
}
```

### 2f. NaN assertions before each write

```python
for entry in node_data["zinb_series"]:
    assert not math.isnan(entry["mu"]),     f"NaN in node {node_id} zinb mu at t={entry['t']}"
    assert not math.isnan(entry["theta"]),  f"NaN in node {node_id} zinb theta at t={entry['t']}"
    assert not math.isnan(entry["std_dev"]),f"NaN in node {node_id} zinb std_dev at t={entry['t']}"
for v in node_data["imputed_series"]:
    assert not math.isnan(v), f"NaN in node {node_id} imputed_series"
```

### 2g. Desk Check — ZINB Tripwire (CLAUDE.md §8)

| Scenario | mu value | Tripwire fires? |
|:---|:---|:---|
| softplus(3.0) + eps | 3.049 > 0 ✓ | No |
| softplus(0.0) + eps | 0.693 + ε > 0 ✓ | No |
| Unclamped 0.0 | 0.0 NOT > 0 | **Yes — crashes immediately** |
| NaN from degenerate grad | NaN NOT > 0 | **Yes — caught by existing `assert not torch.isnan(mu)`** |

All valid model outputs pass. Pathological values crash immediately. ✓

---

## Step 3 — Cross-View Navigation (Floating Button)

Add a Material UI `Fab` (Floating Action Button) to `MapScene.tsx` linking to `/analytics`.

```tsx
// Bottom-right corner, above the TimeSlider
import Fab from '@mui/material/Fab';
import BarChartIcon from '@mui/icons-material/BarChart';
import Link from 'next/link';

<Link href="/analytics" style={{ position: 'absolute', bottom: 96, right: 24, zIndex: 20 }}>
  <Fab size="medium" aria-label="Open analytics" sx={{ bgcolor: 'rgba(30,30,30,0.9)', color: '#60a5fa' }}>
    <BarChartIcon />
  </Fab>
</Link>
```

`MapScene.tsx` is already a Client Component (`"use client"`), so no wrapper change needed.

---

## Step 4 — Master-Detail Analytics Layout

### 4a. New file: `frontend/src/app/analytics/page.tsx`

Client Component (`"use client"`). Fetches `master_index.json` once on mount.

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  AppBar: "NSTAR Analytics"          [← Back to Map]   │
├──────────────────────────────────────────────────────┤
│  [Omnibar — filter by location name or node ID]       │
├──────────────────────────────────────────────────────┤
│  MUI DataGrid (virtualized, ~40vh, multi-select)      │
│  Columns: id | location_name | is_simulated |         │
│           avg_μ | max_μ | avg_σ                       │
├──────────────────────────────────────────────────────┤
│  MUI Tabs                                             │
│  [ Tab 1: Forecast Variance ] [ Tab 2: — ] [Tab 3:—]  │
├──────────────────────────────────────────────────────┤
│  Tab panel content (fills remaining height)           │
└──────────────────────────────────────────────────────┘
```

### 4b. Omnibar

A controlled MUI `TextField` with `variant="outlined"`, `placeholder="Filter nodes…"`.
Filters the DataGrid rows client-side on each keystroke by matching `location_name` (case-insensitive
`includes`) or exact `id` match. Zero external deps.

### 4c. DataGrid columns (computed from `master_index.json`)

| field | headerName | type |
|:---|:---|:---|
| `id` | ID | number |
| `location_name` | Location | string |
| `is_simulated` | Simulated | boolean (checkmark render) |
| `avg_mu` | Avg μ | number (4dp), computed from `node.timeseries` |
| `max_mu` | Peak μ | number (4dp) |
| `avg_sigma` | Avg σ | number (4dp) |

Default sort: `avg_mu` descending.
Multi-select via checkbox column (`checkboxSelection={true}`).
Selected row IDs → `selectedNodeIds: number[]` state drives all tab panels.

### 4d. Tabs scaffold

```tsx
<Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)}>
  <Tab label="Forecast Variance" value={0} />
  <Tab label="Network Influence"  value={1} disabled />
  <Tab label="Self-Healing Log"   value={2} disabled />
</Tabs>
```

Tabs 2 and 3 are disabled (not yet implemented). They will be enabled in future steps.

---

## Step 5 — Tab 1: Forecast Variance

### 5a. Data contract

Tab 1 only uses `master_index.json` data (already loaded). No per-node file fetch needed.

For each selected node, the `timeseries` array provides:
- `t`: timestep index
- `prediction`: μ (ZINB mean)
- `uncertainty`: σ = std_dev (already computed as √(μ + μ²/θ) in the exporter)

**CI derivation:**
```
ci_lower = max(0, prediction - 1.96 × uncertainty)
ci_upper = prediction + 1.96 × uncertainty
```

### 5b. Chart implementation (Recharts `ComposedChart`)

```tsx
<ComposedChart data={chartData}>
  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
  <XAxis dataKey="t" />
  <YAxis />
  <Tooltip />
  <Legend />
  {/* CI band */}
  <Area type="monotone" dataKey="ci_upper" fill="#3b82f680" stroke="none" />
  <Area type="monotone" dataKey="ci_lower" fill="#0f172a"   stroke="none" />
  {/* μ line */}
  <Line type="monotone" dataKey="prediction" stroke="#60a5fa" dot={false} strokeWidth={2} />
</ComposedChart>
```

When multiple nodes are selected: each node gets its own `<Line>` keyed by `node_id`.
CI bands are shown for all selected nodes (each with a distinct hue offset).

**Defensive:** If `selectedNodeIds.length === 0`, render `<Alert severity="info">Select one or more nodes from the table above to plot their forecast variance.</Alert>`.

### 5c. Recharts CI band technique

Recharts `Area` requires a single numeric range. The standard pattern for a CI band is:
1. Build `chartData` with both `ci_lower` and `ci_upper` fields.
2. First `Area` plots `ci_upper` with fill color, stroke none.
3. Second `Area` plots `ci_lower` with the **background color** as fill (effectively clips the bottom).

This is a known Recharts idiom — no custom shape renderers needed.

---

## Step 6 — TypeScript Types (`frontend/src/types/payload.ts`)

Add new interfaces without breaking existing code:

```typescript
// NEW: per-node analytics file
export interface ZINBEntry   { t: number; mu: number; theta: number; std_dev: number; }
export interface HopEdge     { neighbor_id: number; direction: "outgoing" | "incoming"; attention_sequence: number[]; }
export interface NodeDetailPayload {
  id: number; location_name: string; lat: number; lon: number; is_simulated: boolean;
  zinb_series: ZINBEntry[];
  ground_truth_series: (number | null)[];
  imputed_series: number[];
  mask_series: boolean[];
  edges_1hop: HopEdge[];
}

// UPDATED: NodeEntry no longer carries imputed/GT/mask (those are in NodeDetailPayload)
export interface NodeEntry {
  id: number; location_name: string; lat: number; lon: number; is_simulated: boolean;
  timeseries: TimestepEntry[];
}

// ADDED: top-level alias for master_index.json
export interface MasterIndex { metadata: PayloadMetadata; nodes: NodeEntry[]; edges: EdgeEntry[]; }

// KEPT for backward compatibility (NodeInspector currently references UIPayload)
export type UIPayload = MasterIndex;
```

`NodeInspector.tsx` will be updated in Step 7 to use `NodeDetailPayload` via a lazy fetch.

---

## Step 7 — Migrate Existing Components to Sharded Data

### 7a. `MapVisualizer.tsx`

One-line change (line 207):
```typescript
// BEFORE:
const res = await fetch("/data/ui_payload.json");
// AFTER:
const res = await fetch("/data/master_index.json");
```

`UIPayload = MasterIndex`, so no other types need to change.

### 7b. `NodeInspector.tsx`

Replace the monolithic `ui_payload.json` fetch with a lazy per-node fetch:

```typescript
const [nodeDetail, setNodeDetail] = useState<NodeDetailPayload | null>(null);
useEffect(() => {
  if (selectedNodeId === null) { setNodeDetail(null); return; }
  fetch(`/data/nodes/node_${selectedNodeId}.json`)
    .then(r => r.ok ? r.json() : null)
    .then((d: NodeDetailPayload | null) => setNodeDetail(d))
    .catch(() => setNodeDetail(null));
}, [selectedNodeId]);
```

Update display fields: `nodeDetail.zinb_series[currentTimestep].mu` and `.std_dev`.
The existing `NodeChart` displays `ground_truth_series`, `imputed_series`, `mask_series` from `nodeDetail`
— same field names, so the SVG rendering code is unchanged.

Remove the large `UIPayload` state and the `fetched` ref guard (no longer needed).

---

## Step 8 — Package Installation (Manual User Action Required)

**The agent cannot run `docker` commands.** The user must run these on the host:

```bash
docker compose exec frontend-dev npm install recharts @mui/x-data-grid
docker compose exec frontend-dev npm install @mui/icons-material
```

MUI core + emotion are already installed (MapVisualizer uses them).

**Verify (inside container, not via host node_modules):**
```bash
docker compose exec frontend-dev node -e "require('recharts'); require('@mui/x-data-grid'); console.log('OK')"
```

---

## Step 9 — Validation Checkpoints

| Check | Command | Pass criterion |
|:---|:---|:---|
| Python sharding | `python src/export_ui_payload.py` | No assertion failures; files written to `frontend/public/data/` |
| TypeScript types | `docker compose exec frontend-dev npx tsc --noEmit` | 0 errors |
| Map route | `http://localhost:3000/` | Map renders, floating analytics button visible |
| Analytics route | `http://localhost:3000/analytics` | AppBar, Omnibar, DataGrid all render; no console errors |
| Tab 1 | Select 1+ rows in DataGrid | Recharts chart renders μ line + CI band |
| Telemetry | Trigger JS error in console | Entry appears in `/workspace/frontend_runtime_errors.log` |

---

## Step 10 — EXPERIMENTS.md Ledger Update

Per CLAUDE.md §6, append a row upon successful rendering confirmation. **Task is not complete until this row is written.**

---

## Files to Create / Modify

| Path | Action | Description |
|:---|:---|:---|
| `frontend/src/app/api/log/route.ts` | **CREATE** | Telemetry POST handler → appends to `frontend_runtime_errors.log` |
| `frontend/src/app/layout.tsx` | **EDIT** | Add `GlobalErrorReporter` client component |
| `src/export_ui_payload.py` | **EDIT** | Write sharded `master_index.json` + `node_{id}.json`; add theta; ZINB tripwires |
| `frontend/src/types/payload.ts` | **EDIT** | Add `ZINBEntry`, `HopEdge`, `NodeDetailPayload`, `MasterIndex`; simplify `NodeEntry` |
| `frontend/src/components/MapScene.tsx` | **EDIT** | Add MUI Fab linking to `/analytics` |
| `frontend/src/components/MapVisualizer.tsx` | **EDIT** | Fetch `master_index.json` instead of `ui_payload.json` |
| `frontend/src/components/NodeInspector.tsx` | **EDIT** | Replace monolithic fetch with lazy per-node fetch of `node_{id}.json` |
| `frontend/src/app/analytics/page.tsx` | **CREATE** | Omnibar + DataGrid + Tabs layout; Tab 1 Recharts Forecast Variance |
| `EXPERIMENTS.md` | **EDIT** | Append ledger row on successful rendering |

---

## Human-in-the-Loop Pause Points

1. **After Step 3 (navigation button):** Pause — ask user to confirm floating button renders.
2. **After Step 4 (Analytics layout):** Pause — ask user to confirm Omnibar + DataGrid render.
3. **After Step 5 (Tab 1 chart):** Pause — ask user to confirm chart renders and CI bands are visible.

---

**Awaiting explicit user approval before implementation per CLAUDE.md §1.**
