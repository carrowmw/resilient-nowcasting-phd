# Plan: ZINB Calibration Diagnostic UI on Main Map View

**Phase Anchor:** `NSTAR_PHASES.md` → Phase 1 (Static Data Bridge ONLY). No live API.
**North Stars consulted:** `NSTAR_BACKEND_ARCHITECTURE.md`, `NSTAR_FRONTEND_ARCHITECTURE.md`, `NSTAR_MLOPS.md`
**Date:** 2026-05-11
**Branch:** `main`

---

## Pre-flight Observations (from code reading)

| Observation                                                             | File                                     | Impact                                             |
| :---------------------------------------------------------------------- | :--------------------------------------- | :------------------------------------------------- |
| `pi_data[node_id]` already collected (length `num_timesteps`)           | `export_ui_payload.py`                   | Can inject directly into `zinb_series`             |
| `mask_series[node_id]` already collected (length `num_timesteps`)       | `export_ui_payload.py`                   | Alias as `is_missing` inside `zinb_series`         |
| `edge_attention_data: Dict[(src,tgt), List[float]]` available post-loop | `export_ui_payload.py`                   | Source for ego-graph top-K computation             |
| `nodes_meta` carries lat/lon for all nodes (real + simulated)           | `export_ui_payload.py`                   | Embed coordinates into `ego_graph_edges`           |
| Node selection already uses `onClick` (single-click)                    | `MapVisualizer.tsx` ln 88/110            | No code change needed for this sub-task            |
| Analytics Fab is currently at `bottom: 196, right: 24`                  | `MapScene.tsx` ln 32-35                  | Must move to `top: 24, left: 24`                   |
| `edges_1hop` is per-node and not consumed by `MapVisualizer`            | `NodeInspector.tsx`, `MapVisualizer.tsx` | New `ego_graph_edges` field is the correct conduit |
| Recharts is already installed in the frontend container                 | `analytics/page.tsx`                     | Can be imported in `NodeInspector.tsx` immediately |

---

## Step 0 — Auto-commit rollback point

```bash
git add .
git commit -m "Auto-commit before ZINB Diagnostic UI on Map View"
```

---

## Step 1 — Backend: `src/export_ui_payload.py`

### 1a. New constant

```python
TOP_K_EGO_EDGES = 5   # top K incoming + top K outgoing per node
```

### 1b. New helper: `build_ego_graph_edges`

After the `compute_two_extremes` function, add:

```python
def build_ego_graph_edges(
    node_id: int,
    edge_attention_data: Dict[Tuple[int, int], List[float]],
    node_coord_map: Dict[int, Tuple[float, float]],   # {id: (lat, lon)}
    top_k: int = TOP_K_EGO_EDGES,
) -> List[Dict]:
```

**Algorithm:**

1. Iterate `edge_attention_data.items()`.
2. For each `(src, tgt)`:
   - If `tgt == node_id` → incoming edge; if `src == node_id` → outgoing edge.
3. For matched edges: compute `mean_w = float(np.mean(attn_seq))`.
4. Tripwire: `assert not math.isnan(mean_w)`.
5. Skip any edge where either endpoint's lat or lon is `float('nan')` (external data quality; log a warning rather than crashing, since coordinate availability is outside model control).
6. Sort incoming by `mean_w` descending → take top `K`. Same for outgoing.
7. Return combined list of dicts:

```json
{
  "source_id": int,
  "target_id": int,
  "direction": "incoming" | "outgoing",
  "edge_weight": float (4 d.p.),
  "source_lat": float,
  "source_lon": float,
  "target_lat": float,
  "target_lon": float
}
```

### 1c. Desk Check — Ego-Graph Function (CLAUDE.md §8)

| Scenario                                                      | edge_attention_data                     | Expected result                                       |
| :------------------------------------------------------------ | :-------------------------------------- | :---------------------------------------------------- |
| Node 0 has 3 incoming edges, weights [0.9, 0.4, 0.1], top_k=2 | {(1,0):[0.9], (2,0):[0.4], (3,0):[0.1]} | Returns 2 incoming entries, sorted desc: [0.9, 0.4]   |
| Node 0 has 10 outgoing edges, top_k=5                         | {(0,1):[w], …, (0,10):[w]}              | Exactly 5 outgoing entries                            |
| Edge with NaN mean (empty attn_seq)                           | {(1,0):[]}                              | `np.mean([])` → NaN → `math.isnan` → logged + skipped |
| Node has no edges at all                                      | {}                                      | Returns empty list []                                 |

### 1d. Extend `zinb_series` entries to include `pi` and `is_missing`

In the per-node write loop (section 11), update the `zinb_series` construction:

```python
# Explicit alignment tripwire
assert len(mask_series[node_id]) == len(pi_data[node_id]) == num_timesteps, (
    f"TRIPWIRE: temporal misalignment for node {node_id}: "
    f"mask_series={len(mask_series[node_id])}, "
    f"pi_data={len(pi_data[node_id])}, expected={num_timesteps}"
)

zinb_series: List[Dict] = []
for t_idx in range(num_timesteps):
    entry: Dict = {
        "t": t_idx,
        "mu": timeseries_data[node_id][t_idx]["prediction"],
        "theta": theta_data[node_id][t_idx],
        "std_dev": timeseries_data[node_id][t_idx]["uncertainty"],
        "pi": pi_data[node_id][t_idx],          # NEW
        "is_missing": mask_series[node_id][t_idx],  # NEW
    }
    # existing NaN tripwires for mu, theta, std_dev ...
    assert not math.isnan(entry["pi"]), f"NaN in node {node_id} pi at t={t_idx}"
    zinb_series.append(entry)
```

### 1e. Build `ego_graph_edges` and inject into per-node JSON

After the `zinb_series` loop:

```python
# Build coordinate lookup for this invocation
node_coord_map: Dict[int, Tuple[float, float]] = {
    m["id"]: (m["lat"], m["lon"]) for m in nodes_meta
}
ego_edges = build_ego_graph_edges(
    node_id, edge_attention_data, node_coord_map
)

node_data: Dict = {
    ...existing fields...,
    "zinb_series": zinb_series,          # now includes pi + is_missing
    "ego_graph_edges": ego_edges,        # NEW
}
```

### 1f. Per-node schema assertions before write (new checks)

```python
assert isinstance(node_data["ego_graph_edges"], list)
for e in node_data["ego_graph_edges"]:
    assert e["edge_weight"] >= 0.0, f"Negative edge_weight in node {node_id} ego_graph"
    assert e["direction"] in ("incoming", "outgoing")
# Verify pi + is_missing are present in every zinb entry
for entry in node_data["zinb_series"]:
    assert "pi" in entry and "is_missing" in entry, \
        f"Node {node_id}: zinb_series entry missing pi or is_missing"
```

---

## Step 2 — Frontend Types: `frontend/src/types/payload.ts`

### 2a. Extend `ZINBEntry`

```typescript
export interface ZINBEntry {
  t: number;
  mu: number;
  theta: number;
  std_dev: number;
  pi: number; // NEW — zero-inflation probability ∈ (0,1)
  is_missing: boolean; // NEW — true if this timestep was artificially masked before imputation
}
```

### 2b. Add `AttentionEdge`

```typescript
export interface AttentionEdge {
  source_id: number;
  target_id: number;
  direction: "incoming" | "outgoing";
  edge_weight: number; // mean GAT attention weight over all timesteps
  source_lat: number;
  source_lon: number;
  target_lat: number;
  target_lon: number;
}
```

### 2c. Update `NodeDetailPayload`

Add:

```typescript
ego_graph_edges?: AttentionEdge[];  // absent in pre-Iteration-4b payloads
```

---

## Step 3 — Zustand Store: `frontend/src/store/useTimeStore.ts`

Add one field and one action to share ego-graph data between NodeInspector (writer) and MapVisualizer (reader):

```typescript
// In TimeState interface:
egoGraphEdges: AttentionEdge[] | null;
setEgoGraphEdges: (edges: AttentionEdge[] | null) => void;

// In create(...):
egoGraphEdges: null,
setEgoGraphEdges: (edges) => set({ egoGraphEdges: edges }),
```

Import `AttentionEdge` from `@/types/payload` (type-only import to avoid circular deps).

---

## Step 4 — Node Inspector: `frontend/src/components/NodeInspector.tsx`

### 4a. Zustand writes on node-data load

In the `useEffect` that fetches `node_{id}.json`:

```typescript
.then((d: NodeDetailPayload | null) => {
  setNodeDetail(d);
  setEgoGraphEdges(d?.ego_graph_edges ?? null);  // NEW — share with MapVisualizer
})
// On cleanup / null selection:
setEgoGraphEdges(null);
```

### 4b. MNAR Ghosting Timeline (new Recharts section below existing SVG chart)

**Data preparation** (`useMemo`):

```typescript
// Collect contiguous runs of is_missing === true for ReferenceArea spans
const missingRuns = useMemo(() => {
  if (!nodeDetail) return [];
  const runs: { x1: number; x2: number }[] = [];
  let start: number | null = null;
  nodeDetail.zinb_series.forEach(({ t, is_missing }) => {
    if (is_missing && start === null) start = t;
    if (!is_missing && start !== null) {
      runs.push({ x1: start, x2: t - 1 });
      start = null;
    }
  });
  if (start !== null)
    runs.push({ x1: start, x2: nodeDetail.zinb_series.length - 1 });
  return runs;
}, [nodeDetail]);
```

**Chart** (Recharts `LineChart` inside `ResponsiveContainer`, height 110px):

- X axis: `dataKey="t"`, `tick={{ fontSize: 9 }}`
- Y axis: domain `[0, 1]`, `tick={{ fontSize: 9 }}`, width 28
- `CartesianGrid` at low opacity
- One `<Line dataKey="pi" stroke="#a78bfa" dot={false} strokeWidth={1.5} />` (violet, matching π symbol)
- `{missingRuns.map(r => <ReferenceArea key={...} x1={r.x1} x2={r.x2} fill="rgba(239,68,68,0.18)" />)}`
- Vertical cursor `<ReferenceLine x={currentTimestep} stroke="rgba(255,255,255,0.4)" />` (matches existing chart cursor)
- No legend (the chart label above is sufficient)
- Label row above chart: `"π (zero-inflation probability) — red shading = artificially masked"`

**Defensive rendering:** wrap in `{nodeDetail.zinb_series?.[0]?.pi !== undefined ? <chart> : null}` so old payloads without `pi` degrade gracefully.

### 4c. Attention Edge Rank List

Below the pi chart, new section titled "Top Attention Edges":

```
┌──────────────────────────────────────┐
│ Top Attention Edges                  │
│ ← Node 7    ████████░░  0.843        │
│ ← Node 3    ██████░░░░  0.612        │
│ → Node 12   ████░░░░░░  0.441        │
│   (etc.)                             │
└──────────────────────────────────────┘
```

Each row:

- Direction arrow: `←` for incoming, `→` for outgoing (colored: incoming = blue-400, outgoing = amber-400)
- Neighbor ID (source for incoming, target for outgoing)
- A CSS `width: {weight * 100}%` inline-progress bar (max width 80px, height 3px)
- Numeric weight to 3 d.p.

Rendered from `nodeDetail.ego_graph_edges ?? []`, sorted by `edge_weight` descending.

**Defensive rendering:** wrap in `{(nodeDetail.ego_graph_edges?.length ?? 0) > 0 ? <list> : null}`.

---

## Step 5 — Map Visualizer: `frontend/src/components/MapVisualizer.tsx`

### 5a. Hover scaling

Add `hoveredNodeIdRef = useRef<number | null>(null)` at component level.

Extend `buildLayers` signature: `hoveredNodeId: number | null`.

In the radius function:

```typescript
getRadius: (d) => nodeRadius(d, timestep) * (d.id === hoveredNodeId ? 1.6 : 1.0),
```

On both ScatterplotLayers:

```typescript
autoHighlight: true,
highlightColor: [255, 255, 255, 60],
onHover: ({ object }) => {
  const newId = object ? (object as NodeEntry).id : null;
  if (hoveredNodeIdRef.current !== newId) {
    hoveredNodeIdRef.current = newId;
    rebuildLayers();   // helper that calls overlay.setProps(...)
  }
},
```

Extract layer-rebuild logic into a `rebuildLayers()` closure inside the second `useEffect` to avoid duplication.

### 5b. Ego-Graph ArcLayer

Read `egoGraphEdges` from the store (subscribe alongside `currentTimestep`/`selectedNodeId`).

Add to `buildLayers`:

```typescript
const egoArcLayer = new ArcLayer<AttentionEdge>({
  id: "ego-graph-layer",
  data: egoGraphEdges ?? [],
  getSourcePosition: (d) => [d.source_lon, d.source_lat],
  getTargetPosition: (d) => [d.target_lon, d.target_lat],
  getSourceColor: (d) =>
    d.direction === "incoming" ? [96, 165, 250, 220] : [251, 113, 133, 220],
  getTargetColor: (d) =>
    d.direction === "incoming" ? [96, 165, 250, 180] : [251, 113, 133, 180],
  getWidth: (d) => Math.max(2, d.edge_weight * 12),
  widthUnits: "pixels",
  updateTriggers: { data: [selectedNodeId] },
  visible: selectedNodeId !== null && (egoGraphEdges?.length ?? 0) > 0,
});
return [realLayer, simLayer, arcLayer, egoArcLayer];
```

Color convention: **blue** = incoming attention (information flowing in), **rose** = outgoing attention (this node attending outward). Width scales linearly with `edge_weight`.

### 5c. Single-click confirmation

`onClick` is already present on both ScatterplotLayers (lines 88 and 110). **No code change needed.** The plan documents this for traceability.

---

## Step 6 — Map Scene: `frontend/src/components/MapScene.tsx`

Move the Fab `<Link>` from `{bottom: 196, right: 24}` to `{top: 24, left: 24}`.

Also remove the `bottom` and `right` inline style properties and replace with `top` and `left`.

---

## Step 7 — EXPERIMENTS.md Ledger Update

Per CLAUDE.md §6, append a row upon completion.

---

## Files to Create / Modify

| Path                                        | Action   | Description                                                                                                           |
| :------------------------------------------ | :------- | :-------------------------------------------------------------------------------------------------------------------- |
| `src/export_ui_payload.py`                  | **EDIT** | Add `build_ego_graph_edges`; extend `zinb_series` with `pi`+`is_missing`; inject `ego_graph_edges` into per-node JSON |
| `frontend/src/types/payload.ts`             | **EDIT** | Extend `ZINBEntry` (pi, is_missing); add `AttentionEdge`; update `NodeDetailPayload`                                  |
| `frontend/src/store/useTimeStore.ts`        | **EDIT** | Add `egoGraphEdges` + `setEgoGraphEdges`                                                                              |
| `frontend/src/components/NodeInspector.tsx` | **EDIT** | Add pi MNAR timeline (Recharts), ego-graph edge list, write to store on load                                          |
| `frontend/src/components/MapVisualizer.tsx` | **EDIT** | Add ego-graph ArcLayer, hover-scaling, read egoGraphEdges from store                                                  |
| `frontend/src/components/MapScene.tsx`      | **EDIT** | Move Fab to top-left                                                                                                  |
| `EXPERIMENTS.md`                            | **EDIT** | Append ledger row                                                                                                     |

---

## No-op confirmations (strict constraints respected)

- **Zero Host Install:** No new packages. Recharts already in container, Deck.gl ArcLayer already imported in MapVisualizer.
- **Zero Frontend Math:** `build_ego_graph_edges` (sorting, mean weight) is computed in Python. NodeInspector and MapVisualizer are pure renderers of pre-aggregated data.
- **Offensive Tripwires:** alignment assert, NaN tripwire on `pi` in `zinb_series`, non-negative edge weight assert, direction enum assert.
- **No live API:** Static JSON bridge only.
- **Single-click:** Already implemented — confirmed at `MapVisualizer.tsx:88,110`.

---

**Awaiting explicit user approval before implementation per CLAUDE.md §1.**
