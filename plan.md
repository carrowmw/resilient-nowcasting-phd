# Iteration 4c Plan: Dynamic Temporal Ego-Graphs

**Phase Anchor:** `NSTAR_PHASES.md` → Phase 1 (Static Data Bridge ONLY). No live API.
**North Stars consulted:** `NSTAR_BACKEND_ARCHITECTURE.md`, `NSTAR_FRONTEND_ARCHITECTURE.md`
**Date:** 2026-05-11
**Branch:** `main`

---

## Goal

Replace static scalar `edge_weight` with full per-timestep `attention_sequence` arrays in ego-graph
edges, fix MapVisualizer interaction bugs, and enable dynamic top-5 edge ranking in NodeInspector.

---

## Step 0 — Auto-commit rollback point

```bash
git add .
git commit -m "Auto-commit before Iteration 4c — Dynamic Temporal Ego-Graphs"
```

---

## Step 1 — Backend: `src/export_ui_payload.py`

**Constant change:**
```python
TOP_K_EGO_EDGES = 10   # top 10 neighbors (client picks top-5 per timestep)
```

**`build_ego_graph_edges` signature change:**
```python
def build_ego_graph_edges(
    node_id: int,
    edge_attention_data: Dict[Tuple[int, int], List[float]],
    node_coord_map: Dict[int, Tuple[float, float]],
    num_timesteps: int,              # NEW — needed for tripwire
    top_k: int = TOP_K_EGO_EDGES,
) -> List[Dict]:
```

**Schema change — replace scalar with sequence:**
```python
# BEFORE:
"edge_weight": round(mean_w, 6)

# AFTER:
"weight_sequence": [round(w, 6) for w in attn_seq]
```

**Offensive tripwire (added inside loop, before appending):**
```python
assert len(attn_seq) == num_timesteps, (
    f"TRIPWIRE: edge ({src},{tgt}) attention_sequence length "
    f"{len(attn_seq)} != expected {num_timesteps}"
)
```

**Call site update** — pass `num_timesteps` argument.

---

## Step 2 — Frontend Types: `frontend/src/types/payload.ts`

**`AttentionEdge` change:**
```typescript
// BEFORE:
edge_weight: number;   // scalar mean

// AFTER:
weight_sequence: number[];   // full per-timestep attention weights, length = num_timesteps
```

No other interface changes (no other consumer references `edge_weight` directly).

---

## Step 3 — MapVisualizer Bug Fixes: `frontend/src/components/MapVisualizer.tsx`

### Bug A — `onHover` Highlight Lock
Current `onHover` returns `undefined` (implicit). Deck.gl requires `true` to continue hit-testing.
```typescript
// Fix: return true from both ScatterplotLayer onHover callbacks
onHover: ({ object }) => {
  onHoverNode(object ? object.id : null);
  return true;
},
```

### Bug B — ego-graph-layer missing `updateTriggers` for per-timestep visuals

The `getWidth` and `getSourceColor` accessors in `egoArcLayer` reference `timestep` via closure,
but without `updateTriggers` Deck.gl caches the accessors and never re-evaluates them.

```typescript
// Replace scalar edge_weight with weight_sequence[timestep]:
getWidth: (d) => Math.max(2, (d.weight_sequence[timestep] ?? 0) * 14),
getSourceColor: (d) => {
  const w = d.weight_sequence[timestep] ?? 0;
  const alpha = Math.round(80 + w * 175);   // 80–255
  return d.direction === "incoming"
    ? [96, 165, 250, alpha]
    : [251, 113, 133, alpha];
},
getTargetColor: (d) => {
  const w = d.weight_sequence[timestep] ?? 0;
  const alpha = Math.round(50 + w * 110);
  return d.direction === "incoming"
    ? [96, 165, 250, alpha]
    : [251, 113, 133, alpha];
},
updateTriggers: {
  data: [selectedNodeId],
  getWidth: [currentTimestep],            // NEW
  getSourceColor: [currentTimestep],      // NEW
  getTargetColor: [currentTimestep],      // NEW
},
```

The `buildLayers` function already receives `timestep` as a parameter — no signature change needed.

---

## Step 4 — Dynamic Edge Filtering: `frontend/src/components/NodeInspector.tsx`

### `EdgeList` component changes

Subscribe to `currentTimestep` from `useTimeStore`:
```typescript
const currentTimestep = useTimeStore((s) => s.currentTimestep);
```

Dynamic sort + slice via `useMemo`:
```typescript
const { topIncoming, topOutgoing } = useMemo(() => {
  const incoming = edges
    .filter((e) => e.direction === "incoming")
    .sort((a, b) => (b.weight_sequence[currentTimestep] ?? 0)
                   - (a.weight_sequence[currentTimestep] ?? 0))
    .slice(0, 5);
  const outgoing = edges
    .filter((e) => e.direction === "outgoing")
    .sort((a, b) => (b.weight_sequence[currentTimestep] ?? 0)
                   - (a.weight_sequence[currentTimestep] ?? 0))
    .slice(0, 5);
  return { topIncoming: incoming, topOutgoing: outgoing };
}, [edges, currentTimestep]);
```

Display value uses `weight_sequence[currentTimestep]`:
```typescript
const w = edge.weight_sequence[currentTimestep] ?? 0;
// bar width: Math.round(w * 100)%
// numeric: w.toFixed(3)
```

---

## Step 5 — TypeScript Verification

```bash
docker compose exec frontend npx tsc --noEmit
```

Fix any type errors before considering done.

---

## Step 6 — EXPERIMENTS.md Ledger

Append row:
- Date: 2026-05-11
- Branch: main
- Goal: Iteration 4c — Dynamic Temporal Ego-Graphs
- Outcome: Replaced scalar `edge_weight` with `weight_sequence[96]`, increased top-K to 10,
  added tripwire `len(attention_sequence) == num_timesteps`, fixed Deck.gl `onHover` highlight lock
  (return true), added per-timestep `updateTriggers` for ego-arc width/color, dynamic top-5
  incoming+outgoing edge ranking in NodeInspector keyed on `currentTimestep`.

---

## Files to Modify

| Path | Action | Description |
|:-----|:-------|:------------|
| `src/export_ui_payload.py` | EDIT | `TOP_K=10`, `build_ego_graph_edges` returns `weight_sequence`, tripwire |
| `frontend/src/types/payload.ts` | EDIT | `AttentionEdge.edge_weight → weight_sequence: number[]` |
| `frontend/src/components/MapVisualizer.tsx` | EDIT | `onHover` fix, ego-arc per-timestep accessors + updateTriggers |
| `frontend/src/components/NodeInspector.tsx` | EDIT | `EdgeList` useMemo dynamic sort by `weight_sequence[currentTimestep]` |
| `EXPERIMENTS.md` | EDIT | Ledger row |

---

**Awaiting explicit user approval before implementation per CLAUDE.md §1.**
