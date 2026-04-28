# Plan: Phase 1 Analytics Upgrade — Imputation Visualization & Ego-Graph Inspector

**Phase Anchor:** PHASES.md → Phase 1 (Static Data Bridge ONLY). No live API. The
`ui_payload.json` static file remains the single bridge.

**North Stars consulted:** `BACKEND_ARCHITECTURE.md` (StandardScaler / data scaling /
imputation decoder), `FRONTEND_ARCHITECTURE.md` (defensive rendering, simulated
node distinction, ego-graph isolation), `PHASES.md` (Phase 1 constraints).

---

## Pre-flight (Safety First / Branching Protocol per CLAUDE.md §1)

1. **Branch confirmation.** Confirm with user whether to stay on `main` (matches all
   prior ledger entries) or to cut a new feature branch.
2. **Auto-commit rollback point** before any multi-file changes:
   `git add . && git commit -m "Auto-commit before Phase 1 Analytics Upgrade"`.

---

## Step 1 — `src/export_ui_payload.py` (Imputation in Payload)

### 1a. Modify `src/pipeline.py` analysis hook
Currently `return_analysis=True` returns `{edge_index, edge_weight, attention_weights, mu, theta, pi}`
but discards `imputed_features` (drops the first element of `self.decoders(...)`).
Add `imputed_features` to the returned dict — no other call sites depend on this
analysis path.

### 1b. Enable masking in the export run
Currently `RealIoTDataset(... mask_prob=0.0)` — with zero masking, `mask_series`
would be uniformly false, defeating the purpose of visualizing imputation. Set
`mask_prob=0.15` (training default) to inject a deterministic ~15% mask
(seed=42 already set at the top of `main()`).

### 1c. Inverse-transform imputed values into raw-count space (DESK CHECK MANDATE)

`StandardScaler` was fit at dataset init on flattened dynamic features of shape
`(num_times * num_nodes, 3)` — channel order is `[value, Humidity, Solar Radiation]`.
The scaler stores `mean_[3]` and `scale_[3]` (one per feature).

`StandardScaler.transform(x) = (x - mean_) / scale_`
`StandardScaler.inverse_transform(z) = z * scale_ + mean_`

**Desk Check (extreme cases):**

| Scenario | Input (raw count) | Expected scaled | Expected inverse |
| :--- | :--- | :--- | :--- |
| Mean ≈ 50, scale ≈ 25 | x = 100 | (100−50)/25 = **2.0** | 2.0·25 + 50 = **100** ✓ |
| Mean ≈ 50, scale ≈ 25 | x = 0 | (0−50)/25 = **−2.0** | −2.0·25 + 50 = **0** ✓ |
| Mean ≈ 50, scale ≈ 25 | x = 1 (extreme low batch) | (1−50)/25 = **−1.96** | −1.96·25 + 50 = **1** ✓ |
| Pathological scale ≈ 0 | — | divide-by-zero (sklearn auto-bumps to 1.0) | safe |

We extract channel 0 (traffic count) only:
`unscaled = scaled * scaler.scale_[0] + scaler.mean_[0]`

We must **NOT** apply softplus / clamping to the imputed traffic — the imputation
decoder targets `target_impute` which is the *scaled* dynamic features (per
`dataset.py` line 154: `target_impute = dyn_tensor.clone()` after scaling).
After inverse-transform, residual negative reconstructions are honest model
artifacts and must be displayed as-is. (We may `clamp(min=0)` for the *displayed*
ground truth series to match the dataset's `clamp(min=0)` on raw counts.)

**Tripwire:** assert `not torch.isnan(imputed_features).any()` before saving.
Assert `scaler.scale_[0] > 0` before inverse-transform.

### 1d. Per-timestep extraction strategy (96 windows → length-96 arrays)

For each window `idx ∈ [skip, skip + 96)` and each node `n`:
- `imputed_at_t = inverse(imputed_features[0, n, -1, 0])` — last step of the
  window's reconstruction, traffic channel.
- `ground_truth_at_t = unscaled_traffic[idx + seq_len - 1, n]` — pulled directly
  from `dataset.unscaled_traffic` (already in raw-count space; no inverse needed).
  For **simulated nodes**, `ground_truth_at_t = null` (no physical sensor).
- `mask_at_t = node_mask[n, -1, 0]` for real nodes; for simulated nodes,
  `mask_at_t = True` (every input was NaN → fully imputed by definition).

Result: each node carries three arrays of length 96.

### 1e. Updated JSON schema (per node)

```jsonc
{
  "id": 0,
  "location_name": "...",
  "lat": 54.97, "lon": -1.61,
  "is_simulated": false,
  "timeseries": [{ "t": 0, "prediction": ..., "uncertainty": ... }, ...],
  "ground_truth_series": [12.0, 14.5, ..., null|number],   // length 96
  "imputed_series":      [11.8, 14.2, ..., number],         // length 96
  "mask_series":         [false, true, ..., boolean]        // length 96
}
```

Schema assertions on write: every node has all three arrays of length
`num_timesteps`; `is_simulated == true` for exactly `num_simulated` nodes.

---

## Step 2 — Frontend State & Types

### 2a. `frontend/src/types/payload.ts`
Extend `NodeEntry`:
```ts
ground_truth_series: (number | null)[];
imputed_series: number[];
mask_series: boolean[];
```

### 2b. `frontend/src/store/useTimeStore.ts`
Add to the `TimeState` interface and store:
```ts
selectedNodeId: number | null;          // default null
attentionThreshold: number;             // default 0.05
setSelectedNodeId: (id: number | null) => void;
setAttentionThreshold: (v: number) => void;
```
Existing tests (`tests/store.test.ts`) will need new cases; existing tests must
continue to pass.

---

## Step 3 — `frontend/src/components/MapVisualizer.tsx`

### 3a. Hollow-ring vs solid-disc for ScatterplotLayer
Per FRONTEND_ARCHITECTURE.md §A: simulated nodes must be visually distinct.
A single `ScatterplotLayer` accepts only one stroked/filled mode globally (the
flags are layer-wide, not per-row), so we **split into two layers**:
  - `nodes-real` — `data: payload.nodes.filter(n => !n.is_simulated)`,
    `stroked: false, filled: true`.
  - `nodes-simulated` — `data: payload.nodes.filter(n => n.is_simulated)`,
    `stroked: true, filled: false, lineWidthMinPixels: 3,
    getLineColor: getFillColor` (so the ring inherits the uncertainty heat).
Both layers share accessors and `updateTriggers: [timestep]`.

### 3b. Click-to-select node
Add `onClick: ({ object }) => setSelectedNodeId(object?.id ?? null)` on both
node layers, plus a click on empty map deselects (`map.on('click', ...)` falling
through when no Deck.gl object is hit — using `pickable: true`).

### 3c. ArcLayer: ego-graph isolation + threshold filtering
Replace top-N filter with two new filters applied in sequence:

```ts
const { selectedNodeId, attentionThreshold } = useTimeStore(...);
const filtered = payload.edges
  .map(e => ({ edge: e, weight: e.attention_sequence[timestep] ?? 0 }))
  .filter(x => x.weight >= attentionThreshold)
  .filter(x =>
    selectedNodeId == null ||
    x.edge.source === selectedNodeId ||
    x.edge.target === selectedNodeId
  );
```
- `attentionThreshold` from Zustand drops low-weight noise.
- If `selectedNodeId` is set, restrict to ego-graph edges (source OR target ==
  selected).
- Keep an upper cap (e.g. `slice(0, 60)`) as defensive bound when
  `selectedNodeId == null`, to preserve the existing "no hairball" property.
- Add `selectedNodeId, attentionThreshold` to the `useEffect` dependency array
  so layers rebuild when filters change.
- Add both to layer `updateTriggers` for full Deck.gl correctness.

### 3d. Defensive rendering (FRONTEND_ARCHITECTURE.md §2)
If `node.imputed_series` is `undefined` (older payload), gracefully skip the new
inspector wiring rather than crash — `NodeInspector` already guards on
`selectedNodeId == null`, but type narrowing must allow it.

---

## Step 4 — `frontend/src/components/NodeInspector.tsx` (new)

A right-side floating panel; renders only when `selectedNodeId != null`.

**Stack choice:** plain SVG inside React (no new deps). Charting via Recharts is
a fine alternative, but adds ~120 KB; SVG is sufficient for two overlaid
length-96 series and respects the strict-defensive constraint.

**Layout:**
- Container: `position: absolute; right: 16px; top: 16px; width: 360px;`
  Tailwind dark glass panel.
- Header: location_name + `is_simulated` badge.
- SVG plot:
  - X axis: timestep 0–95.
  - Two polylines: ground truth (solid blue, skips `null` points), imputed
    (orange, dashed when `mask_series[t]` is true).
  - Faint vertical bands at masked timesteps (highlight the gap the model is
    healing).
  - Highlight the current `currentTimestep` with a vertical cursor.
- Close button calls `setSelectedNodeId(null)`.

**Defensive behavior** (per FRONTEND_ARCHITECTURE.md):
- If `selectedNode.ground_truth_series` is missing (older payload), hide chart
  and show "Imputation data unavailable" message.
- If `selectedNode == null` (stale id), set selectedNodeId back to null in a
  `useEffect`.

**Mount point:** add `<NodeInspector />` as a sibling of `<MapVisualizer />`
inside `MapScene.tsx`. Per the Next.js 16 App Router rules already used in
`MapScene.tsx`, the parent is a Client Component; the inspector reads from
Zustand directly so no prop wiring needed.

---

## Step 5 — Validation & Ledger

### 5a. Backend verification
- Run `python src/export_ui_payload.py` end-to-end (CPU). Confirm:
  - JSON file written without assertion failures.
  - `nodes[0].ground_truth_series.length == 96`, same for `imputed_series` and
    `mask_series`.
  - Simulated nodes have `mask_series` all true and `ground_truth_series` all
    null.
  - **Numeric desk check:** for one randomly chosen real node + timestep where
    `mask_series[t] == true`, manually re-derive
    `imputed_series[t]` from `imputed_features[0, n, -1, 0] * scale_[0] + mean_[0]`
    and assert exact match (within float tolerance).

### 5b. Frontend verification
- `cd frontend && npx tsc --noEmit` — must show 0 errors.
- `cd frontend && npm run test` — full Vitest suite passes (existing 33 tests +
  new store tests).
- `npm run dev` and visually verify: simulated rings, ego-graph isolation on
  click, threshold slider hides arcs.

### 5c. Ledger update (mandatory before claiming done)
Append a new dated row to `EXPERIMENTS.md` describing the upgrade:
prompt summary, files touched, schema changes, desk-check outcome, and test
results. A task is **not** done until this row is in.

---

## Files Touched (summary)
| Path | Type | Change |
| :--- | :--- | :--- |
| `src/pipeline.py` | edit | add `imputed_features` to analysis return dict |
| `src/export_ui_payload.py` | edit | masking + imputation + inverse-transform + new schema fields |
| `frontend/src/types/payload.ts` | edit | extend `NodeEntry` |
| `frontend/src/store/useTimeStore.ts` | edit | add `selectedNodeId`, `attentionThreshold` |
| `frontend/src/components/MapVisualizer.tsx` | edit | split scatterplot, ego-graph filter, click-select |
| `frontend/src/components/NodeInspector.tsx` | new | side panel chart |
| `frontend/src/components/MapScene.tsx` | edit | mount `<NodeInspector />` |
| `EXPERIMENTS.md` | edit | append ledger row |

## Risks / "Let It Burn" tripwires
- **Scaling shape mismatch** (priority): if `imputed_features` is not the exact
  expected `[B=1, N_total, T=12, D=3]`, an `assert` crashes immediately so we
  can diagnose rather than silently mis-index.
- **Edge map vs node id staleness** when `selectedNodeId` references a node
  that doesn't exist in the payload — guard at lookup time.
- **Next.js 16 breaking changes** — consult `node_modules/next/dist/docs/` if
  any new dynamic-import or RSC issue appears (per `frontend/AGENTS.md`).

---

**Awaiting explicit user approval before implementation per CLAUDE.md §1.**
