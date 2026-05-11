# Plan: Iteration 4 — ZINB Calibration Metrics & Dashboard Upgrade

**Phase Anchor:** `NSTAR_PHASES.md` → Phase 1 (Static Data Bridge ONLY). No live API.
**North Stars consulted:** `NSTAR_BACKEND_ARCHITECTURE.md`, `NSTAR_FRONTEND_ARCHITECTURE.md`, `NSTAR_MLOPS.md`
**Date:** 2026-05-11
**Branch:** `main`

---

## Pre-flight Checks

- `pi` is already returned by the model in analysis mode: `result['pi']` → shape `[1, num_nodes_total]`, values from `torch.sigmoid()` ∈ (0, 1).
- `ground_truth_series[node_id]` can contain `None` (simulated nodes) and `NaN` (real nodes with sensor outages, since `max(0.0, float(NaN)) → NaN`).
- `scipy.special.betainc` is available in the container's Python environment.
- **Branching question per CLAUDE.md §6:** This is a major feature addition. The user should confirm if a new branch is needed, or the work continues on `main`.

---

## Step 1 — Auto-commit rollback point

Before any file edits, execute:
```bash
git add .
git commit -m "Auto-commit before Iteration 4 ZINB Calibration Metrics"
```

---

## Step 2 — Backend: `src/export_ui_payload.py`

### 2a. New import

```python
from scipy.special import betainc
```

### 2b. New accumulator in inference loop setup

```python
pi_data: List[List[float]] = [[] for _ in range(num_nodes_total)]
```

### 2c. In the 24-hour inference loop — capture `pi` with tripwires

After `theta: torch.Tensor = result["theta"][0]`, add:

```python
pi_raw: torch.Tensor = result["pi"][0]   # [num_nodes_total] from sigmoid → (0,1)
assert (pi_raw > 0).all(),  f"t={t}: TRIPWIRE — pi has zero/negative values: {pi_raw.min()}"
assert (pi_raw < 1).all(),  f"t={t}: TRIPWIRE — pi has saturated-to-1 values: {pi_raw.max()}"
assert not torch.isnan(pi_raw).any(), f"t={t}: NaN in pi"

for node_id in range(num_nodes_total):
    pi_data[node_id].append(round(float(pi_raw[node_id]), 6))
```

### 2d. Three new post-loop helper functions

#### `compute_pit_histogram`

**Mathematical basis:**

The ZINB CDF at observed count `y`:

```
F_NB(y; mu, theta) = betainc(theta, floor(y)+1, theta/(theta+mu))
F_ZINB(y)          = pi + (1 - pi) * F_NB(y)
```

For each real node and each timestep where `ground_truth` is not None/NaN, compute the PIT value `u = F_ZINB(y_obs)`. Assert `0 <= u <= 1` (tripwire). Bin 10 equal-width bins across [0, 1]. Return array of `{bin_center, density}` dicts where density = count/total.

**Tripwires:**
- `assert 0.0 <= u <= 1.0` — PIT values outside [0,1] indicate a parametric error.
- `assert theta > 0 and mu > 0` — satisfied by prior per-step assertions; re-checked here.
- If `theta + mu == 0` the script crashes immediately (ZeroDivisionError, not silently suppressed).

```python
def compute_pit_histogram(
    ground_truth_series: List[List],          # [node][t] → float | None
    mu_series: List[List[float]],             # [node][t]
    theta_series: List[List[float]],          # [node][t]
    pi_series: List[List[float]],             # [node][t]
    num_real_nodes: int,
    num_bins: int = 10,
) -> List[Dict]:
```

Returns `[{"bin_center": float, "density": float}, ...]` (10 entries).

#### `compute_zero_inflation_reliability`

For each real node at each timestep where ground truth is not None/NaN:
- Record `(pi_predicted, is_true_zero)` where `is_true_zero = (y_obs == 0.0)`.
- Create 10 equal-width bins for pi ∈ [0, 1].
- For each bin, compute `observed_zero_fraction = count_true_zeros / total_in_bin`.
- Skip bins with 0 observations (set fraction to None, omit from output to avoid charting artifacts).

Returns `[{"pi_center": float, "observed_zero_fraction": float, "count": int}, ...]` (up to 10 entries).

#### `compute_two_extremes`

For one node:

```python
def compute_two_extremes(
    ground_truth: List,          # floats | None | NaN
    mu_predictions: List[float],
) -> Dict:
```

- **Drop mode:** filter to rows where ground truth is not None and not NaN. Compute `MAE_drop = mean(|mu - y|)`, `RMSE_drop = sqrt(mean((mu - y)^2))`. If no valid rows: `None`.
- **Fill mode:** replace None/NaN with `0.0`. Compute `MAE_fill`, `RMSE_fill` over all timesteps.

Returns `{"mae_drop_nan": float|None, "rmse_drop_nan": float|None, "mae_fill_nan": float, "rmse_fill_nan": float}`.

**Tripwires:**
- Assert no negative MAE/RMSE (mathematically impossible but catches implementation bugs).
- Assert all-fill arrays contain no NaN after substitution.

### 2e. Desk Check — PIT Tripwire (CLAUDE.md §8)

| Scenario | y_obs | mu | theta | pi | Expected u | Tripwire fires? |
|:---|:---|:---|:---|:---|:---|:---|
| softplus output, count=5 | 5.0 | 3.0 | 2.0 | 0.1 | 0.1 + 0.9 * betainc(2,6,0.4) ≈ 0.82 | No (∈ [0,1]) |
| High pi near structural zero | 0.0 | 0.5 | 1.0 | 0.95 | 0.95 + 0.05 * betainc(1,1,1/(1.5)) ≈ 0.983 | No |
| Unclamped pi=1.0 | — | — | — | 1.0 | NOT < 1 | **Yes — crashes** |
| NaN ground truth | NaN | — | — | — | Skipped by `None/NaN` guard | No crash |

### 2f. Integration into output files

**`master_index.json`** gains two top-level keys:
```json
{
  "pit_histogram": [{"bin_center": 0.05, "density": 0.09}, ...],
  "zero_inflation_reliability": [{"pi_center": 0.15, "observed_zero_fraction": 0.12, "count": 234}, ...]
}
```

Each node entry in `master_index["nodes"]` gains:
```json
{
  "two_extremes": {
    "mae_drop_nan": 4.21,
    "rmse_drop_nan": 6.87,
    "mae_fill_nan": 5.30,
    "rmse_fill_nan": 8.14
  }
}
```

**`nodes/node_{id}.json`** also gains a `two_extremes` key at the root (same 4 scalars).

### 2g. Schema assertions before write (new checks)

```python
assert len(master_index["pit_histogram"]) == 10
assert len(master_index["zero_inflation_reliability"]) <= 10
for node in master_index["nodes"]:
    te = node.get("two_extremes")
    assert te is not None
    assert te["mae_fill_nan"] >= 0
    assert te["rmse_fill_nan"] >= 0
```

---

## Step 3 — Frontend Types: `frontend/src/types/payload.ts`

Add the following new interfaces (zero breaking changes to existing types):

```typescript
// ZINB Calibration — PIT histogram bin
export interface PITBin {
  bin_center: number;  // midpoint of the [0,1] bin
  density: number;     // fraction of PIT values falling in this bin
}

// ZINB Calibration — Zero-inflation reliability bin
export interface ZeroInflationBin {
  pi_center: number;                // midpoint of the pi bin
  observed_zero_fraction: number;   // actual fraction of true zeros in this bin
  count: number;                    // number of observations in this bin
}

// Per-node Two-Extremes missingness bracket
export interface TwoExtremes {
  mae_drop_nan: number | null;   // null when no valid (non-NaN) ground truths
  rmse_drop_nan: number | null;
  mae_fill_nan: number;          // always present (NaN filled with 0)
  rmse_fill_nan: number;
}
```

Update `NodeEntry` (master_index nodes array) to include:
```typescript
two_extremes?: TwoExtremes;
```

Update `MasterIndex` to include:
```typescript
pit_histogram?: PITBin[];
zero_inflation_reliability?: ZeroInflationBin[];
```

Update `NodeDetailPayload` to include:
```typescript
two_extremes?: TwoExtremes;
```

---

## Step 4 — Frontend Analytics Page: `frontend/src/app/analytics/page.tsx`

### 4a. New imports

```typescript
import { BarChart, Bar, ScatterChart, Scatter, ReferenceLine,
         XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import type { PITBin, ZeroInflationBin, TwoExtremes } from "@/types/payload";
```

### 4b. Update `NodeRow` type

Add four new fields:
```typescript
mae_drop_nan: number | null;
rmse_drop_nan: number | null;
mae_fill_nan: number | null;   // null if two_extremes absent
rmse_fill_nan: number | null;
```

### 4c. Update `rows` useMemo to extract Two-Extremes scalars

```typescript
mae_drop_nan: node.two_extremes?.mae_drop_nan ?? null,
rmse_drop_nan: node.two_extremes?.rmse_drop_nan ?? null,
mae_fill_nan: node.two_extremes?.mae_fill_nan ?? null,
rmse_fill_nan: node.two_extremes?.rmse_fill_nan ?? null,
```

### 4d. Add 4 new columns to `columns` array

| field | headerName | width | type |
|:---|:---|:---|:---|
| `mae_drop_nan` | MAE (drop NaN) | 110 | number, 3dp, nullable |
| `rmse_drop_nan` | RMSE (drop NaN) | 120 | number, 3dp, nullable |
| `mae_fill_nan` | MAE (fill 0) | 100 | number, 3dp |
| `rmse_fill_nan` | RMSE (fill 0) | 110 | number, 3dp |

Null-safe `valueFormatter`: if value is null return `"—"`, else `value.toFixed(3)`.

### 4e. Add Tab 4 to the Tabs scaffold

```tsx
<Tab label="ZINB Calibration" value={3} />
```

(Existing Tabs 1, 2, 3 are unchanged. Tabs 1 and 2 remain disabled.)

### 4f. Tab 3 panel — ZINB Calibration

Two side-by-side charts inside a `Box sx={{ display: 'flex', gap: 2 }}`:

**Left — PIT Histogram (BarChart)**
- Data: `masterIndex.pit_histogram`
- `<Bar dataKey="density" fill="#60a5fa" />`
- X-axis: `bin_center` (0.05 → 0.95)
- Y-axis: density
- A `<ReferenceLine y={0.1} stroke="#f59e0b" strokeDasharray="4 4" />` marks the uniform target (1/10 = 0.1).
- Label above chart: "PIT Histogram — uniform bars indicate perfect calibration"

**Right — Zero-Inflation Reliability (ScatterChart)**
- Data: `masterIndex.zero_inflation_reliability`
- `<Scatter dataKey="observed_zero_fraction" name="Observed fraction" />`
- X-axis: `pi_center` (predicted π)
- Y-axis: `observed_zero_fraction` (actual fraction of zeros)
- `<ReferenceLine` segment: a dashed `y = x` diagonal. Rendered as a `<Line>` in a `<ComposedChart>` overlaid, or as a custom reference line using `<ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} />`.
- Label: "Zero-Inflation Reliability — points on the dashed line indicate perfect calibration"

**Defensive rendering:** Both charts are wrapped in:
```tsx
{!masterIndex.pit_histogram || masterIndex.pit_histogram.length === 0 ? (
  <Alert severity="info">Calibration data not present in this payload. Re-run export_ui_payload.py.</Alert>
) : (
  /* charts */
)}
```

### 4g. Zero math on the frontend

The analytics page performs **zero** statistical computation for the calibration tab. It renders pre-computed arrays from the payload directly. The PIT and Reliability data arrive already aggregated into 10-bin arrays. No CDF, binning, or metric derivation is performed in the browser.

---

## Step 5 — EXPERIMENTS.md Ledger Update

Per CLAUDE.md §6, append a new row upon successful implementation:

| Field | Value |
|:---|:---|
| Date | 2026-05-11 |
| Branch | main |
| Prompt/Goal | Iteration 4: Add PIT histogram, zero-inflation reliability, and Two-Extremes MAE/RMSE to payload + ZINB Calibration tab in analytics dashboard |
| Outcome | 3 new calibration metrics computed in export_ui_payload.py; master_index.json extended with global PIT/reliability arrays and per-node Two-Extremes scalars; payload.ts updated with 3 new interfaces; analytics page gains 4 DataGrid columns and Tab 4 with BarChart + ScatterChart |

---

## Files to Create / Modify

| Path | Action | Description |
|:---|:---|:---|
| `src/export_ui_payload.py` | **EDIT** | Add pi accumulator, 3 calibration functions, inject metrics into JSON outputs |
| `frontend/src/types/payload.ts` | **EDIT** | Add `PITBin`, `ZeroInflationBin`, `TwoExtremes`; update `NodeEntry`, `MasterIndex`, `NodeDetailPayload` |
| `frontend/src/app/analytics/page.tsx` | **EDIT** | Add 4 DataGrid columns, Tab 4 with PIT BarChart + Reliability ScatterChart |
| `EXPERIMENTS.md` | **EDIT** | Append ledger row |

---

## No-op confirmation (strict constraints respected)

- **Zero Host Install:** No `pip install` or `npm install` on host. `scipy` is already in the container's Python environment (required by prior steps; verify with `python -c "from scipy.special import betainc; print('ok')"` if uncertain).
- **Zero Frontend Math:** All binning, CDF computation, and metric arithmetic is done exclusively in `export_ui_payload.py`. The Next.js page is a pure renderer.
- **Offensive Tripwires:** `pi ∈ (0,1)`, `u ∈ [0,1]`, no bare `except`, no silent NaN drops.
- **No live API:** All data flows through the static JSON bridge per Phase 1 constraint.

---

**Awaiting explicit user approval before implementation per CLAUDE.md §1.**
