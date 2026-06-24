# Frontend Bug Report: ZINB Calibration Chart Alignment

## Issue Description
1. **Tooltip Offset:** Hovering over points in the Zero-Inflation Reliability diagram highlights a point two spaces to the right.
2. **Phantom Highlights:** The user can highlight points at exactly `x=0` and `x=1` despite the scatter data following a `0.05 + 0.1n` bin center pattern.
3. **Missing Data:** For certain runs (`solar-butterfly-38`, `pretty-glitter-29`), the points stop abruptly at `x=0.45` or `x=0.75`.

## Bug Identification & Root Cause Analysis

### 1. The Missing Data "Bug" (Expected Behavior)
The absence of points past `x=0.45` is **not a bug**. 
- **Location:** `src/export_ui_payload.py` -> `compute_zero_inflation_reliability()`
- **Logic:** The backend explicitly filters out bins where the total observation count is zero (`if n == 0: continue`). If a model run never outputs a predicted zero-inflation probability (`pi`) greater than 0.45, those higher bins are completely empty and correctly omitted from the payload to avoid charting artifacts.

### 2. The Tooltip Offset & Phantom Highlights
The frontend Recharts implementation uses a `<ComposedChart>` containing two disjoint `<Line>` components:
1. A diagonal reference line fed by a 2-element array: `RELIABILITY_DIAGONAL = [{ pi_center: 0, ... }, { pi_center: 1, ... }]`
2. The scatter dots fed by the payload: `reliabilityData` (containing up to 10 elements).

**Root Cause:** Recharts struggles with `<ComposedChart>` tooltips when child components use disjoint data arrays with overlapping keys but different lengths. It attempts to merge the internal arrays by array index. Because `RELIABILITY_DIAGONAL` has 2 points, it shifts the tooltip index mapping for the scatter plot, resulting in the "off by two" offset. Furthermore, the tooltip actively snaps to the 2 points from the diagonal array (at 0 and 1), creating the phantom highlights.

## Relevant Files
- **Frontend Chart:** `frontend/src/app/analytics/page.tsx` (Lines 944-1035)
- **Backend Generator:** `src/export_ui_payload.py` (Line 464, `compute_zero_inflation_reliability`)

## Proposed Architectural Solution

1. **Remove Disjoint Data:** Eliminate the `RELIABILITY_DIAGONAL` constant entirely to stop Recharts from merging disjoint arrays.
2. **Top-Level Data Binding:** Pass `data={reliabilityData}` directly to the top-level `<ComposedChart>` component so the chart has a single, unified data context.
3. **Replace Diagonal Line Component:** Replace the diagonal `<Line>` component with a Recharts `<ReferenceLine />` using the `segment` property to draw an arbitrary line between two coordinate pairs:
   ```tsx
   <ReferenceLine 
     segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} 
     stroke="#f59e0b" 
     strokeDasharray="6 3" 
   />
   ```
   *Rationale:* A `ReferenceLine` is purely presentational. It does not register as interactive data points, which instantly solves both the tooltip misalignment and the phantom snapping at 0 and 1.
