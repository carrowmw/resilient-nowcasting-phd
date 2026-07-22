# Architectural Design: Configurable Simulated Nodes & Map Toggle

## 1. Overview
The current system hardcodes five simulated node coordinates in `configs/analysis/analysis.yaml`. To support scale-testing and robust zero-shot topological inference visualizations, we must dynamically sample up to 100 simulated nodes directly from the physical road graph intersections, ensuring mathematical reproducibility via a global seed. A frontend toggle will allow the user to isolate physical sensor observations from synthetic simulations in the map view.

## 2. Alignment with North Stars
- **NSTAR_FRONTEND_ARCHITECTURE.md (Decoupled Systems):** The simulated layer toggle must be managed by the centralized `useTimeStore` (Zustand), maintaining a reactive data flow to the `MapVisualizer`'s Deck.gl layers without triggering full React DOM re-renders during playback.
- **NSTAR_MLOPS.md (Reproducibility):** The selection of intersection coordinates must be strictly deterministic. We will enforce the global Python `random.seed(42)` immediately before sampling the `osmnx` graph nodes.
- **NSTAR_BACKEND_ARCHITECTURE.md (Offensive Programming):** We must validate that the configured `num_simulated_nodes` is within bounds (0 to 100). If the OSMnx graph fails to load or lacks sufficient nodes, the system must fail fast with a `RuntimeError`, avoiding silent degradation.

## 3. High-Level Step-by-Step Architecture

### Phase 1: Configuration Updates
1. **Remove Hardcoded Coords:** In `configs/analysis/analysis.yaml` (and `configs/train/train.yaml` if applicable), remove `simulated_node_coords`.
2. **Introduce Parameter:** Add `num_simulated_nodes: 50` (or another default) under the `export` block.

### Phase 2: Dynamic Sampling (Backend)
1. **OSMnx Graph Extraction:** In `src/export_ui_payload.py`, rather than iterating over a config list, invoke `osmnx` to load the walking network within the target bounding box or using the `features.py`'s `OSM_CENTER` and `OSM_DIST`.
2. **Deterministic Sampling:**
   - Extract the `(lat, lon)` coordinates of all nodes in the OSM graph.
   - Enforce `random.seed(42)`.
   - Use `random.sample()` to select `num_simulated_nodes` coordinates.
3. **Tripwires:** Assert that `num_simulated_nodes <= 100` and `num_simulated_nodes <= len(graph_nodes)`. Crash immediately if bounds are violated.

### Phase 3: Frontend Interactive Toggle
1. **State Management (`useTimeStore.ts`):** 
   - Add `showSimulatedNodes` (boolean, default `true`).
   - Add `setShowSimulatedNodes(visible: boolean)` action.
2. **Deck.gl Integration (`MapVisualizer.tsx`):**
   - Bind the `visible` property of `simLayer` (for `nodes-simulated`) to `showSimulatedNodes`.
   - Ensure the visibility toggle updates the GPU-side layer without recreating the data array or causing React thrashing.
3. **UI Controls:** Add a checkbox or toggle switch to the map control panel (e.g., `ModelInfoOverlay` or a dedicated control overlay) to manipulate `showSimulatedNodes`.

## 4. Desk Check Mandate
**Required Desk Check:** Before execution, verify that:
1. `random.sample` correctly yields deterministic arrays across multiple runs.
2. `export_ui_payload.py` cleanly builds the ego-graph for the randomly sampled simulated nodes without crashing, especially since they lack physical traffic data.
3. Ensure no `NaN` coordinates are emitted into the payload.
