# Context for Architect: Simulated Nodes & Map Toggle

## 1. Goal
1. Update payload generation to dynamically calculate intersection coordinates on the road graph, randomly sampling from the full set using the global seed.
2. Make the number of simulated nodes configurable (up to 100) via `configs/analysis/analysis.yaml`.
3. Add a UI toggle for the simulated node layer in the frontend map view.

## 2. Relevant File Paths & Summaries

### `configs/analysis/analysis.yaml`
- Currently hardcodes 5 coordinates under `export.simulated_node_coords`.
- **Target:** Remove the hardcoded list and replace with a `num_simulated_nodes` integer (e.g., 50, max 100).

### `src/export_ui_payload.py`
- Line 85+: Reads constants from `analysis.yaml`.
- Line 786+: `simulated_node_coords = [tuple(c) for c in export_cfg["simulated_node_coords"]]`
- Currently loops through `simulated_node_coords` to inject them into the payload. 
- **Target:** Instead of reading coords from config, read `num_simulated_nodes`. Then, use `osmnx` (similar to how `features.py` does it) to download/load the OSM network, extract nodes (intersections), and randomly sample `num_simulated_nodes` using `random.sample` (with the global seed `42`).

### `src/features.py` (For Reference)
- Uses `osmnx`: `ox.graph_from_point(OSM_CENTER, dist=OSM_DIST, network_type='walk')`.
- Graph nodes have `y` (lat) and `x` (lon) attributes. This logic can be extracted/reused in `export_ui_payload.py` to get the real intersections.

### `frontend/src/store/useTimeStore.ts` (Implicit)
- Zustand store where we need to add a boolean state `showSimulatedNodes` (default `true`) and an action `setShowSimulatedNodes`.

### `frontend/src/components/MapVisualizer.tsx`
- Line 237+: `const simNodes = payload.nodes.filter((n) => roleOf(n) === "simulated");`
- Line 238+: Builds `simLayer` using `ScatterplotLayer`.
- **Target:** Connect to `showSimulatedNodes` from `useTimeStore`. Use `visible: showSimulatedNodes` in the `simLayer` configuration.

### `frontend/src/components/MapOverlay.tsx` or `MapControls.tsx` (Implicit)
- Needs a UI checkbox/toggle to mutate `showSimulatedNodes`.

## 3. High-Level Logic & ASTs

### Data Pipeline (`export_ui_payload.py`)
1. In `main()`, read `num_simulated_nodes` from `export_cfg`.
2. Limit `num_simulated_nodes` to 100.
3. Import `osmnx` and `features.OSM_CENTER`, `features.OSM_DIST`.
4. Fetch/load the graph: `G = ox.graph_from_point(OSM_CENTER, dist=OSM_DIST, network_type='walk')`.
5. Extract node coordinates: `all_nodes = [(data['y'], data['x']) for node, data in G.nodes(data=True)]`.
6. Sample: `simulated_node_coords = random.sample(all_nodes, num_simulated_nodes)`.
7. Proceed with the rest of the generation (which already creates 0-inflated mock data for simulated coords).

### Frontend
1. Add `showSimulatedNodes: boolean` to `useTimeStore`.
2. Update `MapVisualizer.tsx`'s `simLayer` to have `visible: showSimulatedNodes`.
3. Add a Checkbox to the Map UI (e.g., in a control panel) to toggle this state.

## 4. Potential Tripwires & Safeguards
- **Random Seed:** The prompt requires using the "global seed". `export_ui_payload.py` already sets `torch.manual_seed(42)` and `np.random.seed(42)`. We must ensure Python's `random.seed(42)` is also set or used to guarantee deterministic sampling.
- **OSMnx Download:** OSMnx can be slow. We should ensure the graph is cached or only generated once per payload export. `features.py` caches to `data/external/osm_features_cache.json`, but we need raw coordinates (intersections), not just centralities.
- **Payload Size:** Up to 100 simulated nodes means the `ui_payload.json` and time series matrices will grow. However, Phase 1 supports 96 timesteps. 100 additional nodes shouldn't break the browser context.
- **Config:** Update `configs/train/train.yaml` and `configs/analysis/analysis.yaml` to ensure they both use `num_simulated_nodes` instead of the hardcoded `simulated_node_coords` list.

## 5. Defensive Code Alert
- No defensive code observed in this context. The pipeline relies heavily on assertions and offensive programming.