# Implementation Plan: Robust Model Versioning & UI Payload Fixes

## 1. Directory Namespacing Strategy

We have moved away from a flat `models/` directory and implemented a tiered structure:
`models/{wandb_project}/{wandb_run_name}/best_model.pt`

- **Production Namespace**: `models/resilient-iot/{run_name}/`
- **Sweep Namespace**: `models/resilient-iot-sweeps/{study_name}/` (since trials overwrite)
- **Local Namespace**: `models/local/{timestamp}/`

## 2. Source Code Modifications (Already Implemented)

### `src/train.py`
- `run_training` now dynamically resolves the checkpoint path based on `wandb.run` or `study_name`.
- Checkpoints are saved to namespaced subdirectories.

---

## 3. NEW TASKS: UI Payload Enhancements

### Task 3.1: Model Selection in `src/export_ui_payload.py`
- **Problem**: The script currently relies on a hardcoded `CHECKPOINT_PATH = "models/best_pipeline.pt"`. This is incompatible with the new tiered directory structure.
- **Requirement**: Update `parse_args()` to include a `--checkpoint` argument (defaulting to the legacy path for backwards compatibility).
- **Action**: 
    - Replace the `CHECKPOINT_PATH` constant usage in `main()` with `args.checkpoint`.
    - Ensure the script can load models from any namespaced subdirectory (e.g., `models/local/20260516_125722/best_model.pt`).

### Task 3.2: Google Places API for Simulated Nodes
- **Problem**: The 5 simulated nodes (`Simulated_Node_0` to `Simulated_Node_4`) currently have empty semantic tags because they are not in the legacy cache or the template CSV.
- **Requirement**: Fetch real semantic data for these nodes using their coordinates.
- **Action**:
    1. **Integration**: Import `StaticFeatureLoader` from `features`.
    2. **Caching**: Initialize the loader using the existing cache path (`data/external/places_cache.json`).
    3. **Live Query**: For the simulated nodes (coordinates defined in `SIMULATED_NODE_COORDS`), use `loader._fetch_location_data(lat, lon)` if they are missing from the cache.
    4. **Tag Mapping**: Update `load_semantic_tags()` to incorporate these features. The logic should convert raw API counts (commercial density, etc.) into the expected semantic tags (e.g., `high_commercial_density`).
    5. **Optimization**: Continue using cached results for real nodes; do NOT perform redundant API calls for nodes already present in `places_cache.json`.
