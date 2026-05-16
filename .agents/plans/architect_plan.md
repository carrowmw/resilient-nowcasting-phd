# Architecture Plan: Robust Model Versioning & Traceability

## 1. Interactive Study Namespacing
- Use SQLite for persistent Optuna storage instead of ephemeral logs.
- Support `--storage` and `--study-name` CLI arguments in `src/tune.py`.
- Ensure studies are namespaced within the SQLite database to avoid collision.

## 2. Artifact Traceability (Artifact Twinning)
- Every Optuna trial must be stamped with Git metadata.
- **Fields**: `git_commit`, `git_branch`.
- **Implementation**: Use `trial.set_user_attr` within the Optuna objective function.
- **Guard**: `src/tune.py` should exit if the repository is "dirty" to ensure reproducible experiments.

## 3. Tiered Model Storage
- Move away from `models/best_pipeline.pt`.
- **Structure**: `models/{project}/{run_name}/best_model.pt`.
- This ensures that every run (Sweep, Production, Local) has its own unique, non-overwritable artifact path.
- **Downstream Impact**: All scripts that load models (e.g., `src/export_ui_payload.py`) must be updated to support dynamic path selection.
