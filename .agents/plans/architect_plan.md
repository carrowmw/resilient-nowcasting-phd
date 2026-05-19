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

## 4. Database-Driven Hyperparameter Analysis
- **Goal**: Deprecate `src/analyze_sweep.py` in favor of a strictly database-driven analysis tool.
- **Requirement**: Use the Optuna SQLite API for all result retrieval. No log parsing.
- **Traceability Verification**: The analyzer must verify `git_commit` and `branch` from trial metadata to ensure "Artifact Twinning".
- **Workflow Integration**: Support a `promote` command that updates `configs/train.yaml` with the best parameters from a selected study.
- **Documentation**: See `docs/optuna_analysis_transition.md` for detailed behavioral comparison and implementation blueprint.
