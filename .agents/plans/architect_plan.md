# Implementation Plan: Interactive Study Namespacing & Metadata Stamping

Upgrade the tuning framework to support multiple independent experiments through persistent SQLite namespacing and an interactive selection CLI.

## 1. Environment Protection (`.gitignore`)

- **Objective**: Prevent binary database files and journals from polluting the repository.
- **Action**: Append the following to the root `.gitignore`:
  ```text
  # Optuna Databases
  *.db
  *.db-journal
  *.db-wal
  ```

## 2. Interactive Sweep Launcher (`src/tune.py`)

- **Objective**: Prevent parameter-mixing by forcing study selection and stamping trials with Git/Arch metadata.
- **CLI Requirements**:
  - `--study-name`: The specific experiment name.
  - `--storage`: The SQLite connection string (e.g., `sqlite:///optuna.db`).
- **Interactive Discovery (Discovery Logic)**:
  - **Storage Selection**: If `--storage` is missing, the script must scan the current directory for `*.db` files and prompt the user to select one or provide a new filename.
  - **Study Selection**: Once storage is identified, if `--study-name` is missing, the script must call `optuna.get_all_study_summaries(storage)` to display a list of existing studies (including trial counts) for selection, or allow creating a new one.
- **Trial Metadata Stamping**:
  - Inside the Optuna `objective` function, retrieve the current Git commit hash (e.g., via `subprocess`).
  - Use `trial.set_user_attr("git_commit", commit_hash)` and `trial.set_user_attr("branch", branch_name)` to stamp every trial for 1-to-1 traceability.
- **Offensive Programming & Early Stopping**:
  - Do NOT wrap training in `try-except` blocks. Rely on existing tripwires to fail-fast.
  - **IMPLEMENT** Intra-Trial Early Stopping in `src/train.py`. The loop should terminate if `val_loss` does not improve by `early_stop_min_delta` for `early_stop_patience` consecutive epochs.
  - Do NOT use Optuna Pruners. Early stopping is individual to each trial's progress and preserves fairness for slow-converging models.

## 3. Parameter Externalization & Configuration Strategy

- **Objective**: Eliminate hardcoded "magic numbers" and ensure every parameter can be justified.
- **Task**: Move the following hidden parameters from source code into `configs/train.yaml`:
  - **Model**: `num_heads`, `dropout`, `dim_feedforward`, `zinb_eps`
  - **Training**: `grad_clip_max_norm`, `random_seed`
  - **Early Stopping**: `early_stop_patience`, `early_stop_min_delta`
- **Architectural Investigation**:
  The Implementer must investigate the best way to implement configuration moving forward to ensure scalability and justification. Consider:
  - **Unified Config**: A single monolithic config file with switches indicating whether parameters are "Fixed" or "Swept".
  - **Justification Tracking**: How to record the evidence (from `analyze_sweep.py`) that justifies each parameter's value in the final model.

## 4. High-Resolution Analysis (`src/analyze_sweep.py`)

- **Objective**: Support selecting specific studies for comparison and importance analysis.
- **Modification**:
  - Add `--study-name` and `--storage` arguments to the analyzer.
  - Filter the SQLite data for the specific study to ensure correlations are calculated on a clean, incomparable population.

## 5. Documentation (The Tuning Runbook)

- **Objective**: Ensure the tuning process is reproducible and transparent.
- **Action**: Create `docs/tuning.md` containing:
  - CLI usage examples and instructions for the interactive launcher.
  - Instructions for using `optuna-dashboard sqlite:///optuna.db`.
  - A guide to interpreting "Parameter Importance" and "Spearman Correlation" output.

## 6. Verification Plan (Docker Execution)

> [!IMPORTANT]
> All verification commands MUST be run inside the Docker container context.

1. **New Study Creation**: Run `python src/tune.py`. Verify the interactive prompts for storage and study name.
2. **Early Stopping Check**: Run a trial with high LR and verify it terminates before `max_epochs` once the loss plateaus.
3. **Parameter Injection**: Verify that changing `num_heads` or `dropout` in the config successfully changes the model architecture without source edits.
4. **Analysis**: Run `python src/analyze_sweep.py` and confirm it pulls from the correct SQLite study.

## 7. Experiment Ledger

- Update `EXPERIMENTS.md` documenting the move to Study Namespacing, Early Stopping, and Parameter Externalization.
