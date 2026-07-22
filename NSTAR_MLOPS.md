# MLOps & Experimentation Blueprint

## 1. Environment Separation (Docker)

We maintain a strict separation between development and execution using two separate Docker environments defined in `docker-compose.yml`:

- **Development (`Dockerfile.agent`):** Run locally on Mac. Contains CLI tools, Node.js, and bash dependencies for the agent to write, test, and unit-validate code on CPU subsets.
- **Execution (`Dockerfile.cpu`):** Run locally on Mac. A CPU-only image used for quick iteration and debugging of the training pipeline before server deployment.
- **Execution (`Dockerfile.gpu`):** Run on the GPU Server. A headless, optimized CUDA/PyTorch image used strictly for training and Optuna sweeps.
- **Frontend (`Dockerfile.frontend`):** Run locally on Mac. Isolated Node.js environment running the Next.js server.

**Parallel Execution Rule:** Because the GPU runner mounts the local directory (`.:/workspace`), you cannot run multiple branches from the same physical directory. To run experiments in parallel, clone the repository into separate directories on the server and execute `docker compose` independently in each.

### Strict Container Isolation & Anonymous Volumes

This project adheres to a "Zero Host Download" rule. To prevent `npm` dependencies from bleeding onto the host server's filesystem via standard bind mounts, the `frontend-dev` service utilizes an **Anonymous Volume** for `/workspace/frontend/node_modules` and `/workspace/frontend/.next`.

- All frontend packages are baked into the image during the `Dockerfile.frontend` build phase.
- If new packages must be installed dynamically during development, they must be executed _inside_ the frontend container context using `docker compose exec frontend-dev npm install <package>`. Do not run `npm install` from within the `agent-workspace` container, as it lacks the anonymous volume mapping and will corrupt the host filesystem.

### Cross-Container Hot-Reloading

The agent writes sources code (.ts, .py, etc.) inside `agent-workspace`. Because both the agent and frontend containers share the ./frontend bind mount, code changes made by the agent are instantly synced to `frontend-dev`, triggering Next.js hot-reloads. The agent must NEVER run frontend execution commands (npm run dev) inside its own workspace.

### The Sync or Sink Dependency Rule

While Python's global pip install inside the agent-workspace does not pollute the host filesystem, it creates fatal ephemeral container drift. If a new Python package is required for the PyTorch pipeline, it must be installed using docker compose exec agent-workspace pip install <package>. Crucially, any dynamically installed package MUST be immediately appended to ./requirements.txt. Failure to sync dependencies will cause the headless GPU execution environment (Dockerfile.gpu) to fail fast with a ModuleNotFoundError during automated Optuna sweeps.

## 2. File Architecture

We utilize a Centralized Configuration pattern. **No hyperparameters should be hardcoded in Python execution scripts.**

- `configs/train.yaml`: The single source of truth for all runs, containing nested dictionaries for `data`, `model`, `training`, and `loss`.
- `configs/sweep.yaml`: The W&B Optuna search space definitions.
- `src/train.py`: The orchestrator for a single training run. Responsible for model instantiation, W&B initialization, training loops, and validation logic.
- `src/train.py`: Accepts `--config configs/train.yaml`. Loads the dictionary, initializes the W&B run, and executes the pipeline.
- `src/tune.py`: Loads the base `train.yaml`, overwrites the specific parameters Optuna is currently testing, and executes `train.py`.
-

## 3. Weights & Biases (W&B) Logging Standards

- W&B project, sweep project, and artifact collection settings are config-driven and managed via the centralized YAML config files (e.g., `configs/train/train.yaml`).
- `train.py` initializes `wandb.init(project=wandb_project, config=config)` using the configured project name (defaulting to `resilient-iot-v2` for production runs and `resilient-iot-v2-test` for test runs) and logs the full nested config dictionary.
- Required tracked metrics: `train_loss_total`, `train_loss_forecast`, `train_loss_impute`, `val_loss_total`, and human-readable metrics (`val_rmse`, `val_mae`, `val_zero_acc`).
- `val_zero_acc` must be aggregated by keeping a running tally of true zero counts vs. predicted zero counts across the entire epoch to avoid batch-averaging skew.

## 4. Git Branching & Experimentation Strategy

To prevent the repository from becoming an unmanageable bowl of spaghetti, we strictly separate code evolution (infrastructure/features) from configuration/data evolution (hypothesis testing).

### 4.1 Branch Categories

We categorize branches into three distinct tiers:

1. **`main` (The Source of Truth)**
   - Contains stable, tested, and examiner-ready code.
   - Must always be capable of running an end-to-end pipeline.
   - The universal evaluation framework (`train.py`, `tune.py`, metrics calculations) is locked to `main`.

2. **`feat/` (Infrastructure Upgrades)**
   - **Naming Convention:** `feat/add-rope-encoding`, `feat/deckgl-dashboard`, `feat/optuna-integration`.
   - **Purpose:** Actual software engineering changes (new logic, new classes, helper functions, frontend features).
   - **Lifecycle:** Once validated and clean, they are merged into `main` and deleted.

3. **`exp/` (Hypothesis Testing)**
   - **Naming Convention:** `exp/zinb-vs-mse`, `exp/high-sparsity-threshold`.
   - **Purpose:** Hacking/testing specific scientific hypotheses (e.g., swapping a loss function, temporary architectural changes, custom metrics).
   - **Lifecycle:** **Never merge `exp/` branches into `main`.** They serve as immutable historical archives of how results for a specific thesis chapter or model iteration were generated. If an experiment is highly successful, implement its features cleanly in a new `feat/` branch off `main`.
   - **Syncing:** `exp/` branches should regularly `git rebase main` to inherit updates to the universal evaluation framework.

### 4.2 Config-Driven Experiments (No Branching for Parameters)

Do not create branches for parameter tuning (e.g., changing learning rates, attention heads, or batch sizes). Instead:
- Keep the codebase strictly config-driven. All hyperparameters must live in YAML configs under `configs/`.
- When running an experiment, duplicate the baseline config (e.g., `configs/baseline.yaml` -> `configs/exp_8_heads.yaml`), change the values, and run with `--config configs/exp_8_heads.yaml`.
- Commit the new config file to Git to preserve the exact run configuration.

### 4.3 Artifact Twinning & Traceability

Every experiment run and UI payload must follow the "Artifact Twinning" protocol:
- Logs in Weights & Biases (W&B) must automatically record the **Git Hash** of the active branch/commit and the **YAML config path/contents** used.
- This ensures 1-to-1 traceability: a visual curve or dashboard payload can be traced back to the exact code commit and parameter set.

## 5. The Static Data Bridge & Evaluation Constraints

The bridge between the PyTorch Backend and Next.js Frontend is the `export_ui_payload.py` script, which generates `ui_payload.json`.

- **The Visual Holdout Set (No Data Leakage):** The data used to generate the UI payload must be strictly quarantined from the training and validation loops. Visualizing training data in the frontend displays memorization, not predictive inference.
- **Artifact Twinning:** Every generated `ui_payload.json` must be injected with metadata identifying exactly how it was created. This includes the `wandb_run_id`, `git_commit_hash`, and a copy of the `train.yaml` hyperparameters. This ensures a 1-to-1 traceability between a visual frontend artifact and the backend model that generated it.

## 6. AI Agent Orchestration & Roles

Development in this repository is assisted by AI Agents. To maintain code quality and prevent degradation, agents are divided into three distinct roles. Do not allow agents to break character or cross responsibilities.

- **The Interrogator (@interrogator):** Read-only scout. Responsible for searching the codebase, reading files, analyzing logs, and providing context. Never writes code.
- **The Architect (@architect):** System designer. Responsible for cross-referencing requirements with `NSTAR` documents, creating step-by-step plans in `plan.md`, and performing mathematical "Desk Checks" before any code is written. Does not write execution code.
- **The Implementer (@implementer):** The executor. Writes the code based _only_ on the Architect's plan. Operates strictly within the Dockerized environment.

## 7. Environment & Dependency Isolation

This project operates on a strict **Zero Host Download** policy.

- **No Local Environments:** The creation of local virtual environments (e.g., `python -m venv`) is strictly prohibited.
- **Docker Execution:** All code execution, testing, and linting MUST take place inside the designated Docker containers using `docker compose exec [container_name] [command]`.
- **The "Sync or Sink" Rule:** Any Python dependency installed inside the container (`pip install <package>`) MUST be immediately and manually added to `requirements.txt`. Failure to do so will result in container rebuild failures and is considered a critical development violation.
- **Node Modules Quarantine:** Frontend dependencies are handled via Docker anonymous volumes. Do not attempt to read or modify the host's `node_modules` directory.

## 8. The Experiment Ledger

No task, experiment, or significant bug fix is considered complete until it is logged in `EXPERIMENTS.md`.

- Every entry must contain: Date, Git Branch, Objective/Hypothesis, and Outcome/Result.
- AI Agents are instructed to halt workflows and refuse to close out tasks until this ledger is updated.
