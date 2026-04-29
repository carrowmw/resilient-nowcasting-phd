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

## 2. File Architecture

We utilize a Centralized Configuration pattern. **No hyperparameters should be hardcoded in Python execution scripts.**

- `configs/train.yaml`: The single source of truth for all runs, containing nested dictionaries for `data`, `model`, `training`, and `loss`.
- `configs/sweep.yaml`: The W&B Optuna search space definitions.
- `src/train.py`: The orchestrator for a single training run. Responsible for model instantiation, W&B initialization, training loops, and validation logic.
- `src/train.py`: Accepts `--config configs/train.yaml`. Loads the dictionary, initializes the W&B run, and executes the pipeline.
- `src/tune.py`: Loads the base `train.yaml`, overwrites the specific parameters Optuna is currently testing, and executes `train.py`.
-

## 3. Weights & Biases (W&B) Logging Standards

- `train.py` must initialize `wandb.init(project="resilient-iot", config=config_dict)` using the full nested YAML dictionary.
- Required tracked metrics: `train_loss_total`, `train_loss_forecast`, `train_loss_impute`, `val_loss_total`, and human-readable metrics (`val_rmse`, `val_mae`, `val_zero_acc`).
- `val_zero_acc` must be aggregated by keeping a running tally of true zero counts vs. predicted zero counts across the entire epoch to avoid batch-averaging skew.

## 4. Evaluation Branching Strategy

- The evaluation framework (`train.py`, `tune.py`, metrics calculations) must be locked to the `main` branch.
- Experimental model architectures live on their own branches (e.g., `exp/pure-transformer`).
- Experimental branches will `git rebase main` to inherit updates to the universal evaluation framework.

## 5. The Static Data Bridge & Evaluation Constraints

The bridge between the PyTorch Backend and Next.js Frontend is the `export_ui_payload.py` script, which generates `ui_payload.json`.

- **The Visual Holdout Set (No Data Leakage):** The data used to generate the UI payload must be strictly quarantined from the training and validation loops. Visualizing training data in the frontend displays memorization, not predictive inference.
- **Artifact Twinning:** Every generated `ui_payload.json` must be injected with metadata identifying exactly how it was created. This includes the `wandb_run_id`, `git_commit_hash`, and a copy of the `train.yaml` hyperparameters. This ensures a 1-to-1 traceability between a visual frontend artifact and the backend model that generated it.
