# MLOps & Experimentation Blueprint

## 1. Environment Separation (Docker)

We maintain a strict separation between development and execution using two separate Docker environments defined in `docker-compose.yml`:

- **Development (`Dockerfile.agent`):** Run locally on Mac. Contains CLI tools, Node.js, and bash dependencies for the agent to write, test, and unit-validate code on CPU subsets.
- **Execution (`Dockerfile.gpu`):** Run on the GPU Server. A headless, optimized CUDA/PyTorch image used strictly for training and Optuna sweeps.

**Parallel Execution Rule:** Because the GPU runner mounts the local directory (`.:/workspace`), you cannot run multiple branches from the same physical directory. To run experiments in parallel, clone the repository into separate directories on the server and execute `docker compose` independently in each.

## 2. File Architecture

- `src/train.py`: The orchestrator for a single training run. Responsible for model instantiation, W&B initialization, training loops, and validation logic.
- `src/tune.py`: The Optuna orchestrator. Responsible for defining the hyperparameter search space and executing `train.py` across multiple trials using the W&B Optuna integration.
- `configs/`: Directory containing YAML files defining hyperparameter grids.

## 3. Weights & Biases (W&B) Logging Standards

- `train.py` must initialize `wandb.init(project="resilient-iot", config=...)`.
- The training loop must log at the epoch level, not the batch level, to avoid API rate limiting.
- Required tracked metrics: `train_loss_total`, `train_loss_forecast`, `train_loss_impute`, and `val_loss_total`.

## 4. Evaluation Branching Strategy

- The evaluation framework (`train.py`, `tune.py`, metrics calculations) must be locked to the `main` branch.
- Experimental model architectures live on their own branches (e.g., `exp/graphsage`).
- Experimental branches will `git rebase main` to inherit any updates to the universal evaluation framework.
