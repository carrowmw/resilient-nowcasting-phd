# Tuning Runbook

This document explains how to use the Optuna hyperparameter tuning framework for the ResilientIoT model. The tuning process has been upgraded to support multiple independent experiments through persistent SQLite namespacing.

## 1. Quick Start (Interactive Setup)

The easiest way to start a tuning sweep is to use the interactive CLI. Simply run:

```bash
python src/tune.py --use_mock_data
```

*(Omit `--use_mock_data` to run on the real parquet datasets).*

If you do not provide `--storage` and `--study-name`, the script will automatically discover existing SQLite databases in your current directory and prompt you to select one (or create a new one). It will then list the studies within that database and let you select an existing study to resume or create a new one.

## 2. Command-Line Usage (Automated Scripts)

For automated scripts or remote servers, you can pass the namespacing arguments directly to bypass the interactive prompts:

```bash
python src/tune.py \
    --storage sqlite:///my_experiment.db \
    --study-name "dropout-layer-test" \
    --n-trials 10
```

### CLI Arguments

*   `--storage`: The SQLite connection string (e.g., `sqlite:///optuna.db`). If omitted, you will be prompted.
*   `--study-name`: The name of the specific experiment. If omitted, you will be prompted.
*   `--config`: Path to the base YAML config (default: `configs/train.yaml`).
*   `--n-trials`: Number of Optuna trials to run (default: 2).
*   `--use_mock_data`: Use synthetic mock data instead of real data (useful for local CPU testing).

## 3. Configuration Strategy

We maintain a strict separation between base configurations and search spaces:

*   **`configs/train.yaml`**: This file contains the **fixed, base values** for every parameter. If you add a new parameter to the model, you must define its default value here.
*   **`configs/sweep.yaml`**: This file defines **what is being tuned**. It uses **dot-notation** (e.g., `model.dropout`) to map the Optuna suggestion directly into the nested dictionary loaded from `train.yaml`. 

To sweep a new parameter:
1. Ensure it exists in `train.yaml` (e.g., `model: { new_param: 0.5 }`).
2. Add it to `sweep.yaml` using dot notation:
   ```yaml
   parameters:
     model.new_param:
       distribution: uniform
       min: 0.1
       max: 0.9
   ```

## 4. Metadata Stamping and Traceability

Every trial executed by `src/tune.py` is automatically stamped with the current Git commit hash and branch name. 

You can view these stamps in two ways:

1.  **Optuna Dashboard**:
    ```bash
    pip install optuna-dashboard
    optuna-dashboard sqlite:///optuna.db
    ```
    Open the web UI and click on a study. The Git metadata will be visible in the "User Attributes" column of the trial table.

2.  **Analysis Script**: The analysis script (see below) extracts and analyzes all completed trials from the specified study.

## 5. High-Resolution Analysis

To analyze the hyperparameter importance and correlations for a specific study, use the `src/analysis.py` script. You must point it to the correct SQLite database and study name:

```bash
# List all studies in the database
python src/analysis.py --storage sqlite:///optuna.db list

# Perform detailed analysis of a specific study
python src/analysis.py --storage sqlite:///optuna.db analyze my-experiment

# Promote best parameters to a config file
python src/analysis.py --storage sqlite:///optuna.db promote my-experiment --config configs/train/train.yaml
```

### Key Features of Analysis

*   **Artifact Twinning Verification**: The script verifies that each trial is linked to a specific Git commit and branch.
*   **Hyperparameter Importance**: Uses Optuna's built-in importance algorithms to identify which parameters most influence the validation loss.
*   **Parameter Promotion**: Automatically updates your base `train.yaml` with the winning hyperparameters, ensuring a seamless transition from tuning to production training.