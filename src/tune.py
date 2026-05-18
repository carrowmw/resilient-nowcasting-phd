#!/usr/bin/env python3
import argparse
import copy
import logging
import os
import subprocess
import functools

import wandb
import optuna
import yaml
from optuna.integration.wandb import WeightsAndBiasesCallback

from logger import setup_logger
from train import load_config, run_training

# Load sweep configuration globally so the objective function can access it
DEFAULT_SWEEP_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "sweep.yaml")
with open(DEFAULT_SWEEP_CONFIG_PATH, "r") as f:
    DEFAULT_SWEEP_CONFIG = yaml.safe_load(f)

# Default base config path — can be overridden via --config CLI flag
DEFAULT_TRAIN_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "train.yaml"
)

def set_nested_value(d: dict, keys: list, value):
    """Recursively sets a value in a nested dictionary."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def get_git_metadata():
    """Retrieves current Git commit hash and branch name."""
    try:
        # Check if repository is dirty
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if status:
            print("\n[!] ERROR: Repository is dirty (uncommitted changes).")
            print("To ensure 'Artifact Twinning' traceability, you must commit your changes before running a sweep.")
            print("Dirty files:\n" + status)
            exit(1)

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return commit, branch
    except subprocess.CalledProcessError:
        return "unknown", "unknown"

def make_objective(base_config: dict, use_mock_data: bool):
    """
    Factory that returns an Optuna objective bound to a deep-copy of
    base_config so each trial starts from the same YAML defaults and Optuna
    suggestions overwrite only the designated keys.
    """
    def objective(trial):
        """
        Optuna objective function for hyperparameter tuning.
        Dynamically generates suggestions based on configs/sweep.yaml and
        merges them into a fresh copy of the base config dict.
        """
        config = copy.deepcopy(base_config)

        for param_name, param_config in DEFAULT_SWEEP_CONFIG.get("parameters", {}).items():
            dist = param_config.get("distribution")

            if dist == "log_uniform_values":
                val = trial.suggest_float(
                    param_name,
                    param_config["min"],
                    param_config["max"],
                    log=True,
                )
            elif dist == "uniform":
                val = trial.suggest_float(
                    param_name,
                    param_config["min"],
                    param_config["max"],
                )
            elif "values" in param_config:
                # Categorical / discrete choices
                val = trial.suggest_categorical(param_name, param_config["values"])
            else:
                raise ValueError(f"Unsupported distribution or missing 'values': {param_name}")

            # Use dot-notation to set the nested configuration parameter
            keys = param_name.split('.')
            if len(keys) == 1:
                # Fallback for old parameters not using dot notation, try to map them or just set at root
                # This should ideally not happen if sweep.yaml is updated, but for backward compat:
                if param_name in ["lr", "batch_size"]:
                    set_nested_value(config, ["training", param_name], val)
                elif param_name in ["w_forecast", "w_impute"]:
                    set_nested_value(config, ["loss", param_name], val)
                elif param_name in ["sparsity_threshold", "num_layers", "embedding_dim", "hidden_dim"]:
                    set_nested_value(config, ["model", param_name], val)
                elif param_name == "mask_prob":
                    set_nested_value(config, ["data", param_name], val)
                else:
                    config[param_name] = val
            else:
                set_nested_value(config, keys, val)

        # Stamp the trial with Git metadata
        commit, branch = get_git_metadata()
        trial.set_user_attr("git_commit", commit)
        trial.set_user_attr("branch", branch)

        # Execute training loop.
        # init_wandb=False prevents train.py from spawning a separate W&B run;
        # the WeightsAndBiasesCallback manages the run at the study level.
        logger = logging.getLogger("optuna_sweep")
        logger.info(f"Trial {trial.number} parameters: {trial.params}")

        val_loss = run_training(config=config, init_wandb=False, use_mock_data=use_mock_data, study_name=trial.study.study_name)

        logger.info(f"Trial {trial.number} finished with val_loss: {val_loss}")
        return val_loss

    return objective

def interactive_setup():
    """Handles interactive prompts for storage and study name."""
    storage = None
    study_name = None

    # Find local DBs
    local_dbs = [f for f in os.listdir('.') if f.endswith('.db')]

    if local_dbs:
        print("\nFound existing Optuna databases in current directory:")
        for i, db in enumerate(local_dbs):
            print(f"[{i+1}] {db}")
        print(f"[{len(local_dbs)+1}] Create new database")

        while True:
            choice = input(f"Select storage [1-{len(local_dbs)+1}]: ")
            idx = int(choice) - 1
            if 0 <= idx < len(local_dbs):
                storage = f"sqlite:///{local_dbs[idx]}"
                break
            elif idx == len(local_dbs):
                new_db = input("Enter new database name (e.g., optuna.db): ")
                if not new_db.endswith('.db'):
                    new_db += '.db'
                storage = f"sqlite:///{new_db}"
                break
    else:
        new_db = input("\nNo existing databases found. Enter new database name (e.g., optuna.db): ")
        if not new_db.endswith('.db'):
            new_db += '.db'
        storage = f"sqlite:///{new_db}"

    # Study selection
    summaries = optuna.get_all_study_summaries(storage)
    if summaries:
        print(f"\nExisting studies in {storage}:")
        for i, s in enumerate(summaries):
            print(f"[{i+1}] {s.study_name} (Trials: {s.n_trials})")
        print(f"[{len(summaries)+1}] Create new study")

        while True:
            choice = input(f"Select study [1-{len(summaries)+1}]: ")
            idx = int(choice) - 1
            if 0 <= idx < len(summaries):
                # 1. User selected an EXISTING study
                study_name = summaries[idx].study_name
                break
                # 2. User creates a NEW study (allow empty string for WandB auto-name)
            elif idx == len(summaries):
                study_name = input("Enter new study name (leave blank to auto-generate via WandB): ")
                if study_name == "":
                    study_name = None  # Triggers auto-naming in the main block
                break
    else:
        # 3. No studies exist in the DB (allow empty string for WandB auto-name)
        study_name = input(f"\nNo existing studies in {storage}. Enter new study name (leave blank to auto-generate via WandB): ").strip()
        if study_name == "":
            study_name = None  # Triggers auto-naming in the main block

    return storage, study_name

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_TRAIN_CONFIG,
        help="Path to base YAML config (train.yaml). Optuna suggestions overwrite specific keys.",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=DEFAULT_SWEEP_CONFIG,
        help="Path to YAML config (sweep.yaml). Defines hyperparameter search space.",
    )
    parser.add_argument(
        "--use-mock-data",
        action="store_true",
        help="Use synthetic mock data instead of real parquet data (for local CPU integration tests)",
    )
    parser.add_argument("--n-trials", type=int, default=2, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, help="SQLite database connection string (e.g., sqlite:///optuna.db)")
    cli_args = parser.parse_args()

    logger = setup_logger("optuna_sweep")
    logger.info("Starting Optuna Hyperparameter Sweep...")
    if cli_args.use_mock_data:
        logger.info("Mode: MOCK DATA (local integration test)")

    base_config = load_config(cli_args.config)
    logger.info(f"Base config loaded from: {cli_args.config}")

    storage = cli_args.storage
    study_name = cli_args.study_name

    if storage is None or study_name is None:
        storage, study_name = interactive_setup()

    logger.info("Initializing WandB run...")
    wandb_run = wandb.init(
        project="resilient-iot-sweeps",
        name=study_name,
        reinit=True
    )

    if study_name is None:
        study_name = wandb_run.name
        logger.info(f"Aligned Optuna study name with auto-generated WandB run name: '{study_name}'")
    else:
        logger.info(f"Aligned WandB run name with specified study name: '{study_name}'")

    logger.info(f"Using storage: {storage}")
    logger.info(f"Using study name: {study_name}")

    # Initialize the W&B Callback for Optuna
    wandbc = WeightsAndBiasesCallback(
        metric_name="val_loss_total",
        wandb_kwargs={"project": "resilient-iot-sweeps"},
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize"
    )

    # Execute study
    study.optimize(
        make_objective(base_config, cli_args.use_mock_data),
        n_trials=cli_args.n_trials,
        callbacks=[wandbc],
    )

    logger.info("Sweep complete. Best hyperparameters:")
    if len(study.trials) > 0:
        try:
            for key, value in study.best_params.items():
                logger.info(f"    {key}: {value}")
            logger.info(f"Best validation loss: {study.best_value}")
        except ValueError:
            logger.warning("No best trial available yet (e.g., all trials failed).")
    else:
        logger.warning("No trials completed.")
