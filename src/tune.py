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
from dataset import RealIoTDataset
from db_connector import create_connector


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
        if status and os.environ.get("SKIP_GIT_CHECK") != "true":
            print("\n[!] ERROR: Repository is dirty (uncommitted changes).")
            print("To ensure 'Artifact Twinning' traceability, you must commit your changes before running a sweep.")
            print("Dirty files:\n" + status)
            exit(1)

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return commit, branch
    except subprocess.CalledProcessError:
        return "unknown", "unknown"

def make_objective(base_config: dict, sweep_config: dict, use_mock_data: bool, full_dataset=None):
    """
    Factory that returns an Optuna objective bound to a deep-copy of
    base_config so each trial starts from the same YAML defaults and Optuna
    suggestions overwrite only the designated keys.
    """
    def objective(trial):
        """
        Optuna objective function for hyperparameter tuning.
        Dynamically generates suggestions based on sweep_config and
        merges them into a fresh copy of the base config dict.
        """
        config = copy.deepcopy(base_config)

        for param_name, param_config in sweep_config.get("parameters", {}).items():
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

        import torch
        import gc
        try:
            val_loss = run_training(
                config=config, 
                init_wandb=False, 
                use_mock_data=use_mock_data, 
                study_name=trial.study.study_name,
                full_dataset=full_dataset,
                trial_number=trial.number
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Trial {trial.number} failed with CUDA OutOfMemoryError. Pruning trial.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        logger.info(f"Trial {trial.number} finished with val_loss: {val_loss}")
        
        # Explicit garbage collection after each trial
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return val_loss

    return objective

def interactive_setup(storage=None, study_name=None, config_path=None, sweep_config_path=None):
    """Selectively handles interactive prompts for storage, study name, and configuration files."""

    # 1. Storage Selection (only if not provided)
    if storage is None:
        local_dbs = [f for f in os.listdir('.') if f.endswith('.db')]
        if local_dbs:
            print("\nFound existing Optuna databases in current directory:")
            for i, db in enumerate(local_dbs):
                print(f"[{i+1}] {db}")
            print(f"[{len(local_dbs)+1}] Create new database")
            while True:
                choice = input(f"Select storage [1-{len(local_dbs)+1}]: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(local_dbs):
                    storage = f"sqlite:///{local_dbs[idx]}"
                    break
                elif idx == len(local_dbs):
                    new_db = input("Enter new database name (e.g., optuna.db): ").strip()
                    if not new_db.endswith('.db'):
                        new_db += '.db'
                    storage = f"sqlite:///{new_db}"
                    break
        else:
            new_db = input("\nNo existing databases found. Enter new database name (e.g., optuna.db): ").strip()
            if not new_db.endswith('.db'):
                new_db += '.db'
            storage = f"sqlite:///{new_db}"
    # 2. Study Selection (only if not provided)
    if study_name is None:
        summaries = optuna.get_all_study_summaries(storage)
        if summaries:
            print(f"\nExisting studies in {storage}:")
            for i, s in enumerate(summaries):
                print(f"[{i+1}] {s.study_name} (Trials: {s.n_trials})")
            print(f"[{len(summaries)+1}] Create new study")
            while True:
                choice = input(f"Select study [1-{len(summaries)+1}]: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(summaries):
                    study_name = summaries[idx].study_name
                    break
                elif idx == len(summaries):
                    study_name = input("Enter new study name (leave blank to auto-generate via WandB): ").strip()
                    if study_name == "":
                        study_name = None
                    break
        else:
            study_name = input(f"\nNo existing studies in {storage}. Enter new study name (leave blank to auto-generate via WandB): ").strip()
            if study_name == "":
                study_name = None
    # Load list of available configuration files
    configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    train_dir = os.path.join(configs_dir, "train")
    sweeps_dir = os.path.join(configs_dir, "sweeps")
    train_yaml_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(('.yaml', '.yml'))])
    sweep_yaml_files = sorted([os.path.join(sweeps_dir, f) for f in os.listdir(sweeps_dir) if f.endswith(('.yaml', '.yml'))])
    # 3. Base Config Selection (only if not provided)
    if config_path is None:
        default_train = "configs/train/train.yaml"
        if default_train not in train_yaml_files and train_yaml_files:
            default_train = train_yaml_files[0]
        print("\nAvailable Base Configurations:")
        for i, yf in enumerate(train_yaml_files):
            is_default = " (default)" if yf == default_train else ""
            print(f"[{i+1}] {yf}{is_default}")
        print(f"[{len(train_yaml_files)+1}] Enter custom path")
        while True:
            choice = input(f"Select base config [1-{len(train_yaml_files)+1}]: ").strip()
            if choice == "":
                config_path = default_train
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(train_yaml_files):
                    config_path = train_yaml_files[idx]
                    break
                elif idx == len(train_yaml_files):
                    config_path = input("Enter custom config path: ").strip()
                    break
            except ValueError:
                pass
            print("Invalid selection.")
    # 4. Sweep Config Selection (only if not provided)
    if sweep_config_path is None:
        default_sweep = "configs/sweeps/sweep.yaml"
        if default_sweep not in sweep_yaml_files and sweep_yaml_files:
            default_sweep = sweep_yaml_files[0]
        print("\nAvailable Sweep Configurations:")
        for i, yf in enumerate(sweep_yaml_files):
            is_default = " (default)" if yf == default_sweep else ""
            print(f"[{i+1}] {yf}{is_default}")
        print(f"[{len(sweep_yaml_files)+1}] Enter custom path")
        while True:
            choice = input(f"Select sweep config [1-{len(sweep_yaml_files)+1}]: ").strip()
            if choice == "":
                sweep_config_path = default_sweep
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sweep_yaml_files):
                    sweep_config_path = sweep_yaml_files[idx]
                    break
                elif idx == len(sweep_yaml_files):
                    sweep_config_path = input("Enter custom sweep config path: ").strip()
                    break
            except ValueError:
                pass
            print("Invalid selection.")
    return storage, study_name, config_path, sweep_config_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to base YAML config (train.yaml). Optuna suggestions overwrite specific keys.",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
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
    parser.add_argument("--worker-id", type=str, default=None, help="Unique ID for this worker process")
    cli_args = parser.parse_args()

    logger = setup_logger("optuna_sweep")
    logger.info("Starting Optuna Hyperparameter Sweep...")

    selected_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cli_args.use_mock_data:
        if os.environ.get("IS_CHILD_WORKER") == "true":
            selected_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            logger.info(f"Worker process detected. Pinned to GPU(s): {selected_gpus}")
        elif not selected_gpus:
            from gpu_selector import select_gpus
            selected_gpus = select_gpus()

    storage = cli_args.storage
    study_name = cli_args.study_name
    config_path = cli_args.config
    sweep_config_path = cli_args.sweep_config

    gpu_list = [g.strip() for g in selected_gpus.split(",")] if selected_gpus else []
    if len(gpu_list) > 1:
        logger.info(f"Spawning {len(gpu_list)} parallel sweep workers on GPUs: {gpu_list}")

        # Resolve interactive inputs first
        if storage is None or study_name is None or config_path is None or sweep_config_path is None:
            storage, study_name, config_path, sweep_config_path = interactive_setup(
                storage, study_name, config_path, sweep_config_path
            )

        # Config-driven W&B project retrieval
        base_cfg = {}
        if config_path:
            try:
                base_cfg = load_config(config_path)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        sweep_project = base_cfg.get("wandb", {}).get("sweep_project", "resilient-iot-sweeps-v2")

        if study_name is None:
            logger.info("Initializing a temporary WandB run to generate a stable study name...")
            wandb_run = wandb.init(
                project=sweep_project,
                reinit=True
            )
            study_name = wandb_run.name
            wandb_run.finish()
            logger.info(f"Generated stable study name: '{study_name}'")

        import math
        trials_per_worker = math.ceil(cli_args.n_trials / len(gpu_list))

        processes = []
        import sys
        for gpu in gpu_list:
            cmd = [
                sys.executable, "src/tune.py",
                "--config", config_path,
                "--sweep-config", sweep_config_path,
                "--storage", storage,
                "--study-name", study_name,
                "--n-trials", str(trials_per_worker),
                "--worker-id", f"gpu_{gpu}",
            ]
            if cli_args.use_mock_data:
                cmd.append("--use-mock-data")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["IS_CHILD_WORKER"] = "true"

            logger.info(f"Launching worker on GPU {gpu} (runs {trials_per_worker} trials)...")
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)

        for p in processes:
            p.wait()

        logger.info("All parallel sweep workers have finished.")

        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            logger.info("Sweep complete. Best hyperparameters:")
            if len(study.trials) > 0:
                for key, value in study.best_params.items():
                    logger.info(f"    {key}: {value}")
                logger.info(f"Best validation loss: {study.best_value}")
            else:
                logger.warning("No trials completed successfully.")
        except Exception as e:
            logger.error(f"Error loading final study results: {e}")

        sys.exit(0)

    # Single-GPU or CPU Execution Path
    if selected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
        logger.info(f"Isolated sweep execution to GPU(s): {selected_gpus}")

    if cli_args.use_mock_data:
        logger.info("Mode: MOCK DATA (local integration test)")

    # Trigger interactive setup for any values not explicitly provided via CLI flags
    if storage is None or study_name is None or config_path is None or sweep_config_path is None:
        storage, study_name, config_path, sweep_config_path = interactive_setup(
            storage, study_name, config_path, sweep_config_path
        )

    # Dynamically load the chosen configuration files
    base_config = load_config(config_path)
    logger.info(f"Base config loaded from: {config_path}")
    sweep_config = load_config(sweep_config_path)
    logger.info(f"Sweep config loaded from: {sweep_config_path}")

    sweep_project = base_config.get("wandb", {}).get("sweep_project", "resilient-iot-sweeps-v2")

    logger.info("Initializing WandB run...")
    run_name = study_name
    if cli_args.worker_id:
        run_name = f"{study_name}_{cli_args.worker_id}"

    wandb_run = wandb.init(
        project=sweep_project,
        name=run_name,
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
        wandb_kwargs={
            "project": sweep_project,
            "name": run_name,
            "reinit": True
        },
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize"
    )

    full_dataset = None
    if not cli_args.use_mock_data:
        data_cfg = base_config["data"]
        logger.info(f"Worker level: Pre-loading RealIoTDataset once...")
        connector = create_connector(base_config)
        full_dataset = RealIoTDataset(
            connector,
            data_cfg["start_date"],
            data_cfg["end_date"],
            seq_len=data_cfg["seq_len"],
            mask_prob=data_cfg.get("mask_prob", 0.15),
            target_veh_class=data_cfg["target_veh_class"],
            target_dir=data_cfg["target_dir"],
        )

    # Execute study
    study.optimize(
        make_objective(base_config, sweep_config, cli_args.use_mock_data, full_dataset=full_dataset),
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
