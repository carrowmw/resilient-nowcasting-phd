import argparse
import copy
import logging
import os

import optuna
import yaml
from optuna.integration.wandb import WeightsAndBiasesCallback

from logger import setup_logger
from train import load_config, run_training

# Load sweep configuration globally so the objective function can access it
SWEEP_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "sweep.yaml")
with open(SWEEP_CONFIG_PATH, "r") as f:
    SWEEP_CONFIG = yaml.safe_load(f)

# Default base config path — can be overridden via --config CLI flag
DEFAULT_TRAIN_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "train.yaml"
)

# Mapping from sweep.yaml parameter names to their nested location in train.yaml
_PARAM_PATH: dict[str, tuple[str, str]] = {
    "lr":         ("training", "lr"),
    "w_forecast": ("loss",     "w_forecast"),
    "w_impute":   ("loss",     "w_impute"),
}


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

        for param_name, param_config in SWEEP_CONFIG.get("parameters", {}).items():
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
            else:
                raise ValueError(f"Unsupported distribution: {dist} for parameter {param_name}")

            section, key = _PARAM_PATH[param_name]
            config[section][key] = val

        # Execute training loop.
        # init_wandb=False prevents train.py from spawning a separate W&B run;
        # the WeightsAndBiasesCallback manages the run at the study level.
        return run_training(config=config, init_wandb=False, use_mock_data=use_mock_data)

    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_TRAIN_CONFIG,
        help="Path to base YAML config (train.yaml). Optuna suggestions overwrite specific keys.",
    )
    parser.add_argument(
        "--use_mock_data",
        action="store_true",
        help="Use synthetic mock data instead of real parquet data (for local CPU integration tests)",
    )
    parser.add_argument("--n_trials", type=int, default=2, help="Number of Optuna trials")
    cli_args = parser.parse_args()

    logger = setup_logger("optuna_sweep")
    logger.info("Starting Optuna Hyperparameter Sweep...")
    if cli_args.use_mock_data:
        logger.info("Mode: MOCK DATA (local integration test)")

    base_config = load_config(cli_args.config)
    logger.info(f"Base config loaded from: {cli_args.config}")

    # Initialize the W&B Callback for Optuna
    wandbc = WeightsAndBiasesCallback(
        metric_name="val_loss_total",
        wandb_kwargs={"project": "resilient-iot-sweeps"},
    )

    # Create study
    study = optuna.create_study(direction="minimize")

    # Execute study
    study.optimize(
        make_objective(base_config, cli_args.use_mock_data),
        n_trials=cli_args.n_trials,
        callbacks=[wandbc],
    )

    logger.info("Sweep complete. Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    logger.info(f"Best validation loss: {study.best_value}")
