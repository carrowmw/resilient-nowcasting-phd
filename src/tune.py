import yaml
import argparse
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import logging
import os

from train import run_training

# Load sweep configuration globally so the objective function can access it
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "sweep.yaml")
with open(CONFIG_PATH, "r") as f:
    SWEEP_CONFIG = yaml.safe_load(f)

def make_objective(use_mock_data: bool):
    """
    Factory that returns an Optuna objective bound to the requested data mode.
    Separating the factory from __main__ lets us inject --use_mock_data at the
    study level without relying on a global variable.
    """
    def objective(trial):
        """
        Optuna objective function for hyperparameter tuning.
        Dynamically generates suggestions based on configs/sweep.yaml.
        """
        # Define hyperparameter search space dynamically
        params = {}
        for param_name, param_config in SWEEP_CONFIG.get("parameters", {}).items():
            dist = param_config.get("distribution")
            if dist == "log_uniform_values":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["min"],
                    param_config["max"],
                    log=True
                )
            elif dist == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["min"],
                    param_config["max"]
                )
            else:
                raise ValueError(f"Unsupported distribution: {dist} for parameter {param_name}")

        # Map to configuration object for the training runner
        args = argparse.Namespace(
            epochs=2,
            batch_size=4,
            lr=params.get("lr", 1e-3),
            w_forecast=params.get("w_forecast", 1.0),
            w_impute=params.get("w_impute", 1.0),
            use_mock_data=use_mock_data,
            start_date="2023-03-02",
            end_date="2023-04-02",
            seq_len=12,
            veh_class="person",
            dir="all"
        )

        # Execute training loop.
        # init_wandb=False prevents train.py from spawning a separate W&B run,
        # as the WeightsAndBiasesCallback manages the run at the study level.
        val_loss = run_training(args, init_wandb=False)

        return val_loss

    return objective

from logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter sweep")
    parser.add_argument("--use_mock_data", action="store_true",
                        help="Use synthetic mock data instead of real parquet data (for local CPU integration tests)")
    cli_args = parser.parse_args()

    logger = setup_logger("optuna_sweep")
    logger.info("Starting Optuna Hyperparameter Sweep...")
    if cli_args.use_mock_data:
        logger.info("Mode: MOCK DATA (local integration test)")

    # Initialize the W&B Callback for Optuna
    wandbc = WeightsAndBiasesCallback(
        metric_name="val_loss_total",
        wandb_kwargs={"project": "resilient-iot-sweeps"}
    )

    # Create study
    study = optuna.create_study(direction="minimize")

    # Execute study (The Online Local Test)
    # n_trials=2 to ensure the loop runs efficiently and validates the architecture
    study.optimize(make_objective(cli_args.use_mock_data), n_trials=2, callbacks=[wandbc])

    logger.info("Sweep complete. Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    logger.info(f"Best validation loss: {study.best_value}")
