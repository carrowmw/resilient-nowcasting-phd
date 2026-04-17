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

def make_objective(cli_args: argparse.Namespace):
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
            epochs=cli_args.epochs,
            batch_size=cli_args.batch_size,
            lr=params.get("lr", 1e-3),
            w_forecast=params.get("w_forecast", 1.0),
            w_impute=params.get("w_impute", 1.0),
            use_mock_data=cli_args.use_mock_data,
            start_date=cli_args.start_date,
            end_date=cli_args.end_date,
            seq_len=cli_args.seq_len,
            veh_class=cli_args.veh_class,
            dir=cli_args.dir
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
    parser.add_argument("--n_trials", type=int, default=2, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs per trial")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per trial")
    parser.add_argument("--start_date", type=str, default="2023-03-02", help="Start date for real data (inclusive)")
    parser.add_argument("--end_date", type=str, default="2023-04-02", help="End date for real data (exclusive)")
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence length")
    parser.add_argument("--veh_class", type=str, default="person", help="Target vehicle class")
    parser.add_argument("--dir", type=str, default="all", help="Target direction")
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
    study.optimize(make_objective(cli_args), n_trials=cli_args.n_trials, callbacks=[wandbc])

    logger.info("Sweep complete. Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    logger.info(f"Best validation loss: {study.best_value}")
