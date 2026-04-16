import argparse
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import logging

from train import run_training

def objective(trial):
    """
    Optuna objective function for hyperparameter tuning.
    """
    # Define hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    w_forecast = trial.suggest_float("w_forecast", 0.5, 5.0)
    w_impute = trial.suggest_float("w_impute", 0.5, 5.0)
    
    # Map to configuration object for the training runner
    args = argparse.Namespace(
        epochs=2,
        batch_size=4,
        lr=lr,
        w_forecast=w_forecast,
        w_impute=w_impute
    )
    
    # Execute training loop. 
    # init_wandb=False prevents train.py from spawning a separate W&B run,
    # as the WeightsAndBiasesCallback manages the run at the study level.
    val_loss = run_training(args, init_wandb=False)
    
    return val_loss

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Optuna Hyperparameter Sweep...")

    # Initialize the W&B Callback for Optuna
    wandbc = WeightsAndBiasesCallback(
        metric_name="val_loss_total",
        wandb_kwargs={"project": "resilient-iot-sweeps"}
    )
    
    # Create study
    study = optuna.create_study(direction="minimize")
    
    # Execute study (The Online Local Test)
    # n_trials=2 to ensure the loop runs efficiently and validates the architecture
    study.optimize(objective, n_trials=2, callbacks=[wandbc])
    
    logger.info("Sweep complete. Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    logger.info(f"Best validation loss: {study.best_value}")
