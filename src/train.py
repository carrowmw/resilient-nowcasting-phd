import argparse
import copy
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from dataset import RealIoTDataset, custom_collate_real
from logger import setup_logger
from metrics import calculate_metrics
from pipeline import ResilientIoTPipeline


class DivergenceError(Exception):
    """Custom exception raised when model loss diverges (NaNs)."""
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResilientIoTPipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--use_mock_data",
        action="store_true",
        help="Use mock data instead of real parquet data",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_mock_dataloader(
    batch_size: int,
    num_nodes: int,
    seq_len: int,
    time_series_dim: int,
    static_feat_dim: int,
    dataset_size: int,
    mask_prob: float = 0.15,
):
    """
    Creates a mock DataLoader yielding batches that match the pipeline's
    expected dimensional structure.
    """
    class MockPipelineDataset(torch.utils.data.Dataset):
        def __init__(self, size: int):
            self.size = size
            self.num_nodes = num_nodes
            self.static_feat_dim = static_feat_dim
            self.time_series_dim = time_series_dim

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int) -> dict:
            dyn = torch.randn(num_nodes, seq_len, time_series_dim)
            mask = torch.rand_like(dyn) < mask_prob
            dyn_masked = dyn.clone()
            dyn_masked[mask] = float("nan")
            return {
                "static_features": torch.randn(num_nodes, static_feat_dim),
                "dynamic_features": dyn_masked,
                "target_impute": dyn.clone(),
                "target_forecast": torch.randint(0, 50, (num_nodes,)).float(),
                "node_mask": mask,
            }

    dataset = MockPipelineDataset(size=dataset_size)

    def custom_collate(batch_list):
        return {
            "static_features": batch_list[0]["static_features"],
            "dynamic_features": torch.stack([b["dynamic_features"] for b in batch_list]),
            "target_impute": torch.stack([b["target_impute"] for b in batch_list]),
            "target_forecast": torch.stack([b["target_forecast"] for b in batch_list]),
            "node_mask": torch.stack([b["node_mask"] for b in batch_list]),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate), dataset


def run_training(config: dict, init_wandb: bool = True, use_mock_data: bool = False, study_name: str = None) -> float:
    """
    Core execution logic for a single training run.
    Accepts a fully-specified config dictionary (see configs/train.yaml).
    Returns the final validation loss.
    """
    logger = logging.getLogger(__name__)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize W&B only if requested; pass the full nested config so all
    # hyperparameters (data / model / training / loss) are logged together.
    if init_wandb:
        wandb.init(project="resilient-iot", config=config)

    # ------------------------------------------------------------------ #
    # Directory Namespacing Strategy (Robust Model Versioning)           #
    # ------------------------------------------------------------------ #
    if study_name:
        # Sweep Namespace: models/resilient-iot-sweeps/{study_name}/
        project = "resilient-iot-sweeps"
        run_name = study_name
    elif wandb.run is not None:
        # Production Namespace: models/{project}/{run_name}/
        project = wandb.run.project
        run_name = wandb.run.name
    else:
        # Local Namespace: models/local/{timestamp}/
        project = "local"
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_dir = os.path.join("models", project, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    logger.info(f"Checkpointing enabled: {checkpoint_path}")

    # --- ARCHITECT TWINNING: Co-locate Config copy ---
    config_save_path = os.path.join(checkpoint_dir, "config.yaml")
    try:
        with open(config_save_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logger.info(f"Co-located training configuration saved to: {config_save_path}")
    except Exception as e:
        logger.error(f"Failed to save co-located configuration: {e}")

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    if use_mock_data:
        logger.info("Using mock dataloader.")
        train_loader, train_dataset = create_mock_dataloader(
            batch_size=train_cfg["batch_size"],
            num_nodes=model_cfg["num_nodes"],
            seq_len=data_cfg["seq_len"],
            time_series_dim=model_cfg["time_series_dim"],
            static_feat_dim=model_cfg["static_feat_dim"],
            dataset_size=20,
            mask_prob=data_cfg.get("mask_prob", 0.15),
        )
        val_loader, _ = create_mock_dataloader(
            batch_size=train_cfg["batch_size"],
            num_nodes=model_cfg["num_nodes"],
            seq_len=data_cfg["seq_len"],
            time_series_dim=model_cfg["time_series_dim"],
            static_feat_dim=model_cfg["static_feat_dim"],
            dataset_size=8,
            mask_prob=data_cfg.get("mask_prob", 0.15),
        )
    else:
        logger.info(f"Loading RealIoTDataset from {data_cfg['start_date']} to {data_cfg['end_date']}")
        parquet_path = "data/processed/master_dataset_15T.parquet"
        try:
            full_dataset = RealIoTDataset(
                parquet_path,
                data_cfg["start_date"],
                data_cfg["end_date"],
                seq_len=data_cfg["seq_len"],
                mask_prob=data_cfg.get("mask_prob", 0.15),
                target_veh_class=data_cfg["target_veh_class"],
                target_dir=data_cfg["target_dir"],
            )

            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size

            if train_size == 0 or val_size == 0:
                raise ValueError(
                    "Not enough data to split into train and val. Try a wider date range."
                )

            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

            # Propagate dataset-level attributes to the Subset wrapper so the
            # model initializer below can read them uniformly.
            train_dataset.num_nodes = full_dataset.num_nodes
            train_dataset.static_feat_dim = full_dataset.static_feat_dim
            train_dataset.time_series_dim = full_dataset.time_series_dim

            train_loader = DataLoader(
                train_dataset,
                batch_size=train_cfg["batch_size"],
                collate_fn=custom_collate_real,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_cfg["batch_size"],
                collate_fn=custom_collate_real,
            )
        except Exception as e:
            logger.error(f"Failed to load RealIoTDataset: {e}")
            raise

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # For real data the dataset determines num_nodes / feat_dims at        #
    # runtime; for mock data the config values are used directly.          #
    # ------------------------------------------------------------------ #
    model = ResilientIoTPipeline(
        num_nodes=train_dataset.num_nodes,
        static_feat_dim=train_dataset.static_feat_dim,
        time_series_dim=train_dataset.time_series_dim,
        seq_len=data_cfg["seq_len"],
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        sparsity_threshold=model_cfg["sparsity_threshold"],
        num_layers=model_cfg.get("num_layers", 2),
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        dim_feedforward=model_cfg.get("dim_feedforward", 256),
        zinb_eps=model_cfg.get("zinb_eps", 1e-6)
    ).to(device)

    logger.info(
        f"Model initialised — num_nodes={train_dataset.num_nodes}, "
        f"static_feat_dim={train_dataset.static_feat_dim}, "
        f"time_series_dim={train_dataset.time_series_dim}, "
        f"embedding_dim={model_cfg['embedding_dim']}, "
        f"hidden_dim={model_cfg['hidden_dim']}, "
        f"num_layers={model_cfg.get('num_layers', 2)}, "
        f"num_heads={model_cfg.get('num_heads', 4)}"
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    logger.info("Starting training loop...")

    best_val_loss = float("inf")

    epochs = train_cfg["epochs"]
    grad_clip_max_norm = train_cfg.get("grad_clip_max_norm", 1.0)
    early_stop_patience = train_cfg.get("early_stop_patience", 5)
    early_stop_min_delta = train_cfg.get("early_stop_min_delta", 0.001)
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_forecast = 0.0
        epoch_loss_impute = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            loss, loss_impute, loss_forecast, _, _, _, _ = model(batch)
            loss.backward()

            # CRITICAL SAFEGUARD: Gradient Clipping (from NSTAR_BACKEND_ARCHITECTURE.md)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            optimizer.step()

            epoch_loss_total += loss.item()
            epoch_loss_impute += loss_impute.item()
            epoch_loss_forecast += loss_forecast.item()

        avg_train_loss = epoch_loss_total / len(train_loader)
        avg_train_impute = epoch_loss_impute / len(train_loader)
        avg_train_forecast = epoch_loss_forecast / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss_total = 0.0
        val_metrics_accum = {"rmse": 0.0, "mae": 0.0, "impute_mse": 0.0}
        zero_true_total = 0
        zero_correct_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss, _, _, imputed_features, mu, _, pi = model(batch)
                val_loss_total += loss.item()

                batch_metrics = calculate_metrics(
                    target_forecast=batch["target_forecast"],
                    mu=mu,
                    pi=pi,
                    imputed_features=imputed_features,
                    target_impute=batch["target_impute"],
                    node_mask=batch["node_mask"],
                )
                for k in val_metrics_accum:
                    val_metrics_accum[k] += batch_metrics[k]
                zero_true_total += batch_metrics["zero_true_count"]
                zero_correct_total += batch_metrics["zero_correct_count"]

        avg_val_loss = val_loss_total / len(val_loader)
        val_rmse = val_metrics_accum["rmse"] / len(val_loader)
        val_mae = val_metrics_accum["mae"] / len(val_loader)
        val_zero_acc = zero_correct_total / zero_true_total if zero_true_total > 0 else 0.0
        val_impute_mse = val_metrics_accum["impute_mse"] / len(val_loader)

        # Offensive Assertion for Divergence
        if torch.isnan(torch.tensor(avg_val_loss)):
            raise DivergenceError("Validation loss is NaN! The Burn Test Failed.")

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} "
            f"(RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, "
            f"ZeroAcc: {val_zero_acc:.4f} [{zero_correct_total}/{zero_true_total}], "
            f"ImputeMSE: {val_impute_mse:.4f})"
        )

        # Checkpointing and Early Stopping
        if avg_val_loss < (best_val_loss - early_stop_min_delta):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                f"New best model saved to {checkpoint_path} with val_loss: {best_val_loss:.4f}"
            )
        else:
            epochs_no_improve += 1
            logger.info(f"Early stopping counter: {epochs_no_improve} out of {early_stop_patience}")

        # W&B Logging
        if init_wandb or (wandb.run is not None):
            wandb.log(
                {
                    "train_loss_total": avg_train_loss,
                    "train_loss_impute": avg_train_impute,
                    "train_loss_forecast": avg_train_forecast,
                    "val_loss_total": avg_val_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "val_zero_acc": val_zero_acc,
                    "val_zero_true_count": zero_true_total,
                    "val_zero_correct_count": zero_correct_total,
                    "val_impute_mse": val_impute_mse,
                    "epoch": epoch + 1,
                }
            )
            
        if epochs_no_improve >= early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    logger.info("Training complete.")

    # ------------------------------------------------------------------ #
    # W&B Artifact Upload (Phase 1)                                      #
    # ------------------------------------------------------------------ #
    if (init_wandb or (wandb.run is not None)) and os.path.exists(checkpoint_path):
        logger.info(f"Uploading model artifact to W&B from {checkpoint_path}")
        try:
            artifact = wandb.Artifact(
                name="resilient-iot-model",
                type="model",
                description=f"Best model from run {wandb.run.name if wandb.run else 'unknown'}",
                metadata=config
            )
            artifact.add_file(checkpoint_path, name="best_model.pt")
            if os.path.exists(config_save_path):
                artifact.add_file(config_save_path, name="config.yaml")
            
            wandb.log_artifact(artifact)
            logger.info("Model artifact successfully logged to W&B.")
        except Exception as e:
            logger.error(f"Failed to upload W&B artifact: {e}")

    if init_wandb:
        wandb.finish()

    # Force Python to release GPU memory held by local variables
    del model
    del optimizer
    del train_loader
    del val_loader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_val_loss


def main():
    logger = setup_logger("train_pipeline")
    args = parse_args()
    config = load_config(args.config)
    run_training(config, init_wandb=True, use_mock_data=args.use_mock_data)


if __name__ == "__main__":
    main()
