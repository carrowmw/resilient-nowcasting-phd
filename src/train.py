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

from dataset import (
    RealIoTDataset,
    custom_collate_real,
    make_spatial_holdout_mask,
    apply_spatial_holdout,
)
from db_connector import create_connector
from logger import setup_logger
from pipeline import ResilientIoTPipeline
from baselines import LSTMForecaster, GRUForecaster
from harness import (
    MetricAccumulator,
    evaluate_batch,
    make_fixed_holdout_mask,
    save_holdout_manifest,
    load_holdout_manifest,
    manifest_sha,
    save_results_manifest,
)
from harness.adapters import build_adapter


class DivergenceError(Exception):
    """Custom exception raised when model loss diverges (NaNs)."""
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResilientIoTPipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/train.yaml",
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


def run_training(
    config: dict,
    init_wandb: bool = True,
    use_mock_data: bool = False,
    study_name: str = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    full_dataset: RealIoTDataset = None,
    trial_number: int = None
) -> float:
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
    wandb_cfg = config.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "resilient-iot-v2")
    wandb_sweep_project = wandb_cfg.get("sweep_project", "resilient-iot-sweeps-v2")
    wandb_collection = wandb_cfg.get("collection", "resilient-iot-model-v2")

    if init_wandb:
        wandb.init(project=wandb_project, config=config)

    # ------------------------------------------------------------------ #
    # Directory Namespacing Strategy (Robust Model Versioning)           #
    # ------------------------------------------------------------------ #
    if study_name:
        # Sweep Namespace: models/{sweep_project}/{study_name}/
        project = wandb_sweep_project
        if trial_number is not None:
            run_name = f"{study_name}_trial_{trial_number}"
        else:
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
    loss_cfg = config.get("loss", {})

    # HARNESS: model family and evaluation mode.
    model_type = config.get("model_type", "gnn")
    eval_mode = config.get("eval_mode", "temporal")
    assert eval_mode in ("temporal", "inductive"), \
        f"CRITICAL: eval_mode must be temporal | inductive, got {eval_mode!r}"

    # PHASE 2: Spatial kriging configuration
    # ABLATION-SAFE: kriging_active controls holdout masking, decoder evaluation,
    # and W&B telemetry. It depends ONLY on spatial_holdout_prob so that the
    # data masking is identical across ablation runs (w_kriging = 0 vs > 0),
    # making task-specific metrics directly comparable.
    spatial_holdout_prob = data_cfg.get("spatial_holdout_prob", 0.0)
    w_kriging = loss_cfg.get("w_kriging", 0.0)
    w_impute = loss_cfg.get("w_impute", 1.0)
    w_forecast = loss_cfg.get("w_forecast", 1.0)
    kriging_active = spatial_holdout_prob > 0.0
    loss_weights = {"w_impute": w_impute, "w_forecast": w_forecast, "w_kriging": w_kriging}

    if train_loader is not None and val_loader is not None:
        logger.info("Using provided data loaders.")
        train_dataset = train_loader.dataset
    elif full_dataset is not None:
        logger.info("Using provided full_dataset.")
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        # Propagate dataset-level attributes
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
    elif use_mock_data:
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
        connector = create_connector(config)
        try:
            full_dataset = RealIoTDataset(
                connector,
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

            # PHASE 1: Propagate scaler statistics for scale-invariant loss
            train_dataset.traffic_mean = full_dataset.traffic_mean
            train_dataset.traffic_std = full_dataset.traffic_std
            val_dataset.traffic_mean = full_dataset.traffic_mean
            val_dataset.traffic_std = full_dataset.traffic_std

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
            connector.close()
            logger.error(f"Failed to load RealIoTDataset: {e}")
            raise

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # For real data the dataset determines num_nodes / feat_dims at        #
    # runtime; for mock data the config values are used directly.          #
    # HARNESS: the train/val loops below talk only to the ModelAdapter.    #
    # ------------------------------------------------------------------ #
    if model_type == "gnn":
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
            zinb_eps=model_cfg.get("zinb_eps", 1e-6),
            # PHASE 3: physical edge features cache (None for mock data / learned-only fallback).
            edge_features_cache_path=(
                None if use_mock_data else model_cfg.get("edge_features_cache_path")
            ),
        ).to(device)
    else:
        baseline_cls = LSTMForecaster if model_type == "lstm" else GRUForecaster
        model = baseline_cls(
            time_series_dim=train_dataset.time_series_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)

    adapter = build_adapter(model_type, model)
    logger.info(
        f"HARNESS: model_type={model_type}, eval_mode={eval_mode}, "
        f"capabilities={adapter.capabilities}"
    )

    # PHASE 1: Set scaler buffers AFTER moving model to device (GNN only — the
    # baselines carry no scaler buffers; their forecast head trains directly
    # against raw counts).
    if not use_mock_data and model_type == "gnn":
        traffic_mean = train_dataset.traffic_mean
        traffic_std = train_dataset.traffic_std

        logger.info(f"Extracted scaler stats from train_dataset: mean={traffic_mean:.4f}, std={traffic_std:.4f}")

        model.set_scaler_stats(traffic_mean, traffic_std)

        # DEFENSIVE CHECK: Validation set should have similar statistics
        # (both subsets reference the same full_dataset, so they should be identical)
        val_mean_diff = abs(val_dataset.traffic_mean - traffic_mean)
        val_std_diff = abs(val_dataset.traffic_std - traffic_std)
        relative_mean_diff = val_mean_diff / traffic_mean if traffic_mean != 0 else float('inf')
        relative_std_diff = val_std_diff / traffic_std if traffic_std != 0 else float('inf')

        if relative_mean_diff > 0.2 or relative_std_diff > 0.2:
            logger.warning(
                f"Train/val scaler mismatch detected: "
                f"Δmean={val_mean_diff:.2f} ({relative_mean_diff*100:.1f}%), "
                f"Δstd={val_std_diff:.2f} ({relative_std_diff*100:.1f}%)"
            )
        else:
            logger.info(
                f"Train/val scaler stats are consistent: "
                f"Δmean={val_mean_diff:.4f}, Δstd={val_std_diff:.4f}"
            )

    logger.info(
        f"Model initialised — num_nodes={train_dataset.num_nodes}, "
        f"static_feat_dim={train_dataset.static_feat_dim}, "
        f"time_series_dim={train_dataset.time_series_dim}, "
        f"embedding_dim={model_cfg['embedding_dim']}, "
        f"hidden_dim={model_cfg['hidden_dim']}, "
        f"num_layers={model_cfg.get('num_layers', 2)}, "
        f"num_heads={model_cfg.get('num_heads', 4)}"
    )

    # ------------------------------------------------------------------ #
    # HARNESS: holdout configuration                                      #
    #   temporal  — PHASE 2 behavior: seeded val mask, per-batch re-rolls  #
    #               during training (when kriging_active).                 #
    #   inductive — fixed seeded node holdout, blanked in train AND val;   #
    #               excluded from every gradient-producing loss term       #
    #               (zero-shot guarantee); scored at val time only.        #
    # ------------------------------------------------------------------ #
    num_nodes = train_dataset.num_nodes
    inductive_holdout_mask = None   # [N] bool — fixed, never trained on
    loss_node_mask = None           # [N] bool — True = node participates in losses
    holdout_manifest = None

    if eval_mode == "inductive":
        inductive_seed = train_cfg.get("random_seed", 42)
        inductive_frac = data_cfg.get("inductive_holdout_frac", 0.2)
        inductive_holdout_mask = make_fixed_holdout_mask(
            num_nodes=num_nodes,
            holdout_frac=inductive_frac,
            seed=inductive_seed,
            device=device,
        )
        loss_node_mask = ~inductive_holdout_mask

        # Manifest: resolved location IDs, not just the seed — this is the bridge
        # to the exporters and the cross-model comparability key.
        if full_dataset is not None and hasattr(full_dataset, "locations"):
            location_ids = list(full_dataset.locations)
        else:
            location_ids = [f"node_{i}" for i in range(num_nodes)]
            logger.warning("HARNESS: dataset exposes no location ids; using synthetic names in manifest")
        manifest_path = save_holdout_manifest(
            checkpoint_dir, location_ids, inductive_holdout_mask.cpu(),
            seed=inductive_seed, holdout_frac=inductive_frac,
        )
        holdout_manifest = load_holdout_manifest(checkpoint_dir)
        logger.info(
            f"HARNESS inductive holdout: {int(inductive_holdout_mask.sum())}/{num_nodes} nodes "
            f"held out (seed={inductive_seed}), manifest={manifest_path}, "
            f"sha={manifest_sha(holdout_manifest)}"
        )

    # Validation-time holdout to score kriging/zero-shot reconstruction against.
    if eval_mode == "inductive":
        val_holdout_mask = inductive_holdout_mask
    elif kriging_active:
        # PHASE 2: deterministic transductive val mask for stable metric tracking.
        val_holdout_mask = make_spatial_holdout_mask(
            num_nodes=num_nodes,
            holdout_prob=spatial_holdout_prob,
            device=device,
            seed=42,
        )
        logger.info(
            f"PHASE 2 kriging ACTIVE: w_kriging={w_kriging}, "
            f"spatial_holdout_prob={spatial_holdout_prob}, "
            f"validation holdout = {int(val_holdout_mask.sum())}/{num_nodes} nodes"
        )
    else:
        val_holdout_mask = None
        logger.info(
            f"PHASE 2 kriging INACTIVE (w_kriging={w_kriging}, "
            f"spatial_holdout_prob={spatial_holdout_prob})"
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

    import psutil
    memory_threshold_percent = 10.0 # Exit if < 10% RAM available

    for epoch in range(epochs):
        # Soft Memory Limit Check
        mem = psutil.virtual_memory()
        if mem.percent > (100.0 - memory_threshold_percent):
            logger.warning(
                f"Soft memory limit hit (Available: {mem.available / (1024**3):.2f} GB, "
                f"{mem.percent}% used). Terminating training early to prevent OOM crash."
            )
            break

        model.train()

        epoch_loss_total = 0.0
        epoch_loss_forecast = 0.0
        epoch_loss_impute = 0.0
        epoch_loss_kriging = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # PHASE 2: Re-roll a random spatial holdout per batch and blank the held-out
            # nodes' dynamic features so the encoder must rely on neighbour message passing.
            # HARNESS (inductive): the re-roll draws from participating nodes only —
            # the fixed inductive holdout supplies no kriging training signal.
            blank_mask = None
            if kriging_active:
                reroll_mask = make_spatial_holdout_mask(
                    num_nodes=num_nodes,
                    holdout_prob=spatial_holdout_prob,
                    device=device,
                    seed=None,
                )
                if loss_node_mask is not None:
                    reroll_mask = reroll_mask & loss_node_mask
                batch["spatial_holdout_mask"] = reroll_mask
                blank_mask = reroll_mask
            if inductive_holdout_mask is not None:
                blank_mask = (
                    inductive_holdout_mask if blank_mask is None
                    else blank_mask | inductive_holdout_mask
                )
            if blank_mask is not None:
                batch["dynamic_features"] = apply_spatial_holdout(
                    batch["dynamic_features"], blank_mask
                )

            optimizer.zero_grad()
            losses, _ = adapter.step(batch, loss_weights, loss_node_mask=loss_node_mask)
            losses.total.backward()

            # CRITICAL SAFEGUARD: Gradient Clipping (from NSTAR_BACKEND_ARCHITECTURE.md)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            optimizer.step()

            epoch_loss_total += losses.total.item()
            epoch_loss_impute += losses.impute.item()
            epoch_loss_forecast += losses.forecast.item()
            epoch_loss_kriging += losses.kriging.item()

        avg_train_loss = epoch_loss_total / len(train_loader)
        avg_train_impute = epoch_loss_impute / len(train_loader)
        avg_train_forecast = epoch_loss_forecast / len(train_loader)
        avg_train_kriging = epoch_loss_kriging / len(train_loader)

        # Validation Loop — shared harness path (capability-gated metrics).
        model.eval()
        val_loss_total = 0.0
        val_loss_kriging_total = 0.0
        val_accum = MetricAccumulator()

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Apply the fixed validation holdout for deterministic scoring
                # (inductive fixed mask, or PHASE 2 seeded transductive mask).
                if val_holdout_mask is not None:
                    batch["spatial_holdout_mask"] = val_holdout_mask
                    batch["dynamic_features"] = apply_spatial_holdout(
                        batch["dynamic_features"], val_holdout_mask
                    )

                losses, preds = adapter.step(
                    batch, loss_weights, loss_node_mask=loss_node_mask
                )
                val_loss_kriging_total += losses.kriging.item()

                # HARNESS (inductive): model selection must not see holdout-node
                # performance — strip the kriging term from the selection loss.
                if eval_mode == "inductive":
                    val_loss_total += losses.total.item() - w_kriging * losses.kriging.item()
                else:
                    val_loss_total += losses.total.item()

                val_accum.update(
                    evaluate_batch(
                        adapter.capabilities, preds, batch, holdout_mask=val_holdout_mask
                    )
                )

        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_kriging = val_loss_kriging_total / len(val_loader)
        val_metrics = val_accum.summary()
        val_rmse = val_metrics["rmse"]
        val_mae = val_metrics["mae"]
        val_zero_acc = val_metrics["zero_acc"]
        val_impute_mse = val_metrics["impute_mse"]
        val_kriging_msle = val_metrics["kriging_msle"]
        val_kriging_rmse = val_metrics["kriging_rmse"]
        val_kriging_mae = val_metrics["kriging_mae"]

        # Offensive Assertion for Divergence
        if torch.isnan(torch.tensor(avg_val_loss)):
            raise DivergenceError("Validation loss is NaN! The Burn Test Failed.")

        # None-safe formatting: None means "model cannot produce this" — shown
        # as N/A, never as 0.
        def _fmt(v, p=4):
            return "N/A" if v is None else f"{v:.{p}f}"

        kriging_log = (
            f", Kriging(train={avg_train_kriging:.4f}, val={avg_val_kriging:.4f}, "
            f"RMSE={_fmt(val_kriging_rmse, 2)}, MAE={_fmt(val_kriging_mae, 2)}, "
            f"MSLE={_fmt(val_kriging_msle)})"
            if val_holdout_mask is not None else ""
        )
        zero_acc_log = (
            f"{_fmt(val_zero_acc)} "
            f"[{val_metrics['zero_correct_count']}/{val_metrics['zero_true_count']}]"
            if val_zero_acc is not None else "N/A"
        )
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} "
            f"(RMSE: {_fmt(val_rmse, 2)}, MAE: {_fmt(val_mae, 2)}, "
            f"ZeroAcc: {zero_acc_log}, "
            f"ImputeMSE: {_fmt(val_impute_mse)})"
            f"{kriging_log}"
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

        # W&B Logging — capability-gated: None metrics are absent, never 0.0.
        if init_wandb or (wandb.run is not None):
            log_payload = {
                "train_loss_total": avg_train_loss,
                "train_loss_impute": avg_train_impute,
                "train_loss_forecast": avg_train_forecast,
                "val_loss_total": avg_val_loss,
                "epoch": epoch + 1,
            }
            log_payload.update({
                f"val_{k}": v for k, v in val_metrics.items() if v is not None
            })
            # PHASE 2 / inductive: kriging telemetry only when a holdout is scored.
            if val_holdout_mask is not None:
                log_payload.update({
                    "train_loss_kriging": avg_train_kriging,
                    "val_loss_kriging": avg_val_kriging,
                    "val_kriging_nodes": int(val_holdout_mask.sum().item()),
                })
            wandb.log(log_payload)

        # HARNESS: standardized cross-model results row, overwritten each epoch.
        # Joinable across runs on holdout_manifest_sha — identical hash means
        # identical evaluation conditions.
        save_results_manifest(checkpoint_dir, {
            "model_type": model_type,
            "eval_mode": eval_mode,
            "epoch": epoch + 1,
            "seed": train_cfg.get("random_seed", 42),
            "holdout_manifest_sha": manifest_sha(holdout_manifest),
            "val_loss_total": avg_val_loss,
            "metrics": val_metrics,
        })

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
                name=wandb_collection,
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

    if not args.use_mock_data:
        from gpu_selector import select_gpus
        selected_gpus = select_gpus()
        if selected_gpus:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
            logger.info(f"Isolated training execution to GPU(s): {selected_gpus}")

    run_training(config, init_wandb=True, use_mock_data=args.use_mock_data)


if __name__ == "__main__":
    main()
