import argparse
import os
from typing import Callable, List, NamedTuple, Optional, Tuple

# Perform early GPU selection before ANY PyTorch or W&B bindings are loaded
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use_mock_data", action="store_true")
    early_args, _ = parser.parse_known_args()

    if not early_args.use_mock_data:
        existing_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing_gpus is None:
            from gpu_selector import select_gpus
            selected_gpus = select_gpus()
            if selected_gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy
import functools
import logging
import math
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from dataset import (
    RealIoTDataset,
    custom_collate_real,
    make_spatial_holdout_mask,
)
from db_connector import create_connector
from logger import setup_logger
from losses import TrafficScale
from model_factory import build_model
from run_config import resolve_config
from synthetic_connector import SyntheticConnector
from harness import (
    MetricAccumulator,
    evaluate_batch,
    inductive_holdout_ids,
    mask_from_manifest,
    select_network_locations,
    validation_station_ids,
    save_holdout_manifest,
    load_holdout_manifest,
    manifest_sha,
    save_results_manifest,
)
from harness.adapters import build_adapter
from harness.batch_prep import BatchPreparer
from harness.evaluate_v5 import V5Evaluator
from harness.mtl import MTLBalancer


def _worker_init_fn(worker_id: int) -> None:
    # PyTorch reseeds torch per worker but NOT numpy; identical numpy streams across workers would silently correlate augmentation.
    np.random.seed(torch.initial_seed() % 2**32)


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
        help="Use the SyntheticConnector city instead of the real database",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name for W&B and checkpoint directory",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Search for configs/infra.yaml as defaults.
    # Checks 1 and 2 directory levels above the training config, plus repo-relative configs/infra.yaml.
    config_dir = os.path.dirname(os.path.abspath(config_path))
    infra_path = None
    for candidate in [
        os.path.join(config_dir, "..", "infra.yaml"),
        os.path.join(config_dir, "..", "..", "infra.yaml"),
        os.path.join(os.getcwd(), "configs", "infra.yaml"),
    ]:
        norm = os.path.normpath(candidate)
        if os.path.exists(norm):
            infra_path = norm
            break

    if infra_path and os.path.exists(infra_path):
        with open(infra_path, "r") as f:
            infra = yaml.safe_load(f) or {}
        for key, val in infra.items():
            if key not in cfg:
                cfg[key] = val
            elif isinstance(val, dict) and isinstance(cfg.get(key), dict):
                for k, v in val.items():
                    cfg[key].setdefault(k, v)

    return cfg


def chronological_purged_split(
    full_dataset,
    train_split: float = 0.7,
    val_split: float = 0.15,
    purge_steps: int = 13,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, Optional[torch.utils.data.Subset]]:
    """
    Chronological train/val/test split with purge buffers (F-C01 remediation).
    Prevents temporal window overlap leakage between train, validation, and test partitions.
    """
    total_len = len(full_dataset)
    train_size = int(train_split * total_len)
    val_size = int(val_split * total_len)

    train_end = train_size
    val_start = min(train_end + purge_steps, total_len)
    val_end = min(val_start + val_size, total_len)
    test_start = min(val_end + purge_steps, total_len)
    test_end = total_len

    if train_end <= 0 or val_end <= val_start:
        # Fallback if dataset is very short (e.g. mock test): split without purge
        train_end = max(1, int(train_split * total_len))
        val_start = train_end
        val_end = total_len
        test_start = val_end
        test_end = total_len

    train_dataset = torch.utils.data.Subset(full_dataset, list(range(0, train_end)))
    val_dataset = torch.utils.data.Subset(full_dataset, list(range(val_start, val_end)))
    test_dataset = (
        torch.utils.data.Subset(full_dataset, list(range(test_start, test_end)))
        if test_end > test_start
        else None
    )

    for subset in [train_dataset, val_dataset, test_dataset]:
        if subset is not None:
            subset.num_nodes = getattr(full_dataset, "num_nodes", None)
            subset.static_feat_dim = getattr(full_dataset, "static_feat_dim", None)
            subset.time_series_dim = getattr(full_dataset, "time_series_dim", None)
            subset.traffic_mean = getattr(full_dataset, "traffic_mean", None)
            subset.traffic_std = getattr(full_dataset, "traffic_std", None)

    return train_dataset, val_dataset, test_dataset


# Synthetic city used by --use_mock_data (structurally realistic MNAR feed).
MOCK_DAYS = 21


def _network_filter(config: dict) -> Optional[Callable[[List[str]], List[str]]]:
    """Sensor-density runs (data.train_node_count): the stations the dataset keeps.
    Dropped stations leave the network entirely — no inputs, no loss, no graph
    node, no profile donor — as if they were never installed."""
    data_cfg = config["data"]
    if data_cfg["train_node_count"] is None:
        return None
    return functools.partial(
        select_network_locations,
        holdout_frac=data_cfg["inductive_holdout_frac"],
        holdout_seed=data_cfg["inductive_holdout_seed"],
        train_node_count=data_cfg["train_node_count"],
        train_node_seed=data_cfg["train_node_seed"],
    )


def build_dataset(config: dict, use_mock_data: bool) -> RealIoTDataset:
    """The full dataset for a resolved config (shared with tune.py so a pre-loaded
    sweep dataset is built with exactly the run's v5 data options)."""
    data_cfg = config["data"]
    common = dict(
        seq_len=data_cfg["seq_len"],
        target_veh_class=data_cfg["target_veh_class"],
        target_dir=data_cfg["target_dir"],
        train_split=config["train_split"],
        ablate_masks=data_cfg["ablate_masks"],
        log1p_inputs=data_cfg["log1p_inputs"],
        complete_weather=data_cfg["complete_weather"],
        location_filter=_network_filter(config),
    )
    if use_mock_data:
        start = pd.Timestamp(data_cfg["start_date"])
        connector = SyntheticConnector(
            num_nodes=config["model"]["num_nodes"], start_date=str(start), days=MOCK_DAYS,
            seed=config["training"]["random_seed"],
        )
        return RealIoTDataset(connector, str(start), str(start + pd.Timedelta(days=MOCK_DAYS)), **common)
    connector = create_connector(config)
    try:
        return RealIoTDataset(connector, data_cfg["start_date"], data_cfg["end_date"], **common)
    finally:
        # RealIoTDataset.__init__ loads all data into memory; the connector is
        # no longer needed after construction whether it succeeded or failed.
        connector.close()


def _loader(subset, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=batch_size,
        collate_fn=custom_collate_real,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=train,
        worker_init_fn=_worker_init_fn,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _fmt(v, p=4):
    # None means "model cannot produce this" — shown as N/A, never as 0.
    return "N/A" if v is None else f"{v:.{p}f}"


# Test batches that also run the mask-flip / value-perturbation diagnostic.
PERTURB_BATCHES = 8


class NodeMasks(NamedTuple):
    """Inductive node roles for one run (all [N] bool on `device`, or None)."""
    test_holdout: Optional[torch.Tensor]   # fixed test holdout (scored at test as "ind")
    val_stations: Optional[torch.Tensor]   # v6 validation stations (scored at val as "ind")
    unseen: Optional[torch.Tensor]         # never trained on: blanked, loss-excluded, no profile donor
    held_ids: List[str]
    val_ids: List[str]


def build_node_masks(config: dict, location_ids: List[str], all_location_ids: List[str],
                     device: torch.device) -> NodeMasks:
    """Inductive holdout (drawn over the FULL station list, so a run with fewer training
    stations is still scored on the same held-out nodes) plus, when
    data.val_station_count > 0, validation stations drawn from the training pool."""
    data_cfg = config["data"]
    if config["eval_mode"] != "inductive":
        return NodeMasks(None, None, None, [], [])
    held_ids = inductive_holdout_ids(all_location_ids, data_cfg["inductive_holdout_frac"],
                                     data_cfg["inductive_holdout_seed"])
    test_holdout = mask_from_manifest({"holdout_location_ids": held_ids}, location_ids, device=device)
    assert int(test_holdout.sum()) == len(held_ids), "CRITICAL: held-out stations missing from the dataset"
    val_ids, val_stations, unseen = [], None, test_holdout
    if data_cfg["val_station_count"] > 0:
        val_ids = validation_station_ids(location_ids, held_ids, data_cfg["val_station_count"],
                                         data_cfg["val_station_seed"])
        val_stations = mask_from_manifest({"holdout_location_ids": val_ids}, location_ids, device=device)
        assert not bool((val_stations & test_holdout).any()), "CRITICAL: validation and test stations overlap"
        unseen = test_holdout | val_stations
    return NodeMasks(test_holdout, val_stations, unseen, held_ids, val_ids)


def cosine_lr_factor(epoch: int, epochs: int, warmup: int, floor: float) -> float:
    """v6 LR multiplier for 0-based `epoch`: linear warmup to 1 over `warmup` epochs,
    then a half cosine that reaches `floor` at the last epoch of the budget."""
    if epoch < warmup:
        return (epoch + 1) / (warmup + 1)
    progress = min((epoch - warmup) / max(epochs - warmup - 1, 1), 1.0)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_training(
    config: dict,
    init_wandb: bool = True,
    use_mock_data: bool = False,
    study_name: str = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    full_dataset: RealIoTDataset = None,
    trial_number: Optional[int] = None
) -> float:
    """
    Core execution logic for a single training run.
    Accepts a run config (v5, or a v4-era config upgraded by run_config.resolve_config).
    Returns the best validation selection score (selection.metric).
    """
    logger = logging.getLogger(__name__)

    # v5: every key resolved up front (v4 configs get LEGACY_V4_EQUIVALENTS).
    config = resolve_config(config)

    # Reproducibility — seed from config, not hardcoded, so experiments are traceable.
    _seed = config["training"]["random_seed"]
    torch.manual_seed(_seed)
    np.random.seed(_seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    amp_enabled: bool = config["amp"]
    if amp_enabled:
        if device.type == "cuda":
            assert torch.cuda.is_bf16_supported(), \
                "CRITICAL: amp=true requested but this device does not support bf16. " \
                "Set amp: false in train.yaml or use a bf16-capable GPU."
            logger.info("bf16 autocast ENABLED (amp=true)")
        else:
            logger.info("Device is CPU: bf16 autocast disabled for execution")
            amp_enabled = False
    else:
        logger.info("bf16 autocast DISABLED (amp=false)")

    # Initialize W&B only if requested; pass the full nested config so all
    # hyperparameters (data / model / training / loss) are logged together.
    wandb_cfg = config["wandb"]
    wandb_project = wandb_cfg["project"]
    wandb_sweep_project = wandb_cfg["sweep_project"]
    wandb_collection = wandb_cfg["collection"]
    run_name_cfg = wandb_cfg.get("run_name") or wandb_cfg.get("name") or wandb_collection

    if init_wandb:
        wandb.init(project=wandb_project, name=run_name_cfg, config=config)

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
    elif wandb.run is not None and not wandb.run.disabled:
        # Production Namespace: models/{project}/{run_name}/
        project = wandb.run.project
        run_name = wandb.run.name
    else:
        # Local Namespace: models/local/{run name or timestamp}/
        project = "local"
        run_name = f"{run_name_cfg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    checkpoint_dir = os.path.join("models", project, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    logger.info(f"Checkpointing enabled: {checkpoint_path}")

    # --- ARCHITECT TWINNING: Co-locate the RESOLVED config copy ---
    config_save_path = os.path.join(checkpoint_dir, "config.yaml")
    try:
        with open(config_save_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logger.info(f"Co-located training configuration saved to: {config_save_path}")
    except OSError as e:
        logger.error(f"Failed to save co-located configuration: {e}")

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    eval_cfg = config["evaluation"]
    sel_cfg = config["selection"]

    num_workers = train_cfg["num_workers"]
    assert num_workers >= 0, f"CRITICAL: num_workers must be non-negative, got {num_workers}"

    # HARNESS: model family and evaluation mode.
    model_type = config["model_type"]
    eval_mode = config["eval_mode"]
    assert eval_mode in ("temporal", "inductive"), \
        f"CRITICAL: eval_mode must be temporal | inductive, got {eval_mode!r}"
    trainable = model_type != "naive"

    # PHASE 2: Spatial kriging configuration
    # ABLATION-SAFE: kriging_active controls holdout masking, decoder evaluation,
    # and W&B telemetry. It depends ONLY on spatial_holdout_prob so that the
    # data masking is identical across ablation runs (w_kriging = 0 vs > 0),
    # making task-specific metrics directly comparable.
    spatial_holdout_prob = data_cfg["spatial_holdout_prob"]
    w_kriging = loss_cfg["w_kriging"]
    kriging_active = spatial_holdout_prob > 0.0
    loss_weights = {"w_impute": loss_cfg["w_impute"], "w_forecast": loss_cfg["w_forecast"], "w_kriging": w_kriging}

    if train_loader is not None and val_loader is not None:
        logger.info("Using provided data loaders.")
        train_dataset, val_dataset = train_loader.dataset, val_loader.dataset
        assert isinstance(train_dataset, torch.utils.data.Subset), \
            "CRITICAL: provided loaders must wrap Subsets of a RealIoTDataset (v5 tables)"
        full_dataset = train_dataset.dataset
        test_loader, test_dataset = None, None
    else:
        if full_dataset is not None:
            logger.info("Using provided full_dataset with chronological purged splitting (F-C01 fix).")
        else:
            logger.info(
                "Building synthetic dataset (SyntheticConnector)" if use_mock_data
                else f"Loading RealIoTDataset from {data_cfg['start_date']} to {data_cfg['end_date']}"
            )
            full_dataset = build_dataset(config, use_mock_data)
        train_dataset, val_dataset, test_dataset = chronological_purged_split(
            full_dataset,
            train_split=config["train_split"],
            val_split=config["val_split"],
            purge_steps=data_cfg["seq_len"] + 1,
        )
        train_loader = _loader(train_dataset, train_cfg["batch_size"], num_workers, train=True)
        val_loader = _loader(val_dataset, train_cfg["batch_size"], num_workers, train=False)
        test_loader = (
            _loader(test_dataset, train_cfg["batch_size"], num_workers, train=False)
            if test_dataset is not None and len(test_dataset) > 0 else None
        )

    # A pre-built dataset (tune.py, tests) must match this run's data options.
    assert full_dataset.seq_len == data_cfg["seq_len"], "CRITICAL: dataset seq_len != config"
    assert full_dataset.log1p_inputs == data_cfg["log1p_inputs"], "CRITICAL: dataset log1p_inputs != config"
    assert full_dataset.complete_weather == data_cfg["complete_weather"], "CRITICAL: dataset complete_weather != config"
    assert full_dataset.ablate_masks == data_cfg["ablate_masks"], "CRITICAL: dataset ablate_masks != config"

    num_nodes = full_dataset.num_nodes
    location_ids = list(full_dataset.locations)
    if use_mock_data:
        logger.warning("HARNESS: mock mode — synthetic node IDs and a synthetic city")

    # ------------------------------------------------------------------ #
    # HARNESS: holdout configuration                                      #
    #   temporal  — PHASE 2 behavior: seeded val mask, per-batch re-rolls  #
    #               during training (when kriging_active).                 #
    #   inductive — fixed seeded node holdout, blanked in train AND val;   #
    #               excluded from every gradient-producing loss term       #
    #               (zero-shot guarantee); scored at val time only.        #
    # ------------------------------------------------------------------ #
    inductive_holdout_mask = None   # [N] bool — fixed test holdout, never trained on
    loss_node_mask = None           # [N] bool — True = node participates in losses
    holdout_manifest = None
    n_train_nodes = None            # stations the model learns from (inductive mode)

    # v5: the inductive split has its own seed (v4 coupled it to random_seed, so seed
    # replicates silently changed the evaluation nodes). v6: optional validation stations.
    node_masks = build_node_masks(config, location_ids, full_dataset.all_locations, device)
    val_station_mask = node_masks.val_stations
    unseen_mask = node_masks.unseen  # test holdout | validation stations
    if eval_mode == "inductive":
        inductive_seed = data_cfg["inductive_holdout_seed"]
        inductive_frac = data_cfg["inductive_holdout_frac"]
        held_ids = node_masks.held_ids
        inductive_holdout_mask = node_masks.test_holdout
        n_train_nodes = num_nodes - len(held_ids)
        if data_cfg["train_node_count"] is not None:
            assert n_train_nodes == data_cfg["train_node_count"], (
                f"CRITICAL: dataset has {n_train_nodes} training stations, config asks for "
                f"{data_cfg['train_node_count']} (pre-built dataset from another config?)"
            )
        n_train_nodes -= len(node_masks.val_ids)
        loss_node_mask = ~unseen_mask

        # Manifest: resolved location IDs, not just the seed — this is the bridge
        # to the exporters and the cross-model comparability key.
        manifest_path = save_holdout_manifest(
            checkpoint_dir, location_ids, inductive_holdout_mask.cpu(),
            seed=inductive_seed, holdout_frac=inductive_frac,
            val_station_mask=val_station_mask.cpu() if val_station_mask is not None else None,
        )
        holdout_manifest = load_holdout_manifest(checkpoint_dir)
        logger.info(
            f"HARNESS inductive holdout: {int(inductive_holdout_mask.sum())}/{num_nodes} nodes "
            f"held out (seed={inductive_seed}), {len(node_masks.val_ids)} validation stations "
            f"(seed={data_cfg['val_station_seed']}), {n_train_nodes} training stations "
            f"(of {len(full_dataset.all_locations) - len(held_ids)}), manifest={manifest_path}, "
            f"sha={manifest_sha(holdout_manifest)}"
        )

    # Node roles per evaluation pass: the preparer blanks every unseen node; the
    # evaluator scores one set as "ind" and ignores the other (v6 validation stations
    # are invisible at test, the test holdout is invisible at validation).
    val_ind_mask, val_ignore_mask = inductive_holdout_mask, None
    test_ind_mask, test_ignore_mask = inductive_holdout_mask, None
    if val_station_mask is not None:
        val_ind_mask, val_ignore_mask = val_station_mask, inductive_holdout_mask
        test_ignore_mask = val_station_mask

    # Validation-time holdout to score kriging/zero-shot reconstruction against.
    if eval_mode == "inductive":
        val_holdout_mask = unseen_mask
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

    if eval_mode != "inductive":
        # Temporal mode scores the kriging holdout as "ind" (v5 behaviour).
        val_ind_mask = test_ind_mask = val_holdout_mask

    # Pre-compute constant that only requires a single sync (mask is fixed for the whole run).
    val_holdout_node_count = int(val_holdout_mask.sum().item()) if val_holdout_mask is not None else 0

    # v5: one device-side batch builder for every pass (masking, encoding, holdout, covariates).
    preparer = BatchPreparer(
        config, full_dataset, device,
        inductive_mask=unseen_mask,
        loss_node_mask=loss_node_mask,
        eval_holdout_mask=val_holdout_mask,
        static_features=full_dataset.static_features.to(device),
        location_ids=location_ids,
    )
    scale = TrafficScale(
        torch.tensor(full_dataset.traffic_mean, device=device),
        torch.tensor(full_dataset.traffic_std, device=device),
        data_cfg["log1p_inputs"],
    )

    # ------------------------------------------------------------------ #
    # Model — HARNESS: the loops below talk only to the ModelAdapter.      #
    # ------------------------------------------------------------------ #
    model = build_model(
        config, full_dataset, device,
        covariate_dim=preparer.covariates.dim,
        edge_features_cache_path=None if use_mock_data else model_cfg.get("edge_features_cache_path"),
    )
    adapter = build_adapter(model_type, model, config, scale=scale, covariates=preparer.covariates)
    logger.info(
        f"HARNESS: model_type={model_type}, eval_mode={eval_mode}, "
        f"capabilities={adapter.capabilities}"
    )
    logger.info(
        f"Model initialised — num_nodes={num_nodes}, "
        f"static_feat_dim={full_dataset.static_feat_dim}, "
        f"time_series_dim={full_dataset.time_series_dim}, covariate_dim={preparer.covariates.dim}, "
        f"hidden_dim={model_cfg['hidden_dim']}, num_layers={model_cfg['num_layers']}"
    )

    balancer = MTLBalancer(loss_cfg["mtl_balancing"], model, train_cfg["grad_cosine_every"]).to(device)
    optimizer = (
        torch.optim.Adam(list(model.parameters()) + balancer.extra_parameters(), lr=train_cfg["lr"])
        if trainable else None
    )

    epochs = train_cfg["epochs"]
    # v6 protocol: per-epoch cosine schedule (stepped after each epoch) and a minimum
    # number of epochs before early stopping may trigger.
    scheduler = None
    if trainable and train_cfg["lr_schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, functools.partial(
            cosine_lr_factor, epochs=epochs, warmup=train_cfg["warmup_epochs"], floor=train_cfg["lr_min_frac"],
        ))
    min_epochs = train_cfg["min_epochs"]
    grad_clip_max_norm = train_cfg["grad_clip_max_norm"]
    early_stop_patience = train_cfg["early_stop_patience"]
    early_stop_min_delta = train_cfg["early_stop_min_delta"]
    accum_steps = train_cfg["grad_accum_steps"]
    pretrain_epochs = loss_cfg["impute_pretrain_epochs"]
    selection_metric = sel_cfg["metric"]
    eval_seed = eval_cfg["eval_mask_seed"]

    def _gen(offset: int) -> torch.Generator:
        # Fresh, identically-seeded generator per evaluation pass: every epoch (and
        # every run) scores the same artificial gaps and synthetic outages.
        return torch.Generator(device=device).manual_seed(eval_seed + offset)

    def _eval_pass(loader, weights, evaluator, legacy_accum, perturb_batches=0):
        """Main evaluation pass. Returns averaged loss terms (kriging stripped from
        the total in inductive mode — selection never sees holdout-node targets)."""
        sums = {"total": 0.0, "impute": 0.0, "forecast": 0.0, "kriging": 0.0, "forecast_nll_obs": 0.0}
        gen = _gen(0)
        for i, raw in enumerate(loader):
            batch = preparer.prepare(_to_device(raw, device), train=False, gen=gen)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                losses, preds = adapter.step(batch, weights, loss_node_mask=loss_node_mask)
                if i < perturb_batches:
                    evaluator.update_perturbation(adapter, batch, preds)
            total = losses.total.detach()
            if eval_mode == "inductive":
                total = total - weights["w_kriging"] * losses.kriging.detach()
            sums["total"] = sums["total"] + total
            sums["impute"] = sums["impute"] + losses.impute.detach()
            sums["forecast"] = sums["forecast"] + losses.forecast.detach()
            sums["kriging"] = sums["kriging"] + losses.kriging.detach()
            sums["forecast_nll_obs"] = sums["forecast_nll_obs"] + losses.components["forecast_nll_obs"].detach()
            evaluator.update(preds, batch)
            legacy_accum.update(evaluate_batch(adapter.capabilities, preds, batch, holdout_mask=val_holdout_mask))
        n = max(len(loader), 1)
        return {k: float(v) / n for k, v in sums.items()}

    def _so_pass(loader, length: int, evaluator):
        gen = _gen(length)
        for raw in loader:
            batch = preparer.prepare(_to_device(raw, device), train=False, gen=gen, so_length=length)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                preds = adapter.predict(batch)
            evaluator.update_so(length, preds, batch)

    def _selection_score(avg: dict, v5_metrics: dict) -> float:
        if selection_metric == "val_loss_total":
            return avg["total"]
        if selection_metric == "obs_loss":
            return avg["forecast_nll_obs"]
        so_rmse = v5_metrics[f"so_{sel_cfg['so_length']}_forecast_rmse"]
        assert so_rmse is not None, "CRITICAL: obs_loss_plus_so selection but no synthetic-outage rows scored"
        return avg["forecast_nll_obs"] + sel_cfg["so_lambda"] * so_rmse / full_dataset.count_std_raw

    logger.info(
        f"Starting training loop (selection={selection_metric}, grad_accum={accum_steps}, "
        f"mtl={loss_cfg['mtl_balancing']}, impute_pretrain_epochs={pretrain_epochs})..."
    )

    best_score = float("inf")
    best_epoch = None
    best_val_metrics, best_val_v5 = None, None
    # v6: second checkpoint selected on validation-station (unseen) RMSE.
    checkpoint_ind_path = os.path.join(checkpoint_dir, "best_model_ind.pt")
    best_ind_score, best_epoch_ind, best_val_v5_ind = float("inf"), None, None
    epochs_no_improve = 0
    opt_step = 0
    last_score = float("nan")

    import psutil
    memory_threshold_percent = 10.0 # Exit if < 10% RAM available

    for epoch in range(epochs):
        # Soft Memory Limit Check — loud failure so the artifact is not silently truncated.
        mem = psutil.virtual_memory()
        if mem.percent > (100.0 - memory_threshold_percent):
            raise RuntimeError(
                f"CRITICAL: Soft memory limit hit at epoch {epoch + 1}/{epochs} "
                f"(Available: {mem.available / (1024**3):.2f} GB, {mem.percent}% used). "
                "Aborting to prevent OOM crash — reduce batch_size or use a smaller dataset window."
            )

        # R6: imputation-only warm-up epochs (forecast weight 0, no checkpointing).
        pretraining = epoch < pretrain_epochs
        weights = dict(loss_weights, w_forecast=0.0) if pretraining else loss_weights

        epoch_sums = {"total": 0.0, "impute": 0.0, "forecast": 0.0, "kriging": 0.0}
        grad_cosines = []
        n_train = 0
        if trainable:
            model.train()
            optimizer.zero_grad()
            for micro, raw in enumerate(train_loader):
                batch = preparer.prepare(_to_device(raw, device), train=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    losses, _ = adapter.step(batch, weights, loss_node_mask=loss_node_mask)
                total = balancer.total(losses, weights)
                if (micro + 1) % accum_steps == 0:
                    cos = balancer.grad_cosine(losses, opt_step)
                    if cos is not None:
                        grad_cosines.append(cos)
                # backward outside autocast — always in full precision
                balancer.backward(total, losses, weights, lambda t: t / accum_steps)

                if (micro + 1) % accum_steps == 0:
                    # CRITICAL SAFEGUARD: Gradient Clipping (from NSTAR_BACKEND_ARCHITECTURE.md)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    opt_step += 1

                epoch_sums["total"] = epoch_sums["total"] + total.detach()
                epoch_sums["impute"] = epoch_sums["impute"] + losses.impute.detach()
                epoch_sums["forecast"] = epoch_sums["forecast"] + losses.forecast.detach()
                epoch_sums["kriging"] = epoch_sums["kriging"] + losses.kriging.detach()
                n_train += 1
            optimizer.zero_grad()  # drop a trailing partial accumulation window
        avg_train = {k: float(v) / max(n_train, 1) for k, v in epoch_sums.items()}
        assert not math.isnan(avg_train["total"]), "CRITICAL FAILURE: train loss is NaN"
        epoch_lr = optimizer.param_groups[0]["lr"] if trainable else None
        if scheduler is not None:
            scheduler.step()

        # Validation — shared harness path (capability-gated metrics).
        model.eval()
        val_accum = MetricAccumulator()
        val_eval = V5Evaluator(full_dataset, preparer.covariates, adapter.capabilities, config,
                               val_ind_mask, scale, device, ignore_mask=val_ignore_mask)
        with torch.no_grad():
            avg_val = _eval_pass(val_loader, weights, val_eval, val_accum)
            if selection_metric == "obs_loss_plus_so":
                _so_pass(val_loader, sel_cfg["so_length"], val_eval)
        val_metrics = val_accum.summary()
        val_v5 = val_eval.summary()
        avg_val_loss = avg_val["total"]
        last_score = _selection_score(avg_val, val_v5)

        # Offensive Assertion for Divergence
        if math.isnan(avg_val_loss) or math.isnan(last_score):
            raise DivergenceError("Validation loss is NaN! The Burn Test Failed.")

        kriging_log = (
            f", Kriging(train={avg_train['kriging']:.4f}, val={avg_val['kriging']:.4f}, "
            f"RMSE={_fmt(val_v5.get('kriging_raw_rmse'), 2)})"
            if val_holdout_mask is not None else ""
        )
        logger.info(
            f"Epoch {epoch+1}/{epochs}{' [impute pretrain]' if pretraining else ''} - "
            f"Train Loss: {avg_train['total']:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Score({selection_metric}): {last_score:.4f} "
            f"(obsRMSE: {_fmt(val_v5.get('obs_all_rmse'), 2)}, peakRMSE: {_fmt(val_v5.get('peak_rmse'), 2)}, "
            f"collapse: {_fmt(val_v5.get('collapse_rate'), 3)}, "
            f"gapMass: {_fmt(val_v5.get('real_gap_mass_ratio'), 3)}, "
            f"ImputeRMSE: {_fmt(val_v5.get('impute_raw_rmse'), 2)})"
            f"{kriging_log}"
        )

        # Checkpointing and Early Stopping (never during imputation pre-training).
        if not pretraining:
            if last_score < (best_score - early_stop_min_delta):
                best_score = last_score
                best_epoch = epoch + 1
                best_val_metrics, best_val_v5 = val_metrics, val_v5
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"New best model saved to {checkpoint_path} with {selection_metric}: {best_score:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Early stopping counter: {epochs_no_improve} out of {early_stop_patience}")
            if val_station_mask is not None:
                ind_score = val_v5.get("obs_ind_rmse")
                assert ind_score is not None, "CRITICAL: no observed validation-station targets to select on"
                if ind_score < best_ind_score:
                    best_ind_score, best_epoch_ind, best_val_v5_ind = ind_score, epoch + 1, val_v5
                    torch.save(model.state_dict(), checkpoint_ind_path)
                logger.info(f"Validation stations: indRMSE {ind_score:.2f} "
                            f"(best {best_ind_score:.2f} at epoch {best_epoch_ind}, lr {_fmt(epoch_lr, 6)})")

        # W&B Logging — capability-gated: None metrics are absent, never 0.0.
        if init_wandb or (wandb.run is not None):
            log_payload = {
                "train_loss_total": avg_train["total"],
                "train_loss_impute": avg_train["impute"],
                "train_loss_forecast": avg_train["forecast"],
                "val_loss_total": avg_val_loss,
                "val_loss_forecast_nll_obs": avg_val["forecast_nll_obs"],
                "val_selection_score": last_score,
                "epoch": epoch + 1,
            }
            if epoch_lr is not None:
                log_payload["lr"] = epoch_lr
            log_payload.update({f"val_{k}": v for k, v in val_metrics.items() if v is not None})
            log_payload.update({f"val_{k}": v for k, v in val_v5.items() if v is not None})
            if grad_cosines:
                log_payload["train_grad_cosine_forecast_impute"] = float(np.mean(grad_cosines))
            if balancer.log_vars is not None:
                log_payload.update({f"mtl_log_var_{k}": float(v) for k, v in balancer.log_vars.items()})
            # PHASE 2 / inductive: kriging telemetry only when a holdout is scored.
            if val_holdout_mask is not None:
                log_payload.update({
                    "train_loss_kriging": avg_train["kriging"],
                    "val_loss_kriging": avg_val["kriging"],
                    "val_kriging_nodes": val_holdout_node_count,
                })
            wandb.log(log_payload)

        # HARNESS: standardized cross-model results row, overwritten each epoch.
        # Joinable across runs on holdout_manifest_sha — identical hash means
        # identical evaluation conditions.
        save_results_manifest(checkpoint_dir, {
            "pipeline_version": config["pipeline_version"],
            "model_type": model_type,
            "eval_mode": eval_mode,
            "epoch": epoch + 1,
            "seed": train_cfg["random_seed"],
            "holdout_manifest_sha": manifest_sha(holdout_manifest),
            "val_loss_total": avg_val_loss,
            "selection_metric": selection_metric,
            "selection_score": last_score,
            "metrics": val_metrics,
            "v5_metrics": val_v5,
            "mask_rng": "cuda",  # augmentation RNG stream moved to GPU (§14 medium-risk)
        })

        if not trainable:
            logger.info("Non-learned baseline: a single evaluation epoch is the whole run.")
            break
        if epochs_no_improve >= early_stop_patience:
            if epoch + 1 >= min_epochs:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break
            logger.info(f"Early stopping deferred: min_epochs={min_epochs} not reached")

    logger.info("Training complete.")

    # ------------------------------------------------------------------ #
    # Final Evaluation on Held-Out Test Set (F-C01 / Forensic Audit)     #
    # ------------------------------------------------------------------ #
    test_metrics, test_v5, test_v5_ind = None, None, None
    assert os.path.exists(checkpoint_path), \
        "CRITICAL: no checkpoint was saved (every epoch was imputation pre-training?)"

    def _test_eval():
        model.eval()
        test_accum = MetricAccumulator()
        test_eval = V5Evaluator(
            full_dataset, preparer.covariates, adapter.capabilities, config, test_ind_mask, scale, device,
            diagnostics=eval_cfg["diagnostics"], store_series=eval_cfg["diagnostics"], ignore_mask=test_ignore_mask,
        )
        with torch.no_grad():
            avg = _eval_pass(
                test_loader, loss_weights, test_eval, test_accum,
                perturb_batches=PERTURB_BATCHES if eval_cfg["diagnostics"] else 0,
            )
            for length in eval_cfg["so_lengths"]:
                _so_pass(test_loader, length, test_eval)
        legacy = test_accum.summary()
        legacy["test_loss"] = avg["total"]
        legacy["test_loss_kriging"] = avg["kriging"]
        legacy["test_loss_forecast_nll_obs"] = avg["forecast_nll_obs"]
        return legacy, test_eval.summary(), avg

    has_test = test_loader is not None and len(test_loader) > 0
    if has_test and val_station_mask is not None:
        assert os.path.exists(checkpoint_ind_path), "CRITICAL: no validation-station checkpoint was saved"
        logger.info(f"Evaluating the validation-station checkpoint (epoch {best_epoch_ind}) on the test partition...")
        model.load_state_dict(torch.load(checkpoint_ind_path, map_location=device))
        _, test_v5_ind, _ = _test_eval()
        logger.info(
            f"Final Test Metrics [val-station checkpoint] -> obsRMSE: {_fmt(test_v5_ind.get('obs_all_rmse'), 2)} "
            f"(trans {_fmt(test_v5_ind.get('obs_trans_rmse'), 2)}, ind {_fmt(test_v5_ind.get('obs_ind_rmse'), 2)}) | "
            f"SO12 RMSE: {_fmt(test_v5_ind.get('so_12_forecast_rmse'), 2)}"
        )
        if init_wandb or (wandb.run is not None):
            for k, v in test_v5_ind.items():
                if v is not None:
                    wandb.run.summary[f"test_indckpt_{k}"] = v

    logger.info(f"Restoring best checkpoint from {checkpoint_path} (epoch {best_epoch}) for final evaluation.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if has_test:
        logger.info(f"Running final evaluation on held-out test partition ({len(test_dataset)} samples)...")
        test_metrics, test_v5, avg_test = _test_eval()

        logger.info(
            f"Final Test Metrics -> Loss: {avg_test['total']:.4f} | "
            f"obsRMSE: {_fmt(test_v5.get('obs_all_rmse'), 2)} (trans {_fmt(test_v5.get('obs_trans_rmse'), 2)}, "
            f"ind {_fmt(test_v5.get('obs_ind_rmse'), 2)}) | peakRMSE: {_fmt(test_v5.get('peak_rmse'), 2)} | "
            f"collapse: {_fmt(test_v5.get('collapse_rate'), 3)} | "
            f"SO12 RMSE: {_fmt(test_v5.get('so_12_forecast_rmse'), 2)} | CRPS: {_fmt(test_v5.get('crps'), 3)}"
        )
        if init_wandb or (wandb.run is not None):
            test_payload = {f"final_test_{k}": v for k, v in test_metrics.items() if v is not None}
            test_payload.update({f"final_test_{k}": v for k, v in test_v5.items() if v is not None})
            wandb.log(test_payload)
            for k, v in list(test_metrics.items()) + list(test_v5.items()):
                if v is not None:
                    wandb.run.summary[f"test_{k}"] = v

    save_results_manifest(checkpoint_dir, {
        "pipeline_version": config["pipeline_version"],
        "model_type": model_type,
        "eval_mode": eval_mode,
        "best_val_loss": best_score,
        "best_epoch": best_epoch,
        "seed": train_cfg["random_seed"],
        "holdout_manifest_sha": manifest_sha(holdout_manifest),
        "num_nodes": num_nodes,
        "train_node_count": n_train_nodes,
        "selection_metric": selection_metric,
        "selection_score": best_score,
        "metrics": best_val_metrics,
        "v5_metrics": best_val_v5,
        "test_metrics": test_metrics,
        "v5_test_metrics": test_v5,
        # v6 validation stations: ids, the unseen-station checkpoint and its scores.
        "val_station_ids": node_masks.val_ids,
        "best_epoch_ind": best_epoch_ind,
        "best_val_ind_rmse": best_ind_score if best_epoch_ind is not None else None,
        "v5_metrics_ind_ckpt": best_val_v5_ind,
        "v5_test_metrics_ind_ckpt": test_v5_ind,
        "mask_rng": "cuda",
    })

    # ------------------------------------------------------------------ #
    # W&B Artifact Upload (Phase 1)                                      #
    # ------------------------------------------------------------------ #
    logger.info(f"Uploading model artifact to W&B from {checkpoint_path}")

    assert wandb.run is not None, "FATAL: Attempted to upload a model artifact without an active W&B run. Manually uploaded artifacts are strictly forbidden."
    assert os.path.exists(checkpoint_path), f"FATAL: Checkpoint path does not exist: {checkpoint_path}"

    artifact = wandb.Artifact(
        name=wandb_collection,
        type="model",
        description=f"Best model from run {wandb.run.name}",
        metadata=config
    )
    artifact.add_file(checkpoint_path, name="best_model.pt")
    checkpoint_ind_path = os.path.join(checkpoint_dir, "best_model_ind.pt")
    if os.path.exists(checkpoint_ind_path):
        artifact.add_file(checkpoint_ind_path, name="best_model_ind.pt")

    if os.path.exists(config_save_path):
        artifact.add_file(config_save_path, name="config.yaml")

    # Upload holdout manifest if it exists (for inductive runs)
    manifest_path = os.path.join(checkpoint_dir, "holdout_manifest.json")
    if os.path.exists(manifest_path):
        artifact.add_file(manifest_path, name="holdout_manifest.json")

    wandb.log_artifact(artifact)
    logger.info(f"Model artifact successfully logged to W&B run: {wandb.run.name}")

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

    return best_score


def main():
    logger = setup_logger("train_pipeline")
    args = parse_args()
    config = load_config(args.config)
    if args.run_name:
        config.setdefault("wandb", {})["run_name"] = args.run_name

    # CUDA_VISIBLE_DEVICES is already set by the early setup block
    if not args.use_mock_data:
        logger.info(f"Isolated training execution to GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")

    run_training(config, init_wandb=True, use_mock_data=args.use_mock_data)


if __name__ == "__main__":
    main()
