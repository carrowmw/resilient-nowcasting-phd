"""
Model construction from a resolved run config (v5).

Single place where config keys become constructor arguments, so train.py, the
desk checks and the tests build identical modules. Every key is read with
cfg["key"] (the config has been through run_config.resolve_config).
"""
import logging

import torch

from baselines import GRUForecaster, LSTMForecaster, NaiveBaseline
from pipeline import ResilientIoTPipeline

logger = logging.getLogger(__name__)


def baseline_input_dim(config: dict, dataset, covariate_dim: int) -> int:
    if config["model"]["baseline_inputs"] == "dynamic":
        return dataset.time_series_dim
    # dynamic + 2 mask channels + covariates + broadcast static features
    return dataset.time_series_dim + 2 + covariate_dim + dataset.static_feat_dim


def build_model(
    config: dict,
    dataset,
    device: torch.device,
    covariate_dim: int,
    edge_features_cache_path,
) -> torch.nn.Module:
    """dataset: the full RealIoTDataset (scaler stats, v5 tables).
    edge_features_cache_path: None for synthetic/mock data (no physical cache)."""
    model_type = config["model_type"]
    data_cfg, model_cfg, loss_cfg = config["data"], config["model"], config["loss"]

    if model_type == "gnn":
        model = ResilientIoTPipeline(
            num_nodes=dataset.num_nodes,
            static_feat_dim=dataset.static_feat_dim,
            time_series_dim=dataset.time_series_dim,
            seq_len=data_cfg["seq_len"],
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            sparsity_threshold=model_cfg["sparsity_threshold"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            dropout=model_cfg["dropout"],
            dim_feedforward=model_cfg["dim_feedforward"],
            zinb_eps=model_cfg["zinb_eps"],
            edge_features_cache_path=edge_features_cache_path,
            use_positional_encoding=model_cfg["use_positional_encoding"],
            covariate_dim=covariate_dim,
            sensor_embedding_dim=model_cfg["sensor_embedding_dim"],
            sensor_embedding_dropout=model_cfg["sensor_embedding_dropout"],
            temporal_key_padding=model_cfg["temporal_key_padding"],
            gat_drop_missing_sources=model_cfg["gat_drop_missing_sources"],
            graph_mode=model_cfg["graph_mode"],
            forecast_likelihood=loss_cfg["forecast_likelihood"],
            impute_likelihood=loss_cfg["impute_likelihood"],
            quantiles=model_cfg["quantiles"],
            outage_head_input=model_cfg["outage_head_input"],
            mask_channels=data_cfg["mask_channels"],
            holdout_night_channel=data_cfg["holdout_night_channel"],
            detach_impute_latent=loss_cfg["detach_impute_latent"],
            traffic_log1p=data_cfg["log1p_inputs"],
        ).to(device)
        model.set_scaler_stats(dataset.traffic_mean, dataset.traffic_std)
        if model.node_missing_rate is not None:
            model.node_missing_rate.copy_(dataset.node_missing_rate.to(device))
    elif model_type in ("lstm", "gru"):
        cls = LSTMForecaster if model_type == "lstm" else GRUForecaster
        model = cls(
            time_series_dim=dataset.time_series_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            input_dim=baseline_input_dim(config, dataset, covariate_dim),
            head=model_cfg["baseline_head"],
        ).to(device)
    elif model_type == "naive":
        model = NaiveBaseline(model_cfg["naive_kind"]).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r} (expected gnn | lstm | gru | naive)")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model_factory: built {model_type} ({n_params} parameters)")
    return model
