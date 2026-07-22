import torch
import numpy as np
from typing import Dict, Union

def calculate_metrics(
    target_forecast: torch.Tensor,
    mu: torch.Tensor,
    pi: torch.Tensor,
    imputed_features: torch.Tensor,
    target_impute: torch.Tensor,
    node_mask: torch.Tensor
) -> Dict[str, Union[float, int]]:
    """
    Calculates human-readable metrics for the ResilientIoTPipeline.

    Args:
        target_forecast: [batch_size, num_nodes] Ground truth future counts.
        mu: [batch_size, num_nodes] Predicted ZINB mean.
        pi: [batch_size, num_nodes] Predicted zero-inflation probability.
        imputed_features: [batch_size, num_nodes, seq_len, time_series_dim] Reconstructed features.
        target_impute: [batch_size, num_nodes, seq_len, time_series_dim] Ground truth historical features.
        node_mask: [batch_size, num_nodes, seq_len, time_series_dim] Mask where NaNs were artificially applied.

    Returns:
        Dict with keys: rmse (float), mae (float), impute_mse (float),
        zero_true_count (int), zero_correct_count (int).
    """
    assert target_forecast.dim() == 2, \
        f"CRITICAL: target_forecast must be [B, N], got {tuple(target_forecast.shape)}"
    assert mu.dim() == 2, \
        f"CRITICAL: mu must be [B, N], got {tuple(mu.shape)}"
    assert pi.dim() == 2, \
        f"CRITICAL: pi must be [B, N], got {tuple(pi.shape)}"
    assert imputed_features.dim() == 4, \
        f"CRITICAL: imputed_features must be [B, N, T, D], got {tuple(imputed_features.shape)}"
    assert target_impute.dim() == 4, \
        f"CRITICAL: target_impute must be [B, N, T, D], got {tuple(target_impute.shape)}"
    assert node_mask.dim() == 4, \
        f"CRITICAL: node_mask must be [B, N, T, D], got {tuple(node_mask.shape)}"
    assert target_forecast.shape == mu.shape == pi.shape, \
        f"CRITICAL: forecast shape mismatch: target={tuple(target_forecast.shape)}, mu={tuple(mu.shape)}, pi={tuple(pi.shape)}"
    assert imputed_features.shape == target_impute.shape == node_mask.shape, \
        f"CRITICAL: imputation shape mismatch: imputed={tuple(imputed_features.shape)}, target={tuple(target_impute.shape)}, mask={tuple(node_mask.shape)}"

    # 1. RMSE and MAE (Forecasting)
    # mu is predicted mean count
    mse_forecast = torch.mean((target_forecast - mu)**2)
    rmse = torch.sqrt(mse_forecast).item()
    mae = torch.mean(torch.abs(target_forecast - mu)).item()

    # 2. Zero-Accuracy counts (aggregated by caller to avoid batch-averaging bias)
    # zero_true_count: number of positions where target_forecast == 0
    # zero_correct_count: among those, how many have pi > 0.5
    zero_indices = (target_forecast == 0)
    zero_true_count: int = int(zero_indices.sum().item())
    if zero_true_count > 0:
        zero_correct_count: int = int((pi[zero_indices] > 0.5).sum().item())
    else:
        zero_correct_count = 0

    # 3. Imputation MSE
    # Calculate MSE strictly where node_mask is True (the artificial NaNs)
    if node_mask.any():
        # Ensure node_mask matches the shape of imputed_features/target_impute
        # Based on src/train.py mock data, node_mask has the same shape as dynamic_features [B, N, T, D]
        diff_sq = (imputed_features - target_impute)**2
        impute_mse = torch.mean(diff_sq[node_mask]).item()
    else:
        impute_mse = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "zero_true_count": zero_true_count,
        "zero_correct_count": zero_correct_count,
        "impute_mse": impute_mse
    }


def compute_msle(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Mean Squared Logarithmic Error as a plain float metric (scale-invariant).

    Mirrors decoders.scale_invariant_mse but returns a detached Python float for
    logging. Inputs are clamped to >= 0; no division is performed.
    """
    pred_safe = torch.clamp(pred, min=0.0)
    target_safe = torch.clamp(target, min=0.0)
    log_pred = torch.log1p(pred_safe + eps)
    log_target = torch.log1p(target_safe + eps)
    return torch.mean((log_pred - log_target) ** 2).item()


def calculate_kriging_metrics(
    kriging_pred: torch.Tensor,
    kriging_target: torch.Tensor,
    traffic_channel: int = 0,
) -> Dict[str, float]:
    """
    Spatial-kriging validation metrics over the held-out nodes (Component 2).

    Args:
        kriging_pred:    [B, K, T, D] reconstructed series for held-out nodes.
        kriging_target:  [B, K, T, D] ground-truth series for the same nodes.
        traffic_channel: channel index treated as count data; MSLE applies here
                         (scale-invariant), matching the loss design in docs §5.5.

    Returns:
        Dict with keys: kriging_msle (traffic channel, scale-invariant),
        kriging_rmse (all channels), kriging_mae (all channels).

    NOTE: Phase 2 infrastructure only — not yet called from the training loop.
    """
    assert kriging_pred.shape == kriging_target.shape, \
        f"CRITICAL: kriging metric shape mismatch: {kriging_pred.shape} vs {kriging_target.shape}"

    # No held-out nodes in this batch -> nothing to score.
    if kriging_pred.numel() == 0:
        return {"kriging_msle": 0.0, "kriging_rmse": 0.0, "kriging_mae": 0.0}

    traffic_pred = kriging_pred[..., traffic_channel]
    traffic_target = kriging_target[..., traffic_channel]
    kriging_msle = compute_msle(traffic_pred, traffic_target)

    diff = kriging_pred - kriging_target
    kriging_rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    kriging_mae = torch.mean(torch.abs(diff)).item()

    return {
        "kriging_msle": kriging_msle,
        "kriging_rmse": kriging_rmse,
        "kriging_mae": kriging_mae,
    }
