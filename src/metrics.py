import torch
import numpy as np
from typing import Dict

def calculate_metrics(
    target_forecast: torch.Tensor, 
    mu: torch.Tensor, 
    pi: torch.Tensor, 
    imputed_features: torch.Tensor, 
    target_impute: torch.Tensor, 
    node_mask: torch.Tensor
) -> Dict[str, float]:
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
        Dict[str, float]: Dictionary of scalar metrics.
    """
    # 1. RMSE and MAE (Forecasting)
    # mu is predicted mean count
    mse_forecast = torch.mean((target_forecast - mu)**2)
    rmse = torch.sqrt(mse_forecast).item()
    mae = torch.mean(torch.abs(target_forecast - mu)).item()
    
    # 2. Zero-Accuracy
    # If target_forecast == 0, calculate what percentage of the time pi > 0.5
    zero_indices = (target_forecast == 0)
    if zero_indices.any():
        pi_zero_pred = pi[zero_indices] > 0.5
        zero_acc = torch.mean(pi_zero_pred.float()).item()
    else:
        zero_acc = 1.0 # Or NaN? Let's use 1.0 if no zeros to avoid NaN logging
        
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
        "zero_acc": zero_acc,
        "impute_mse": impute_mse
    }
