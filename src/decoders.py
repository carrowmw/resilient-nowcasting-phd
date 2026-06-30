import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DecoupledDecoders(nn.Module):
    """
    DecoupledDecoders: Multi-task heads for imputation and uncertainty-aware forecasting.
    """
    def __init__(self, hidden_dim: int, time_series_dim: int, dropout: float = 0.1, zinb_eps: float = 1e-6):
        super(DecoupledDecoders, self).__init__()

        # Reproducibility
        torch.manual_seed(42)

        self.hidden_dim = hidden_dim
        self.time_series_dim = time_series_dim
        self.zinb_eps = zinb_eps

        # Imputation Decoder: Reconstructs the entire sequence.
        # Deepened to a 2-layer MLP to give the model enough capacity to recover
        # masked sensor readings from the latent graph representation.
        self.impute_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, time_series_dim)
        )

        # Forecasting Decoder: Predicts ZINB parameters (mu, theta, pi).
        # Deepened to match imputation head capacity and reduce gradient starvation.
        self.forecast_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 3)
        )

    def forward(self, latent_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Args:
            latent_embeddings: [batch_size, num_nodes, seq_len, hidden_dim]
        Returns:
            imputed_features: [batch_size, num_nodes, seq_len, time_series_dim]
            mu: [batch_size, num_nodes]
            theta: [batch_size, num_nodes]
            pi: [batch_size, num_nodes]
            tripwire_logs: list of dicts {'node_id', 'event_type'}
        """
        assert latent_embeddings.dim() == 4, f"Expected 4D embeddings, got {latent_embeddings.shape}"
        batch_size, num_nodes, seq_len, hidden_dim = latent_embeddings.shape

        # 1. Imputation (across all time steps)
        imputed_features = self.impute_layer(latent_embeddings)

        # 2. Forecasting (using the final time step's embedding)
        last_step_embedding = latent_embeddings[:, :, -1, :] # [batch_size, num_nodes, hidden_dim]
        forecast_logits = self.forecast_layer(last_step_embedding) # [batch_size, num_nodes, 3]

        # Split raw logits
        mu_logits = forecast_logits[:, :, 0]
        theta_logits = forecast_logits[:, :, 1]
        pi_logits = forecast_logits[:, :, 2]

        # Apply ZINB constraints (Tripwires from NSTAR_BACKEND_ARCHITECTURE.md)
        pi = torch.sigmoid(pi_logits)
        mu = F.softplus(mu_logits) + self.zinb_eps
        theta = F.softplus(theta_logits) + self.zinb_eps

        # Detect Offesive Programming safeguard activations
        tripwire_logs = []
        pi_clamp = (pi < self.zinb_eps * 10) | (pi > 1.0 - self.zinb_eps * 10)
        mu_clamp = (F.softplus(mu_logits) < self.zinb_eps * 10)
        theta_clamp = (F.softplus(theta_logits) < self.zinb_eps * 10)

        for b in range(batch_size):
            for n in range(num_nodes):
                if pi_clamp[b, n]:
                    tripwire_logs.append({'node_id': n, 'event_type': 'PI_CLAMP'})
                if mu_clamp[b, n]:
                    tripwire_logs.append({'node_id': n, 'event_type': 'MU_CLAMP'})
                if theta_clamp[b, n]:
                    tripwire_logs.append({'node_id': n, 'event_type': 'THETA_CLAMP'})

        return imputed_features, mu, theta, pi, tripwire_logs

class SpatialKrigingDecoder(nn.Module):
    """
    Spatial kriging head (Spatial Estimation Architecture, Component 2).

    Reconstructs the full time series for spatially held-out nodes from their latent
    embeddings — embeddings that are shaped purely by message passing from observable
    neighbours, since the held-out nodes' own dynamic features are blanked during
    training (IGNNK-style node masking, docs/spatial_estimation_architecture.md §4).

    Input:  latent_embeddings [B, N, T, H], holdout_mask [N] (bool)
    Output: reconstructed series for the held-out nodes [B, K, T, D]
            where K = holdout_mask.sum() and D = time_series_dim.

    NOTE: Phase 2 infrastructure only. This head is instantiated but kept dormant
    (the loss weight w_kriging defaults to 0.0). Wiring it into the pipeline,
    dataset holdout masking, and compute_loss happens on exp/spatial-kriging-head.
    """
    def __init__(self, hidden_dim: int, time_series_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_series_dim = time_series_dim
        # Mirrors the imputation head's capacity (doc §4.3).
        self.kriging_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, time_series_dim),
        )

    def forward(self, latent_embeddings: torch.Tensor, holdout_mask: torch.Tensor) -> torch.Tensor:
        assert latent_embeddings.dim() == 4, \
            f"Expected 4D embeddings [B, N, T, H], got {latent_embeddings.shape}"
        b, n, t, h = latent_embeddings.shape
        assert holdout_mask.dim() == 1 and holdout_mask.shape[0] == n, \
            f"holdout_mask must be [num_nodes={n}], got {tuple(holdout_mask.shape)}"
        assert holdout_mask.dtype == torch.bool, \
            f"holdout_mask must be boolean, got {holdout_mask.dtype}"
        assert h == self.hidden_dim, \
            f"hidden_dim mismatch: embeddings {h} != decoder {self.hidden_dim}"

        held_out = latent_embeddings[:, holdout_mask, :, :]   # [B, K, T, H]
        reconstructed = self.kriging_head(held_out)           # [B, K, T, D]
        return reconstructed


def scale_invariant_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean Squared Logarithmic Error (MSLE) — a scale-invariant reconstruction loss
    (Spatial Estimation Architecture, Component 3, docs §5).

        L = mean( (log1p(pred) - log1p(target))^2 )

    Penalises *relative* error rather than absolute magnitude, so quiet low-count
    nodes contribute on par with high-variance city-centre nodes rather than being
    drowned out. Intended for the raw (non-negative) count domain; inputs are clamped
    to >= 0 before log1p.

    Numerical safety: there is no division anywhere, so MSLE cannot raise a
    division-by-zero, and log1p is finite for every non-negative input. NaNs in the
    input propagate to the output by design (caught downstream by the compute_loss
    NaN tripwire) rather than being silently masked.

    Args:
        pred:   predicted values, any shape broadcastable-equal to target.
        target: ground-truth values, same shape as pred.
        eps:    small additive constant for numerical headroom inside log1p.
    Returns:
        Scalar MSLE loss.
    """
    assert pred.shape == target.shape, \
        f"CRITICAL: scale_invariant_mse shape mismatch: {pred.shape} vs {target.shape}"
    pred_safe = torch.clamp(pred, min=0.0)
    target_safe = torch.clamp(target, min=0.0)
    log_pred = torch.log1p(pred_safe + eps)
    log_target = torch.log1p(target_safe + eps)
    return F.mse_loss(log_pred, log_target)


def compute_zinb_nll(y_true: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """
    Negative Log-Likelihood for Zero-Inflated Negative Binomial Distribution.

    Formula:
    If y=0: P(y) = pi + (1-pi) * (theta / (theta + mu))^theta
    If y>0: P(y) = (1-pi) * [Gamma(y+theta)/(y!Gamma(theta))] * (theta/(theta+mu))^theta * (mu/(theta+mu))^y
    """
    eps = 1e-6
    y_true = y_true.float()

    # Clamp log operations to eps
    def safe_log(x):
        return torch.log(x.clamp(min=eps))

    # NB log-likelihood components
    # log(theta / (theta + mu))
    log_theta_mu_ratio = safe_log(theta) - safe_log(theta + mu)
    # log(mu / (theta + mu))
    log_mu_mu_ratio = safe_log(mu) - safe_log(theta + mu)

    # Case y > 0
    # log P(y|NB) = logGamma(y+theta) - logGamma(y+1) - logGamma(theta) + theta*log(theta/(theta+mu)) + y*log(mu/(theta+mu))
    ll_nb = (torch.lgamma(y_true + theta)
             - torch.lgamma(y_true + 1)
             - torch.lgamma(theta)
             + theta * log_theta_mu_ratio
             + y_true * log_mu_mu_ratio)

    # Case y = 0
    # P(y=0|ZINB) = pi + (1-pi) * exp(theta * log(theta/(theta+mu)))
    # For numerical stability, we use log-sum-exp or handle it carefully
    # log_p_zero_nb = theta * log_theta_mu_ratio
    ll_zero = safe_log(pi + (1 - pi) * torch.exp(theta * log_theta_mu_ratio))

    # Case y > 0 (ZINB)
    # log P(y|ZINB) = log(1-pi) + ll_nb
    ll_nonzero = safe_log(1 - pi) + ll_nb

    # Combine cases
    result = torch.where(y_true < eps, ll_zero, ll_nonzero)

    return -torch.mean(result)

def compute_loss(imputed_features: torch.Tensor,
                 target_impute: torch.Tensor,
                 mu: torch.Tensor,
                 theta: torch.Tensor,
                 pi: torch.Tensor,
                 target_forecast: torch.Tensor,
                 traffic_mean: torch.Tensor,      # PHASE 1: scaler buffer
                 traffic_std: torch.Tensor,       # PHASE 1: scaler buffer
                 w_impute: float = 1.0,
                 w_forecast: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint Loss with Channel-Specific Treatment (PHASE 1):
    - Traffic (channel 0): MSLE on raw counts (inverse transform)
    - Weather (channels 1-2): MSE on scaled features
    - Forecasting: ZINB NLL on raw counts
    """

    # --- 1. Imputation Loss: Split Traffic vs Weather ---

    # Extract traffic channel (dim 0)
    traffic_pred_scaled = imputed_features[:, :, :, 0]      # [B, N, T]
    traffic_target_scaled = target_impute[:, :, :, 0]       # [B, N, T]

    # Inverse transform to raw counts
    traffic_pred_raw = (traffic_pred_scaled * traffic_std) + traffic_mean
    traffic_target_raw = (traffic_target_scaled * traffic_std) + traffic_mean

    # Clamp to non-negative (count domain)
    traffic_pred_raw = torch.clamp(traffic_pred_raw, min=0.0)
    traffic_target_raw = torch.clamp(traffic_target_raw, min=0.0)

    # Compute TRUE MSLE on raw counts
    loss_impute_traffic = scale_invariant_mse(traffic_pred_raw, traffic_target_raw)

    # Weather channels: MSE on scaled features (no inverse transform)
    weather_pred = imputed_features[:, :, :, 1:]      # [B, N, T, 2]
    weather_target = target_impute[:, :, :, 1:]       # [B, N, T, 2]
    loss_impute_weather = F.mse_loss(weather_pred, weather_target)

    # Combined imputation loss
    loss_impute = loss_impute_traffic + loss_impute_weather

    # --- 2. Forecasting Loss (ZINB NLL) - UNCHANGED ---
    loss_forecast = compute_zinb_nll(target_forecast, mu, theta, pi)

    # --- 3. Total Weighted Loss ---
    total_loss = (w_impute * loss_impute) + (w_forecast * loss_forecast)

    # --- 4. OFFENSIVE ASSERTIONS (Tripwires) ---
    assert not torch.isnan(loss_impute_traffic), "CRITICAL: Traffic MSLE is NaN"
    assert not torch.isnan(loss_impute_weather), "CRITICAL: Weather MSE is NaN"
    assert not torch.isnan(loss_forecast), "CRITICAL: Forecast loss is NaN"
    assert not torch.isnan(total_loss), "CRITICAL: Total loss is NaN"
    assert not torch.isinf(total_loss), "CRITICAL: Total loss is Inf"

    return total_loss, loss_impute, loss_forecast
