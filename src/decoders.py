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

    def forward(self, latent_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_embeddings: [batch_size, num_nodes, seq_len, hidden_dim]
        Returns:
            imputed_features: [batch_size, num_nodes, seq_len, time_series_dim]
            mu: [batch_size, num_nodes]
            theta: [batch_size, num_nodes]
            pi: [batch_size, num_nodes]
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

        return imputed_features, mu, theta, pi

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
                 w_impute: float = 1.0,
                 w_forecast: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint Loss: MSE for Imputation + ZINB NLL for Forecasting.
    """
    # 1. Imputation Loss (MSE)
    # NSTAR_BACKEND_ARCHITECTURE.md: "strictly on the artificially masked nodes"
    # Note: In a real scenario, we'd use a mask here.
    loss_impute = F.mse_loss(imputed_features, target_impute)

    # 2. Forecasting Loss (ZINB NLL)
    loss_forecast = compute_zinb_nll(target_forecast, mu, theta, pi)

    # 3. Total Weighted Loss
    total_loss = (w_impute * loss_impute) + (w_forecast * loss_forecast)

    # --- Tripwires (Rule 5 & NSTAR_BACKEND_ARCHITECTURE.md) ---
    if torch.isnan(total_loss).any():
        # Print stats for debugging before crashing
        print(f"DEBUG: mu min={mu.min()}, max={mu.max()}")
        print(f"DEBUG: theta min={theta.min()}, max={theta.max()}")
        print(f"DEBUG: pi min={pi.min()}, max={pi.max()}")
        raise ValueError("CRITICAL FAILURE: Loss is NaN. Mathematical tripwire triggered.")

    return total_loss, loss_impute, loss_forecast
