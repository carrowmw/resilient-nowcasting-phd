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

        # Split raw logits and cast to float32 so parameter activations (sigmoid, softplus)
        # and clamp boundaries (1.0 - zinb_eps) are evaluated at full precision,
        # preventing bfloat16 precision limits (mantissa eps ~ 0.0078) from rounding 1.0 - eps to 1.0.
        mu_logits = forecast_logits[:, :, 0].float()
        theta_logits = forecast_logits[:, :, 1].float()
        pi_logits = forecast_logits[:, :, 2].float()

        # Apply ZINB constraints (Tripwires from NSTAR_BACKEND_ARCHITECTURE.md)
        pi = torch.clamp(torch.sigmoid(pi_logits), min=self.zinb_eps, max=1.0 - self.zinb_eps)
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


def compute_kriging_loss(kriging_pred: torch.Tensor,
                         target_impute: torch.Tensor,
                         holdout_mask: torch.Tensor,
                         traffic_mean: torch.Tensor,
                         traffic_std: torch.Tensor) -> torch.Tensor:
    """
    Spatial-kriging reconstruction loss on the held-out nodes (Component 2, docs §5.5).

    Mirrors the imputation head's channel-specific treatment so the two losses are
    directly comparable:
      - Traffic (channel 0): TRUE MSLE on raw counts (inverse-transformed from scaled).
      - Weather (channels 1-2): MSE on scaled features.

    Args:
        kriging_pred:  [B, K, T, D] reconstructed series for held-out nodes
                       (K = holdout_mask.sum()).
        target_impute: [B, N, T, D] ground-truth series over the full spatial grid.
        holdout_mask:  [N] boolean, True = held out.
        traffic_mean:  scalar scaler buffer (raw-count mean).
        traffic_std:   scalar scaler buffer (raw-count std).

    Returns:
        Scalar kriging loss (traffic MSLE + weather MSE). Returns 0.0 when no nodes
        are held out (avoids NaN from F.mse_loss over an empty tensor).
    """
    target_holdout = target_impute[:, holdout_mask, :, :]  # [B, K, T, D]

    assert kriging_pred.shape == target_holdout.shape, \
        f"CRITICAL: kriging shape mismatch: pred {tuple(kriging_pred.shape)} " \
        f"vs target {tuple(target_holdout.shape)}"

    # Empty-holdout guard: a random training mask can select zero nodes on small N.
    # F.mse_loss over an empty tensor returns NaN, so short-circuit to a clean zero
    # that still participates in the graph on the correct device.
    if kriging_pred.numel() == 0:
        return torch.zeros((), device=kriging_pred.device)

    # Traffic channel (ch 0): inverse transform to raw counts, then TRUE MSLE.
    traffic_pred_scaled = kriging_pred[:, :, :, 0]
    traffic_target_scaled = target_holdout[:, :, :, 0]

    traffic_pred_raw = (traffic_pred_scaled * traffic_std) + traffic_mean
    traffic_target_raw = (traffic_target_scaled * traffic_std) + traffic_mean

    traffic_pred_raw = torch.clamp(traffic_pred_raw, min=0.0)
    traffic_target_raw = torch.clamp(traffic_target_raw, min=0.0)

    loss_traffic = scale_invariant_mse(traffic_pred_raw, traffic_target_raw)

    # Weather channels (ch 1-2): MSE on scaled features (no inverse transform).
    weather_pred = kriging_pred[:, :, :, 1:]
    weather_target = target_holdout[:, :, :, 1:]
    loss_weather = F.mse_loss(weather_pred, weather_target)

    loss_kriging = loss_traffic + loss_weather

    # OFFENSIVE ASSERTION
    assert not torch.isnan(loss_kriging), "CRITICAL: Kriging loss is NaN"

    return loss_kriging


def compute_zinb_nll(y_true: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """
    Negative Log-Likelihood for Zero-Inflated Negative Binomial Distribution.

    Formula:
    If y=0: P(y) = pi + (1-pi) * (theta / (theta + mu))^theta
    If y>0: P(y) = (1-pi) * [Gamma(y+theta)/(y!Gamma(theta))] * (theta/(theta+mu))^theta * (mu/(theta+mu))^y
    """
    eps = 1e-6
    mu = mu.float()
    theta = theta.float()
    pi = pi.float().clamp(min=eps, max=1.0 - eps)
    y_true = y_true.float()

    assert (mu > 0).all(), "CRITICAL: mu must be strictly positive before safe_log"
    assert (theta > 0).all(), "CRITICAL: theta must be strictly positive before safe_log"
    assert (pi > 0).all() and (pi < 1).all(), "CRITICAL: pi must be in (0, 1)"

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
                 kriging_pred: Optional[torch.Tensor] = None,          # PHASE 2
                 spatial_holdout_mask: Optional[torch.Tensor] = None,  # PHASE 2
                 w_impute: float = 1.0,
                 w_forecast: float = 1.0,
                 w_kriging: float = 0.0,
                 loss_node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint Loss with Channel-Specific Treatment (PHASE 1 + PHASE 2):
    - Traffic (channel 0): MSLE on raw counts (inverse transform)
    - Weather (channels 1-2): MSE on scaled features
    - Forecasting: ZINB NLL on raw counts
    - Kriging (PHASE 2): MSLE/MSE on spatially held-out nodes (active when w_kriging > 0)

    loss_node_mask ([N] bool, True = node participates): inductive-mode zero-shot
    guarantee. Excluded nodes' targets contribute to NO gradient-producing loss
    term. The kriging term always reads full (unsliced) targets: during training
    the holdout must be drawn from participating nodes only (asserted whenever
    gradients are enabled); during no-grad validation the holdout is deliberately
    the loss-excluded fixed inductive set, which is scoring, not leakage.

    Returns:
        (total_loss, loss_impute, loss_forecast, loss_kriging)
        loss_kriging is 0.0 when w_kriging == 0.0 (kriging path bypassed).
    """
    assert imputed_features.dim() == 4, \
        f"CRITICAL: imputed_features must be [B, N, T, D], got {tuple(imputed_features.shape)}"
    assert target_impute.dim() == 4, \
        f"CRITICAL: target_impute must be [B, N, T, D], got {tuple(target_impute.shape)}"
    assert mu.dim() == 2, f"CRITICAL: mu must be [B, N], got {tuple(mu.shape)}"
    assert theta.dim() == 2, f"CRITICAL: theta must be [B, N], got {tuple(theta.shape)}"
    assert pi.dim() == 2, f"CRITICAL: pi must be [B, N], got {tuple(pi.shape)}"
    assert target_forecast.dim() == 2, \
        f"CRITICAL: target_forecast must be [B, N], got {tuple(target_forecast.shape)}"
    assert imputed_features.shape[:2] == target_impute.shape[:2] == mu.shape == theta.shape == pi.shape == target_forecast.shape, \
        "CRITICAL: compute_loss tensor shapes are inconsistent"

    target_impute_full = target_impute
    if loss_node_mask is not None:
        assert loss_node_mask.dtype == torch.bool and loss_node_mask.dim() == 1, \
            f"CRITICAL: loss_node_mask must be 1D bool, got {loss_node_mask.dtype} {tuple(loss_node_mask.shape)}"
        assert loss_node_mask.any(), "CRITICAL: loss_node_mask excludes every node"
        if spatial_holdout_mask is not None and torch.is_grad_enabled():
            assert not (spatial_holdout_mask & ~loss_node_mask).any(), \
                "CRITICAL: kriging holdout contains loss-excluded (inductive) nodes — target leakage"
        imputed_features = imputed_features[:, loss_node_mask]
        target_impute = target_impute[:, loss_node_mask]
        mu = mu[:, loss_node_mask]
        theta = theta[:, loss_node_mask]
        pi = pi[:, loss_node_mask]
        target_forecast = target_forecast[:, loss_node_mask]

    # --- 1. Imputation Loss: Split Traffic vs Weather ---

    # Extract traffic channel (dim 0)
    traffic_pred_scaled = imputed_features[:, :, :, 0]      # [B, N, T]
    traffic_target_scaled = target_impute[:, :, :, 0]       # [B, N, T]

    # Inverse transform to raw counts
    traffic_pred_raw = (traffic_pred_scaled * traffic_std) + traffic_mean
    traffic_target_raw = (traffic_target_scaled * traffic_std) + traffic_mean

    # Clamp to non-negative (count domain). Non-negativity of the target is asserted
    # at dataset construction time (dataset.py) where raw counts are loaded.
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

    # --- 3. Kriging Loss (ABLATION-SAFE: always computed when predictions exist) ---
    # loss_kriging is ALWAYS calculated when kriging_pred is supplied so that
    # W&B telemetry and ablation comparisons work even when w_kriging == 0.0.
    # It only enters the optimisation objective when w_kriging > 0.0, preserving
    # the gradient bypass (no autograd path to kriging decoder when ablated).
    loss_kriging = torch.zeros((), device=imputed_features.device)
    if kriging_pred is not None:
        assert spatial_holdout_mask is not None, \
            "CRITICAL: kriging_pred supplied but spatial_holdout_mask is None"
        loss_kriging = compute_kriging_loss(
            kriging_pred=kriging_pred,
            target_impute=target_impute_full,
            holdout_mask=spatial_holdout_mask,
            traffic_mean=traffic_mean,
            traffic_std=traffic_std,
        )

    # --- 4. Total Weighted Loss ---
    # Only add kriging term when w_kriging > 0 to keep the autograd graph clean.
    total_loss = (w_impute * loss_impute) + (w_forecast * loss_forecast)
    if w_kriging > 0.0:
        total_loss = total_loss + (w_kriging * loss_kriging)

    # --- 5. OFFENSIVE ASSERTIONS (Tripwires) ---
    assert not torch.isnan(loss_impute_traffic), "CRITICAL: Traffic MSLE is NaN"
    assert not torch.isnan(loss_impute_weather), "CRITICAL: Weather MSE is NaN"
    assert not torch.isnan(loss_forecast), "CRITICAL: Forecast loss is NaN"
    assert not torch.isnan(loss_kriging), "CRITICAL: Kriging loss is NaN"
    assert not torch.isnan(total_loss), "CRITICAL: Total loss is NaN"
    assert not torch.isinf(total_loss), "CRITICAL: Total loss is Inf"

    return total_loss, loss_impute, loss_forecast, loss_kriging
