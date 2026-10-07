"""
v5 joint loss (R1 target hygiene, R4 likelihoods, R5 magnitude terms).

compute_v5_losses() consumes the pipeline's output dict plus the prepared batch
and the resolved `loss` config section. Under LEGACY_V4_EQUIVALENTS
(run_config.py) it computes exactly what decoders.compute_loss computed in v4
(asserted by tests/test_v5_losses.py::test_legacy_equivalence).

Routing masks (all from the UN-ablated observation copy, C12):
  target_observed [B, N]     forecast truth exists
  input_observed  [B, N, T]  input-window truth exists
  art             [B, N, T]  artificially hidden traffic steps
  target_outage   [B, N]     forecast made from inside an artificial outage
"""
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn.functional as F

from likelihoods import (
    EPS, band_logpmf, count_logpmf, nb_log_p0, nb_logpmf, safe_log,
)


class TrafficScale(NamedTuple):
    mean: torch.Tensor
    std: torch.Tensor
    log1p: bool

    def to_raw(self, scaled: torch.Tensor) -> torch.Tensor:
        domain = scaled * self.std + self.mean
        return (torch.expm1(domain) if self.log1p else domain).clamp_min(0.0)

    def to_scaled(self, raw: torch.Tensor) -> torch.Tensor:
        domain = torch.log1p(raw) if self.log1p else raw
        return (domain - self.mean) / self.std


class V5Losses(NamedTuple):
    total: torch.Tensor
    impute: torch.Tensor
    forecast: torch.Tensor
    kriging: torch.Tensor
    components: Dict[str, torch.Tensor]


def masked_mean(x: torch.Tensor, mask: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean of x over mask (optionally weighted). Empty mask -> graph-connected 0."""
    m = mask.to(x.dtype)
    if weight is not None:
        m = m * weight
    denom = m.sum()
    # where, not x * m: excluded slots may hold +-inf (ztnb at y = 0) and inf * 0 = NaN.
    x = torch.where(mask, x, torch.zeros_like(x))
    return torch.where(denom > 0, (x * m).sum() / denom.clamp_min(EPS), (x * 0.0).sum())


def msle_elem(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    """Elementwise term of decoders.scale_invariant_mse (identical arithmetic)."""
    return (torch.log1p(pred_raw.clamp_min(0.0) + 1e-6) - torch.log1p(target_raw.clamp_min(0.0) + 1e-6)) ** 2


def impute_traffic_elem(pred_raw, target_raw, loss_cfg) -> torch.Tensor:
    """(1 - a) MSLE + a Huber(raw, delta) / delta^2, with the optional low-end band."""
    elem = msle_elem(pred_raw, target_raw)
    alpha = loss_cfg["impute_huber_mix"]
    if alpha > 0.0:
        delta = loss_cfg["impute_huber_delta"]
        huber = F.huber_loss(pred_raw, target_raw, reduction="none", delta=delta) / (delta ** 2)
        elem = (1.0 - alpha) * elem + alpha * huber
    if loss_cfg["low_end_tolerance"]:
        inside = (target_raw <= 2.0) & ((pred_raw - target_raw).abs() <= 1.0)
        elem = torch.where(inside, torch.zeros_like(elem), elem)
    return elem


def pinball(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """diff = y - prediction."""
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def mixture_terms(y, mu, theta, q, observed, artificial, loss_cfg, count_ll=None):
    """Elementwise outage-mixture NLL.
      observed (sensor reported y):      -log(1-q) - log NB(y)
      artificial outage, y known:        -log NB(y) - w_art log q      (when w_art > 0)
      missing:                           -w_missing log(q + (1-q) NB(0))
    """
    if count_ll is None:
        count_ll = nb_logpmf(y, mu, theta)
    obs_nll = -safe_log(1.0 - q) - count_ll
    w_art = loss_cfg["w_artificial_outage"]
    if w_art > 0.0:
        art_nll = -count_ll - w_art * safe_log(q)
        obs_nll = torch.where(artificial & observed, art_nll, obs_nll)
    miss_nll = -loss_cfg["w_missing_lik"] * safe_log(q + (1.0 - q) * torch.exp(nb_log_p0(mu, theta)))
    return torch.where(observed, obs_nll, miss_nll), obs_nll


def _forecast_loss(out, batch, loss_cfg, sel, components):
    lik = loss_cfg["forecast_likelihood"]
    y = batch["target_forecast"].float()
    obs = batch["target_observed"]
    mu, theta = out["mu"].float(), out["theta"].float()
    pi = out["pi"].float() if out["pi"] is not None else None
    q = out["q"].float() if out["q"] is not None else None
    point = out["point"].float()
    if sel is not None:
        y, obs, mu, theta, point = y[:, sel], obs[:, sel], mu[:, sel], theta[:, sel], point[:, sel]
        pi = pi[:, sel] if pi is not None else None
        q = q[:, sel] if q is not None else None

    participate = obs if loss_cfg["target_mask"] == "observed" else torch.ones_like(obs)

    # Magnitude weights on observed slots, normalised to mean 1 over them (R5).
    alpha = loss_cfg["magnitude_weight_alpha"]
    weight = None
    if alpha > 0.0:
        w = 1.0 + (y.clamp_min(0.0) / loss_cfg["magnitude_weight_y0"]) ** alpha
        w_mean = masked_mean(w, obs).detach().clamp_min(EPS)
        weight = torch.where(obs, w / w_mean, torch.ones_like(w))

    count_ll = count_logpmf(lik, y, mu, theta, pi)
    if loss_cfg["low_end_tolerance"]:
        count_ll = torch.where(obs & (y <= 2.0), band_logpmf(lik, y, mu, theta, pi), count_ll)

    if lik == "outage_mixture":
        artificial = batch["target_outage"][:, sel] if sel is not None else batch["target_outage"]
        elem, obs_nll = mixture_terms(y, mu, theta, q, obs, artificial, loss_cfg, count_ll)
        lik_loss = masked_mean(elem, participate | ~obs, weight)
        nll_obs = masked_mean(obs_nll, obs)
    else:
        elem = -count_ll
        lik_loss = masked_mean(elem, participate, weight)
        nll_obs = masked_mean(elem, obs)
    components["forecast_lik"] = lik_loss
    components["forecast_nll_obs"] = nll_obs
    total = lik_loss

    y0 = loss_cfg["magnitude_weight_y0"]
    if loss_cfg["w_underpred"] > 0.0:
        under = masked_mean(pinball(y - point, loss_cfg["underpred_tau"]), obs) / y0
        components["underpred"] = under
        total = total + loss_cfg["w_underpred"] * under
    if out["quantiles"] is not None:
        qs = out["quantiles"].float()
        qs = qs[:, sel] if sel is not None else qs
        levels = out["quantile_levels"]
        qloss = sum(masked_mean(pinball(y - qs[..., i], tau), obs) for i, tau in enumerate(levels)) / (len(levels) * y0)
        components["quantile"] = qloss
        total = total + loss_cfg["w_quantile"] * qloss
    if loss_cfg["outage_prior_weight"] > 0.0:
        rl = batch["rl_target_raw"][:, sel] if sel is not None else batch["rl_target_raw"]
        night = batch["night_flags"][:, -1].view(-1, 1).expand_as(rl)
        short_night = night & (rl >= 0) & (rl <= 4)
        long_gap = rl > 96
        prior = torch.where(long_gap, torch.full_like(q, 0.95), torch.full_like(q, 0.05))
        bce = F.binary_cross_entropy(q.clamp(EPS, 1 - EPS), prior, reduction="none")
        prior_loss = masked_mean(bce, short_night | long_gap)
        components["outage_prior"] = prior_loss
        total = total + loss_cfg["outage_prior_weight"] * prior_loss
    return total


def _impute_loss(out, batch, loss_cfg, scale: TrafficScale, sel, components):
    imputed = out["imputed"].float()
    target = batch["target_impute"].float()
    in_obs = batch["input_observed"]
    art = batch["art"]
    node_mask = batch["node_mask"]
    params = out["impute_traffic_params"]
    if sel is not None:
        imputed, target, in_obs, art, node_mask = (
            imputed[:, sel], target[:, sel], in_obs[:, sel], art[:, sel], node_mask[:, sel]
        )
        params = tuple(p[:, sel] for p in params) if params is not None else None

    target_raw = scale.to_raw(target[..., 0])
    positions = loss_cfg["impute_positions"]
    if positions == "all":
        traffic_mask = torch.ones_like(in_obs)
        weather_mask = torch.ones_like(node_mask[..., 1:])
    else:
        traffic_mask = art & in_obs
        weather_mask = node_mask[..., 1:]
    aux_w = loss_cfg["impute_aux_weight"] if positions == "masked_observed_plus_aux" else 0.0
    aux_mask = in_obs & ~art

    if loss_cfg["impute_likelihood"] == "outage_mixture":
        mu_t, theta_t, q_t = (p.float() for p in params)
        y = target_raw.round()
        elem, obs_nll = mixture_terms(y, mu_t, theta_t, q_t, in_obs, art, loss_cfg)
        traffic = masked_mean(elem, traffic_mask | ~in_obs)
        if aux_w > 0.0:
            traffic = traffic + aux_w * masked_mean(obs_nll, aux_mask)
    else:
        elem = impute_traffic_elem(scale.to_raw(imputed[..., 0]), target_raw, loss_cfg)
        traffic = masked_mean(elem, traffic_mask)
        if aux_w > 0.0:
            traffic = traffic + aux_w * masked_mean(elem, aux_mask)
    weather = masked_mean((imputed[..., 1:] - target[..., 1:]) ** 2, weather_mask)
    components["impute_traffic"] = traffic
    components["impute_weather"] = weather
    return traffic + weather


def _kriging_loss(out, batch, loss_cfg, scale: TrafficScale, components):
    pred = out["kriging_pred"].float()
    holdout = batch["spatial_holdout_mask"]
    target = batch["target_impute"][:, holdout].float()
    assert pred.shape == target.shape, \
        f"CRITICAL: kriging shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
    if pred.numel() == 0:
        return torch.zeros((), device=pred.device)
    target_raw = scale.to_raw(target[..., 0])
    if loss_cfg["kriging_observed_only"]:
        mask = batch["input_observed"][:, holdout]
    else:
        mask = torch.ones(target.shape[:3], dtype=torch.bool, device=pred.device)
    params = out["kriging_traffic_params"]
    if params is not None:
        mu_k, theta_k = (p.float() for p in params)
        traffic = masked_mean(-nb_logpmf(target_raw.round(), mu_k, theta_k), mask)
    else:
        traffic = masked_mean(msle_elem(scale.to_raw(pred[..., 0]), target_raw), mask)
    weather = masked_mean((pred[..., 1:] - target[..., 1:]) ** 2, mask.unsqueeze(-1).expand_as(pred[..., 1:]))
    components["kriging_traffic"] = traffic
    return traffic + weather


def compute_v5_losses(
    out: dict,
    batch: dict,
    loss_cfg: dict,
    scale: TrafficScale,
    weights: Dict[str, float],
    loss_node_mask: Optional[torch.Tensor] = None,
) -> V5Losses:
    sel = None
    if loss_node_mask is not None:
        assert loss_node_mask.dtype == torch.bool and loss_node_mask.dim() == 1, \
            f"CRITICAL: loss_node_mask must be 1D bool, got {loss_node_mask.dtype} {tuple(loss_node_mask.shape)}"
        assert loss_node_mask.any(), "CRITICAL: loss_node_mask excludes every node"
        holdout = batch.get("spatial_holdout_mask")
        if holdout is not None and torch.is_grad_enabled():
            assert not (holdout & ~loss_node_mask).any(), \
                "CRITICAL: kriging holdout contains loss-excluded (inductive) nodes — target leakage"
        sel = loss_node_mask

    components: Dict[str, torch.Tensor] = {}
    loss_impute = _impute_loss(out, batch, loss_cfg, scale, sel, components)
    loss_forecast = _forecast_loss(out, batch, loss_cfg, sel, components)
    loss_kriging = torch.zeros((), device=loss_forecast.device)
    if out["kriging_pred"] is not None:
        assert batch.get("spatial_holdout_mask") is not None, \
            "CRITICAL: kriging_pred supplied but spatial_holdout_mask is None"
        loss_kriging = _kriging_loss(out, batch, loss_cfg, scale, components)
    components["kriging"] = loss_kriging

    total = weights["w_impute"] * loss_impute + weights["w_forecast"] * loss_forecast
    if weights["w_kriging"] > 0.0:
        total = total + weights["w_kriging"] * loss_kriging

    assert not torch.isnan(loss_impute), "CRITICAL: Impute loss is NaN"
    assert not torch.isnan(loss_forecast), "CRITICAL: Forecast loss is NaN"
    assert not torch.isnan(loss_kriging), "CRITICAL: Kriging loss is NaN"
    assert not torch.isnan(total), "CRITICAL: Total loss is NaN"
    assert not torch.isinf(total), "CRITICAL: Total loss is Inf"
    return V5Losses(total, loss_impute, loss_forecast, loss_kriging, components)
