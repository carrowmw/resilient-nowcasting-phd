"""
Elementwise count likelihoods for the v5 forecast / imputation heads (R4).

All functions take fp32 tensors of identical shape and return elementwise
log-probabilities (no reduction). Parameter conventions match decoders.py:
mu > 0, theta > 0, pi / q in (0, 1).

  zinb            P(0) = pi + (1-pi) NB(0);  P(y>0) = (1-pi) NB(y)        (v4)
  nb              NB(y)                                                     (R4a)
  ztnb            NB(y) / (1 - NB(0)),  y >= 1                              (R4b)
  outage_mixture  observed y>=1: (1-q) NB(y);  missing: q + (1-q) NB(0)     (R4)

For the outage mixture, `mu` is the counterfactual count with the sensor
working; q is P(outage | context).
"""
import torch

EPS = 1e-6


def safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp(min=EPS))


def nb_log_p0(mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return theta * (safe_log(theta) - safe_log(theta + mu))


def nb_logpmf(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    log_theta_ratio = safe_log(theta) - safe_log(theta + mu)
    log_mu_ratio = safe_log(mu) - safe_log(theta + mu)
    return (torch.lgamma(y + theta) - torch.lgamma(y + 1) - torch.lgamma(theta)
            + theta * log_theta_ratio + y * log_mu_ratio)


def zinb_logpmf(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Elementwise form of decoders.compute_zinb_nll (identical arithmetic)."""
    pi = pi.clamp(min=EPS, max=1.0 - EPS)
    ll_nb = nb_logpmf(y, mu, theta)
    ll_zero = safe_log(pi + (1 - pi) * torch.exp(nb_log_p0(mu, theta)))
    ll_nonzero = safe_log(1 - pi) + ll_nb
    return torch.where(y < EPS, ll_zero, ll_nonzero)


def ztnb_logpmf(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Zero-truncated NB. 1 - NB(0) is clamped to >= EPS: as mu -> 0 the
    truncation normaliser vanishes and the unclamped NLL would blow up."""
    log_norm = safe_log(1.0 - torch.exp(nb_log_p0(mu, theta)))
    ll = nb_logpmf(y, mu, theta) - log_norm
    return torch.where(y < 0.5, torch.full_like(ll, float("-inf")), ll)


def count_logpmf(likelihood: str, y, mu, theta, pi=None) -> torch.Tensor:
    """Log-pmf of the *count* distribution used on observed slots (the outage
    mixture's count component is plain NB)."""
    if likelihood == "zinb":
        return zinb_logpmf(y, mu, theta, pi)
    if likelihood == "ztnb":
        return ztnb_logpmf(y, mu, theta)
    assert likelihood in ("nb", "outage_mixture"), f"CRITICAL: unknown likelihood {likelihood!r}"
    return nb_logpmf(y, mu, theta)


def band_logpmf(likelihood: str, y, mu, theta, pi=None) -> torch.Tensor:
    """log P(max(0, y-1) <= Y <= y+1) — the low-end tolerance band (R5)."""
    terms = [count_logpmf(likelihood, y, mu, theta, pi), count_logpmf(likelihood, y + 1, mu, theta, pi)]
    lower = count_logpmf(likelihood, (y - 1).clamp_min(0), mu, theta, pi)
    terms.append(torch.where(y >= 1, lower, torch.full_like(lower, float("-inf"))))
    return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


def point_forecast(likelihood: str, mu, theta, pi=None) -> torch.Tensor:
    """Point prediction scored by RMSE/MAE.
    zinb: E[Y] = mu (1 - pi) (F-C06); nb: mu; ztnb: E[Y | Y >= 1] = mu / (1 - NB(0));
    outage_mixture: mu — the counterfactual count with the sensor working (R4/C11)."""
    if likelihood == "zinb":
        return mu * (1.0 - pi)
    if likelihood == "ztnb":
        return mu / (1.0 - torch.exp(nb_log_p0(mu, theta))).clamp(min=EPS)
    return mu


def prob_zero(likelihood: str, mu, theta, pi=None, q=None) -> torch.Tensor:
    """P(the slot reads zero / is empty). For the outage mixture this is
    P(missing) = q + (1 - q) NB(0) — the quantity comparable to ZINB's zero mass."""
    p0 = torch.exp(nb_log_p0(mu, theta))
    if likelihood == "zinb":
        return pi + (1.0 - pi) * p0
    if likelihood == "outage_mixture":
        return q + (1.0 - q) * p0
    return p0


def predictive_pmf_grid(likelihood: str, mu, theta, pi, max_count: int) -> torch.Tensor:
    """pmf over k = 0..max_count of the count distribution, [..., K + 1].
    Mass beyond max_count is folded into the last bin so the CDF reaches 1."""
    k = torch.arange(max_count + 1, device=mu.device, dtype=mu.dtype)
    shape = mu.shape + (1,)
    lp = count_logpmf(
        likelihood, k.view(*([1] * mu.dim()), -1).expand(*mu.shape, -1),
        mu.reshape(shape), theta.reshape(shape), None if pi is None else pi.reshape(shape),
    )
    pmf = torch.exp(lp)
    tail = (1.0 - pmf.sum(dim=-1)).clamp_min(0.0)
    pmf[..., -1] = pmf[..., -1] + tail
    return pmf
