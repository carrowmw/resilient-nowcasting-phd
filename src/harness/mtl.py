"""
Multi-task loss balancing (zero-collapse report R6 / Phase 6).

loss.mtl_balancing
  none         total = sum_i w_i L_i                              (v4)
  uncertainty  total = sum_i 0.5 exp(-s_i) w_i L_i + 0.5 s_i      (Kendall et al. 2018;
               s_i are learnable log-variances added to the optimiser)
  pcgrad       per-task gradients of w_i L_i; a task's gradient is projected off
               any other task gradient it conflicts with (negative dot), in random
               order, then summed (Yu et al. 2020)

training.grad_cosine_every = k > 0 logs cos(grad forecast, grad impute) on the
shared (non-decoder) parameters every k optimiser steps — the R6 diagnostic for
whether imputation fights forecasting.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from harness.contract import Losses

_HEAD_PREFIXES = ("decoders.", "kriging_decoder.", "forecast_head.", "recon_head.")


def _task_terms(losses: Losses, weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    terms = {}
    for name, loss, w in (
        ("forecast", losses.forecast, weights["w_forecast"]),
        ("impute", losses.impute, weights["w_impute"]),
        ("kriging", losses.kriging, weights["w_kriging"]),
    ):
        if w > 0.0 and loss.requires_grad:
            terms[name] = w * loss
    return terms


class MTLBalancer(nn.Module):
    def __init__(self, mode: str, model: nn.Module, grad_cosine_every: int):
        super().__init__()
        assert mode in ("none", "uncertainty", "pcgrad"), f"Unknown mtl_balancing: {mode}"
        self.mode = mode
        self.grad_cosine_every = grad_cosine_every
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shared = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith(_HEAD_PREFIXES)
        ]
        # One learnable log-variance per task; unused tasks keep a zero gradient.
        self.log_vars = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(())) for k in ("forecast", "impute", "kriging")
        }) if mode == "uncertainty" else None

    def extra_parameters(self) -> List[nn.Parameter]:
        return list(self.log_vars.values()) if self.log_vars is not None else []

    def total(self, losses: Losses, weights: Dict[str, float]) -> torch.Tensor:
        """The scalar that is backpropagated (and logged as the train objective)."""
        if self.mode != "uncertainty":
            return losses.total
        terms = _task_terms(losses, weights)
        assert terms, "CRITICAL: uncertainty weighting with no active task"
        out = 0.0
        for k, t in terms.items():
            s = self.log_vars[k]
            out = out + 0.5 * torch.exp(-s) * t + 0.5 * s
        return out

    def _grads(self, loss: torch.Tensor, params) -> List[torch.Tensor]:
        gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for g, p in zip(gs, params)]

    def grad_cosine(self, losses: Losses, step: int) -> Optional[float]:
        if self.grad_cosine_every <= 0 or step % self.grad_cosine_every != 0:
            return None
        if not (losses.forecast.requires_grad and losses.impute.requires_grad) or not self.shared:
            return None
        gf = torch.cat([g.flatten() for g in self._grads(losses.forecast.float(), self.shared)])
        gi = torch.cat([g.flatten() for g in self._grads(losses.impute.float(), self.shared)])
        denom = gf.norm() * gi.norm()
        return float((gf @ gi) / denom) if denom > 0 else None

    def backward(self, total: torch.Tensor, losses: Losses, weights: Dict[str, float], scale_fn) -> None:
        """scale_fn: GradScaler.scale (or identity) — applied to every backpropagated term."""
        if self.mode != "pcgrad":
            scale_fn(total).backward()
            return
        terms = _task_terms(losses, weights)
        if len(terms) < 2:
            scale_fn(total).backward()
            return
        flat = []
        for t in terms.values():
            gs = self._grads(scale_fn(t), self.params)
            flat.append(torch.cat([g.flatten() for g in gs]))
        projected = []
        for i, gi in enumerate(flat):
            g = gi.clone()
            for j in torch.randperm(len(flat)).tolist():
                if j == i:
                    continue
                dot = g @ flat[j]
                if dot < 0:
                    g = g - dot / flat[j].norm().pow(2).clamp_min(1e-20) * flat[j]
            projected.append(g)
        combined = torch.stack(projected).sum(0)
        offset = 0
        for p in self.params:
            n = p.numel()
            g = combined[offset:offset + n].view_as(p).to(p.dtype)
            p.grad = g if p.grad is None else p.grad + g
            offset += n
