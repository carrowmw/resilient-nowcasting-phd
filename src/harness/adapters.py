"""
Model adapters — the seam between the harness and concrete models.

All model-specific wrangling lives here so the train/val loops and exporters
depend only on the Predictions/Losses/Capabilities contract, never on a model's
internals. Loss functions legitimately differ per model (ZINB NLL vs MSE), so
val_loss is per-model only (early stopping / checkpointing); cross-model
comparison happens exclusively through capability-gated metrics.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from harness.contract import Capabilities, Losses, Predictions


class ModelAdapter(ABC):
    capabilities: Capabilities

    def __init__(self, module: torch.nn.Module):
        self.module = module

    @abstractmethod
    def step(
        self,
        batch: dict,
        weights: Dict[str, float],
        loss_node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Losses, Predictions]:
        """One forward pass producing losses AND predictions (single forward —
        splitting predict/loss would double the compute for the RIoT pipeline).

        loss_node_mask: [N] bool, True = node participates in the loss. In
        inductive mode the fixed holdout nodes are False — their targets must
        never produce gradient (the zero-shot guarantee).
        """

    @abstractmethod
    def predict(self, batch: dict) -> Predictions:
        """Inference-only pass (exporters): no targets, no losses."""


class RIoTAdapter(ModelAdapter):
    """Wraps ResilientIoTPipeline unchanged. Absorbs the pipeline's
    weights-in-batch convention so the harness batch stays a pure data contract."""

    capabilities = Capabilities(
        uncertainty=True, zero_inflation=True, imputation=True, kriging=True, graph=True,
    )

    def step(self, batch, weights, loss_node_mask=None):
        model_batch = dict(batch)
        model_batch["w_impute"] = weights["w_impute"]
        model_batch["w_forecast"] = weights["w_forecast"]
        model_batch["w_kriging"] = weights["w_kriging"]
        if loss_node_mask is not None:
            model_batch["loss_node_mask"] = loss_node_mask

        (loss, loss_impute, loss_forecast, loss_kriging,
         imputed, mu, theta, pi, tripwires, kriging_pred) = self.module(model_batch)

        losses = Losses(total=loss, impute=loss_impute,
                        forecast=loss_forecast, kriging=loss_kriging)
        preds = Predictions(
            mu=mu, theta=theta, pi=pi, imputed=imputed,
            kriging_pred=kriging_pred, tripwires=tuple(tripwires),
        )
        return losses, preds

    def predict(self, batch):
        result = self.module(dict(batch), return_analysis=True)
        return Predictions(
            mu=result["mu"],
            theta=result["theta"],
            pi=result["pi"],
            imputed=result["imputed_features"],
            kriging_pred=result["kriging_pred"],
            edge_index=result["edge_index"],
            attention=result["attention_weights"],
            tripwires=tuple(result["tripwire_logs"]),
        )


class RecurrentBaselineAdapter(ModelAdapter):
    """LSTM/GRU baselines. No uncertainty or zero-inflation head exists, so
    theta/pi are None — the harness reports those metrics as null rather than
    scoring fabricated parameters. Imputation and kriging are naive-but-genuine
    attempts via the reconstruction head."""

    capabilities = Capabilities(
        uncertainty=False, zero_inflation=False, imputation=True, kriging=True, graph=False,
    )

    def _forward(self, batch):
        dynamic = torch.nan_to_num(batch["dynamic_features"], nan=0.0)
        mu, recon = self.module(dynamic)
        holdout_mask = batch.get("spatial_holdout_mask")
        kriging_pred = recon[:, holdout_mask, :, :] if holdout_mask is not None else None
        return mu, recon, kriging_pred

    def step(self, batch, weights, loss_node_mask=None):
        mu, recon, kriging_pred = self._forward(batch)

        target_forecast = batch["target_forecast"]
        target_impute = batch["target_impute"]

        if loss_node_mask is not None:
            assert loss_node_mask.any(), "CRITICAL: loss_node_mask excludes every node"
            mu_l = mu[:, loss_node_mask]
            target_forecast = target_forecast[:, loss_node_mask]
            recon_l = recon[:, loss_node_mask]
            target_impute = target_impute[:, loss_node_mask]
        else:
            mu_l, recon_l = mu, recon

        # Keep MSE losses in FP32 to match ZINB carve-out in pipeline.py.
        with torch.autocast(device_type=mu.device.type, enabled=False):
            loss_forecast = F.mse_loss(mu_l.float(), target_forecast.float())
            loss_impute = F.mse_loss(recon_l.float(), target_impute.float())
        # The reconstruction loss already trains the head that produces
        # kriging_pred, so there is no separate kriging objective.
        loss_kriging = torch.zeros((), device=mu.device)

        total = (
            weights["w_forecast"] * loss_forecast
            + weights["w_impute"] * loss_impute
        )
        assert not torch.isnan(total), "CRITICAL: baseline total loss is NaN"

        losses = Losses(total=total, impute=loss_impute,
                        forecast=loss_forecast, kriging=loss_kriging)
        preds = Predictions(mu=mu, imputed=recon, kriging_pred=kriging_pred)
        return losses, preds

    def predict(self, batch):
        mu, recon, kriging_pred = self._forward(batch)
        return Predictions(mu=mu, imputed=recon, kriging_pred=kriging_pred)


def build_adapter(model_type: str, module: torch.nn.Module) -> ModelAdapter:
    if model_type == "gnn":
        return RIoTAdapter(module)
    if model_type in ("lstm", "gru"):
        return RecurrentBaselineAdapter(module)
    raise ValueError(f"Unknown model_type: {model_type!r} (expected gnn | lstm | gru)")
