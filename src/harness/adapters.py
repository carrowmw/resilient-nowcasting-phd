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
from likelihoods import nb_log_p0, nb_logpmf
from losses import TrafficScale, compute_v5_losses, masked_mean


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
    weights-in-batch convention so the harness batch stays a pure data contract.

    config=None -> legacy v4 tuple path (exporters, legacy tests).
    config=<resolved run config> -> v5 dict path: batches come from
    harness.batch_prep.BatchPreparer and losses from losses.compute_v5_losses."""

    capabilities = Capabilities(
        uncertainty=True, zero_inflation=True, imputation=True, kriging=True, graph=True,
    )

    def __init__(self, module, config: Optional[dict] = None):
        super().__init__(module)
        self.config = config
        if config is not None:
            lik = config["loss"]["forecast_likelihood"]
            # Instance attribute shadows the class default: only a ZINB head has a
            # genuine zero-inflation pi; only the mixture has an outage probability.
            self.capabilities = Capabilities(
                uncertainty=True, zero_inflation=lik == "zinb", imputation=True,
                kriging=True, graph=True, outage=lik == "outage_mixture",
            )

    def _predictions(self, out) -> Predictions:
        return Predictions(
            mu=out["mu"], theta=out["theta"], pi=out["pi"], imputed=out["imputed"],
            kriging_pred=out["kriging_pred"], edge_index=out["edge_index"],
            attention=out["attention"], tripwires=tuple(out["tripwires"]),
            point=out["point"], p_zero=out["p_zero"], outage_q=out["q"],
            quantiles=out["quantiles"], quantile_levels=tuple(out["quantile_levels"]),
            imputed_raw=out["imputed_raw"],
            nll_params={
                "likelihood": self.config["loss"]["forecast_likelihood"],
                "mu": out["mu"], "theta": out["theta"], "pi": out["pi"], "q": out["q"],
            },
        )

    def step(self, batch, weights, loss_node_mask=None):
        if self.config is not None:
            out = self.module(batch, return_outputs=True)
            with torch.autocast(device_type=out["mu"].device.type, enabled=False):
                v5 = compute_v5_losses(out, batch, self.config["loss"], self.module.scale, weights, loss_node_mask)
            losses = Losses(total=v5.total, impute=v5.impute, forecast=v5.forecast,
                            kriging=v5.kriging, components=v5.components)
            return losses, self._predictions(out)

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
        if self.config is not None:
            return self._predictions(self.module(batch, return_outputs=True, return_analysis=True))
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

    def __init__(self, module, config: Optional[dict] = None, scale: Optional[TrafficScale] = None):
        """config=None -> v4 behaviour. With a resolved config the adapter honours
        model.baseline_inputs / model.baseline_head and loss.target_mask /
        loss.impute_positions (the less-confounded baseline of report Phase 7)."""
        super().__init__(module)
        self.config = config
        self.scale = scale
        if config is not None:
            assert scale is not None, "CRITICAL: v5 baseline adapter needs the traffic scale"
            self.inputs = config["model"]["baseline_inputs"]
            self.head = config["model"]["baseline_head"]
            assert self.head == module.head, \
                f"CRITICAL: baseline_head={self.head} but module head={module.head}"
            self.capabilities = Capabilities(
                uncertainty=self.head == "nb", zero_inflation=False, imputation=True,
                kriging=True, graph=False,
            )

    def _inputs(self, batch):
        dynamic = torch.nan_to_num(batch["dynamic_features"], nan=0.0)
        if self.config is None or self.inputs == "dynamic":
            return dynamic
        B, N, T, _ = dynamic.shape
        masks = batch["structural_masks"].to(dynamic.dtype)
        blank = batch["blank_mask"]
        if blank is not None:
            masks = masks.clone()
            masks[:, blank, :, 0] = 1.0
        parts = [dynamic, masks]
        if batch["covariates"] is not None:
            parts.append(batch["covariates"][:, :, :T].to(dynamic.dtype))
        static = batch["static_features"].to(dynamic.dtype)
        parts.append(static.view(1, N, 1, -1).expand(B, N, T, static.shape[-1]))
        return torch.cat(parts, dim=-1)

    def _forward(self, batch):
        mu, theta, recon = self.module.forward_heads(self._inputs(batch))
        holdout_mask = batch.get("spatial_holdout_mask")
        kriging_pred = recon[:, holdout_mask, :, :] if holdout_mask is not None else None
        return mu, theta, recon, kriging_pred

    def _predictions(self, mu, theta, recon, kriging_pred) -> Predictions:
        if self.config is None:
            return Predictions(mu=mu, imputed=recon, kriging_pred=kriging_pred)
        nb = self.head == "nb"
        return Predictions(
            mu=mu, theta=theta, imputed=recon, kriging_pred=kriging_pred, point=mu,
            p_zero=torch.exp(nb_log_p0(mu, theta)) if nb else None,
            imputed_raw=self.scale.to_raw(recon[..., 0].float()),
            nll_params={"likelihood": "nb", "mu": mu, "theta": theta, "pi": None, "q": None} if nb else None,
        )

    def _step_v5(self, batch, weights, loss_node_mask):
        mu, theta, recon, kriging_pred = self._forward(batch)
        loss_cfg = self.config["loss"]
        y = batch["target_forecast"]
        obs = batch["target_observed"]
        target_impute = batch["target_impute"]
        art, in_obs, node_mask = batch["art"], batch["input_observed"], batch["node_mask"]
        sel = loss_node_mask if loss_node_mask is not None else torch.ones_like(obs[0])
        assert sel.any(), "CRITICAL: loss_node_mask excludes every node"

        with torch.autocast(device_type=mu.device.type, enabled=False):
            mu_f, y_f = mu[:, sel].float(), y[:, sel].float()
            participate = obs[:, sel] if loss_cfg["target_mask"] == "observed" else torch.ones_like(obs[:, sel])
            if self.head == "nb":
                elem = -nb_logpmf(y_f, mu_f, theta[:, sel].float())
            else:
                elem = (mu_f - y_f) ** 2
            loss_forecast = masked_mean(elem, participate)
            nll_obs = masked_mean(elem, obs[:, sel])

            recon_l, tgt_l = recon[:, sel].float(), target_impute[:, sel].float()
            sq = (recon_l - tgt_l) ** 2
            if loss_cfg["impute_positions"] == "all":
                loss_impute = sq.mean()
            else:
                traffic_mask = art[:, sel] & in_obs[:, sel]
                loss_impute = masked_mean(sq[..., 0], traffic_mask) + masked_mean(sq[..., 1:], node_mask[:, sel][..., 1:])
                if loss_cfg["impute_positions"] == "masked_observed_plus_aux":
                    aux = in_obs[:, sel] & ~art[:, sel]
                    loss_impute = loss_impute + loss_cfg["impute_aux_weight"] * masked_mean(sq[..., 0], aux)
        loss_kriging = torch.zeros((), device=mu.device)
        total = weights["w_forecast"] * loss_forecast + weights["w_impute"] * loss_impute
        assert not torch.isnan(total), "CRITICAL: baseline total loss is NaN"
        losses = Losses(total=total, impute=loss_impute, forecast=loss_forecast, kriging=loss_kriging,
                        components={"forecast_lik": loss_forecast, "forecast_nll_obs": nll_obs,
                                    "kriging": loss_kriging})
        return losses, self._predictions(mu, theta, recon, kriging_pred)

    def step(self, batch, weights, loss_node_mask=None):
        if self.config is not None:
            return self._step_v5(batch, weights, loss_node_mask)
        mu, _, recon, kriging_pred = self._forward(batch)

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
        return self._predictions(*self._forward(batch))


class NaiveAdapter(ModelAdapter):
    """Non-learned floor (baselines.NaiveBaseline). Needs a v5 prepared batch.

    profile:  forecast = seasonal slot-of-week median (train slice) of the node;
              blanked nodes get the inductive substitute (never their own data).
    last_obs: last traffic value visible to the model in the input window,
              falling back to the profile when nothing is visible.
    Imputation / kriging output = the profile over the input window. Losses are
    zero tensors (there is nothing to train); the harness skips the train loop.
    """

    capabilities = Capabilities(
        uncertainty=False, zero_inflation=False, imputation=True, kriging=True, graph=False,
    )

    def __init__(self, module, scale: TrafficScale, covariates):
        super().__init__(module)
        self.kind = module.kind
        self.scale = scale
        self.covariates = covariates

    def _predict(self, batch) -> Predictions:
        clean = batch["target_impute"]
        B, N, T, D = clean.shape
        profile = self.covariates.profile_raw(batch["time_idx"], batch["blank_mask"])   # [B, N, T + 1]
        point = profile[..., T]
        if self.kind == "last_obs":
            visible = ~batch["input_missing"]                                    # [B, N, T]
            steps = torch.arange(T, device=clean.device).view(1, 1, T)
            last = torch.where(visible, steps, torch.full_like(steps, -1)).max(dim=-1).values
            raw = self.scale.to_raw(clean[..., 0].float())
            last_val = raw.gather(-1, last.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            point = torch.where(last >= 0, last_val, point)
        imputed = torch.zeros_like(clean, dtype=torch.float32)  # weather: scaled mean
        imputed[..., 0] = self.scale.to_scaled(profile[..., :T])
        holdout = batch["spatial_holdout_mask"]
        kriging_pred = imputed[:, holdout] if holdout is not None else None
        return Predictions(mu=point, imputed=imputed, kriging_pred=kriging_pred, point=point,
                           imputed_raw=profile[..., :T])

    def step(self, batch, weights, loss_node_mask=None):
        preds = self._predict(batch)
        zero = torch.zeros((), device=preds.mu.device)
        return Losses(total=zero, impute=zero, forecast=zero, kriging=zero,
                      components={"forecast_nll_obs": zero, "kriging": zero}), preds

    def predict(self, batch):
        return self._predict(batch)


def build_adapter(
    model_type: str,
    module: torch.nn.Module,
    config: Optional[dict] = None,
    scale: Optional[TrafficScale] = None,
    covariates=None,
) -> ModelAdapter:
    """config=None keeps the v4 adapters (exporters, legacy tests); train.py
    passes the resolved run config, the dataset traffic scale and the
    BatchPreparer's CovariateBuilder."""
    if model_type == "gnn":
        return RIoTAdapter(module, config)
    if model_type in ("lstm", "gru"):
        return RecurrentBaselineAdapter(module, config, scale)
    if model_type == "naive":
        assert config is not None and covariates is not None, "CRITICAL: naive baseline is v5-only"
        return NaiveAdapter(module, scale, covariates)
    raise ValueError(f"Unknown model_type: {model_type!r} (expected gnn | lstm | gru | naive)")
