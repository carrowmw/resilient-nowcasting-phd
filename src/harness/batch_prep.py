"""
Device-side batch preparation shared by the train / val / test / synthetic-outage
passes (v5). Turns a collated dataset batch into the full model batch:

  1. artificial gaps for self-supervised imputation (R2: iid | block | transplant |
     tail | mixed; v4's iid-over-all-channels when mask_weather=true),
  2. the "missing" encoding (R2: unify_missing_encoding — real gaps, artificial
     gaps and synthetic outages all become NaN -> 0 plus mask channel 0 = 1),
  3. holdout blanking (v4 semantics: per-batch kriging re-roll during training,
     the fixed inductive set always; the fixed val/test holdout at evaluation),
  4. covariates (R3) and the routing masks consumed by losses / metrics.

Loss and metric routing always comes from the dataset's UN-ablated observation
mask (input_observed / target_observed, C12).
"""
from typing import Optional

import torch

from covariates import CovariateBuilder
from dataset import apply_spatial_holdout
from masking import GapMasker, synthetic_outage_mask


class BatchPreparer:
    def __init__(
        self,
        config: dict,
        dataset,
        device: torch.device,
        inductive_mask: Optional[torch.Tensor],
        loss_node_mask: Optional[torch.Tensor],
        eval_holdout_mask: Optional[torch.Tensor],
        static_features: torch.Tensor,
        location_ids: list,
    ):
        data_cfg = config["data"]
        self.device = device
        self.seq_len = data_cfg["seq_len"]
        self.mask_prob = data_cfg["mask_prob"]
        self.legacy_mask = data_cfg["mask_strategy"] == "iid" and data_cfg["mask_weather"]
        self.observed_only = data_cfg["artificial_mask_observed_only"]
        self.unify = data_cfg["unify_missing_encoding"]
        self.complete_weather = data_cfg["complete_weather"]
        self.spatial_holdout_prob = data_cfg["spatial_holdout_prob"]
        self.kriging_active = self.spatial_holdout_prob > 0.0
        self.inductive_mask = inductive_mask
        self.loss_node_mask = loss_node_mask
        self.eval_holdout_mask = eval_holdout_mask
        self.static_features = static_features
        self.location_ids = location_ids
        self.so_node_frac = config["evaluation"]["so_node_frac"]
        self.weather_zero_scaled = dataset.weather_zero_scaled.to(device)
        # Raw count 0 in the traffic channel's scaled domain (log1p(0) = 0 too).
        self.traffic_zero_scaled = -dataset.traffic_mean / dataset.traffic_std

        self.covariates = CovariateBuilder(dataset, data_cfg, inductive_mask, device)
        self.masker = None if self.legacy_mask else GapMasker(
            strategy=data_cfg["mask_strategy"],
            mask_prob=self.mask_prob,
            seq_len=self.seq_len,
            gap_length_bank=dataset.gap_length_bank,
            transplant_bank=dataset.transplant_bank,
            block_len_dist=data_cfg["mask_block_len_dist"],
            block_mean_len=data_cfg["mask_block_mean_len"],
            mask_mix=data_cfg["mask_mix"],
            device=device,
        )

    def _rand(self, shape, gen):
        return torch.rand(shape, device=self.device, generator=gen)

    def prepare(
        self,
        batch: dict,
        train: bool,
        gen: Optional[torch.Generator] = None,
        so_length: Optional[int] = None,
    ) -> dict:
        """
        Args:
            batch:     collated batch already on device.
            train:     True -> training holdout semantics (kriging re-roll | inductive).
            gen:       seeded generator for evaluation passes (None = global RNG).
            so_length: evaluation-only synthetic outage of this many steps, ending at
                       the last input step, on a so_node_frac sample of eligible rows.
                       No artificial imputation gaps are drawn in an SO pass.
        """
        batch = dict(batch)
        clean = batch["dynamic_features"]
        B, N, T, D = clean.shape
        in_obs = batch["input_observed"]
        tgt_obs = batch["target_observed"]
        time_idx = batch["time_idx"]
        night = self.covariates.night_flags(time_idx)                          # [B, T + 1]

        # ---- 1. artificial gaps ------------------------------------------------
        if so_length is not None:
            node_mask = torch.zeros_like(clean, dtype=torch.bool)
            target_outage = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        elif self.legacy_mask:
            node_mask = self._rand(clean.shape, gen) < self.mask_prob
            if self.observed_only:
                node_mask[..., 0] &= in_obs
            target_outage = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        else:
            gaps = self.masker.sample(B, N, gen)                                 # [B, N, T + 1]
            art_t = gaps[..., :T]
            if self.observed_only:
                art_t = art_t & in_obs
            node_mask = torch.zeros_like(clean, dtype=torch.bool)
            node_mask[..., 0] = art_t
            target_outage = gaps[..., T] & tgt_obs
        art = node_mask[..., 0]

        target_impute = clean.clone()
        x = clean.masked_fill(node_mask, float("nan"))
        masks = batch["structural_masks"].clone()
        night_in = night[:, :T].view(B, 1, T)

        if self.unify:
            # Real gaps become NaN -> 0 (scaled), exactly like artificial gaps, and
            # artificial gaps raise the mask channels exactly like real ones.
            x[..., 0] = x[..., 0].masked_fill(~in_obs, float("nan"))
            masks[..., 0] = torch.maximum(masks[..., 0], art.to(masks.dtype))
            masks[..., 1] = torch.maximum(masks[..., 1], (art & night_in).to(masks.dtype))

        # ---- synthetic outage (evaluation) ---------------------------------------
        so_slots, so_selected, outage_start = None, None, None
        if so_length is not None:
            k = min(so_length, T)
            eligible = tgt_obs.clone()
            if self.eval_holdout_mask is not None:
                eligible &= ~self.eval_holdout_mask.view(1, N)
            so_selected = synthetic_outage_mask(B, N, T, so_length, self.so_node_frac, eligible, gen, self.device)
            tail = torch.arange(T, device=self.device).view(1, 1, T) >= (T - k)
            so_slots = so_selected.unsqueeze(-1) & tail                          # [B, N, T]
            # Encode the hidden steps exactly as this run encodes real gaps.
            fill = float("nan") if self.unify else self.traffic_zero_scaled
            x[..., 0] = x[..., 0].masked_fill(so_slots, fill)
            if not self.complete_weather:
                x[..., 1:] = torch.where(so_slots.unsqueeze(-1), self.weather_zero_scaled.view(1, 1, 1, -1), x[..., 1:])
            masks[..., 0] = torch.where(so_slots, torch.ones_like(masks[..., 0]), masks[..., 0])
            masks[..., 1] = torch.where(so_slots, night_in.expand(B, N, T).to(masks.dtype), masks[..., 1])
            target_t = time_idx.view(B, 1) + T
            outage_start = torch.where(so_selected, target_t - so_length, target_t + 1)
            target_outage = so_selected

        # ---- 3. holdout blanking (v4 semantics) ----------------------------------
        spatial_holdout_mask, blank_mask = None, None
        if train:
            if self.kriging_active:
                reroll = torch.rand(N, device=self.device) < self.spatial_holdout_prob
                if self.loss_node_mask is not None:
                    reroll = reroll & self.loss_node_mask
                spatial_holdout_mask = reroll
                blank_mask = reroll
            if self.inductive_mask is not None:
                blank_mask = self.inductive_mask if blank_mask is None else (blank_mask | self.inductive_mask)
        elif self.eval_holdout_mask is not None:
            spatial_holdout_mask = self.eval_holdout_mask
            blank_mask = self.eval_holdout_mask
        if blank_mask is not None:
            x = apply_spatial_holdout(x, blank_mask)

        input_missing = ~in_obs | art
        if so_slots is not None:
            input_missing = input_missing | so_slots
        if blank_mask is not None:
            input_missing = input_missing | blank_mask.view(1, N, 1)

        # ---- 4. covariates ---------------------------------------------------------
        covariates, rl_target_raw = self.covariates.build(time_idx, input_missing, blank_mask, outage_start)

        batch.update({
            "static_features": self.static_features,
            "location_ids": self.location_ids,
            "dynamic_features": x,
            "target_impute": target_impute,
            "node_mask": node_mask,
            "structural_masks": masks,
            "art": art,
            "target_outage": target_outage,
            "input_missing": input_missing,
            "spatial_holdout_mask": spatial_holdout_mask,
            "blank_mask": blank_mask,
            "covariates": covariates,
            "rl_target_raw": rl_target_raw,
            "night_flags": night,
            "so_slots": so_slots,
            "so_selected": so_selected,
        })
        return batch
