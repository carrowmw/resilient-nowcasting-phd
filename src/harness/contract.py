"""
The harness data contract.

Contract In  — the batch dict produced by dataset collate functions. The harness
               owns holdout blanking and never smuggles hyperparameters (loss
               weights) through the batch; adapters absorb any model-internal
               impedance mismatch.
Contract Out — Predictions / Losses. Absence of a capability is expressed as
               None in Predictions and as a device-placed zero *tensor* in
               Losses (the train loop calls .item() on every component).
"""
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

import torch

# Data keys the harness guarantees on every batch (custom_collate_real).
# spatial_holdout_mask / simulated_nodes_mask / loss_node_mask are optional and
# injected by the harness, never by datasets.
BATCH_DATA_KEYS = (
    "static_features",
    "dynamic_features",
    "target_impute",
    "target_forecast",
    "node_mask",
    "location_ids",
)


@dataclass(frozen=True)
class Capabilities:
    """What a model can genuinely produce. A 'naive' but real attempt (e.g. the
    recurrent baselines' reconstruction head) counts as capable — the point is
    that every reported number is honestly computed, never fabricated."""
    uncertainty: bool = False     # theta (dispersion) is a real estimate
    zero_inflation: bool = False  # pi is a real probability
    imputation: bool = False      # imputed [B, N, T, D] is produced
    kriging: bool = False         # kriging_pred [B, K, T, D] is produced
    graph: bool = False           # edge_index / attention exist
    outage: bool = False          # v5: outage_q is a real P(sensor outage) (outage mixture)


class Predictions(NamedTuple):
    mu: torch.Tensor                          # [B, N] point forecast — required
    theta: Optional[torch.Tensor] = None      # [B, N] ZINB dispersion
    pi: Optional[torch.Tensor] = None         # [B, N] zero-inflation probability
    imputed: Optional[torch.Tensor] = None    # [B, N, T, D] reconstruction
    kriging_pred: Optional[torch.Tensor] = None  # [B, K, T, D] holdout reconstruction
    edge_index: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None
    tripwires: tuple = ()
    # v5 additions (None for models / likelihoods that cannot produce them).
    point: Optional[torch.Tensor] = None      # [B, N] scored point forecast (raw counts)
    p_zero: Optional[torch.Tensor] = None     # [B, N] P(slot reads zero / is empty)
    outage_q: Optional[torch.Tensor] = None   # [B, N] outage probability (outage mixture)
    quantiles: Optional[torch.Tensor] = None  # [B, N, Q] forecast quantiles (raw counts)
    quantile_levels: tuple = ()
    imputed_raw: Optional[torch.Tensor] = None  # [B, N, T] imputed traffic (raw counts)
    nll_params: Optional[Dict] = None         # {"likelihood", "mu", "theta", "pi"} for CRPS / PIT


class Losses(NamedTuple):
    total: torch.Tensor     # backprop target
    impute: torch.Tensor    # zero tensor when unsupported — never a python float
    forecast: torch.Tensor
    kriging: torch.Tensor
    components: Optional[Dict[str, torch.Tensor]] = None   # v5 per-term breakdown
