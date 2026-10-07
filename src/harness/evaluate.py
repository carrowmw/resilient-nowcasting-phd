"""
Capability-gated validation and metrics.

One evaluation path used verbatim by the val loop and the exporters. Each metric
is computed only if the adapter's Capabilities claim it; otherwise it is None
(serialized as null, rendered as N/A, absent from W&B) — never 0.0.
"""
from typing import Dict, Optional

import torch

from harness.contract import Capabilities, Predictions
from metrics import calculate_kriging_metrics


def validate_predictions(caps: Capabilities, preds: Predictions) -> None:
    """Capability-gated tripwires. Replaces the exporter-inline ZINB assertions
    (which crash on any model that legitimately lacks a capability)."""
    assert preds.mu is not None, "CRITICAL: mu (point forecast) is required of every model"
    assert not torch.isnan(preds.mu).any(), "CRITICAL: NaN in mu"

    if caps.uncertainty:
        assert preds.theta is not None, "CRITICAL: uncertainty claimed but theta is None"
        assert (preds.theta > 0).all(), f"ZINB tripwire — theta non-positive: {preds.theta.min()}"
        assert not torch.isnan(preds.theta).any(), "CRITICAL: NaN in theta"
        assert (preds.mu > 0).all(), f"ZINB tripwire — mu non-positive: {preds.mu.min()}"
    else:
        assert preds.theta is None, "CRITICAL: theta supplied without the uncertainty capability"

    if caps.zero_inflation:
        assert preds.pi is not None, "CRITICAL: zero_inflation claimed but pi is None"
        assert (preds.pi > 0).all(), f"TRIPWIRE — pi has zero/negative: {preds.pi.min()}"
        assert (preds.pi < 1).all(), f"TRIPWIRE — pi saturated to 1: {preds.pi.max()}"
        assert not torch.isnan(preds.pi).any(), "CRITICAL: NaN in pi"
    else:
        assert preds.pi is None, "CRITICAL: pi supplied without the zero_inflation capability"

    if caps.imputation:
        assert preds.imputed is not None, "CRITICAL: imputation claimed but imputed is None"
        assert not torch.isnan(preds.imputed).any(), "CRITICAL: NaN in imputed"

    # Graph outputs are per-call optional even for graph-capable models (the RIoT
    # training-mode forward skips them; only the analysis/predict path emits them).
    if not caps.graph:
        assert preds.edge_index is None and preds.attention is None, \
            "CRITICAL: graph tensors supplied without the graph capability"


def evaluate_batch(
    caps: Capabilities,
    preds: Predictions,
    batch: dict,
    holdout_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[float]]:
    """Per-batch metrics. None means 'this model cannot produce this metric'."""
    validate_predictions(caps, preds)

    out: Dict[str, Optional[float]] = {}

    # Forecast RMSE / MAE — point prediction evaluated as E[Y] = mu * (1 - pi) (F-C06 fix)
    target_forecast = batch["target_forecast"]
    if preds.point is not None:
        # v5: the likelihood-specific scored point (likelihoods.point_forecast).
        point_pred = preds.point
    elif caps.zero_inflation and preds.pi is not None:
        point_pred = preds.mu * (1.0 - preds.pi)
    else:
        point_pred = preds.mu

    out["rmse"] = torch.sqrt(torch.mean((target_forecast - point_pred) ** 2)).item()
    out["mae"] = torch.mean(torch.abs(target_forecast - point_pred)).item()
    out["rmse_active"] = torch.sqrt(torch.mean((target_forecast - preds.mu) ** 2)).item()

    # Zero-accuracy counts — gated on a real zero-inflation head.
    if caps.zero_inflation:
        zero_idx = target_forecast == 0
        out["zero_true_count"] = int(zero_idx.sum().item())
        out["zero_correct_count"] = (
            int((preds.pi[zero_idx] > 0.5).sum().item()) if out["zero_true_count"] > 0 else 0
        )
    else:
        out["zero_true_count"] = None
        out["zero_correct_count"] = None

    # Imputation MSE on the artificially masked entries.
    if caps.imputation:
        node_mask = batch["node_mask"]
        if node_mask.any():
            diff_sq = (preds.imputed - batch["target_impute"]) ** 2
            out["impute_mse"] = torch.mean(diff_sq[node_mask]).item()
        else:
            out["impute_mse"] = 0.0
    else:
        out["impute_mse"] = None

    # Kriging reconstruction metrics on held-out nodes.
    if caps.kriging and preds.kriging_pred is not None and holdout_mask is not None:
        kriging_target = batch["target_impute"][:, holdout_mask, :, :]
        out.update(calculate_kriging_metrics(preds.kriging_pred, kriging_target))
    else:
        out["kriging_msle"] = None
        out["kriging_rmse"] = None
        out["kriging_mae"] = None

    return out


class MetricAccumulator:
    """Accumulates per-batch metric dicts and averages them, preserving Nones.

    Count-type keys (zero_true_count / zero_correct_count) are summed and folded
    into a single zero_acc ratio to avoid batch-averaging bias.
    """

    _COUNT_KEYS = ("zero_true_count", "zero_correct_count")

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._batches = 0

    def update(self, batch_metrics: Dict[str, Optional[float]]) -> None:
        self._batches += 1
        for k, v in batch_metrics.items():
            if v is None:
                self._sums.setdefault(k, None)
                assert self._sums[k] is None, \
                    f"CRITICAL: metric '{k}' switched between None and numeric mid-run"
            else:
                assert self._sums.get(k, 0.0) is not None, \
                    f"CRITICAL: metric '{k}' switched between None and numeric mid-run"
                self._sums[k] = self._sums.get(k, 0.0) + v

    def summary(self) -> Dict[str, Optional[float]]:
        assert self._batches > 0, "CRITICAL: MetricAccumulator.summary() before any update()"
        out: Dict[str, Optional[float]] = {}
        for k, total in self._sums.items():
            if k in self._COUNT_KEYS:
                continue
            out[k] = None if total is None else total / self._batches

        zero_true = self._sums.get("zero_true_count")
        zero_correct = self._sums.get("zero_correct_count")
        if zero_true is None:
            out["zero_acc"] = None
            out["zero_true_count"] = None
            out["zero_correct_count"] = None
        else:
            out["zero_acc"] = (zero_correct / zero_true) if zero_true > 0 else 0.0
            out["zero_true_count"] = int(zero_true)
            out["zero_correct_count"] = int(zero_correct)
        return out
