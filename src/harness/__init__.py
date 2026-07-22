"""
Generic evaluation harness shared by train.py (train/val loops) and the payload
exporters. Models participate through ModelAdapter wrappers that emit the
Predictions/Losses contract; metrics are capability-gated so a model that cannot
produce a quantity reports null, never a fabricated number.
"""
from harness.adapters import ModelAdapter, RIoTAdapter, RecurrentBaselineAdapter, build_adapter
from harness.contract import Capabilities, Predictions, Losses, BATCH_DATA_KEYS
from harness.evaluate import validate_predictions, MetricAccumulator, evaluate_batch
from harness.holdout import (
    make_fixed_holdout_mask,
    save_holdout_manifest,
    load_holdout_manifest,
    mask_from_manifest,
    manifest_sha,
    save_results_manifest,
)

__all__ = [
    "ModelAdapter",
    "RIoTAdapter",
    "RecurrentBaselineAdapter",
    "build_adapter",
    "Capabilities",
    "Predictions",
    "Losses",
    "BATCH_DATA_KEYS",
    "validate_predictions",
    "MetricAccumulator",
    "evaluate_batch",
    "make_fixed_holdout_mask",
    "save_holdout_manifest",
    "load_holdout_manifest",
    "mask_from_manifest",
    "manifest_sha",
    "save_results_manifest",
]
