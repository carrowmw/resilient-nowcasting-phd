"""
Forecaster factory for the quality track — mirrors harness/adapters.py's
build_adapter(): the config names a model spec, the registry constructs it.
Zero hyperparameters live in code; everything comes from configs/quality/*.yaml.
"""
import logging
from typing import Dict, Optional, Sequence

from quality.contract import ALL_FEATURE_GROUPS, SensorForecaster
from quality.forecasters import (
    ARIMAForecaster,
    GRUSensorForecaster,
    LSTMSensorForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from quality.transformer_forecaster import (
    MaskedTransformerSensorForecaster,
    TransformerSensorForecaster,
)

logger = logging.getLogger(__name__)

KNOWN_TYPES = ("persistence", "seasonal_naive", "arima", "lstm", "gru", "transformer", "transformer_masked")


def build_forecaster(
    model_key: str,
    spec: Dict,
    seed: int = 42,
    device: Optional[str] = None,
    feature_groups_override: Optional[Sequence[str]] = None,
) -> SensorForecaster:
    """
    Args:
        model_key: name of the model entry in the config (used for labeling).
        spec: the model's config block; must contain 'type'.
        feature_groups_override: ablation runs substitute the feature set of
            an LSTM spec without redefining the whole model block.
    """
    assert "type" in spec, f"CRITICAL: model spec {model_key!r} has no 'type'"
    mtype = spec["type"]
    assert mtype in KNOWN_TYPES, \
        f"CRITICAL: unknown forecaster type {mtype!r}, expected one of {KNOWN_TYPES}"

    if mtype == "persistence":
        forecaster = PersistenceForecaster()

    elif mtype == "seasonal_naive":
        forecaster = SeasonalNaiveForecaster(
            seasonality_days=float(spec["seasonality_days"]),
        )

    elif mtype == "arima":
        forecaster = ARIMAForecaster(
            order=spec.get("order", (2, 0, 2)),
            seasonal_order=spec.get("seasonal_order", (0, 0, 0, 0)),
            trend=spec.get("trend", "c"),
            min_fit_steps=spec.get("min_fit_steps", 192),
            max_eval_history=spec.get("max_eval_history", 2688),
        )

    elif mtype == "lstm":
        groups = list(feature_groups_override or spec["feature_groups"])
        assert groups and groups[0] == "count", \
            f"CRITICAL: LSTM feature groups must start with 'count', got {groups}"
        unknown = set(groups) - set(ALL_FEATURE_GROUPS)
        assert not unknown, f"CRITICAL: unknown feature groups {unknown}"
        forecaster = LSTMSensorForecaster(
            feature_groups=groups,
            hidden_dim=spec["hidden_dim"],
            num_layers=spec["num_layers"],
            dropout=spec["dropout"],
            lr=spec["lr"],
            batch_size=spec["batch_size"],
            epochs=spec["epochs"],
            patience=spec["patience"],
            scheduler_step_size=spec.get("scheduler_step_size", 5),
            scheduler_gamma=spec.get("scheduler_gamma", 0.7),
            fractional_time=spec.get("fractional_time", True),
            seed=seed,
            device=device,
        )

    elif mtype == "gru":
        groups = list(feature_groups_override or spec["feature_groups"])
        assert groups and groups[0] == "count", \
            f"CRITICAL: GRU feature groups must start with 'count', got {groups}"
        unknown = set(groups) - set(ALL_FEATURE_GROUPS)
        assert not unknown, f"CRITICAL: unknown feature groups {unknown}"
        forecaster = GRUSensorForecaster(
            feature_groups=groups,
            hidden_dim=spec["hidden_dim"],
            num_layers=spec["num_layers"],
            dropout=spec["dropout"],
            lr=spec["lr"],
            batch_size=spec["batch_size"],
            epochs=spec["epochs"],
            patience=spec["patience"],
            scheduler_step_size=spec.get("scheduler_step_size", 5),
            scheduler_gamma=spec.get("scheduler_gamma", 0.7),
            fractional_time=spec.get("fractional_time", True),
            seed=seed,
            device=device,
        )

    elif mtype == "transformer":
        groups = list(feature_groups_override or spec["feature_groups"])
        assert groups and groups[0] == "count", \
            f"CRITICAL: Transformer feature groups must start with 'count', got {groups}"
        unknown = set(groups) - set(ALL_FEATURE_GROUPS)
        assert not unknown, f"CRITICAL: unknown feature groups {unknown}"
        forecaster = TransformerSensorForecaster(
            feature_groups=groups,
            d_model=spec["d_model"],
            nhead=spec["nhead"],
            num_encoder_layers=spec["num_encoder_layers"],
            dim_feedforward=spec["dim_feedforward"],
            dropout=spec["dropout"],
            lr=spec["lr"],
            batch_size=spec["batch_size"],
            epochs=spec["epochs"],
            patience=spec["patience"],
            scheduler_step_size=spec.get("scheduler_step_size", 5),
            scheduler_gamma=spec.get("scheduler_gamma", 0.7),
            fractional_time=spec.get("fractional_time", True),
            seed=seed,
            device=device,
        )

    elif mtype == "transformer_masked":
        groups = list(feature_groups_override or spec["feature_groups"])
        assert groups and groups[0] == "count", \
            f"CRITICAL: MaskedTransformer feature groups must start with 'count', got {groups}"
        unknown = set(groups) - set(ALL_FEATURE_GROUPS)
        assert not unknown, f"CRITICAL: unknown feature groups {unknown}"
        forecaster = MaskedTransformerSensorForecaster(
            feature_groups=groups,
            d_model=spec["d_model"],
            nhead=spec["nhead"],
            num_encoder_layers=spec["num_encoder_layers"],
            dim_feedforward=spec["dim_feedforward"],
            dropout=spec["dropout"],
            lr=spec["lr"],
            batch_size=spec["batch_size"],
            epochs=spec["epochs"],
            patience=spec["patience"],
            scheduler_step_size=spec.get("scheduler_step_size", 5),
            scheduler_gamma=spec.get("scheduler_gamma", 0.7),
            fractional_time=spec.get("fractional_time", True),
            seed=seed,
            device=device,
        )

    # Ablation variants keep the base model_key as name; the runner records
    # the substituted feature set in a dedicated column.
    forecaster.name = model_key
    logger.info(f"Built forecaster {model_key!r} ({mtype}), capabilities={forecaster.capabilities}")
    return forecaster

