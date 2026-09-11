"""
Forecaster factory: config `type` string → concrete class.

Required (scientific) hyperparameters are read with spec["key"] so a missing
key fails loudly; only non-scientific plumbing has defaults. The config key
becomes the run label (forecaster.name = model_key) — class-default names are
placeholders.
"""
from typing import Dict, Optional

from spatial.contract import GraphForecaster
from spatial.models.gat import GATForecaster, LearnedTopologyGATForecaster
from spatial.models.temporal import (
    HistoricalAverageForecaster,
    PerNodeRNNForecaster,
    PerNodeTransformerForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)

KNOWN_TYPES = ("persistence", "seasonal_naive", "historical_average",
               "lstm", "transformer", "gat", "gat_learned")


def build_forecaster(
    model_key: str,
    spec: Dict,
    seed: int = 42,
    device: Optional[str] = None,
    inputs_cfg: Optional[Dict] = None,
) -> GraphForecaster:
    assert "type" in spec, f"CRITICAL: model {model_key!r} spec missing 'type'"
    mtype = spec["type"]
    assert mtype in KNOWN_TYPES, \
        f"CRITICAL: unknown model type {mtype!r} for {model_key!r} " \
        f"(known: {KNOWN_TYPES})"

    if mtype == "persistence":
        forecaster: GraphForecaster = PersistenceForecaster()
    elif mtype == "seasonal_naive":
        forecaster = SeasonalNaiveForecaster(seasonality_days=float(spec["seasonality_days"]))
    elif mtype == "historical_average":
        forecaster = HistoricalAverageForecaster()
    else:
        assert inputs_cfg is not None, \
            f"CRITICAL: trainable model {model_key!r} needs the inputs config"
        if mtype == "lstm":
            forecaster = PerNodeRNNForecaster(spec, seed=seed, device=device,
                                              inputs_cfg=inputs_cfg)
        elif mtype == "transformer":
            forecaster = PerNodeTransformerForecaster(spec, seed=seed, device=device,
                                                      inputs_cfg=inputs_cfg)
        elif mtype == "gat":
            forecaster = GATForecaster(spec, seed=seed, device=device,
                                       inputs_cfg=inputs_cfg)
        else:  # gat_learned
            forecaster = LearnedTopologyGATForecaster(spec, seed=seed, device=device,
                                                      inputs_cfg=inputs_cfg)

    forecaster.name = model_key
    return forecaster
