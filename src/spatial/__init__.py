"""
Spatial dependence track (thesis Chapter 6 / Experiment 2 / RO2).

A self-contained experiment package answering: does explicitly modelling
inter-sensor spatial dependencies via GATv2 improve short-term pedestrian-count
forecasting over purely temporal baselines — and by how much?

Deliberately decoupled from both src/harness (the full RIoT pipeline, Exp 3)
and src/quality (the per-sensor track, Exp 1). It borrows their philosophy —
capability-gated forecasters, offensive asserts, config-driven runs,
reproducible manifests — but defines its own contract for the multi-node
graph-batch research unit [B, N, T, D].

Deviation register vs the legacy GAT notebooks (docs/context/GAT/, context
only, no code reused):
  - Loss held constant (masked MSE, scaled space) across ALL models in the
    primary comparison; the legacy work compared GAT+ZINB against MSE
    baselines, confounding loss with architecture.
  - Seeds are the replication unit (one network = one fit; the quality track's
    ~100 per-sensor fits provided replication this track does not have).
  - Targets are never imputed: losses and metrics mask to observed targets.
  - Graph depth (gat_layers) is a real ablation axis; the legacy encoder used
    a single GATv2 layer regardless of its num_layers setting.
  - The identity graph (self-loops only) is the matched-capacity control that
    isolates message passing at equal parameter count.

Permitted external imports: quality.stats (pure statistical functions),
graph_learner.DynamicGraphLearner (learned-topology module), features
(STATIC_FEATURE_COLS).
"""
from spatial.contract import (
    FitReport,
    GraphContext,
    GraphForecaster,
    GraphSpec,
    GraphSplit,
    SpatialCapabilities,
)

__all__ = [
    "FitReport",
    "GraphContext",
    "GraphForecaster",
    "GraphSpec",
    "GraphSplit",
    "SpatialCapabilities",
]
