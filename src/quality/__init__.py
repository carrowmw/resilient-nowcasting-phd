"""
Quality-aware per-sensor forecasting track (MDPI Sensors paper rebuild).

This package is deliberately decoupled from the multi-node GNN harness
(src/harness): the research unit here is a single directional sensor series,
not a [B, N, T, D] node batch. It reuses the harness *philosophy* —
capability-gated forecasters, offensive asserts, config-driven runs, and
reproducible manifests — via its own lightweight contract.
"""
from quality.contract import QualityCapabilities, PerSensorSplit, SensorContext, FitReport
from quality.loader import SensorSeriesStore, sensor_series_id
from quality.quality_metrics import (
    SequenceSpan,
    find_consecutive_sequences,
    compute_completeness,
    compute_sequence_stats,
    count_usable_windows,
    build_quality_table,
    build_sensor_manifest,
)

__all__ = [
    "QualityCapabilities",
    "PerSensorSplit",
    "SensorContext",
    "FitReport",
    "SensorSeriesStore",
    "sensor_series_id",
    "SequenceSpan",
    "find_consecutive_sequences",
    "compute_completeness",
    "compute_sequence_stats",
    "count_usable_windows",
    "build_quality_table",
    "build_sensor_manifest",
]
