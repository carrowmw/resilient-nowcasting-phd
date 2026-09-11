"""Outlier Event Analysis (OEA) for the ResilientIoT spatiotemporal GNN.

A cache-backed, reproducible pipeline that pulls per-node inference payloads
from Cloudflare R2 and training curves from Weights & Biases, then runs the
eight outlier-analysis tests described in
``docs/experiments/outlier_analysis_prompt.md``.

Entry point: ``docs/experiments/outlier_analysis/run_analysis.py``.
"""

__all__ = ["config", "data_io", "loading", "analysis", "plotting", "report"]
