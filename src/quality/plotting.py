"""
Shared figure styling and paper figures for the quality track.

Reviewer requirement: "increase the font size and improve figure readability".
Every figure in the track goes through apply_paper_style() so readability is
fixed globally, not per figure.
"""
import logging
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PAPER_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


def plot_error_vs_quality(
    results: pd.DataFrame,
    quality_col: str,
    error_col: str,
    out_path: str,
    segmented_fit=None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Per-sensor scatter of error vs a quality metric, optionally overlaying the
    broken-stick fit and its breakpoint CI (stats.SegmentedFit).
    """
    apply_paper_style()
    assert quality_col in results.columns and error_col in results.columns, \
        f"CRITICAL: missing columns {quality_col!r}/{error_col!r}"
    x = results[quality_col].to_numpy(dtype=float)
    y = results[error_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=42, alpha=0.75, edgecolor="k", linewidth=0.4)

    if segmented_fit is not None:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ys = (
            segmented_fit.intercept
            + segmented_fit.slope_below * xs
            + (segmented_fit.slope_above - segmented_fit.slope_below)
            * np.maximum(0.0, xs - segmented_fit.breakpoint)
        )
        ax.plot(xs, ys, color="crimson", lw=2, label="segmented fit")
        ax.axvline(segmented_fit.breakpoint, color="crimson", ls="--", lw=1.5,
                   label=f"breakpoint = {segmented_fit.breakpoint:.1f}")
        ax.axvspan(*segmented_fit.breakpoint_ci, color="crimson", alpha=0.12,
                   label="breakpoint 95% CI")
        ax.legend()

    ax.set_xlabel(xlabel or quality_col)
    ax.set_ylabel(ylabel or error_col)
    if title:
        ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Figure saved: {out_path}")


def plot_periodogram(
    period_hours: np.ndarray,
    power: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
    mark_periods: tuple = (12.0, 24.0, 168.0),
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(period_hours, power, lw=1.2)
    for p in mark_periods:
        if period_hours.min() <= p <= period_hours.max():
            ax.axvline(p, color="gray", ls=":", lw=1)
            ax.annotate(f"{p:g} h", (p, ax.get_ylim()[1] * 0.9), fontsize=11, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("Normalized LS power")
    if title:
        ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Figure saved: {out_path}")


def plot_importances(importances: Dict[str, Dict[str, float]], out_path: str,
                     title: str = "Permutation importance (exploratory)") -> None:
    apply_paper_style()
    names = list(importances.keys())
    means = np.array([importances[k]["mean"] for k in names])
    stds = np.array([importances[k]["std"] for k in names])
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 2))
    ax.barh(np.array(names)[order], means[order], xerr=stds[order], color="steelblue")
    ax.set_xlabel("Δ R² when permuted")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Figure saved: {out_path}")
