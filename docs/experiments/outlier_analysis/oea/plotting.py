"""Shared plotting style + small helpers.  All figures render head-less (Agg)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

# Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#111111",
}
SEQ = [PALETTE[c] for c in ("blue", "orange", "green", "red", "purple", "sky")]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})


def new_fig(w: float = 8.0, h: float = 4.5):
    return plt.subplots(figsize=(w, h))


def save(fig, name: str) -> str:
    """Save ``fig`` to the figures dir; return the filename (for the report)."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return name


def rel_fig(name: str) -> str:
    """Path used inside the markdown report (report lives in OUTPUT_DIR)."""
    return f"figures/{name}"
