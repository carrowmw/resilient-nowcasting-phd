"""Static configuration for the outlier-event analysis.

Everything that is a knob or a piece of domain knowledge lives here so the rest
of the package stays free of magic numbers.
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PKG_DIR.parent                      # docs/experiments/outlier_analysis
CACHE_DIR = EXPERIMENT_DIR / "cache"                 # raw R2 / W&B payloads (gitignored)
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "report.md"

# --------------------------------------------------------------------------- #
# Data-source coordinates
# --------------------------------------------------------------------------- #
R2_PUBLIC_URL = os.environ.get(
    "R2_PUBLIC_URL", "https://pub-b102fa91750348f78a6385248654688a.r2.dev"
).rstrip("/")

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "rappel-velvet9310-newcastle-university")
WANDB_PROJECT = "resilient-iot-v3"
WANDB_PROJECT_LEGACY = "resilient-iot-v2"

DEFAULT_RUN_ID = "logical-surf-18"   # overridden by registry.default_run_id at runtime

# --------------------------------------------------------------------------- #
# Signal geometry
# --------------------------------------------------------------------------- #
STEPS_PER_DAY = 96          # 15-minute cadence
STEP_MINUTES = 15
ZINB_EPS = 1e-6

# Night window (local) used to reason about structural zeros / MNAR boundaries.
NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 5

# --------------------------------------------------------------------------- #
# Test thresholds
# --------------------------------------------------------------------------- #
RESIDUAL_SIGMA = 3.0        # flag |gt-mu| > k*sigma of the node residual distribution
MIN_EVENT_STEPS = 1         # min consecutive flagged steps to count as a temporal event
EVENT_GAP_TOLERANCE = 1     # merge flagged steps separated by <= this many normal steps
NETWORK_WIDE_FRAC = 0.30    # >= this fraction of real nodes affected -> "network-wide"
SPATIAL_NEIGHBOUR_KM = 1.0  # nodes within this distance count as a spatial cluster
TOP_N_EVENTS = 20

HIGH_TRAFFIC_MEAN = 50.0    # ch0 mean count/interval for "city-centre" sensor (Test 7)
LOW_TRAFFIC_MEAN = 5.0      # ch0 mean count/interval for "suburban" sensor (Test 7)

RAMP_UP_HOURS = 6           # pre-event window for build-up dynamics (Test 4)


# --------------------------------------------------------------------------- #
# Known Newcastle calendar (deterministic features for the May-2026 window).
#
# Real football fixtures / concerts for a future window are not knowable here, so
# we cross-reference the deterministic calendar (UK bank holidays, weekends) and
# leave fixture confirmation as a manual follow-up.  These are the facts we *can*
# assert for 2026.
# --------------------------------------------------------------------------- #
UK_BANK_HOLIDAYS_2026 = {
    _dt.date(2026, 5, 4): "Early May bank holiday",
    _dt.date(2026, 5, 25): "Spring bank holiday",
}


@dataclass
class AnalysisConfig:
    """Runtime configuration resolved from CLI args."""

    run_id: str = DEFAULT_RUN_ID
    wandb_run_id: str | None = None
    max_days: int | None = None            # limit number of dates (None = all available)
    include_simulated_nodes: bool = True   # download simulated node files too (heavy)
    max_workers: int = 24
    refresh: bool = False                  # ignore cache and re-download
    skip_wandb: bool = False
    dates: list[str] = field(default_factory=list)   # populated from spatial metadata

    @property
    def run_cache_dir(self) -> Path:
        return CACHE_DIR / self.run_id


def ensure_dirs() -> None:
    for d in (CACHE_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
