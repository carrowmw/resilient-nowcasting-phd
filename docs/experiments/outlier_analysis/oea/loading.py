"""Turn cached R2 payloads into tidy, analysis-ready numpy structures.

The core object is :class:`Dataset`: node metadata plus, for every node, a single
contiguous time series spanning all available dates (15-minute cadence).  Real
nodes carry ground truth and full ZINB parameters; simulated nodes carry the
predictions that reached them purely via message passing.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

from . import config, data_io


@dataclass
class NodeSeries:
    node_id: int
    location_name: str
    lat: float
    lon: float
    role: str                     # "observed" | "holdout" | "simulated"
    is_simulated: bool
    semantic_tags: list[str]
    # Time series (length T = n_dates * 96); NaN where a value is unavailable.
    gt: np.ndarray                # ground-truth raw counts (real nodes only)
    mu: np.ndarray                # ZINB mean / point prediction
    theta: np.ndarray            # ZINB dispersion
    pi: np.ndarray               # ZINB zero-inflation probability
    std: np.ndarray              # predictive std dev / uncertainty
    imputed: np.ndarray          # imputation head reconstruction
    is_missing: np.ndarray       # bool: model flagged step as missing
    two_extremes: dict = field(default_factory=dict)   # date -> {mae/rmse drop/fill}

    @property
    def residual(self) -> np.ndarray:
        """Absolute forecast residual |gt - mu| (NaN where gt missing)."""
        return np.abs(self.gt - self.mu)

    @property
    def zinb_variance(self) -> np.ndarray:
        """NB variance mu + mu^2/theta (ignores zero-inflation inflation term)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.mu + (self.mu ** 2) / self.theta

    def mean_count(self) -> float:
        return float(np.nanmean(self.gt)) if np.isfinite(self.gt).any() else np.nan


@dataclass
class Dataset:
    cfg: config.AnalysisConfig
    spatial: dict
    dates: list[str]
    timestamps: np.ndarray                 # datetime64[m], length T
    nodes: dict[int, NodeSeries]
    edge_topology: list[dict]
    _page_cache: dict = field(default_factory=dict, repr=False)

    # ---- id groupings ---------------------------------------------------- #
    @property
    def real_ids(self) -> list[int]:
        return [i for i, n in self.nodes.items() if not n.is_simulated]

    @property
    def observed_ids(self) -> list[int]:
        return [i for i, n in self.nodes.items() if n.role == "observed"]

    @property
    def holdout_ids(self) -> list[int]:
        return [i for i, n in self.nodes.items() if n.role == "holdout"]

    @property
    def sim_ids(self) -> list[int]:
        return [i for i, n in self.nodes.items() if n.is_simulated]

    @property
    def T(self) -> int:
        return len(self.timestamps)

    # ---- lazy per-date page derivatives ---------------------------------- #
    def _page(self, date: str) -> dict | None:
        if date not in self._page_cache:
            self._page_cache[date] = data_io.load_page(self.cfg.run_id, date)
        return self._page_cache[date]

    def attention_lookup(self, date: str) -> dict[tuple[int, int], np.ndarray]:
        """(source, target) -> attention_sequence[96] for one date."""
        page = self._page(date)
        out: dict[tuple[int, int], np.ndarray] = {}
        if not page:
            return out
        for e in page.get("edge_attention", []):
            seq = e.get("attention_sequence")
            if seq is not None:
                out[(e["source"], e["target"])] = np.asarray(seq, dtype=float)
        return out

    def tripwires(self, date: str) -> list[dict]:
        page = self._page(date)
        return page.get("tripwire_audit_trail", []) if page else []

    def attention_entropy(self, date: str) -> np.ndarray:
        page = self._page(date)
        if not page:
            return np.full(config.STEPS_PER_DAY, np.nan)
        return np.asarray(page.get("attention_entropy_series", []), dtype=float)

    def date_index(self, date: str) -> int:
        return self.dates.index(date)

    def step_slice(self, date: str) -> slice:
        i = self.date_index(date)
        return slice(i * config.STEPS_PER_DAY, (i + 1) * config.STEPS_PER_DAY)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def _build_timestamps(dates: list[str]) -> np.ndarray:
    stamps = []
    for d in dates:
        base = _dt.datetime.strptime(d, "%Y-%m-%d")
        for t in range(config.STEPS_PER_DAY):
            stamps.append(base + _dt.timedelta(minutes=config.STEP_MINUTES * t))
    return np.array(stamps, dtype="datetime64[m]")


def _extract_zinb(node_json: dict, key: str, n: int) -> np.ndarray:
    arr = _nan(n)
    for rec in node_json.get("zinb_series", []):
        t = rec.get("t")
        if t is not None and 0 <= t < n and rec.get(key) is not None:
            arr[t] = rec[key]
    return arr


def _extract_missing(node_json: dict, n: int) -> np.ndarray:
    arr = np.zeros(n, dtype=bool)
    for rec in node_json.get("zinb_series", []):
        t = rec.get("t")
        if t is not None and 0 <= t < n:
            arr[t] = bool(rec.get("is_missing", False))
    return arr


def build_dataset(cfg: config.AnalysisConfig) -> Dataset:
    spatial = data_io.load_spatial(cfg.run_id)
    meta = spatial["metadata"]
    dates = list(meta.get("available_dates", []))
    if cfg.max_days:
        dates = dates[: cfg.max_days]
    T = len(dates) * config.STEPS_PER_DAY
    spd = config.STEPS_PER_DAY

    nodes: dict[int, NodeSeries] = {}
    for nrec in spatial["nodes"]:
        nid = nrec["id"]
        nodes[nid] = NodeSeries(
            node_id=nid,
            location_name=nrec.get("location_name", f"node_{nid}"),
            lat=nrec.get("lat", np.nan),
            lon=nrec.get("lon", np.nan),
            role=nrec.get("node_role", "observed"),
            is_simulated=bool(nrec.get("is_simulated", False)),
            semantic_tags=nrec.get("semantic_tags", []),
            gt=_nan(T), mu=_nan(T), theta=_nan(T), pi=_nan(T),
            std=_nan(T), imputed=_nan(T),
            is_missing=np.zeros(T, dtype=bool),
        )

    # Fill from per-node files (real nodes, and simulated if downloaded).
    for di, date in enumerate(dates):
        sl = slice(di * spd, (di + 1) * spd)
        page = data_io.load_page(cfg.run_id, date)
        page_pred = {}
        if page:
            for k, series in page.get("node_timeseries", {}).items():
                arr_mu = _nan(spd); arr_sd = _nan(spd)
                for rec in series:
                    t = rec.get("t")
                    if t is not None and 0 <= t < spd:
                        arr_mu[t] = rec.get("prediction", np.nan)
                        arr_sd[t] = rec.get("uncertainty", np.nan)
                page_pred[int(k)] = (arr_mu, arr_sd)

        for nid, ns in nodes.items():
            nj = data_io.load_node(cfg.run_id, date, nid)
            if nj is None:
                # Fall back to page-level predictions (covers simulated nodes).
                if nid in page_pred:
                    ns.mu[sl], ns.std[sl] = page_pred[nid]
                continue
            gt = _nan(spd)
            for i, v in enumerate(nj.get("ground_truth_series", [])[:spd]):
                gt[i] = np.nan if v is None else v
            imp = _nan(spd)
            for i, v in enumerate(nj.get("imputed_series", [])[:spd]):
                imp[i] = np.nan if v is None else v
            ns.gt[sl] = gt
            ns.imputed[sl] = imp
            ns.mu[sl] = _extract_zinb(nj, "mu", spd)
            ns.theta[sl] = _extract_zinb(nj, "theta", spd)
            ns.pi[sl] = _extract_zinb(nj, "pi", spd)
            ns.std[sl] = _extract_zinb(nj, "std_dev", spd)
            ns.is_missing[sl] = _extract_missing(nj, spd)
            te = nj.get("two_extremes")
            if te:
                ns.two_extremes[date] = te
            # If mu still NaN (older payloads), use page prediction.
            if np.isnan(ns.mu[sl]).all() and nid in page_pred:
                ns.mu[sl], ns.std[sl] = page_pred[nid]

    return Dataset(
        cfg=cfg,
        spatial=spatial,
        dates=dates,
        timestamps=_build_timestamps(dates),
        nodes=nodes,
        edge_topology=spatial.get("edge_topology", []),
    )


# --------------------------------------------------------------------------- #
# Geospatial helper
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))
