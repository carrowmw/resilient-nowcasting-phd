"""The eight outlier-analysis tests.

Design: :func:`compute_shared` runs once and produces the artefacts every test
reuses (per-node residual z-scores, outlier flags, and the event catalogue).
Each ``test_N`` function then consumes ``ds`` + ``shared`` and returns a plain
dict of results (JSON-friendly) plus the filenames of any figures it wrote.
"""
from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import config, plotting
from .loading import Dataset, haversine_km

P = plotting.PALETTE


# =========================================================================== #
# Shared artefacts
# =========================================================================== #
@dataclass
class Event:
    event_id: int
    start_idx: int
    end_idx: int
    start_ts: str
    end_ts: str
    date: str
    affected_nodes: list[int]
    peak_frac: float
    classification: str
    severity: float
    peak_residual: float
    peak_node: int
    calendar: str


@dataclass
class Shared:
    flags: dict[int, np.ndarray]          # node_id -> bool[T]  (outlier step?)
    z: dict[int, np.ndarray]              # node_id -> residual z-score[T]
    sigma: dict[int, float]               # node_id -> residual sigma
    n_flagged: np.ndarray                 # [T] count of real nodes flagged
    frac: np.ndarray                      # [T] fraction of real nodes flagged
    events: list[Event] = field(default_factory=list)


def _residual_sigma(x: np.ndarray) -> float:
    """Std-dev of a node's residual distribution (the σ in the prompt's '3σ' rule).

    Residuals are heavy-tailed and heteroscedastic (they scale with traffic
    volume), so a MAD-based scale over-flags the normal diurnal peak.  The plain
    standard deviation is both faithful to the prompt and yields a ~1-3% flag
    rate consistent with genuine outlier events.  A tiny floor guards against
    degenerate all-zero residual series.
    """
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float(np.std(x)) if x.size else np.nan
    s = float(np.std(x))
    return s if s > 1e-6 else 1e-6


def _calendar_label(date: str) -> str:
    d = _dt.datetime.strptime(date, "%Y-%m-%d").date()
    parts = [d.strftime("%A")]
    if d in config.UK_BANK_HOLIDAYS_2026:
        parts.append(config.UK_BANK_HOLIDAYS_2026[d])
    elif d.weekday() >= 5:
        parts.append("weekend")
    return ", ".join(parts)


def compute_shared(ds: Dataset) -> Shared:
    flags: dict[int, np.ndarray] = {}
    zmap: dict[int, np.ndarray] = {}
    sigma: dict[int, float] = {}
    real = ds.real_ids

    for nid in real:
        ns = ds.nodes[nid]
        res = ns.residual
        s = _residual_sigma(res)
        sigma[nid] = s
        z = np.where(np.isfinite(res) & (s > 0), res / s, np.nan)
        zmap[nid] = z
        flags[nid] = np.nan_to_num(z, nan=0.0) > config.RESIDUAL_SIGMA

    T = ds.T
    n_flagged = np.zeros(T, dtype=int)
    n_valid = np.zeros(T, dtype=int)
    for nid in real:
        n_flagged += flags[nid].astype(int)
        n_valid += np.isfinite(ds.nodes[nid].gt).astype(int)
    frac = np.where(n_valid > 0, n_flagged / np.maximum(n_valid, 1), 0.0)

    shared = Shared(flags=flags, z=zmap, sigma=sigma, n_flagged=n_flagged, frac=frac)
    shared.events = _build_events(ds, shared)
    return shared


def _build_events(ds: Dataset, sh: Shared) -> list[Event]:
    """Merge per-timestep flag activity into temporally contiguous network events."""
    active = sh.n_flagged > 0
    spd = config.STEPS_PER_DAY
    tol = config.EVENT_GAP_TOLERANCE

    events: list[Event] = []
    t = 0
    T = ds.T
    while t < T:
        if not active[t]:
            t += 1
            continue
        start = t
        end = t
        gap = 0
        j = t + 1
        while j < T:
            if active[j]:
                end = j
                gap = 0
            else:
                gap += 1
                if gap > tol:
                    break
            j += 1
        # window [start, end]
        window = slice(start, end + 1)
        affected: dict[int, float] = {}
        peak_res = 0.0
        peak_node = -1
        for nid in ds.real_ids:
            fl = sh.flags[nid][window]
            if fl.any():
                zpk = float(np.nanmax(sh.z[nid][window]))
                affected[nid] = zpk
                res = ds.nodes[nid].residual[window]
                rp = float(np.nanmax(res)) if np.isfinite(res).any() else 0.0
                if rp > peak_res:
                    peak_res, peak_node = rp, nid
        peak_frac = float(np.nanmax(sh.frac[window]))
        classification = _classify(ds, list(affected), peak_frac)
        date = str(ds.timestamps[start])[:10]
        ev = Event(
            event_id=len(events),
            start_idx=start, end_idx=end,
            start_ts=str(ds.timestamps[start]),
            end_ts=str(ds.timestamps[end]),
            date=date,
            affected_nodes=sorted(affected, key=lambda k: -affected[k]),
            peak_frac=round(peak_frac, 4),
            classification=classification,
            severity=round(float(sum(affected.values())), 3),
            peak_residual=round(peak_res, 2),
            peak_node=peak_node,
            calendar=_calendar_label(date),
        )
        events.append(ev)
        t = end + 1
    events.sort(key=lambda e: -e.severity)
    for i, e in enumerate(events):
        e.event_id = i
    return events


def _classify(ds: Dataset, affected: list[int], peak_frac: float) -> str:
    if peak_frac >= config.NETWORK_WIDE_FRAC:
        return "network-wide"
    if len(affected) <= 1:
        return "isolated"
    # cluster if spatially contiguous (all pairwise within a small radius of centroid)
    lats = np.array([ds.nodes[n].lat for n in affected])
    lons = np.array([ds.nodes[n].lon for n in affected])
    clat, clon = np.nanmean(lats), np.nanmean(lons)
    dists = [haversine_km(clat, clon, la, lo) for la, lo in zip(lats, lons)]
    if np.nanmax(dists) <= config.SPATIAL_NEIGHBOUR_KM * 2:
        return "cluster"
    return "multi-focal"


# =========================================================================== #
# ZINB distribution helpers
# =========================================================================== #
def zinb_central_interval(mu, theta, pi, level):
    """Return (lo, hi) integer bounds of the central `level` ZINB interval.

    NB parameterisation: mean=mu, var=mu+mu^2/theta -> n=theta, p=theta/(theta+mu).
    Zero-inflation: F(k) = pi + (1-pi) F_nb(k).
    """
    mu = np.asarray(mu, float); theta = np.asarray(theta, float); pi = np.asarray(pi, float)
    p = np.clip(theta / (theta + mu), 1e-9, 1 - 1e-9)
    n = np.clip(theta, 1e-6, None)
    a = (1 - level) / 2.0
    lo_target = np.clip((a - pi) / np.maximum(1 - pi, 1e-9), 0.0, 1.0)
    hi_target = np.clip((1 - a - pi) / np.maximum(1 - pi, 1e-9), 0.0, 1.0)
    lo = stats.nbinom.ppf(lo_target, n, p)
    hi = stats.nbinom.ppf(hi_target, n, p)
    lo = np.where(a <= pi, 0.0, lo)          # zero-inflation absorbs the lower tail
    return lo, hi


def zinb_variance(mu, theta):
    mu = np.asarray(mu, float); theta = np.asarray(theta, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mu + mu ** 2 / theta


# =========================================================================== #
# TEST 1 — Outlier detection & cataloguing
# =========================================================================== #
def test1_catalogue(ds: Dataset, sh: Shared) -> dict:
    events = sh.events
    from collections import Counter
    cls_counts = Counter(e.classification for e in events)
    top = events[: config.TOP_N_EVENTS]

    catalogue = [{
        "rank": i + 1,
        "event_id": e.event_id,
        "start": e.start_ts, "end": e.end_ts,
        "duration_steps": e.end_idx - e.start_idx + 1,
        "n_affected": len(e.affected_nodes),
        "peak_frac": e.peak_frac,
        "classification": e.classification,
        "severity": e.severity,
        "peak_residual": e.peak_residual,
        "peak_node": e.peak_node,
        "peak_node_name": ds.nodes[e.peak_node].location_name if e.peak_node >= 0 else "",
        "calendar": e.calendar,
    } for i, e in enumerate(top)]

    # -- Figure A: network outlier activity over time -----------------------
    fig, ax = plotting.new_fig(11, 3.6)
    ts = ds.timestamps.astype("datetime64[s]").astype(_dt.datetime)
    ax.fill_between(ts, sh.frac * 100, color=P["blue"], alpha=0.5, lw=0)
    ax.axhline(config.NETWORK_WIDE_FRAC * 100, color=P["red"], ls="--", lw=1,
               label=f"network-wide threshold ({int(config.NETWORK_WIDE_FRAC*100)}%)")
    for e in top[:8]:
        ax.axvline(ts[e.start_idx], color=P["orange"], lw=0.8, alpha=0.7)
    ax.set_ylabel("% real nodes flagged")
    ax.set_title("Network-wide outlier activity (share of real nodes > 3σ)")
    ax.legend(loc="upper right", fontsize=8)
    figA = plotting.save(fig, "t1_network_activity.png")

    # -- Figure B: spatial distribution of outlier burden -------------------
    fig, ax = plotting.new_fig(7.2, 6.2)
    counts = {nid: int(sh.flags[nid].sum()) for nid in ds.real_ids}
    lats = [ds.nodes[n].lat for n in ds.real_ids]
    lons = [ds.nodes[n].lon for n in ds.real_ids]
    c = [counts[n] for n in ds.real_ids]
    sizes = [20 + 6 * counts[n] for n in ds.real_ids]
    # simulated nodes as small grey markers for context
    ax.scatter([ds.nodes[n].lon for n in ds.sim_ids],
               [ds.nodes[n].lat for n in ds.sim_ids],
               s=8, color=P["grey"], alpha=0.4, label="simulated")
    sc = ax.scatter(lons, lats, c=c, s=sizes, cmap="inferno", edgecolor="k",
                    linewidth=0.3, label="real")
    fig.colorbar(sc, ax=ax, label="# outlier timesteps")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("Spatial distribution of outlier burden (real nodes)")
    ax.legend(loc="upper right", fontsize=8)
    figB = plotting.save(fig, "t1_spatial_burden.png")

    top_nodes = sorted(counts, key=lambda k: -counts[k])[:5]
    return {
        "n_events": len(events),
        "class_counts": dict(cls_counts),
        "total_flagged_steps": int(sh.n_flagged.sum()),
        "mean_pct_flagged": round(float(np.mean(sh.frac) * 100), 3),
        "max_pct_flagged": round(float(np.max(sh.frac) * 100), 3),
        "catalogue": catalogue,
        "busiest_nodes": [
            {"node": n, "name": ds.nodes[n].location_name, "n_outliers": counts[n]}
            for n in top_nodes
        ],
        "figures": [figA, figB],
    }


# =========================================================================== #
# TEST 2 — ZINB distribution behaviour during outliers
# =========================================================================== #
def test2_zinb(ds: Dataset, sh: Shared) -> dict:
    out_mask, base_mask = [], []
    mu_all, th_all, pi_all, gt_all = [], [], [], []
    for nid in ds.real_ids:
        ns = ds.nodes[nid]
        valid = np.isfinite(ns.gt) & np.isfinite(ns.mu) & np.isfinite(ns.theta) & np.isfinite(ns.pi)
        fl = sh.flags[nid] & valid
        base = (~sh.flags[nid]) & valid
        mu_all.append(ns.mu); th_all.append(ns.theta); pi_all.append(ns.pi); gt_all.append(ns.gt)
        out_mask.append(fl); base_mask.append(base)

    mu = np.concatenate(mu_all); th = np.concatenate(th_all)
    pi = np.concatenate(pi_all); gt = np.concatenate(gt_all)
    fout = np.concatenate(out_mask); fbase = np.concatenate(base_mask)

    var = zinb_variance(mu, th)

    def _stats(mask):
        return {
            "n": int(mask.sum()),
            "mean_variance": _fnum(np.nanmean(var[mask])),
            "mean_pi": _fnum(np.nanmean(pi[mask])),
            "mean_mu": _fnum(np.nanmean(mu[mask])),
            "mean_theta": _fnum(np.nanmean(th[mask])),
        }

    out_s, base_s = _stats(fout), _stats(fbase)

    # Coverage at nominal 50/90/99%
    cov = {}
    for lvl in (0.5, 0.9, 0.99):
        lo, hi = zinb_central_interval(mu, th, pi, lvl)
        inside = (gt >= lo) & (gt <= hi)
        cov[f"{int(lvl*100)}"] = {
            "outlier": _fnum(np.nanmean(inside[fout])),
            "baseline": _fnum(np.nanmean(inside[fbase])),
            "nominal": lvl,
        }

    # -- Figure: coverage bars + variance/pi comparison ---------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    levels = ["50", "90", "99"]
    x = np.arange(len(levels)); w = 0.35
    ax1.bar(x - w/2, [cov[l]["baseline"]*100 for l in levels], w,
            label="baseline", color=P["blue"])
    ax1.bar(x + w/2, [cov[l]["outlier"]*100 for l in levels], w,
            label="outlier", color=P["red"])
    ax1.plot(x, [float(l) for l in levels], "k_", ms=22, mew=2, label="nominal")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{l}%" for l in levels])
    ax1.set_ylabel("empirical coverage (%)")
    ax1.set_title("ZINB interval coverage: outlier vs baseline")
    ax1.legend(fontsize=8)

    # Variance and pi live on very different scales -> separate y-axes.
    ax2b = ax2.twinx()
    ax2.bar([0 - w/2], [base_s["mean_variance"]], w, color=P["blue"], label="baseline")
    ax2.bar([0 + w/2], [out_s["mean_variance"]], w, color=P["red"], label="outlier")
    ax2b.bar([1 - w/2], [base_s["mean_pi"]], w, color=P["blue"])
    ax2b.bar([1 + w/2], [out_s["mean_pi"]], w, color=P["red"])
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["variance (μ+μ²/θ)", "π (zero-infl.)"])
    ax2.set_ylabel("mean ZINB variance"); ax2b.set_ylabel("mean π")
    ax2.set_yscale("log")
    ax2.set_title("ZINB variance & zero-inflation (outlier vs baseline)")
    ax2.legend(fontsize=8, loc="upper left")
    figC = plotting.save(fig, "t2_zinb_behaviour.png")

    return {
        "outlier": out_s, "baseline": base_s,
        "variance_ratio": _fnum(out_s["mean_variance"] / base_s["mean_variance"]
                                if base_s["mean_variance"] else np.nan),
        "pi_ratio": _fnum(out_s["mean_pi"] / base_s["mean_pi"] if base_s["mean_pi"] else np.nan),
        "coverage": cov,
        "figures": [figC],
    }


# =========================================================================== #
# TEST 3 — Outlier signal propagation to simulated nodes
# =========================================================================== #
def test3_propagation(ds: Dataset, sh: Shared) -> dict:
    # Confirm the architectural constraint empirically (out-edges masked).
    date0 = ds.dates[0]
    att0 = ds.attention_lookup(date0)
    sim_out = sum(1 for (s, _t) in att0 if ds.nodes[s].is_simulated)
    sim_in = sum(1 for (_s, t) in att0 if ds.nodes[t].is_simulated)

    # Build incoming-edge map: sim_node -> list of real source neighbours.
    incoming = defaultdict(list)
    for (s, t) in att0:
        if ds.nodes[t].is_simulated and not ds.nodes[s].is_simulated:
            incoming[t].append(s)

    att_shifts, mu_shifts, decay_pts, lag_votes = [], [], [], []
    examples = []

    # Use isolated/cluster events (clean source attribution) among the top events.
    focal_events = [e for e in sh.events if e.classification in ("isolated", "cluster")][:40]
    for e in focal_events:
        date = e.date
        att = ds.attention_lookup(date)
        loc = ds.step_slice(date)
        base_local = np.ones(config.STEPS_PER_DAY, bool)
        ev_lo = e.start_idx - loc.start
        ev_hi = e.end_idx - loc.start
        if ev_lo < 0 or ev_hi >= config.STEPS_PER_DAY:
            continue
        ev_local = np.zeros(config.STEPS_PER_DAY, bool)
        ev_local[max(0, ev_lo): ev_hi + 1] = True
        base_local = ~ev_local

        for src in e.affected_nodes[:3]:                 # top affected real nodes
            src_node = ds.nodes[src]
            for sim in ds.sim_ids:
                key = (src, sim)
                if key not in att:
                    continue
                aseq = att[key]
                a_ev = np.nanmean(aseq[ev_local]) if ev_local.any() else np.nan
                a_base = np.nanmean(aseq[base_local]) if base_local.any() else np.nan
                if np.isfinite(a_ev) and np.isfinite(a_base) and a_base > 0:
                    att_shifts.append((a_ev - a_base) / a_base)
                # prediction response of sim node
                mu_sim = ds.nodes[sim].mu[loc]
                m_ev = np.nanmean(mu_sim[ev_local])
                m_base = np.nanmean(mu_sim[base_local])
                s_base = np.nanstd(mu_sim[base_local])
                if np.isfinite(m_ev) and np.isfinite(m_base) and s_base > 1e-6:
                    dz = (m_ev - m_base) / s_base
                    mu_shifts.append(dz)
                    dist = haversine_km(src_node.lat, src_node.lon,
                                        ds.nodes[sim].lat, ds.nodes[sim].lon)
                    decay_pts.append((dist, abs(dz)))
                    # latency: lag of max cross-corr between src residual and sim mu
                    lag = _best_lag(ds.nodes[src].residual[loc], mu_sim)
                    if lag is not None:
                        lag_votes.append(lag)
        if len(examples) < 1 and incoming:
            examples.append(e.event_id)

    att_shifts = np.array(att_shifts, float)
    mu_shifts = np.array(mu_shifts, float)

    # -- Figure: attention shift dist + spatial decay -----------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if att_shifts.size:
        ax1.hist(np.clip(att_shifts, -1, 3), bins=40, color=P["purple"], alpha=0.8)
        ax1.axvline(np.nanmedian(att_shifts), color=P["red"], ls="--",
                    label=f"median {np.nanmedian(att_shifts):+.2f}")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("relative attention shift (real→sim) during event")
    ax1.set_ylabel("count"); ax1.set_title("Attention re-weighting toward outlier source")
    ax1.legend(fontsize=8)

    if decay_pts:
        d = np.array([p[0] for p in decay_pts]); v = np.array([p[1] for p in decay_pts])
        ax2.scatter(d, v, s=10, alpha=0.4, color=P["green"])
        # binned mean
        bins = np.linspace(0, np.nanpercentile(d, 95), 9)
        idx = np.digitize(d, bins)
        bm = [np.nanmean(v[idx == i]) if np.any(idx == i) else np.nan
              for i in range(1, len(bins))]
        ax2.plot((bins[:-1] + bins[1:]) / 2, bm, "-o", color=P["red"], label="binned mean")
        ax2.set_xlabel("physical distance src→sim (km)")
        ax2.set_ylabel("|Δμ| of sim node (z)")
        ax2.set_title("Spatial decay of propagated response")
        ax2.legend(fontsize=8)
    figD = plotting.save(fig, "t3_propagation.png")

    return {
        "sim_outgoing_edges": sim_out,
        "sim_incoming_edges": sim_in,
        "sim_nodes_with_incoming": len(incoming),
        "mean_incoming_neighbours": _fnum(np.mean([len(v) for v in incoming.values()]) if incoming else 0),
        "n_att_pairs": int(att_shifts.size),
        "attention_shift_median": _fnum(np.nanmedian(att_shifts) if att_shifts.size else np.nan),
        "attention_shift_pos_frac": _fnum(np.mean(att_shifts > 0) if att_shifts.size else np.nan),
        "n_mu_pairs": int(mu_shifts.size),
        "mu_shift_median_abs": _fnum(np.nanmedian(np.abs(mu_shifts)) if mu_shifts.size else np.nan),
        "mu_shift_frac_gt_0p5z": _fnum(np.mean(np.abs(mu_shifts) > 0.5) if mu_shifts.size else np.nan),
        "median_lag_steps": _fnum(np.median(lag_votes) if lag_votes else np.nan),
        "figures": [figD],
    }


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 4):
    """Lag (in steps) maximising corr(a[t], b[t+lag]); None if degenerate."""
    a = np.nan_to_num(a - np.nanmean(a)); b = np.nan_to_num(b - np.nanmean(b))
    if np.allclose(a, 0) or np.allclose(b, 0):
        return None
    best, best_c = 0, -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            c = np.dot(a, b)
        else:
            c = np.dot(a[:-lag], b[lag:])
        if c > best_c:
            best_c, best = c, lag
    return best


# =========================================================================== #
# TEST 4 — Build-up dynamics
# =========================================================================== #
def test4_buildup(ds: Dataset, sh: Shared) -> dict:
    ramp = config.RAMP_UP_HOURS * 4          # steps (15-min)
    post = ramp
    curves = defaultdict(list)               # signal -> list of aligned arrays
    used = 0
    # Major events: network-wide first, then highest severity.
    major = sorted(sh.events, key=lambda e: (e.classification != "network-wide", -e.severity))
    for e in major[:25]:
        nid = e.peak_node
        if nid < 0:
            continue
        onset = e.start_idx
        lo = onset - ramp
        hi = onset + post
        if lo < 0 or hi >= ds.T:
            continue
        ns = ds.nodes[nid]
        seg = slice(lo, hi + 1)
        def _norm(arr):
            a = arr[seg].astype(float)
            base = np.nanmean(a[:ramp])
            return a - base if np.isfinite(base) else a
        curves["mu"].append(_norm(ns.mu))
        curves["theta"].append(_norm(ns.theta))
        curves["pi"].append(_norm(ns.pi))
        imp_res = np.abs(ns.gt - ns.imputed)
        curves["impute_res"].append(_norm(imp_res))
        # attention entropy for that date, aligned
        date = e.date
        ent = ds.attention_entropy(date)
        loc0 = ds.step_slice(date).start
        ent_full = np.full(ds.T, np.nan)
        ent_full[loc0:loc0 + len(ent)] = ent
        curves["att_entropy"].append(_norm(ent_full))
        used += 1

    axis = (np.arange(-ramp, post + 1) * config.STEP_MINUTES) / 60.0   # hours rel. onset

    def _mean(sig):
        arr = np.vstack(curves[sig]) if curves[sig] else np.zeros((1, len(axis)))
        return np.nanmean(arr, axis=0)

    # pre-event ramp slope (are signals rising before onset?)
    def _pre_slope(sig):
        m = _mean(sig)[:ramp + 1]
        if np.isfinite(m).sum() < 3:
            return np.nan
        x = axis[:ramp + 1]
        good = np.isfinite(m)
        return float(np.polyfit(x[good], m[good], 1)[0])

    # -- Figure: event-aligned composite curves -----------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for sig, col, lab in [("mu", P["blue"], "μ (mean)"),
                          ("theta", P["green"], "θ (dispersion)"),
                          ("impute_res", P["orange"], "imputation |resid|"),
                          ("att_entropy", P["purple"], "attention entropy")]:
        m = _mean(sig)
        # scale each to unit range for shared axis
        rng = np.nanmax(np.abs(m)) or 1.0
        ax.plot(axis, m / rng, color=col, label=lab)
    ax.axvline(0, color=P["red"], ls="--", lw=1, label="event onset")
    ax.set_xlabel("hours relative to event onset")
    ax.set_ylabel("baseline-subtracted (normalised)")
    ax.set_title(f"Event-aligned build-up & recovery (n={used} major events)")
    ax.legend(fontsize=8, ncol=2)
    figE = plotting.save(fig, "t4_buildup.png")

    # recovery: steps for mu to fall back below 25% of its peak-abs post-onset
    mu_mean = _mean("mu")
    peak = np.nanmax(np.abs(mu_mean[ramp:]))
    recov = np.nan
    if peak > 0:
        post_curve = np.abs(mu_mean[ramp:])
        below = np.where(post_curve < 0.25 * peak)[0]
        # first index after the peak
        pk = int(np.nanargmax(post_curve))
        after = below[below > pk]
        if after.size:
            recov = float(after[0] * config.STEP_MINUTES / 60.0)

    return {
        "n_events_used": used,
        "pre_event_slopes_per_hour": {
            "mu": _fnum(_pre_slope("mu")),
            "theta": _fnum(_pre_slope("theta")),
            "pi": _fnum(_pre_slope("pi")),
            "impute_res": _fnum(_pre_slope("impute_res")),
            "att_entropy": _fnum(_pre_slope("att_entropy")),
        },
        "mu_recovery_hours": _fnum(recov),
        "figures": [figE],
    }


# =========================================================================== #
# TEST 5 — Kriging performance stratified by outlier exposure
# =========================================================================== #
def test5_kriging(ds: Dataset, sh: Shared) -> dict:
    # Neighbour map: holdout node -> observed neighbours (from edge topology).
    obs = set(ds.observed_ids)
    neigh = defaultdict(set)
    for e in ds.edge_topology:
        s, t = e["source"], e["target"]
        if t in obs and s != t:
            neigh[s].add(t)          # s receives from observed t
        if s in obs and s != t:
            neigh[t].add(s)

    exp_k, norm_k, exp_i, norm_i = [], [], [], []
    for nid in ds.holdout_ids:
        ns = ds.nodes[nid]
        valid = np.isfinite(ns.gt) & np.isfinite(ns.mu)
        # neighbour-outlier exposure at each step
        exposed = np.zeros(ds.T, bool)
        for m in neigh.get(nid, ()):
            if m in sh.flags:
                exposed |= sh.flags[m]
        kr_err = np.abs(ns.gt - ns.mu)
        im_err = np.abs(ns.gt - ns.imputed)
        exp_k.append(kr_err[valid & exposed]); norm_k.append(kr_err[valid & ~exposed])
        exp_i.append(im_err[valid & exposed]); norm_i.append(im_err[valid & ~exposed])

    def _cat(chunks):
        arr = np.concatenate(chunks) if chunks else np.array([])
        arr = arr[np.isfinite(arr)]
        return arr

    ek, nk, ei, ni = map(_cat, (exp_k, norm_k, exp_i, norm_i))

    def _rmse(a):
        return _fnum(np.sqrt(np.mean(a ** 2)) if a.size else np.nan)

    def _ci(a):
        if a.size < 5:
            return [np.nan, np.nan]
        boots = [np.sqrt(np.mean(np.random.choice(a, a.size) ** 2)) for _ in range(300)]
        return [_fnum(np.percentile(boots, 2.5)), _fnum(np.percentile(boots, 97.5))]

    res = {
        "kriging_rmse_normal": _rmse(nk), "kriging_rmse_exposed": _rmse(ek),
        "kriging_rmse_normal_ci": _ci(nk), "kriging_rmse_exposed_ci": _ci(ek),
        "impute_rmse_normal": _rmse(ni), "impute_rmse_exposed": _rmse(ei),
        "n_normal": int(nk.size), "n_exposed": int(ek.size),
        "kriging_degradation_ratio": _fnum(_rmse(ek) / _rmse(nk) if nk.size and np.mean(nk**2) else np.nan),
        "impute_degradation_ratio": _fnum(_rmse(ei) / _rmse(ni) if ni.size and np.mean(ni**2) else np.nan),
    }

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(2); w = 0.35
    kr = [res["kriging_rmse_normal"], res["kriging_rmse_exposed"]]
    im = [res["impute_rmse_normal"], res["impute_rmse_exposed"]]
    ax.bar(x - w/2, kr, w, color=P["blue"], label="kriging head")
    ax.bar(x + w/2, im, w, color=P["orange"], label="imputation head")
    ax.set_xticks(x); ax.set_xticklabels(["normal", "neighbour-outlier"])
    ax.set_ylabel("RMSE (raw counts)")
    ax.set_title("Holdout-node error vs neighbour outlier exposure")
    ax.legend(fontsize=8)
    figF = plotting.save(fig, "t5_kriging.png")
    res["figures"] = [figF]
    return res


# =========================================================================== #
# TEST 6 — Tripwire correlation with outliers
# =========================================================================== #
def test6_tripwire(ds: Dataset, sh: Shared) -> dict:
    from collections import Counter
    by_type = Counter()
    per_node = Counter()
    per_hour = Counter()
    gt_by_type = defaultdict(list)
    frac_by_type = defaultdict(list)   # fraction that coincide w/ a flagged outlier
    night_by_type = defaultdict(list)

    for date in ds.dates:
        loc0 = ds.step_slice(date).start
        for tw in ds.tripwires(date):
            et = tw.get("event_type", "UNKNOWN")
            nid = tw.get("node_id")
            t = tw.get("timestep")
            if nid is None or t is None:
                continue
            gidx = loc0 + t
            by_type[et] += 1
            per_node[(nid, et)] += 1
            hour = (t * config.STEP_MINUTES) // 60
            per_hour[(et, hour)] += 1
            night = hour >= config.NIGHT_START_HOUR or hour < config.NIGHT_END_HOUR
            night_by_type[et].append(night)
            if nid in ds.nodes and 0 <= gidx < ds.T:
                g = ds.nodes[nid].gt[gidx]
                if np.isfinite(g):
                    gt_by_type[et].append(float(g))
                if nid in sh.flags:
                    frac_by_type[et].append(bool(sh.flags[nid][gidx]))

    # concentration: node vs time.  Gini over per-node counts (per type).
    def _gini(counts):
        c = np.sort(np.array(counts, float))
        if c.size == 0 or c.sum() == 0:
            return np.nan
        n = c.size
        return _fnum((2 * np.sum((np.arange(1, n + 1)) * c) / (n * c.sum())) - (n + 1) / n)

    type_summ = {}
    for et in by_type:
        node_counts = [v for (n, e), v in per_node.items() if e == et]
        gts = gt_by_type.get(et, [])
        type_summ[et] = {
            "count": by_type[et],
            "mean_gt": _fnum(np.mean(gts) if gts else np.nan),
            "median_gt": _fnum(np.median(gts) if gts else np.nan),
            "frac_at_night": _fnum(np.mean(night_by_type[et]) if night_by_type[et] else np.nan),
            "frac_coincide_outlier": _fnum(np.mean(frac_by_type[et]) if frac_by_type[et] else np.nan),
            "node_gini": _gini(node_counts),
            "n_distinct_nodes": len(node_counts),
        }

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    types = list(by_type)
    ax1.bar(types, [by_type[t] for t in types], color=[P["red"], P["blue"], P["green"], P["orange"]][:len(types)])
    ax1.set_ylabel("count"); ax1.set_title("Tripwire events by type")
    ax1.tick_params(axis="x", rotation=15)
    # gt distribution by type (box)
    data = [gt_by_type[t] for t in types if gt_by_type[t]]
    labs = [t for t in types if gt_by_type[t]]
    if data:
        ax2.boxplot(data, tick_labels=labs, showfliers=False)
        ax2.set_ylabel("ground-truth count at event")
        ax2.set_title("Traffic level at tripwire (by type)")
        ax2.tick_params(axis="x", rotation=15)
    figG = plotting.save(fig, "t6_tripwire.png")

    return {"by_type": type_summ, "total": int(sum(by_type.values())), "figures": [figG]}


# =========================================================================== #
# TEST 7 — Scale-invariance stress test
# =========================================================================== #
def test7_scale(ds: Dataset, sh: Shared) -> dict:
    means = {nid: ds.nodes[nid].mean_count() for nid in ds.real_ids}
    high = [n for n in means if means[n] and means[n] > config.HIGH_TRAFFIC_MEAN]
    # "active suburban": low-traffic but not a near-dead sensor (mean in [1, LOW)).
    low = [n for n in means if means[n] and 1.0 <= means[n] < config.LOW_TRAFFIC_MEAN]
    high.sort(key=lambda n: -means[n]); low.sort(key=lambda n: means[n])

    def _pick(cands):
        # prefer a node that actually has flagged outliers
        for n in cands:
            if sh.flags[n].sum() > 0:
                return n
        return cands[0] if cands else None

    hi = _pick(high); lo = _pick(low)

    def _profile(nid):
        if nid is None:
            return None
        ns = ds.nodes[nid]
        fl = sh.flags[nid]
        gt, mu = ns.gt, ns.mu
        msle = (np.log1p(np.clip(gt, 0, None)) - np.log1p(np.clip(mu, 0, None))) ** 2
        se = (gt - mu) ** 2
        m = fl & np.isfinite(gt) & np.isfinite(mu)
        return {
            "node": nid, "name": ns.location_name, "mean_count": _fnum(means[nid]),
            "n_outliers": int(fl.sum()),
            "mean_msle_outlier": _fnum(np.nanmean(msle[m]) if m.any() else np.nan),
            "mean_se_outlier": _fnum(np.nanmean(se[m]) if m.any() else np.nan),
            "peak_gt": _fnum(np.nanmax(gt[m]) if m.any() else np.nan),
        }

    hp, lp = _profile(hi), _profile(lo)

    # Kriging equalisation: same idea over holdout nodes (mu = kriging recon).
    hmeans = {nid: ds.nodes[nid].mean_count() for nid in ds.holdout_ids}
    def _krig_profile(pred):
        vals = []
        for nid, mc in hmeans.items():
            if not np.isfinite(mc):
                continue
            ns = ds.nodes[nid]; fl = sh.flags.get(nid, np.zeros(ds.T, bool))
            m = fl & np.isfinite(ns.gt) & np.isfinite(ns.mu)
            if m.sum() < 3:
                continue
            msle = (np.log1p(np.clip(ns.gt, 0, None)) - np.log1p(np.clip(ns.mu, 0, None))) ** 2
            vals.append((mc, float(np.nanmean(msle[m]))))
        return vals
    krig_pts = _krig_profile(None)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if hp and lp:
        x = np.arange(2); w = 0.35
        ax1.bar(x - w/2, [lp["mean_msle_outlier"], hp["mean_msle_outlier"]], w,
                color=P["green"], label="MSLE (scale-inv.)")
        ax1b = ax1.twinx()
        ax1b.bar(x + w/2, [lp["mean_se_outlier"], hp["mean_se_outlier"]], w,
                 color=P["grey"], alpha=0.7, label="raw SE")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"low\n{lp['name'][:14]}", f"high\n{hp['name'][:14]}"])
        ax1.set_ylabel("mean MSLE at outliers"); ax1b.set_ylabel("mean raw SE")
        ax1.set_title("Loss contribution: low vs high-traffic node")
        ax1.legend(loc="upper left", fontsize=8); ax1b.legend(loc="upper right", fontsize=8)
    if krig_pts:
        mc = np.array([p[0] for p in krig_pts]); ms = np.array([p[1] for p in krig_pts])
        ax2.scatter(mc, ms, s=25, color=P["blue"])
        if mc.size > 2:
            r = stats.spearmanr(mc, ms).correlation
            ax2.set_title(f"Kriging MSLE at outliers vs node mean (ρ={r:+.2f})")
        ax2.set_xlabel("holdout node mean count"); ax2.set_ylabel("mean kriging MSLE @ outliers")
    figH = plotting.save(fig, "t7_scale.png")

    ratio = None
    if hp and lp and hp["mean_msle_outlier"] and lp["mean_msle_outlier"]:
        ratio = _fnum(hp["mean_msle_outlier"] / lp["mean_msle_outlier"])
    se_ratio = None
    if hp and lp and hp["mean_se_outlier"] and lp["mean_se_outlier"]:
        se_ratio = _fnum(hp["mean_se_outlier"] / lp["mean_se_outlier"])

    return {
        "high_node": hp, "low_node": lp,
        "msle_ratio_high_over_low": ratio,
        "se_ratio_high_over_low": se_ratio,
        "n_high_candidates": len(high), "n_low_candidates": len(low),
        "figures": [figH],
    }


# =========================================================================== #
# TEST 8 — Two-extremes divergence as outlier indicator
# =========================================================================== #
def test8_two_extremes(ds: Dataset, sh: Shared) -> dict:
    rows = []                        # (role, gap, n_outliers, is_sim)
    for nid, ns in ds.nodes.items():
        for date, te in ns.two_extremes.items():
            fill = te.get("rmse_fill_nan")
            drop = te.get("rmse_drop_nan")
            gap = (fill - drop) if (fill is not None and drop is not None) else np.nan
            loc = ds.step_slice(date)
            n_out = int(sh.flags[nid][loc].sum()) if nid in sh.flags else 0
            rows.append((ns.role, float(gap) if gap is not None else np.nan, n_out))

    roles = ("observed", "holdout", "simulated")
    gap_by_role = {r: [g for (rr, g, _n) in rows if rr == r and np.isfinite(g)] for r in roles}

    all_gaps = np.array([g for (_r, g, _n) in rows if np.isfinite(g)])
    nonzero_frac = float(np.mean(np.abs(all_gaps) > 1e-9)) if all_gaps.size else 0.0
    degenerate = nonzero_frac < 0.001    # fill == drop everywhere -> export is degenerate

    # correlation gap vs #outliers (real nodes with a gt)
    real_rows = [(g, n) for (rr, g, n) in rows if rr in ("observed", "holdout") and np.isfinite(g)]
    corr = None
    if len(real_rows) > 5:
        g = np.array([x[0] for x in real_rows]); n = np.array([x[1] for x in real_rows])
        if np.std(g) > 0 and np.std(n) > 0:
            corr = _fnum(stats.spearmanr(g, n).correlation)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    data = [gap_by_role[r] for r in roles if gap_by_role[r]]
    labs = [r for r in roles if gap_by_role[r]]
    if data:
        ax1.boxplot(data, tick_labels=labs, showfliers=False)
        ax1.axhline(0, color="k", lw=0.8)
        ax1.set_ylabel("gap = rmse_fill − rmse_drop")
        ax1.set_title("Two-extremes gap by node role")
    if real_rows:
        g = np.array([x[0] for x in real_rows]); n = np.array([x[1] for x in real_rows])
        ax2.scatter(n + np.random.uniform(-0.2, 0.2, n.size), g, s=12, alpha=0.4, color=P["purple"])
        ax2.set_xlabel("# outlier steps that day (node)")
        ax2.set_ylabel("two-extremes gap")
        _ct = "n/a" if corr is None else f"{corr:.2f}"
        ax2.set_title(f"Gap vs outlier count (Spearman ρ={_ct})")
    figI = plotting.save(fig, "t8_two_extremes.png")

    return {
        "median_gap_by_role": {r: _fnum(np.median(v) if v else np.nan) for r, v in gap_by_role.items()},
        "mean_gap_by_role": {r: _fnum(np.mean(v) if v else np.nan) for r, v in gap_by_role.items()},
        "spearman_gap_vs_outliers": corr,
        "n_node_days": len(rows),
        "n_real_node_days_with_gap": len(real_rows),
        "nonzero_gap_frac": _fnum(nonzero_frac),
        "degenerate_export": degenerate,
        "figures": [figI],
    }


# --------------------------------------------------------------------------- #
def _fnum(x):
    """Coerce numpy scalars/NaN to plain JSON-friendly floats."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf):
        return None
    return round(xf, 5)
