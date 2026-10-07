#!/usr/bin/env python3
"""Build the real-data caches behind every figure in the Dstl presentation.

Every figure in ``figures/`` reads a small ``figures/data/<name>.js`` file that
assigns one global (``window.DATA_<NAME>``). Script-tag data (rather than
``fetch`` of JSON) keeps the deck working when opened straight from disk.

Sources, all inside this repository:

  docs/experiments/riot_v5/data/runs.csv          Experiment 3 (RIoT v5, 103 runs)
  artifacts/payload_staging/runs/<run>/            dashboard payloads (forecasts, intervals)
  artifacts/quality_full_v2/                       Experiment 1 (110 streams)
  artifacts/spatial_full/aggregated/               Experiment 2 (54 stations)
  data/traffic_20230108_20260608.parquet           raw Urban Observatory pedestrian counts
  logs/scaling/suite.log                           density-scaling suite (in progress)

Run from the repository root or anywhere else:

  python3 dstl_presentation/scripts/build_figure_data.py
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures" / "data"
PAYLOADS = ROOT / "artifacts" / "payload_staging" / "runs"
RUNS_CSV = ROOT / "docs" / "experiments" / "riot_v5" / "data" / "runs.csv"
QUALITY = ROOT / "artifacts" / "quality_full_v2"
SPATIAL = ROOT / "artifacts" / "spatial_full" / "aggregated"
TRAFFIC = ROOT / "data" / "traffic_20230108_20260608.parquet"
SCALING_LOG = ROOT / "logs" / "scaling" / "suite.log"

GRID_START = pd.Timestamp("2023-01-08")
GRID_END = pd.Timestamp("2026-06-07 23:45")


def write(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    path = OUT / f"{name}.js"
    path.write_text(f"window.DATA_{name.upper()}={body};\n")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KB)")


def clean(x, nd=3):
    """Round floats and turn NaN/inf into None so the JSON stays valid."""
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(x) else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ── Raw pedestrian counts ────────────────────────────────────────────────────


def load_station_series() -> pd.Series:
    """Pedestrian counts summed over directions per station and 15-min slot.

    This is the ``target_dir: all`` view the RIoT pipeline trains on. The raw
    table only contains reported rows, so an absent slot is a missing value.
    """
    table = ds.dataset(TRAFFIC).to_table(
        columns=["dt", "location", "value"], filter=pc.field("veh_class") == "person"
    )
    df = table.to_pandas(categories=["location"])
    n_rows, n_zero = len(df), int((df["value"] == 0).sum())
    df["dt"] = df["dt"].dt.floor("15min")
    s = df.groupby(["location", "dt"], observed=True)["value"].sum()
    s.attrs.update(n_direction_rows=n_rows, n_explicit_zeros=n_zero)
    return s


def build_availability(s: pd.Series, network: dict) -> None:
    """Station x week completeness heat map for the 81 Experiment 3 stations."""
    roles = {n["name"]: n["role"] for n in network["nodes"]}
    weeks = pd.date_range(GRID_START, GRID_END, freq="7D")
    slots_per_week = 7 * 96
    rows = []
    for loc in s.index.get_level_values(0).unique():
        x = s.loc[loc]
        wk = ((x.index - GRID_START) // pd.Timedelta("7D")).astype(int)
        counts = np.bincount(wk, minlength=len(weeks))[: len(weeks)]
        rows.append(
            {
                "name": loc,
                "role": roles.get(loc, "unused"),
                "completeness": round(len(x) / (len(weeks) * slots_per_week), 3),
                "weeks": [round(c / slots_per_week, 2) for c in counts],
            }
        )
    rows.sort(key=lambda r: (-r["completeness"]))
    total_slots = len(pd.date_range(GRID_START, GRID_END, freq="15min")) * len(rows)
    write(
        "availability",
        {
            "weekStarts": [w.strftime("%Y-%m-%d") for w in weeks],
            "stations": rows,
            "summary": {
                "stations": len(rows),
                "slots": total_slots,
                "observedSlots": int(s.size),
                "missingFrac": round(1 - s.size / total_slots, 3),
                "directionRows": s.attrs["n_direction_rows"],
                "explicitZeros": s.attrs["n_explicit_zeros"],
            },
        },
    )


def day_snippet(s: pd.Series, loc: str, start: str, days: int) -> dict:
    """A regular 15-min window with nulls where the sensor did not report."""
    grid = pd.date_range(pd.Timestamp(start), periods=days * 96, freq="15min")
    x = s.loc[loc].reindex(grid)
    return {
        "station": loc,
        "start": grid[0].strftime("%Y-%m-%dT%H:%M"),
        "values": [None if pd.isna(v) else int(v) for v in x.to_numpy()],
    }


def weekly_snippet(s: pd.Series, loc: str) -> dict:
    """Daily totals and daily completeness over a station's full life."""
    x = s.loc[loc]
    days = pd.date_range(x.index.min().normalize(), x.index.max().normalize(), freq="D")
    tot = x.groupby(x.index.normalize()).sum().reindex(days)
    cnt = x.groupby(x.index.normalize()).size().reindex(days).fillna(0)
    return {
        "station": loc,
        "start": days[0].strftime("%Y-%m-%d"),
        "dailyTotal": [None if pd.isna(v) else int(v) for v in tot.to_numpy()],
        "dailyCompleteness": [round(c / 96, 2) for c in cnt.to_numpy()],
    }


def best_window(s: pd.Series, loc: str, score, days: int = 2, months=None) -> str:
    """Pick the start day whose ``days``-day window maximises ``score(values)``."""
    x = s.loc[loc]
    first, last = x.index.min().normalize(), x.index.max().normalize()
    best, best_day = -np.inf, first
    for day in pd.date_range(first, last - pd.Timedelta(days=days), freq="D"):
        if months and day.month not in months:
            continue
        grid = pd.date_range(day, periods=days * 96, freq="15min")
        v = x.reindex(grid).to_numpy(dtype=float)
        sc = score(v, grid)
        if sc > best:
            best, best_day = sc, day
    return best_day.strftime("%Y-%m-%d")


def build_quality_examples(s: pd.Series) -> None:
    """Real snippets illustrating each data-quality failure mode."""

    def short_gaps(v, grid):
        day = (grid.hour >= 8) & (grid.hour < 20)
        miss = np.isnan(v)
        if miss[day].mean() > 0.5 or miss[day].mean() < 0.15:
            return -np.inf
        # many transitions = many short gaps, but the day must still be mostly observed
        return np.abs(np.diff(miss[day].astype(int))).sum()

    def night_only(v, grid):
        night = grid.hour < 5
        day = (grid.hour >= 8) & (grid.hour < 22)
        return np.isnan(v[night]).mean() - 3 * np.isnan(v[day]).mean()

    def fragmented(v, grid):
        miss = np.isnan(v)
        if miss.mean() > 0.75 or miss.mean() < 0.4:
            return -np.inf
        return np.abs(np.diff(miss.astype(int))).sum()

    def surge(v, grid):
        if np.isnan(v).mean() > 0.1:
            return -np.inf
        return np.nanmax(v)

    ex = {}
    loc = "NclTenthAveSecondAve"
    ex["intermittent"] = day_snippet(s, loc, best_window(s, loc, short_gaps, 2), 2)
    loc = "NclShieldsRdRobinsonStNorth"
    ex["nullzero"] = day_snippet(s, loc, best_window(s, loc, night_only, 2), 2)
    loc = "NclNorhamRdCoquetAveANPR"
    ex["fragmentation"] = day_snippet(s, loc, best_window(s, loc, fragmented, 2), 2)
    loc = "NclHeberSt"
    ex["surge"] = day_snippet(s, loc, best_window(s, loc, surge, 2), 2)
    ex["failure"] = weekly_snippet(s, "NclQuaysideUnderHighLevelBridge")
    ex["degraded"] = weekly_snippet(s, "NclStephensonRdHeatonRdNewtonRdSE3")
    # A healthy neighbour for scale on the surge example
    ex["surgeReference"] = {
        "station": "NclHeberSt",
        "medianProfile": weekday_median_profile(s, "NclHeberSt", ex["surge"]["start"]),
    }
    ex["stats"] = gap_stats(s)
    write("quality_examples", ex)


def gap_stats(s: pd.Series) -> dict:
    """Gap structure within each station's active life (first to last report)."""
    lengths, persist_num, persist_den, missing = [], 0, 0, 0
    for loc in s.index.get_level_values(0).unique():
        idx = s.loc[loc].index
        grid = pd.date_range(idx.min(), idx.max(), freq="15min")
        miss = ~np.isin(grid, idx)
        d = np.diff(np.r_[0, miss.astype(int), 0])
        lengths += list(np.where(d == -1)[0] - np.where(d == 1)[0])
        persist_num += int((miss[1:] & miss[:-1]).sum())
        persist_den += int(miss[:-1].sum())
        missing += int(miss.sum())
    L = np.array(lengths)
    return {
        "gapRuns": int(len(L)),
        "medianGapSteps": float(np.median(L)),
        "shareMissingInGapsOver3h": round(float(L[L > 12].sum() / L.sum()), 3),
        "pMissingGivenPrevMissing": round(persist_num / persist_den, 3),
        "activeLifeMissingFrac": round(missing / (missing + int(s.size)), 3),
    }


def weekday_median_profile(s: pd.Series, loc: str, start: str) -> list:
    """Median count per 15-min slot for the same weekdays as the snippet."""
    x = s.loc[loc]
    out = []
    for d in range(2):
        dow = (pd.Timestamp(start) + pd.Timedelta(days=d)).dayofweek
        sel = x[x.index.dayofweek == dow]
        prof = sel.groupby(sel.index.hour * 4 + sel.index.minute // 15).median()
        out += [clean(prof.get(i), 1) for i in range(96)]
    return out


def build_profiles(s: pd.Series, network: dict) -> None:
    """Normalised hour-of-week profiles: how differently each street behaves."""
    names = [n["name"] for n in network["nodes"]]
    rows = []
    for loc in names:
        if loc not in s.index.get_level_values(0):
            continue
        x = s.loc[loc]
        x = x[x.index >= "2024-09-01"]
        if len(x) < 96 * 120:
            continue
        how = x.index.dayofweek * 24 + x.index.hour
        prof = x.groupby(how).mean().reindex(range(168))
        if prof.isna().mean() > 0.05:
            continue
        prof = prof.interpolate(limit_direction="both")
        peak = prof.max()
        rows.append(
            {
                "name": loc,
                "mean": round(float(x.mean()), 1),
                "profile": [round(v / peak, 3) for v in prof.to_numpy()],
                "weekendRatio": round(float(prof[120:].mean() / prof[:120].mean()), 2),
                "peakHour": int(prof.idxmax()),
            }
        )
    rows.sort(key=lambda r: r["weekendRatio"])
    write("profiles", {"since": "2024-09-01", "stations": rows})


# ── Experiment 3 dashboard payloads ──────────────────────────────────────────


def load_json(path: Path):
    return json.loads(path.read_text())


def build_network() -> dict:
    sp = load_json(PAYLOADS / "v5_full-s42" / "spatial.json")
    nodes = [
        {"id": n["id"], "name": n["location_name"], "lat": n["lat"], "lon": n["lon"], "role": n["node_role"]}
        for n in sp["nodes"]
    ]
    edges = [[e["source"], e["target"]] for e in sp["edge_topology"]]
    net = {"nodes": nodes, "edges": edges}
    write("network", net)
    return net


def build_outage_example() -> None:
    """A real 7.5 h daytime outage: V5F vs V5F without unified encoding."""
    day, node = "2026-05-01", "43"
    out = {"date": day, "series": {}}
    for run, key in [("v5_full-s42", "v5f"), ("loo_no_unify-s42", "noUnify"), ("e2_1_unify_iid-s42", "e21")]:
        page = load_json(PAYLOADS / run / day / "page.json")
        s = page["node_series"][node]
        out["series"][key] = {k: s[k] for k in ("point", "lo", "hi", "q50", "q90", "outage_q") if s.get(k) is not None}
        if key == "v5f":
            out["truth"] = s["truth"]
            out["profile"] = s["profile"]
            out["timestamps"] = page["timestamps"]
            meta = load_json(PAYLOADS / run / day / "nodes" / f"node_{node}.json")
            out["station"] = meta["location_name"]
            out["inputMissingFrac"] = meta["input_missing_frac"]
    write("outage_example", out)


def build_outage_awareness() -> None:
    """Outage probability and interval coverage vs fraction of the input window missing."""
    root = PAYLOADS / "v5_full-s42"
    roles = {n["id"]: n["node_role"] for n in load_json(root / "spatial.json")["nodes"]}
    acc = {}
    for day in sorted(p for p in root.iterdir() if p.is_dir()):
        page = load_json(day / "page.json")
        for nid, s in page["node_series"].items():
            if roles[int(nid)] != "observed":
                continue
            mf = load_json(day / "nodes" / f"node_{nid}.json")["input_missing_frac"]
            for t in range(len(mf)):
                b = int(round(mf[t] * 12))
                a = acc.setdefault(b, {"n": 0, "q": 0.0, "nObs": 0, "cov": 0})
                a["n"] += 1
                a["q"] += s["outage_q"][t]
                y = s["truth"][t]
                if y is not None:
                    a["nObs"] += 1
                    a["cov"] += int(s["lo"][t] <= y <= s["hi"][t])
    bins = []
    for b in sorted(acc):
        a = acc[b]
        bins.append(
            {
                "missingSteps": b,
                "n": a["n"],
                "meanOutageQ": round(a["q"] / a["n"], 3),
                "nObserved": a["nObs"],
                "coverage80": round(a["cov"] / a["nObs"], 3) if a["nObs"] else None,
            }
        )
    write("outage_awareness", {"run": "v5_full-s42", "period": "May 2026 (test)", "bins": bins})


# ── Experiment 3 results table ───────────────────────────────────────────────

RUN_COLUMNS = [
    "run", "experiment", "group", "desc", "duration_h", "best_epoch", "epochs_run",
    "test_obs_all_rmse", "test_obs_all_mae", "test_obs_trans_rmse", "test_obs_ind_rmse",
    "test_peak_rmse", "test_collapse_rate", "test_real_gap_mass_ratio", "test_crps",
    "test_q90_coverage", "test_q50_coverage", "test_kriging_raw_rmse", "test_legacy_rmse",
    "test_ind_pred_pairwise_corr_mean", "test_ind_corr_mean",
    *[f"test_so_{L}_forecast_rmse" for L in (1, 4, 12, 48, 96)],
    *[f"test_so_{L}_mass_ratio" for L in (1, 4, 12, 48, 96)],
    *[f"test_so_{L}_collapse" for L in (1, 4, 12, 48, 96)],
    *[f"test_pit_b{i}" for i in range(10)],
    *[f"test_bias_d{i}" for i in range(10)],
    "test_so_12_outage_auc", "test_so_96_outage_auc",
]


def build_v5_runs() -> None:
    with RUNS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    runs = []
    for r in rows:
        rec = {}
        for c in RUN_COLUMNS:
            v = r.get(c, "")
            if c in ("run", "experiment", "group", "desc"):
                rec[c] = v
            else:
                try:
                    rec[c] = clean(float(v), 4)
                except ValueError:
                    rec[c] = None
        runs.append(rec)
    gpu_h = sum(r["duration_h"] or 0 for r in runs)
    write("v5_runs", {"runs": runs, "summary": {"n": len(runs), "gpuHours": round(gpu_h, 1)}})


# ── Experiment 1 and 2 ───────────────────────────────────────────────────────


def build_exp1() -> None:
    r = pd.read_parquet(QUALITY / "analysis" / "results.parquet")
    r = r[(r["ablation"] == "") & (r["interpolation"].isin(["none", "knn"]))]
    models = ["persistence", "snaive_weekly", "arima", "lstm_uv", "lstm_mv", "transformer_mv"]
    r = r[r["model"].isin(models)]
    pts = []
    for row in r.itertuples():
        pts.append(
            {
                "s": row.sensor_id,
                "m": row.model,
                "h": int(row.horizon),
                "i": row.interpolation,
                "c": clean(row.completeness_active),
                "L": clean(row.avg_seq_len_hours),
                "cv": clean(row.cv_rmse_raw_real_only, 1),  # already a percentage
                "rmse": clean(row.rmse_raw_real_only, 2),
            }
        )
    thresholds = {}
    for m in models:
        for h in (8, 24):
            p = QUALITY / "analysis" / f"threshold_cvrmse_{m}_h{h}.json"
            if p.exists():
                thresholds[f"{m}_h{h}"] = {k: clean(v, 4) if not isinstance(v, list) else [clean(x, 3) for x in v]
                                           for k, v in load_json(p).items()}
    rf = load_json(QUALITY / "analysis" / "rf_exploratory_lstm_mv_h8.json")
    corr = pd.read_csv(QUALITY / "analysis" / "correlations_cvrmse_lstm_mv_h8.csv")
    summary = (
        r.groupby(["model", "horizon", "interpolation"])["rmse_raw_real_only"].median().round(2).reset_index()
        .to_dict(orient="records")
    )
    manifest = load_json(QUALITY / "sensor_manifest.json")
    write(
        "exp1",
        {
            "points": pts,
            "thresholds": thresholds,
            "rf": {"oobR2": clean(rf["oob_r2"]), "importance": {k: clean(v["mean"], 4) for k, v in rf["permutation_importance"].items()}},
            "spearman": {row.metric: clean(row.rho) for row in corr.itertuples()},
            "medianRmse": summary,
            "nCandidate": manifest["n_candidate_series"],
            "nViable": manifest["n_viable_series"],
        },
    )


def build_exp2() -> None:
    mc = pd.read_csv(SPATIAL / "model_comparison.csv")
    keep = {
        ("persistence", "identity"): "Persistence",
        ("snaive_weekly", "identity"): "Seasonal naïve (weekly)",
        ("hist_avg", "identity"): "Historical average",
        ("lstm", "identity"): "LSTM",
        ("transformer", "identity"): "Transformer",
        ("gat_plain", "gabriel"): "GAT, no temporal encoder",
        ("gat_transformer", "gabriel"): "GAT + Transformer",
        ("gat_lstm", "gabriel"): "GAT + LSTM",
    }
    tier = {"Persistence": "naive", "Seasonal naïve (weekly)": "naive", "Historical average": "naive",
            "LSTM": "temporal", "Transformer": "temporal"}
    rows = []
    for (m, g), label in keep.items():
        sub = mc[(mc["model"] == m) & (mc["graph_variant"] == g)]
        if sub.empty:
            continue
        rows.append(
            {
                "label": label,
                "tier": tier.get(label, "spatial"),
                "mae": {int(h): clean(v, 3) for h, v in zip(sub["horizon"], sub["mae_raw_median"])},
                "sd": {int(h): clean(v, 3) for h, v in zip(sub["horizon"], sub["mae_raw_median_sd"])},
            }
        )
    write("exp2", {"rows": rows, "stations": 54, "seeds": 3})


# ── Density scaling (in progress) ────────────────────────────────────────────


def build_scaling() -> None:
    text = SCALING_LOG.read_text() if SCALING_LOG.exists() else ""
    naive = {int(n): float(v) for n, v in re.findall(r"Naive donor baseline for N_train=(\d+): ~([\d.]+)", text)}
    done = {
        int(n): {"ind": float(i), "all": float(a)}
        for n, i, a in re.findall(r"scaling_n(\d+)-s42 DONE .*?Inductive RMSE: ([\d.]+), All RMSE: ([\d.]+)", text)
    }
    with RUNS_CSV.open() as f:
        by_exp = {r["experiment"]: r for r in csv.DictReader(f)}
    full = by_exp["e3_3d_profile_inductive_zero"]
    naive_full = by_exp["naive_profile"]
    points = []
    for n in (15, 25, 35, 45, 57):
        p = {"n": n, "naive": naive.get(n), "model": None, "status": "running"}
        if n in done:
            p.update(model=done[n]["ind"], status="done", source="scaling suite")
        if n == 57:
            p.update(model=round(float(full["test_obs_ind_rmse"]), 2), naive=round(float(naive_full["test_obs_ind_rmse"]), 2),
                     status="done", source="v5 suite (E3.3d)")
        points.append(p)
    write("scaling", {"heldOut": 24, "points": points})


def main() -> None:
    network = build_network()
    build_outage_example()
    build_outage_awareness()
    build_v5_runs()
    build_exp1()
    build_exp2()
    build_scaling()
    s = load_station_series()
    build_availability(s, network)
    build_quality_examples(s)
    build_profiles(s, network)


if __name__ == "__main__":
    main()
