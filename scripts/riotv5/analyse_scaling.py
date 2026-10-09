#!/usr/bin/env python3
"""
Analyse the sensor-density scaling sweep (scripts/riotv5/generate_scaling_configs.py).

For each arm (GNN = e3_3d stack, naive = seasonal-profile donor baseline) fits

    RMSE_ind(N) = A * N^(-alpha) + RMSE_inf

to the held-out-station test RMSE (v5_test_metrics.obs_ind_rmse, 24 frozen stations)
as a function of N = number of training stations, and locates the crossover N* above
which the GNN beats the naive heuristic.

Uncertainty: parametric bootstrap. Each arm's noise sd is the pooled within-N sd of
replicate runs (extra --seeds / --draws) when they exist, otherwise the fit residual
sd (n - 3 dof — with 5 single-seed points that is 2 dof, so treat the band as rough).
--gnn-noise-sd overrides the GNN value, e.g. with a seed sd measured elsewhere.

  N*      point-fit crossover: smallest N from which the GNN curve stays below naive
  N*_95   "definitive" crossover: smallest N from which P_boot(GNN < naive) >= 0.95

N is also reported as sensor density (training stations per km^2 of the convex hull
of all 81 stations) and as the mean distance from a held-out station to its nearest
training station.

Inputs:   <logs_dir>/status/*.json -> results.json + holdout_manifest.json per run
          data/traffic_sensors.json (station coordinates)
Outputs:  docs/experiments/riot_v5_scaling/
            data/scaling_runs.csv      one row per run
            data/scaling_fit.json      fits, bootstrap summaries, crossover
            figures/scaling_crossover.png
            scaling_summary.md

Usage:
  python scripts/riotv5/analyse_scaling.py
  python scripts/riotv5/analyse_scaling.py --mock             # logs/v5_scaling_mock (pipeline check)
  python scripts/riotv5/analyse_scaling.py --gnn-noise-sd 1.5
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "experiments" / "riot_v5_scaling"
SENSORS = ROOT / "data" / "traffic_sensors.json"
METRIC = "obs_ind_rmse"
# v5 suite runs these sweep points replicate at N = 57 (identical configs).
V5_ANCHORS = {"gnn": ("e3_3d_profile_inductive_zero-s42", 39.0927), "naive": ("naive_profile-s42", 42.6336)}
N_GRID = np.unique(np.round(np.geomspace(5, 400, 600)).astype(int)).astype(float)
ALPHA_MAX = 4.0

# Same validated categorical slots as plot_v5_results.py.
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, INK, INK2 = "#9a9893", "#1f1f1f", "#5c5b57"
ARM_STYLE = {"gnn": (BLUE, "GNN (e3_3d, zero-shot)"), "naive": (ORANGE, "Naive profile donors")}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 2, "savefig.dpi": 160, "figure.dpi": 160,
    "text.color": INK,
})


# --------------------------------------------------------------------------- #
# Loading                                                                       #
# --------------------------------------------------------------------------- #
def load_runs(logs_dir: Path) -> pd.DataFrame:
    rows, held_sets = [], {}
    for f in sorted((logs_dir / "status").glob("*.json")):
        st = json.loads(f.read_text())
        if st["status"] != "COMPLETED" or not st["name"].startswith("scale_"):
            continue
        res_path = Path(st["results"])
        res_path = res_path if res_path.is_absolute() else ROOT / res_path
        res = json.loads(res_path.read_text())
        man = json.loads((res_path.parent / "holdout_manifest.json").read_text())
        arm, n_tag = st["experiment"].split("_")[1:3]
        n_train = res["train_node_count"]
        assert n_train == int(n_tag[1:]) == len(man["train_location_ids"]), \
            f"{st['name']}: station count mismatch (results {n_train}, name {n_tag})"
        held_sets[st["name"]] = tuple(sorted(man["holdout_location_ids"]))
        draw = int(st["experiment"].split("_d")[1]) if "_d" in st["experiment"] else 0
        rows.append(dict(
            run=st["name"], arm=arm, n_train=n_train, seed=st["seed"], draw=draw,
            rmse=res["v5_test_metrics"][METRIC],
            trans_rmse=res["v5_test_metrics"]["obs_trans_rmse"],
            nrmse_median_ind=res["v5_test_metrics"]["node_nrmse_median_ind"],
            best_epoch=res["best_epoch"], duration_h=st["duration_sec"] / 3600.0, mock=bool(st.get("mock")),
            train_ids=man["train_location_ids"], holdout_ids=man["holdout_location_ids"],
        ))
    assert rows, f"no completed scale_* runs under {logs_dir}/status"
    distinct = set(held_sets.values())
    assert len(distinct) == 1, (
        "CRITICAL: runs were scored on different held-out stations — the sweep is not comparable: "
        + ", ".join(f"{k}: {len(v)}" for k, v in held_sets.items())
    )
    return pd.DataFrame(rows).sort_values(["arm", "n_train", "draw", "seed"]).reset_index(drop=True)


def check_reference_holdout(runs: pd.DataFrame) -> str:
    """The sweep must use the same 24 stations as the v5 suite."""
    ref = ROOT / "models" / "resilient-iot-v5" / V5_ANCHORS["gnn"][0] / "holdout_manifest.json"
    if runs.mock.any():
        return "mock data (synthetic stations; not checked against the v5 suite)"
    if not ref.exists():
        return "v5 reference manifest not available locally (not checked)"
    ref_ids = sorted(json.loads(ref.read_text())["holdout_location_ids"])
    assert sorted(runs.holdout_ids.iloc[0]) == ref_ids, "CRITICAL: sweep held-out stations differ from the v5 suite's"
    return "identical to the v5 suite (seed 42)"


# --------------------------------------------------------------------------- #
# Geography: density and nearest-sensor distance                                #
# --------------------------------------------------------------------------- #
def station_coords() -> dict:
    return {e["location"]: (e["lat"], e["lon"]) for e in json.loads(SENSORS.read_text())}


def _project_km(latlon: np.ndarray, lat0: float) -> np.ndarray:
    return np.stack([np.radians(latlon[:, 1]) * 6371.0 * math.cos(math.radians(lat0)),
                     np.radians(latlon[:, 0]) * 6371.0], axis=1)


def add_geography(runs: pd.DataFrame):
    """Returns (runs with nearest_km column, hull area km^2) or (runs, None) when the
    stations have no coordinates (mock city)."""
    coords = station_coords()
    every = sorted(set(runs.holdout_ids.iloc[0]) | set().union(*runs.train_ids.map(set)))
    if not all(loc in coords for loc in every):
        return runs.assign(nearest_km=np.nan), None
    lat0 = float(np.mean([coords[l][0] for l in every]))
    xy = {l: _project_km(np.array([coords[l]]), lat0)[0] for l in every}
    area = ConvexHull(np.array([xy[l] for l in every])).volume  # 2-D "volume" = area
    nearest = []
    for _, r in runs.iterrows():
        train = np.array([xy[l] for l in r.train_ids])
        held = np.array([xy[l] for l in r.holdout_ids])
        nearest.append(float(np.linalg.norm(held[:, None] - train[None], axis=-1).min(axis=1).mean()))
    return runs.assign(nearest_km=nearest), float(area)


# --------------------------------------------------------------------------- #
# Power-law fits                                                                #
# --------------------------------------------------------------------------- #
def power_law(n, a, alpha, r_inf):
    return a * np.power(n, -alpha) + r_inf


def fit_arm(n: np.ndarray, y: np.ndarray, r_inf_bounds=(35.0, 44.0), balance_n: bool = True):
    """Least-squares fit with A >= 0, 0 <= alpha <= ALPHA_MAX, r_inf bounded.
    Weights are balanced so each N has equal aggregate weight."""
    if len(np.unique(n)) < 4:
        return None
    r0 = float(np.clip(y.min() * 0.95, r_inf_bounds[0], r_inf_bounds[1]))
    a0 = max(float(y.max() - r0), 1e-3) * float(n.min()) ** 0.5
    sigma = None
    if balance_n:
        counts = {val: float((n == val).sum()) for val in np.unique(n)}
        sigma = np.array([np.sqrt(counts[val]) for val in n], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            p, _ = curve_fit(power_law, n, y, sigma=sigma, absolute_sigma=False,
                             p0=[a0, 0.5, r0],
                             bounds=([0.0, 0.0, r_inf_bounds[0]], [np.inf, ALPHA_MAX, r_inf_bounds[1]]),
                             maxfev=20000)
        except RuntimeError:
            return None
    return p


def noise_sd(n: np.ndarray, y: np.ndarray, params) -> tuple:
    """(sd, source). Pooled within-N replicate sd if any N has >= 2 runs, else residual sd."""
    df = pd.DataFrame({"n": n, "y": y})
    reps = df.groupby("n").y.agg(["var", "count"])
    reps = reps[reps["count"] >= 2]
    if len(reps):
        pooled = float(np.sqrt((reps["var"] * (reps["count"] - 1)).sum() / (reps["count"] - 1).sum()))
        return pooled, f"pooled replicate sd over {int(reps['count'].sum())} runs at {len(reps)} N"
    dof = max(len(y) - 3, 1)
    resid = y - power_law(n, *params)
    return float(np.sqrt((resid ** 2).sum() / dof)), f"fit residual sd ({dof} dof)"


def crossover(delta: np.ndarray) -> float:
    """Smallest grid N from which delta = GNN - naive stays < 0. -inf if the GNN is
    better over the whole grid, +inf if it is never better at the top of the grid."""
    if delta[-1] >= 0:
        return math.inf
    above = np.where(delta >= 0)[0]
    return -math.inf if len(above) == 0 else float(N_GRID[above[-1] + 1])


def bootstrap(fits: dict, data: dict, sds: dict, n_boot: int, rng: np.random.Generator,
              r_inf_bounds=(35.0, 44.0), balance_n: bool = True):
    curves = {arm: [] for arm in fits}
    params = {arm: [] for arm in fits}
    xs = []
    for _ in range(n_boot):
        draw = {}
        for arm, p in fits.items():
            n = data[arm][0]
            y = power_law(n, *p) + rng.normal(0.0, sds[arm], size=len(n))
            draw[arm] = fit_arm(n, y, r_inf_bounds=r_inf_bounds, balance_n=balance_n)
        if any(v is None for v in draw.values()):
            continue
        for arm, pb in draw.items():
            params[arm].append(pb)
            curves[arm].append(power_law(N_GRID, *pb))
        xs.append(crossover(power_law(N_GRID, *draw["gnn"]) - power_law(N_GRID, *draw["naive"])))
    return ({arm: np.array(c) for arm, c in curves.items()}, {arm: np.array(p) for arm, p in params.items()},
            np.array(xs))


def ci90(x: np.ndarray) -> list:
    """5th/95th percentile without interpolating across +-inf (crossover samples can be infinite)."""
    return [float(np.percentile(x, q, method="inverted_cdf")) for q in (5, 95)]


def jsonable(o):
    """Strict JSON: +-inf / nan become strings rather than the non-standard Infinity / NaN."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else str(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return o


def fmt_n(x: float, lo: float, hi: float) -> str:
    if x == -math.inf:
        return f"< {lo:.0f} (GNN better at every N in the grid)"
    if x == math.inf:
        return f"> {hi:.0f} (not reached)"
    return f"{x:.0f}"


# --------------------------------------------------------------------------- #
# Figure                                                                        #
# --------------------------------------------------------------------------- #
def plot(runs, fits, boot, p_better, n_star, n_star95, area, path: Path,
         ylim_bottom: float = 36.0, ylim_top: float = 52.0):
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    n_obs = runs.n_train.unique()
    lo, hi = max(5.0, n_obs.min() / 2), min(400.0, n_obs.max() * 3)
    view = (N_GRID >= lo) & (N_GRID <= hi)
    for arm, (color, label) in ARM_STYLE.items():
        sub = runs[runs.arm == arm]
        if arm in boot and len(boot[arm]):
            q = np.percentile(boot[arm], [5, 95], axis=0)
            ax.fill_between(N_GRID[view], q[0][view], q[1][view], color=color, alpha=0.14, lw=0)
        if fits.get(arm) is not None:
            p = fits[arm]
            inside = view & (N_GRID >= n_obs.min()) & (N_GRID <= n_obs.max())
            ax.plot(N_GRID[view], power_law(N_GRID[view], *p), color=color, lw=1.2, ls=(0, (4, 3)))
            ax.plot(N_GRID[inside], power_law(N_GRID[inside], *p), color=color, lw=2)
            ax.axhline(p[2], color=color, lw=0.8, ls=":", alpha=0.8)
            ax.annotate(f"{label}\nα={p[1]:.2f}, RMSE∞={p[2]:.1f}", (N_GRID[view][-1], power_law(N_GRID[view][-1], *p)),
                        xytext=(-4, 10 if arm == "naive" else -24), textcoords="offset points",
                        ha="right", fontsize=8, color=INK)
        
        in_bounds = sub[sub.rmse <= ylim_top]
        outliers = sub[sub.rmse > ylim_top]
        ax.scatter(in_bounds.n_train, in_bounds.rmse, s=42, color=color, edgecolor="white", linewidth=1.5,
                   zorder=4, label=label, marker="o" if arm == "gnn" else "s")
        for _, r in outliers.iterrows():
            ax.scatter([r.n_train], [ylim_top - 0.6], s=48, marker="^", color=color, edgecolor="black", lw=1.2, zorder=5)
            ax.annotate(f"▲ {r.rmse:.1f} (seed {r.seed} diverged)",
                        (r.n_train, ylim_top - 0.6), xytext=(-8, -3),
                        textcoords="offset points", ha="right", va="center", fontsize=7.5, color=color, fontweight="bold")
                        
    for arm, (run, value) in V5_ANCHORS.items():
        ax.scatter([57], [value], s=60, facecolor="none", edgecolor=ARM_STYLE[arm][0], lw=1.2, zorder=3)
    ax.scatter([], [], s=60, facecolor="none", edgecolor=GRAY, lw=1.2, label="v5 suite run (N=57 replicate)")
    ax.set_ylim(ylim_bottom, ylim_top)
    ax.set_ylabel("Held-out station RMSE (counts / 15 min)")
    ax.set_title(f"Zero-shot RMSE at {len(runs.holdout_ids.iloc[0])} frozen held-out stations vs training-station count",
                 loc="left", pad=34 if area else 6)
    ax.legend(loc="upper right")

    if len(p_better):
        axd.plot(N_GRID[view], p_better[view], color=BLUE, lw=2)
        axd.axhline(0.95, color=GRAY, lw=0.8, ls="--")
        axd.text(lo * 1.03, 0.955, "0.95", fontsize=7, color=INK2, va="bottom")
        axd.axhline(0.5, color=GRAY, lw=0.6, ls=":")
    notes = []
    for x, style, text, y in ((n_star, ":", "N*", 0.08), (n_star95, "--", "N*₉₅", 0.22)):
        if math.isfinite(x) and lo <= x <= hi:
            for a in (ax, axd):
                a.axvline(x, color=INK2, lw=1, ls=style)
            axd.annotate(f"{text} = {x:.0f}", (x, y), xytext=(4, 0), textcoords="offset points",
                         fontsize=8, color=INK)
        elif not math.isnan(x):
            notes.append(f"{text} {'> ' + format(N_GRID[-1], '.0f') if x > 0 else '< ' + format(N_GRID[0], '.0f')}"
                         + (" (not reached)" if x > 0 else ""))
    if notes:
        axd.text(0.99, 0.04, "; ".join(notes), transform=axd.transAxes, ha="right", fontsize=8, color=INK2)
    axd.axvspan(n_obs.min(), n_obs.max(), color="#f0efec", zorder=0, lw=0)
    axd.set_ylim(-0.02, 1.02)
    axd.set_ylabel("P(GNN beats naive)")
    axd.set_xscale("log")
    axd.set_xlim(lo, hi)
    ticks = [t for t in (5, 10, 15, 25, 35, 45, 57, 100, 200, 400) if lo <= t <= hi]
    axd.set_xticks(ticks)
    axd.set_xticklabels([str(t) for t in ticks])
    axd.set_xlabel("Training stations N (shaded: sampled range; dashed curves: extrapolation)")
    if area:
        top = ax.secondary_xaxis("top", functions=(lambda n: n / area, lambda d: d * area))
        top.set_xlabel("Training-station density (stations / km²)", color=INK2)
        top.tick_params(colors=INK2)
        dens = [d for d in (0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20) if lo <= d * area <= hi]
        top.xaxis.set_major_locator(FixedLocator(dens))
        top.xaxis.set_minor_locator(NullLocator())
        top.xaxis.set_major_formatter(FuncFormatter(lambda d, _: f"{d:g}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=ROOT / "logs" / "v5_scaling")
    ap.add_argument("--mock", action="store_true", help="read <logs-dir>_mock (smoke runs)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--gnn-noise-sd", type=float, default=None, help="override the GNN noise sd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ylim-bottom", type=float, default=36.0, help="bottom y-limit for the RMSE plot (chops unphysical 0)")
    ap.add_argument("--ylim-top", type=float, default=52.0, help="top y-limit for the RMSE plot")
    ap.add_argument("--max-valid-rmse", type=float, default=80.0, help="threshold to exclude diverged runs from fits")
    ap.add_argument("--rmse-inf-min", type=float, default=35.0, help="minimum bound for RMSE_inf asymptote")
    ap.add_argument("--rmse-inf-max", type=float, default=44.0, help="maximum bound for RMSE_inf asymptote")
    args = ap.parse_args()
    logs_dir = args.logs_dir.with_name(args.logs_dir.name + "_mock") if args.mock else args.logs_dir

    runs = load_runs(logs_dir)
    holdout_check = check_reference_holdout(runs)
    runs, area = add_geography(runs)

    r_inf_bounds = (args.rmse_inf_min, args.rmse_inf_max)
    data, fits, sds = {}, {}, {}
    for arm in ARM_STYLE:
        sub = runs[(runs.arm == arm) & (runs.rmse <= args.max_valid_rmse)]
        data[arm] = (sub.n_train.to_numpy(float), sub.rmse.to_numpy(float))
        fits[arm] = fit_arm(*data[arm], r_inf_bounds=r_inf_bounds, balance_n=True)
        if fits[arm] is not None:
            sds[arm] = noise_sd(*data[arm], fits[arm])
    if args.gnn_noise_sd is not None and "gnn" in sds:
        sds["gnn"] = (args.gnn_noise_sd, "--gnn-noise-sd")

    table = (runs.groupby(["n_train", "arm"]).agg(rmse=("rmse", "mean"), sd=("rmse", "std"), runs=("rmse", "size"),
                                                  nearest_km=("nearest_km", "mean"))
             .unstack("arm"))
    fitted = all(fits.get(a) is not None for a in ARM_STYLE)
    summary = {"metric": f"v5_test_metrics.{METRIC}", "held_out_stations": len(runs.holdout_ids.iloc[0]),
               "holdout_check": holdout_check, "hull_area_km2": area, "runs": len(runs), "fits": {}}
    boot, p_better, xs = {}, np.array([]), np.array([])
    n_star = n_star95 = math.nan
    lo_obs, hi_obs = runs.n_train.min(), runs.n_train.max()
    if fitted:
        rng = np.random.default_rng(args.seed)
        boot, boot_params, xs = bootstrap(fits, data, {a: sds[a][0] for a in sds}, args.n_boot, rng,
                                          r_inf_bounds=r_inf_bounds, balance_n=True)
        delta_boot = boot["gnn"] - boot["naive"]
        p_better = (delta_boot < 0).mean(axis=0)
        n_star = crossover(power_law(N_GRID, *fits["gnn"]) - power_law(N_GRID, *fits["naive"]))
        n_star95 = crossover(0.95 - p_better)  # smallest N from which P >= 0.95 holds
        finite = xs[np.isfinite(xs)]
        for arm in ARM_STYLE:
            a, alpha, r_inf = fits[arm]
            summary["fits"][arm] = {
                "A": a, "alpha": alpha, "rmse_inf": r_inf, "noise_sd": sds[arm][0], "noise_source": sds[arm][1],
                "alpha_at_bound": bool(alpha <= 1e-6 or alpha >= ALPHA_MAX - 1e-6),
                "boot_alpha_ci90": ci90(boot_params[arm][:, 1]),
                "boot_rmse_inf_ci90": ci90(boot_params[arm][:, 2]),
            }
        summary["crossover"] = {
            "n_star": n_star, "n_star_95": n_star95,
            "boot_n_star_median": float(np.median(xs)) if len(xs) else None,
            "boot_n_star_ci90": ci90(xs) if len(xs) else None,
            "boot_draws": int(len(xs)),
            "boot_frac_never": float(np.mean(xs == math.inf)) if len(xs) else None,
            "boot_frac_always": float(np.mean(xs == -math.inf)) if len(xs) else None,
            "boot_frac_finite": float(len(finite) / max(len(xs), 1)),
            "density_n_star_per_km2": n_star / area if area and math.isfinite(n_star) else None,
            "density_n_star95_per_km2": n_star95 / area if area and math.isfinite(n_star95) else None,
            "extrapolated": not (lo_obs <= n_star <= hi_obs) if math.isfinite(n_star) else True,
            "grid": [float(N_GRID[0]), float(N_GRID[-1])],
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data").mkdir(exist_ok=True)
    runs.drop(columns=["train_ids", "holdout_ids"]).to_csv(args.out / "data" / "scaling_runs.csv", index=False)
    (args.out / "data" / "scaling_fit.json").write_text(json.dumps(jsonable(summary), indent=2))
    fig_path = args.out / "figures" / "scaling_crossover.png"
    plot(runs, fits, boot, p_better, n_star, n_star95, area, fig_path,
         ylim_bottom=args.ylim_bottom, ylim_top=args.ylim_top)
    write_markdown(args.out / "scaling_summary.md", runs, table, summary, fitted, area)

    print(table.round(2).to_string())
    print(json.dumps(jsonable(summary.get("crossover", {"crossover": "not fitted (< 4 distinct N per arm)"})), indent=2))
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")


def write_markdown(path: Path, runs, table, s, fitted, area):
    lines = ["# Spatial sparsity scaling: GNN vs naive profile donors", "",
             f"Metric: `{s['metric']}` on {s['held_out_stations']} frozen held-out stations "
             f"(holdout {s['holdout_check']}). {s['runs']} runs.", ""]
    if area:
        lines.append(f"Network area (convex hull of all stations): {area:.1f} km².")
        lines.append("")
    lines += ["| N_train | density /km² | GNN RMSE | naive RMSE | Δ (GNN − naive) | mean nearest-train km |",
              "|---:|---:|---:|---:|---:|---:|"]
    for n, row in table.iterrows():
        g, nv = row.get(("rmse", "gnn"), np.nan), row.get(("rmse", "naive"), np.nan)
        km = pd.Series([row.get(("nearest_km", "gnn"), np.nan), row.get(("nearest_km", "naive"), np.nan)]).mean()
        dens = f"{n / area:.2f}" if area else "–"
        lines.append(f"| {n} | {dens} | {g:.2f} | {nv:.2f} | {g - nv:+.2f} | "
                     + (f"{km:.2f}" if np.isfinite(km) else "–") + " |")
    lines.append("")
    if not fitted:
        lines.append("Power-law fit skipped: each arm needs runs at >= 4 distinct N.")
    else:
        lines += ["RMSE(N) = A · N^(−α) + RMSE∞  (bootstrap 90% intervals in brackets)", "",
                  "| arm | A | α | RMSE∞ | noise sd (source) |", "|---|---:|---:|---:|---|"]
        for arm, f in s["fits"].items():
            flag = " (at bound — not identified)" if f["alpha_at_bound"] else ""
            (a_lo, a_hi), (r_lo, r_hi) = f["boot_alpha_ci90"], f["boot_rmse_inf_ci90"]
            lines.append(f"| {arm} | {f['A']:.2f} | {f['alpha']:.3f} [{a_lo:.2f}, {a_hi:.2f}]{flag} | "
                         f"{f['rmse_inf']:.2f} [{r_lo:.1f}, {r_hi:.1f}] | {f['noise_sd']:.2f} ({f['noise_source']}) |")
        c = s["crossover"]
        lo, hi = c["grid"]
        lines += ["", f"- Point-fit crossover N* = {fmt_n(c['n_star'], lo, hi)}"
                  + (" — **extrapolated beyond the sampled range**" if c["extrapolated"] else ""),
                  f"- Definitive crossover N*₉₅ (P(GNN better) ≥ 0.95 from there on) = {fmt_n(c['n_star_95'], lo, hi)}"]
        if c["boot_n_star_ci90"]:
            q = [fmt_n(v, lo, hi) for v in c["boot_n_star_ci90"]]
            lines.append(f"- Bootstrap N* 90% interval: [{q[0]}, {q[1]}]; never crosses in "
                         f"{100 * c['boot_frac_never']:.0f}% of resamples, GNN always better in "
                         f"{100 * c['boot_frac_always']:.0f}%")
        if c["density_n_star_per_km2"] is not None:
            lines.append(f"- As density: N* ≈ {c['density_n_star_per_km2']:.2f} training stations / km²")
    lines += ["", "![crossover](figures/scaling_crossover.png)", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
