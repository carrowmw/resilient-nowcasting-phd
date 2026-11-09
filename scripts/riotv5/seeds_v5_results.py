#!/usr/bin/env python3
"""
Seed analysis for the RIoT v5 suite (tier 3: seeds 43 / 44 of the tier-1 core).

Applies the statistical protocol of docs/design_reports/zero_collapse_investigation.md:
3 seeds per configuration, mean ± std, and a difference is real if it exceeds 2 × the pooled
seed std and points the same way on validation and test.

Pooled seed std is estimated per model family (V4R / P1–P4 / V5) from every configuration that
has three seeds, relative (CV) for RMSE-type metrics and absolute for rates / ratios. Each
comparison is scored as z = Δ / (σ · sqrt(1/n_a + 1/n_b)); with 3 v 3 seeds the protocol's
"Δ > 2σ" is z > 2.45, and that z threshold is used for every comparison so that single-seed
screens (1 v 3) and the deterministic naive baselines (3 v 0) get the same false-positive rate.

Inputs:  docs/experiments/riot_v5/data/runs.csv, epochs.csv, grad_cosine_history.csv
Outputs: docs/experiments/riot_v5/data/seed_summary.csv      mean / std / n per experiment
         docs/experiments/riot_v5/data/seed_noise.csv        pooled seed std per family
         docs/experiments/riot_v5/data/seed_comparisons.csv  every claimed comparison, s42 vs 3-seed
         docs/experiments/riot_v5/figures/seeds/*.png

Usage: python scripts/riotv5/seeds_v5_results.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "experiments" / "riot_v5" / "data"
FIG = ROOT / "docs" / "experiments" / "riot_v5" / "figures" / "seeds"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY, GRAY_LIGHT, MID, INK, INK2 = "#9a9893", "#d6d4cf", "#f0efec", "#1f1f1f", "#5c5b57"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 10.5, "lines.linewidth": 2.2, "savefig.dpi": 240, "figure.dpi": 240,
    "text.color": INK,
})

# key -> (column suffix, kind, decimals). kind "rel" = compare as % change, "abs" = absolute change.
METRICS = {
    "all": ("obs_all_rmse", "rel", 2), "trans": ("obs_trans_rmse", "rel", 2),
    "ind": ("obs_ind_rmse", "rel", 2), "mae": ("obs_all_mae", "rel", 2), "peak": ("peak_rmse", "rel", 1),
    "so12": ("so_12_forecast_rmse", "rel", 2), "so96": ("so_96_forecast_rmse", "rel", 2),
    "crps": ("crps", "rel", 2), "krig": ("kriging_raw_rmse", "rel", 1), "impute": ("impute_raw_rmse", "rel", 1),
    "nll": ("nll_obs", "rel", 2), "legacy": ("legacy_rmse", "rel", 2),
    "so96m": ("so_96_mass_ratio", "abs", 2), "coll": ("collapse_rate", "abs", 3),
    "gapm": ("real_gap_mass_ratio", "abs", 2), "q90": ("q90_coverage", "abs", 2),
    "bias9": ("bias_d9", "abs", 1), "underpred": ("peak_underpred_rate", "abs", 3),
    "ind_pred_corr": ("ind_pred_pairwise_corr_mean", "abs", 2), "ind_corr": ("ind_corr_mean", "abs", 2),
    "beats_peer": ("ind_beats_peer_frac", "abs", 2),
}
Z_REAL, Z_SUGG = 2.45, 1.5
# Bands the seed-42 report used ("noise" upper edge, "clear" lower edge), for the verdict-change table.
OLD_BANDS = {"all": (3, 6), "mae": (3, 6), "peak": (3, 6), "crps": (3, 6), "so12": (3, 6), "so96": (3, 6),
             "krig": (3, 6), "trans": (5, 8), "ind": (5, 8), "coll": (0.02, 0.02), "so96m": (0.10, 0.10),
             "gapm": (0.15, 0.15)}

NAIVE = {"naive_profile", "naive_last_obs"}
BASES = [("V4R", "v4r_full"), ("P2B", "e1_6_complete_weather"), ("P3B", "e2_7_mixed_030"),
         ("P4B", "e3_stack"), ("P5B", "e4_4_anchored"), ("V5F", "v5_full")]


def family(group):
    if group == "thesis_v4":
        return "V4R"
    if group in ("p7_v5_suite", "p6_mtl", "loo"):
        return "V5"
    return "P1-P4"


def load():
    r = pd.read_csv(DATA / "runs.csv").copy()
    r["fam"] = r.group.map(family)
    ep = pd.read_csv(DATA / "epochs.csv")
    ep = ep.merge(r[["run", "experiment", "seed"]], on="run")
    return r, ep


def summarise(r):
    agg = {}
    for split in ("test", "val"):
        for k, (c, _, _) in METRICS.items():
            col = f"{split}_{c}"
            if col in r:
                agg[f"{split}_{k}_mean"] = (col, "mean")
                agg[f"{split}_{k}_std"] = (col, "std")
    s = r.groupby("experiment").agg(group=("group", "first"), fam=("fam", "first"), n=("seed", "size"),
                                    best_epoch_mean=("best_epoch", "mean"), best_epoch_std=("best_epoch", "std"),
                                    gpu_h=("duration_h", "sum"), **agg)
    return s


def pooled_noise(r):
    """Pooled within-config seed std per family: CV (%) for rel metrics, absolute for abs metrics."""
    s3 = r[r.groupby("experiment").seed.transform("size") == 3]
    out = []
    for fam, g in list(s3.groupby("fam")) + [("ALL", s3)]:
        row = {"fam": fam, "configs": g.experiment.nunique()}
        for split in ("test", "val"):
            for k, (c, kind, _) in METRICS.items():
                col = f"{split}_{c}"
                if col not in g or g[col].isna().all():
                    continue
                gg = g.dropna(subset=[col])
                gg = gg[gg.groupby("experiment")[col].transform("size") == 3]
                if kind == "rel":
                    dev = np.log(gg[col]) - np.log(gg[col]).groupby(gg.experiment).transform("mean")
                    sd = 100 * np.sqrt((dev ** 2).sum() / (2 * gg.experiment.nunique()))
                else:
                    dev = gg[col] - gg[col].groupby(gg.experiment).transform("mean")
                    sd = np.sqrt((dev ** 2).sum() / (2 * gg.experiment.nunique()))
                row[f"{split}_{k}"] = sd
        out.append(row)
    return pd.DataFrame(out).set_index("fam")


def seed42_offset(r):
    """Seed-42 value vs the mean of seeds 43/44, per config; seed main effects."""
    s3 = r[r.groupby("experiment").seed.transform("size") == 3]
    rows = []
    for e, x in s3.groupby("experiment"):
        x = x.set_index("seed")
        row = {"experiment": e, "fam": x.fam.iloc[0], "ep42": x.best_epoch[42],
               "ep_other": x.best_epoch[[43, 44]].mean()}
        for k, (c, kind, _) in METRICS.items():
            v = x[f"test_{c}"]
            if v.isna().any():
                continue
            row[k] = 100 * (v[42] / v[[43, 44]].mean() - 1) if kind == "rel" else v[42] - v[[43, 44]].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")


# --------------------------------------------------------------- comparisons
def comparisons():
    """(section, label, experiment, reference). Reference is the base the seed-42 report used."""
    C = []
    chain = [b for _, b in BASES]
    names = dict((b, n) for n, b in BASES)
    for a, b in zip(chain[1:], chain[:-1]):
        C.append(("chain", f"{names[a]} vs {names[b]}", a, b))
    C.append(("chain", "V5F vs V4R", "v5_full", "v4r_full"))
    C.append(("chain", "P3B vs V4R", "e2_7_mixed_030", "v4r_full"))
    for e in ["e1_1_obs_forecast", "e1_2_obs_impute", "e1_3_obs_kriging", "e1_4_selection", "e1_5_nb"]:
        C.append(("p1", f"{e} vs V4R", e, "v4r_full"))
    for a in ["no_impute", "no_kriging", "no_masks", "no_pe", "lstm", "bs32", "dropout03", "night_zero",
              "total_only", "night_only", "identity_graph", "lstm_rich", "no_masks_no_impute", "bs128_steps4x",
              "bs128_peak_weighted", "bs32_lr_div4"]:
        C.append(("thesis_v4r", f"v4r_{a}", f"v4r_{a}", "v4r_full"))
    for a in ["no_impute", "no_kriging", "no_masks", "no_pe", "lstm", "bs32", "dropout03", "lstm_plain",
              "identity_graph", "iid_masks", "no_outage_head", "night_only", "total_only", "no_masks_no_impute",
              "no_calendar_profile", "bs32_lr_div4"]:
        C.append(("v5_suite", f"v5_{a}", f"v5_{a}", "v5_full"))
    for e in ["e4_1_ztnb", "e4_2_mixture_latent", "e4_3_mixture_missingness", "e4_3b_mixture_detached"]:
        C.append(("p4", e, e, "e3_stack"))
    for e in ["e4_5_prior", "e4_6_wmiss025", "e4_7_mixture_all_heads"]:
        C.append(("p4", e, e, "e4_4_anchored"))
    for e in ["e6_1_uncertainty", "e6_2_pcgrad", "e6_3_impute_pretrain", "e6_4_detached_impute",
              "e6_5_v4_impute_aux", "e5_8_full_plus_underpred"]:
        C.append(("p6", e, e, "v5_full"))
    for e in ["e2_1_unify_iid", "e2_1b_iid_traffic", "e2_2_block", "e2_2b_block_geometric", "e2_3_transplant",
              "e2_4_tail", "e2_5_mixed_015", "e2_7_mixed_030", "e2_8_mixed_050"]:
        C.append(("p2", e, e, "e1_6_complete_weather"))
    for e in ["e3_1_calendar", "e3_2_log1p", "e3_3b_profile_inductive_neighbour", "e3_3c_profile_inductive_static",
              "e3_3d_profile_inductive_zero", "e3_4_runlength", "e3_5_sensor_emb", "e3_6_seq48", "e3_7_seq96",
              "e3_7b_lags", "e3_9_key_padding", "e3_10_gat_drop_missing", "e3_stack", "e3_stack_seq48"]:
        C.append(("p3", e, e, "e2_7_mixed_030"))
    for e in ["e5_1_alpha05", "e5_2_alpha10", "e5_3a_underpred01", "e5_3b_underpred05", "e5_4_quantiles",
              "e5_5_huber", "e5_6_low_tol"]:
        C.append(("p5", e, e, "e4_4_anchored"))
    for e in ["loo_no_unify", "loo_no_sensor_emb", "loo_no_runlength", "loo_no_profile", "loo_no_calendar",
              "loo_no_gat_drop", "loo_mask_015", "loo_no_quantiles", "loo_unanchored", "loo_no_log1p",
              "loo_no_magnitude", "loo_no_huber", "loo_no_key_padding", "loo_no_complete_weather",
              "loo_night_leak", "loo_val_loss_selection"]:
        C.append(("loo", e, e, "v5_full"))
    for e in ["v5_full", "v4r_full", "e2_1_unify_iid", "e3_3d_profile_inductive_zero", "v5_no_outage_head",
              "v5_lstm"]:
        C.append(("naive", f"{e} vs naive_profile", e, "naive_profile"))
    return C


def compare(r, noise):
    rows = []
    by = {e: g for e, g in r.groupby("experiment")}
    for sec, label, a, b in comparisons():
        ga, gb = by[a], by[b]
        na, nb = (0 if a in NAIVE else len(ga)), (0 if b in NAIVE else len(gb))
        fams = [x for x, n in ((ga.fam.iloc[0], na), (gb.fam.iloc[0], nb)) if n]
        row = {"section": sec, "label": label, "exp": a, "ref": b, "n_exp": len(ga), "n_ref": len(gb)}
        for k, (c, kind, _) in METRICS.items():
            for split in ("test", "val"):
                col = f"{split}_{c}"
                if col not in r or ga[col].isna().all() or gb[col].isna().all():
                    continue
                ma, mb = ga[col].mean(), gb[col].mean()
                a42 = ga.loc[ga.seed == 42, col].iloc[0]
                b42 = gb.loc[gb.seed == 42, col].iloc[0]
                if kind == "rel":
                    d, d42 = 100 * (ma / mb - 1), 100 * (a42 / b42 - 1)
                else:
                    d, d42 = ma - mb, a42 - b42
                row[f"{k}_d" if split == "test" else f"val_{k}_d"] = d
                if split == "test":
                    row[f"{k}_a"], row[f"{k}_b"], row[f"{k}_d42"] = ma, mb, d42
                    row[f"{k}_sda"] = ga[col].std() if na > 1 else np.nan
                    sig = np.sqrt(np.mean([noise.loc[f, f"test_{k}"] ** 2 for f in fams]))
                    se = sig * np.sqrt((1 / na if na else 0) + (1 / nb if nb else 0))
                    row[f"{k}_sigma"], row[f"{k}_z"] = sig, d / se if se > 0 else np.nan
        for k in METRICS:
            z = row.get(f"{k}_z", np.nan)
            if np.isnan(z):
                continue
            vd = row.get(f"val_{k}_d", np.nan)
            agree = np.isnan(vd) or np.sign(vd) == np.sign(row[f"{k}_d"])
            row[f"{k}_val_agree"] = agree if not np.isnan(vd) else np.nan
            row[f"{k}_verdict"] = ("real" if abs(z) >= Z_REAL and agree else
                                   "suggestive" if abs(z) >= Z_SUGG else "noise")
            if k in OLD_BANDS:
                lo, hi = OLD_BANDS[k]
                d42 = abs(row[f"{k}_d42"])
                row[f"{k}_old"] = "noise" if d42 <= lo else ("clear" if d42 > hi else "suggestive")
        rows.append(row)
    return pd.DataFrame(rows)


def epoch_confound(r):
    """Within-config relation between selected epoch and test RMSE (seeds only differ by randomness)."""
    s3 = r[(r.groupby("experiment").seed.transform("size") == 3) & (r.model_type != "lstm")].copy()
    out = {}
    for fam, g in list(s3.groupby("fam")) + [("ALL", s3)]:
        g = g.copy()
        g["ep_dm"] = g.best_epoch - g.groupby("experiment").best_epoch.transform("mean")
        res = {}
        for k in ("trans", "all", "ind"):
            col = f"test_{METRICS[k][0]}"
            y = 100 * (np.log(g[col]) - np.log(g[col]).groupby(g.experiment).transform("mean"))
            res[k] = stats.spearmanr(g.ep_dm, y)
            slope = np.polyfit(g.ep_dm, y, 1)[0]
            res[k + "_slope10"] = 10 * slope
        res["n"] = len(g)
        out[fam] = res
    return out


# ------------------------------------------------------------------- figures
def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f"figures/seeds/{name}")


def fig_noise(off, noise):
    """S1: seed-42 offset per config, and noise per metric vs the seed-42 report's bands."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    keys = ["all", "trans", "ind", "peak", "so12", "crps"]
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        v = off[k].dropna()
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=12, color=BLUE, alpha=0.6, zorder=3)
        m, se = v.mean(), v.std() / np.sqrt(len(v))
        ax.errorbar(i + 0.32, m, yerr=1.96 * se, color=INK, marker="D", ms=4, capsize=3, lw=1.2, zorder=4)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(len(keys)), keys)
    ax.set_ylabel("seed 42 vs mean(43, 44), %")
    ax.set_title("Seed 42 vs seeds 43/44, 53 configs\n(diamond = mean ± 95% CI)", loc="left", fontsize=9)
    ax = axes[1]
    keys = ["all", "mae", "peak", "so12", "crps", "trans", "ind", "krig"]
    fams = ["V5", "V4R", "P1-P4"]
    x = np.arange(len(keys))
    w = 0.26
    for j, (f, c) in enumerate(zip(fams, (ORANGE, BLUE, YELLOW))):
        thr = [Z_REAL * np.sqrt(2) * noise.loc[f, f"test_{k}"] for k in keys]
        ax.bar(x + (j - 1) * w, thr, width=w, color=c, label=f, zorder=2)
    for xi, k in zip(x, keys):
        lo, hi = OLD_BANDS[k]
        ax.hlines(lo, xi - 0.42, xi + 0.42, color=INK, lw=1.2, ls=":", zorder=3)
        ax.hlines(hi, xi - 0.42, xi + 0.42, color=INK, lw=1.2, zorder=3)
    ax.set_xticks(x, keys)
    ax.set_ylabel("|Δ| needed, %")
    ax.set_title("|Δ| needed to call a 1-seed v 1-seed difference real (bars, z 2.45)\n"
                 "vs the seed-42 report's 'noise' (dotted) and 'clear' (solid) edges", loc="left", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "figS1_seed_noise.png")


def fig_chain(r, summ):
    """S2: base chain with seed spread; seed 42 highlighted."""
    names = [n for n, _ in BASES]
    runs = [e for _, e in BASES]
    panels = [("all", "All-node RMSE"), ("trans", "Transductive RMSE"), ("ind", "Held-out RMSE"),
              ("peak", "Peak RMSE"), ("so12", "SO@12 RMSE"), ("so96m", "SO@96 mass ratio"),
              ("coll", "Real-gap collapse rate"), ("gapm", "Real-gap mass ratio"), ("crps", "CRPS")]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.4))
    x = np.arange(len(runs))
    naive = r.set_index("experiment").loc["naive_profile"]
    for ax, (k, title) in zip(axes.flat, panels):
        col = f"test_{METRICS[k][0]}"
        for xi, e in zip(x, runs):
            v = r[r.experiment == e].set_index("seed")[col]
            ax.scatter([xi] * 2, v[[43, 44]], s=20, color=GRAY_LIGHT, edgecolor=GRAY, lw=0.6, zorder=3)
            ax.scatter([xi], v[42], s=20, color=ORANGE, zorder=4)
        m = summ.loc[runs, f"test_{k}_mean"].values
        ax.plot(x, m, color=BLUE, marker="o", ms=5.5, zorder=5)
        for xi, yi in zip(x, m):
            ax.annotate(f"{yi:.2f}" if abs(yi) < 5 else f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9.5, color=INK2)
        if k in ("so96m", "gapm"):
            ax.axhline(1, color=GRAY, ls="--", lw=1)
        elif not np.isnan(naive[col]) and k != "coll":
            ax.axhline(naive[col], color=GRAY, ls="--", lw=1)
            ax.annotate("naive profile", (0, naive[col]), xytext=(4, 3), textcoords="offset points",
                        fontsize=9.5, color=INK2)
        ax.set_xticks(x, names)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))
    fig.suptitle("Base chain, 3-seed mean (blue), seed 42 (orange), seeds 43/44 (gray)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "figS2_base_chain_seeds.png")


def _forest(ax, labels, d, d42, ci, title, band=None):
    y = np.arange(len(labels))
    if band is not None:
        ax.axvspan(-band, band, color=MID, zorder=0)
    sig = np.nan_to_num(np.abs(d)) > ci
    ax.errorbar(d[~sig], y[~sig], xerr=ci[~sig], fmt="o", color=GRAY, ms=5, capsize=2.5, lw=1.2, zorder=3)
    ax.errorbar(d[sig], y[sig], xerr=ci[sig], fmt="o", color=BLUE, ms=5, capsize=2.5, lw=1.4, zorder=3)
    ax.scatter(d42, y, marker="x", s=32, color=ORANGE, zorder=4, label="seed 42 only")
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")


def fig_thesis(cmp):
    """S3: thesis Experiment 3 ablations, 3-seed mean ± protocol threshold."""
    thesis = {"no_impute": 7.56, "no_kriging": 3.25, "no_masks": 9.62, "no_pe": 4.48, "lstm": 22.42,
              "bs32": -7.29, "dropout03": 7.50}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, pre, title in ((axes[0], "v4r", "V4R"), (axes[1], "v5", "V5F")):
        sec = "thesis_v4r" if pre == "v4r" else "v5_suite"
        c = cmp[cmp.section == sec].set_index("exp").loc[[f"{pre}_{a}" for a in thesis]]
        thr = Z_REAL * c.all_sigma * np.sqrt(2 / 3)
        _forest(ax, list(thesis), c.all_d.values, c.all_d42.values, thr.values,
                f"{title}: Δ all-node RMSE vs full (%), 3-seed mean ± 2σ threshold")
        if pre == "v4r":
            ax.scatter(list(thesis.values()), np.arange(len(thesis)), marker="s", s=26, color=GRAY_LIGHT,
                       edgecolor=INK2, lw=0.7, zorder=2, label="thesis (legacy metric)")
    axes[0].legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    save(fig, "figS3_thesis_ablations_seeds.png")


def fig_screens_rebased(cmp):
    """S4: LOO and P3 single-seed screens re-scored against the 3-seed base mean."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2))
    for ax, sec, k, title in ((axes[0], "loo", "so12", "LOO: Δ SO@12 vs V5F mean (%)"),
                              (axes[1], "loo", "ind", "LOO: Δ held-out RMSE vs V5F mean (%)"),
                              (axes[2], "p3", "all", "P3: Δ all-node RMSE vs P3B mean (%)")):
        c = cmp[cmp.section == sec].copy()
        c = c[c.exp != c.ref]
        c = c.sort_values(f"{k}_d", ascending=False)
        thr = Z_REAL * c[f"{k}_sigma"] * np.sqrt(1 + 1 / 3)
        labs = [e.replace("loo_", "").replace("profile_inductive_", "profile_") for e in c.exp]
        d, d42 = c[f"{k}_d"].values.copy(), c[f"{k}_d42"].values.copy()
        lim = 32 if sec == "p3" else 16
        for i, v in enumerate(d):
            if abs(v) > lim:  # off-scale (no_unify on SO@12): clip and label the true value
                ax.annotate(f"{labs[i]}: {v:+.0f}% →", (lim * 0.97, i), ha="right", va="center", fontsize=7,
                            color=INK2)
                d[i], d42[i] = np.nan, np.nan
                labs[i] = labs[i] + " *"
        _forest(ax, labs, d, d42, thr.values, title)
        ax.set_xlim(-lim, lim)
    axes[0].legend(loc="lower right")
    fig.suptitle("Single-seed screens vs the 3-seed base (dot = Δ vs base mean; bar = z 2.45 threshold for 1 v 3; "
                 "× = Δ vs seed-42 base)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "figS4_screens_rebased.png")


def fig_epoch(r):
    """S5: within-config best-epoch vs RMSE (seed differences only)."""
    s3 = r[(r.groupby("experiment").seed.transform("size") == 3) & (r.model_type != "lstm")].copy()
    s3["ep_dm"] = s3.best_epoch - s3.groupby("experiment").best_epoch.transform("mean")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, k in zip(axes, ("trans", "all", "ind")):
        col = f"test_{METRICS[k][0]}"
        s3["y"] = 100 * (np.log(s3[col]) - np.log(s3[col]).groupby(s3.experiment).transform("mean"))
        for (f, g), c in zip(s3.groupby("fam"), (YELLOW, BLUE, ORANGE)):
            ax.scatter(g.ep_dm, g.y, s=14, color=c, alpha=0.8, label=f, zorder=3)
        rho = stats.spearmanr(s3.ep_dm, s3.y)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axvline(0, color=GRAY, lw=0.8)
        ax.set_xlabel("selected epoch − config mean")
        ax.set_ylabel(f"test {k} RMSE vs config mean (%)")
        ax.set_title(f"{k}: Spearman ρ = {rho.statistic:.2f} (p = {rho.pvalue:.1g}, n = {len(s3)})", loc="left",
                     fontsize=9)
    axes[0].legend(loc="upper right")
    fig.suptitle("Same config, different seed: does stopping earlier mean a worse model?", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "figS5_epoch_within_config.png")


def fig_ranks(r):
    """S6: seed-42 value vs mean of seeds 43/44, all 53 seeded configs."""
    s3 = r[r.groupby("experiment").seed.transform("size") == 3]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, k in zip(axes, ("all", "ind", "so12", "coll")):
        col = f"test_{METRICS[k][0]}"
        p = s3.pivot(index="experiment", columns="seed", values=col)
        x, y = p[42], p[[43, 44]].mean(axis=1)
        fam = s3.groupby("experiment").fam.first().loc[p.index]
        for (f, idx), c in zip(fam.groupby(fam).groups.items(), (YELLOW, BLUE, ORANGE)):
            ax.scatter(x.loc[idx], y.loc[idx], s=14, color=c, label=f, zorder=3)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.8, ls="--")
        rho = stats.spearmanr(x, y).statistic
        ax.set_title(f"{k}: rank ρ = {rho:.2f}", loc="left", fontsize=9)
        ax.set_xlabel("seed 42")
    axes[0].set_ylabel("mean of seeds 43, 44")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "figS6_seed42_vs_others.png")


def fig_interaction(summ):
    """S7: masks × imputation 2 × 2 with seed s.d."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, pre, title in ((axes[0], "v4r", "V4R"), (axes[1], "v5", "V5F")):
        cells = {(1, 1): f"{pre}_full", (0, 1): f"{pre}_no_masks", (1, 0): f"{pre}_no_impute",
                 (0, 0): f"{pre}_no_masks_no_impute"}
        for imp, c, lab, dx in ((1, BLUE, "imputation loss on", -0.03), (0, ORANGE, "imputation loss off", 0.03)):
            m = [summ.loc[cells[(k, imp)], "test_all_mean"] for k in (1, 0)]
            s = [summ.loc[cells[(k, imp)], "test_all_std"] for k in (1, 0)]
            ax.errorbar(np.array([0, 1]) + dx, m, yerr=s, color=c, marker="o", ms=5, capsize=3, label=lab)
            for xi, yi in zip([0, 1], m):
                ax.annotate(f"{yi:.2f}", (xi + dx, yi), xytext=(8, 0), textcoords="offset points", fontsize=7,
                            color=INK2, va="center")
        ax.set_xticks([0, 1], ["mask channels on", "mask channels off"])
        ax.set_xlim(-0.3, 1.45)
        ax.set_title(f"{title}: all-node RMSE, 3-seed mean ± s.d.", loc="left", fontsize=9)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "figS7_masks_x_impute_seeds.png")


def fig_cosine(r):
    path = DATA / "grad_cosine_history.csv"
    gc = pd.read_csv(path).merge(r[["run", "experiment", "seed", "group", "model_type"]], on="run")
    m = gc.groupby(["experiment", "seed"]).cos.mean().unstack()
    m = m[m.notna().sum(axis=1) == 3]
    grp = r.groupby("experiment").group.first().loc[m.index]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    order = ["thesis_v4", "p1_hygiene", "p2_masking", "p3_inputs", "p4_likelihood", "p6_mtl", "p7_v5_suite"]
    for i, g in enumerate(order):
        idx = grp.index[grp == g]
        for j, e in enumerate(idx):
            xx = i + (j - len(idx) / 2) * 0.6 / max(len(idx), 1)
            ax.plot([xx] * 3, m.loc[e].values, color=GRAY_LIGHT, lw=0.8, zorder=2)
            ax.scatter([xx] * 3, m.loc[e].values, s=7, color=BLUE, zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(len(order)), [g.split("_")[0] if g != "thesis_v4" else "thesis" for g in order])
    ax.set_ylabel("mean cos(∇forecast, ∇impute)")
    ax.set_title("Gradient cosine, 3 seeds per config (vertical line = seed range)", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "figS8_grad_cosine_seeds.png")
    return m


def main():
    r, ep = load()
    summ = summarise(r)
    noise = pooled_noise(r)
    off = seed42_offset(r)
    cmp = compare(r, noise)
    summ.to_csv(DATA / "seed_summary.csv", float_format="%.5g")
    noise.to_csv(DATA / "seed_noise.csv", float_format="%.5g")
    cmp.to_csv(DATA / "seed_comparisons.csv", index=False, float_format="%.5g")
    print("wrote seed_summary.csv, seed_noise.csv, seed_comparisons.csv")
    fig_noise(off, noise)
    fig_chain(r, summ)
    fig_thesis(cmp)
    fig_screens_rebased(cmp)
    fig_epoch(r)
    fig_ranks(r)
    fig_interaction(summ)
    if (DATA / "grad_cosine_history.csv").exists():
        fig_cosine(r)
    conf = epoch_confound(r)
    for f, v in conf.items():
        print("epoch confound", f, {k: (round(x.statistic, 2), float(f"{x.pvalue:.2g}")) if hasattr(x, "pvalue")
                                    else round(x, 2) for k, x in v.items()})


if __name__ == "__main__":
    main()
