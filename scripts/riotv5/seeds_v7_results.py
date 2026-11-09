#!/usr/bin/env python3
"""
Seed analysis for the RIoT v7 stage (docs/experiments/riot_v6/riot_v6_seed_report.md §9.2, recs 1-6).

8 experiments x seeds 42-47 = 48 runs, all V6 lineage on the v5 57-station training set with no
validation stations (tier 2, the gated stratified-validation tier, was skipped: rec2_gate.json).

Protocol of the v5 / v6 seed reports, generalised to 6 seeds: pooled within-config seed s.d. over
the 8 v7 configs (log-CV % for RMSE-type metrics, absolute for rates), z = Δ / (σ · sqrt(1/n_a + 1/n_b));
real if |z| ≥ 2.45 and the validation window agrees in sign, suggestive if |z| ≥ 1.5. Every v7 arm
shares seeds 42-47, so each comparison also gets a paired test on the per-seed log ratio and the
range of Δ when any one seed is left out.

With no validation stations, the validation split scores the 24 held-out test stations as "ind"
(src/train.py), so val held-out RMSE is the same stations in an earlier window: the sign check on
held-out claims is a temporal replication, not an independent station set. Selection
(obs_loss_plus_so) does not read the held-out stations.

Cross-stage (†): the v6 runs trained on 52 stations; only held-out and kriging are like-for-like,
and only test metrics are compared.

Inputs:  docs/experiments/riot_v7/data/runs.csv, epochs.csv, rec2_gate.json, rec6_stop_list.json;
         docs/experiments/riot_v6/data/runs.csv; docs/experiments/riot_v5/data/runs.csv (naive profile)
Outputs: docs/experiments/riot_v7/data/seed_summary.csv, seed_noise.csv, seed_comparisons.csv,
         factorial.csv, seed_effects.csv, epoch_stats.csv
         docs/experiments/riot_v7/figures/*.png

Usage: python scripts/riotv5/seeds_v7_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeds_v5_results as s5  # noqa: E402
import seeds_v6_results as s6  # noqa: E402  (extended METRICS, forest plot)

plt = s5.plt
ROOT = s5.ROOT
DATA = ROOT / "docs" / "experiments" / "riot_v7" / "data"
FIG = ROOT / "docs" / "experiments" / "riot_v7" / "figures"
METRICS = s6.METRICS
Z_REAL, Z_SUGG = s5.Z_REAL, s5.Z_SUGG
NAIVE = s5.NAIVE
BLUE, ORANGE, AQUA, YELLOW, GRAY, GRAY_LIGHT, INK2 = s5.BLUE, s5.ORANGE, s5.AQUA, s5.YELLOW, s5.GRAY, s5.GRAY_LIGHT, s5.INK2
HEAD = ["all", "trans", "ind", "mae", "peak", "so12", "crps", "krig"]


# --------------------------------------------------------------------- data
def load():
    r7 = pd.read_csv(DATA / "runs.csv").copy().assign(stage="v7", fam="V7")
    r6 = pd.read_csv(s6.DATA / "runs.csv").copy()
    r6 = r6[r6.model_type == "gnn"].copy().assign(stage="v6", fam=lambda d: d.experiment.map(s6.family6))
    r5 = pd.read_csv(s5.DATA / "runs.csv").copy()
    naive = r5[r5.experiment.isin(NAIVE)].copy().assign(stage="v5", fam="naive")
    r = pd.concat([r7, r6, naive], ignore_index=True)
    return r, r7


def pooled_noise(r):
    """Pooled within-config seed s.d. per family for any number of seeds (ddof 1 per config)."""
    x = r[r.groupby("experiment").seed.transform("size") >= 3]
    out = []
    for fam, g in x.groupby("fam"):
        row = {"fam": fam, "configs": g.experiment.nunique(), "runs": len(g)}
        for split in ("test", "val"):
            for k, (c, kind, _) in METRICS.items():
                col = f"{split}_{c}"
                if col not in g or g[col].isna().all():
                    continue
                gg = g.dropna(subset=[col])
                v = np.log(gg[col]) if kind == "rel" else gg[col]
                dev = v - v.groupby(gg.experiment).transform("mean")
                dof = (gg.groupby("experiment").size() - 1).sum()
                row[f"{split}_{k}"] = (100 if kind == "rel" else 1) * np.sqrt((dev ** 2).sum() / dof)
        out.append(row)
    return pd.DataFrame(out).set_index("fam")


def summarise(r):
    agg = {}
    for split in ("test", "val"):
        for k, (c, _, _) in METRICS.items():
            col = f"{split}_{c}"
            if col in r:
                agg[f"{split}_{k}_mean"] = (col, "mean")
                agg[f"{split}_{k}_std"] = (col, "std")
    return r.groupby("experiment").agg(stage=("stage", "first"), group=("group", "first"), fam=("fam", "first"),
                                       n=("seed", "size"), best_epoch_mean=("best_epoch", "mean"),
                                       best_epoch_std=("best_epoch", "std"), epochs_run_mean=("epochs_run", "mean"),
                                       gpu_h=("duration_h", "sum"), **agg)


# -------------------------------------------------------------- comparisons
COMPARISONS = [
    # rec 1: static-ridge confirmation on 57 training stations
    ("static", "static-ridge profile", "v7_profile_static", "v7_base"),
    ("static", "static-ridge, 150 epochs", "v7_profile_static_e150", "v7_base_e150"),
    ("static", "static-ridge + block masks vs block masks", "v7_static_block", "v7_block"),
    # rec 3: block masks (2x2 with tier 1)
    ("block", "block masks", "v7_block", "v7_base"),
    ("block", "block masks on static-ridge", "v7_static_block", "v7_profile_static"),
    ("block", "static-ridge + block vs base", "v7_static_block", "v7_base"),
    # rec 4: weather extremes
    ("weather", "weather extremes at target (perfect forecast)", "v7_weather_target", "v7_base"),
    ("weather", "weather extremes at origin (deployable)", "v7_weather_origin", "v7_base"),
    ("weather", "origin vs target", "v7_weather_origin", "v7_weather_target"),
    # rec 5: 150-epoch budget
    ("epochs", "150 epochs (base)", "v7_base_e150", "v7_base"),
    ("epochs", "150 epochs (static-ridge)", "v7_profile_static_e150", "v7_profile_static"),
    ("epochs", "static-ridge 150 ep vs base 100 ep", "v7_profile_static_e150", "v7_base"),
    # cross-stage replication (†, test only): 57 vs 52 training stations, 6 v 3 seeds
    ("cross", "v7 base vs v6 base", "v7_base", "v6_base"),
    ("cross", "v7 static-ridge vs v6 static-ridge", "v7_profile_static", "v6_profile_static"),
    ("cross", "v7 weather-target vs v6 weather extremes", "v7_weather_target", "v6_weather_extremes"),
    ("cross", "v7 block vs v6 E2.2 block (P-family)", "v7_block", "v6p_e2_2_block"),
    # naive profile (deterministic, held-out and kriging comparable)
    *[("naive", f"{e} vs naive profile", e, "naive_profile") for e in
      ["v7_base", "v7_profile_static", "v7_static_block", "v7_weather_origin", "v7_base_e150",
       "v7_profile_static_e150"]],
]


def _delta(a, b, kind):
    return 100 * (a / b - 1) if kind == "rel" else a - b


def compare(r, noise):
    rows = []
    by = {e: g.set_index("seed") for e, g in r.groupby("experiment")}
    for sec, label, a, b in COMPARISONS:
        ga, gb = by[a], by[b]
        nb = 0 if b in NAIVE else len(gb)
        cross = ga.stage.iloc[0] != gb.stage.iloc[0] and b not in NAIVE
        row = {"section": sec, "label": label, "exp": a, "ref": b, "n_exp": len(ga), "n_ref": nb, "cross": cross}
        paired = not cross and nb and set(ga.index) == set(gb.index)
        for k, (c, kind, _) in METRICS.items():
            col = f"test_{c}"
            if col not in r or ga[col].isna().all() or gb[col].isna().all():
                continue
            ma, mb = ga[col].mean(), gb[col].mean()
            d = _delta(ma, mb, kind)
            row[f"{k}_a"], row[f"{k}_b"], row[f"{k}_d"] = ma, mb, d
            row[f"{k}_sda"] = ga[col].std()
            nk = f"test_{k}"
            sig = [noise.loc[ga.fam.iloc[0], nk] if nk in noise else np.nan,
                   noise.loc[gb.fam.iloc[0], nk] if nb and nk in noise else 0.0]
            se = np.sqrt(sig[0] ** 2 / len(ga) + (sig[1] ** 2 / nb if nb else 0))
            row[f"{k}_z"] = d / se if se > 0 else np.nan
            # validation window: same stations, earlier period (naive has val rows from v5)
            vcol = f"val_{c}"
            vd = np.nan
            if not cross and vcol in r and ga[vcol].notna().any() and gb[vcol].notna().any():
                vd = _delta(ga[vcol].mean(), gb[vcol].mean(), kind)
            row[f"val_{k}_d"] = vd
            if paired:
                ps = (np.log(ga[col]) - np.log(gb.loc[ga.index, col])) * 100 if kind == "rel" else \
                    ga[col] - gb.loc[ga.index, col]
                row[f"{k}_pt"] = stats.ttest_1samp(ps, 0).statistic
                row[f"{k}_pwins"] = int((ps < 0).sum())
                loo = [_delta(ga.drop(s)[col].mean(), gb.drop(s)[col].mean(), kind) for s in ga.index]
                row[f"{k}_loo_min"], row[f"{k}_loo_max"] = min(loo), max(loo)
            z = row[f"{k}_z"]
            agree = np.isnan(vd) or np.sign(vd) == np.sign(d)
            row[f"{k}_verdict"] = ("real" if abs(z) >= Z_REAL and agree else
                                   "suggestive" if abs(z) >= Z_SUGG else "noise")
        rows.append(row)
    return pd.DataFrame(rows)


def factorial(r7, noise):
    """2x2 profile (zero / static-ridge) x masks (mixed / block) on shared seeds 42-47, log scale.
    Main effect = mean of the two simple effects; interaction = difference of the simple effects."""
    cells = {("zero", "mixed"): "v7_base", ("static", "mixed"): "v7_profile_static",
             ("zero", "block"): "v7_block", ("static", "block"): "v7_static_block"}
    out = []
    for k in HEAD + ["so96m"]:
        c, kind, _ = METRICS[k]
        col = f"test_{c}"
        m = {key: r7[r7.experiment == e][col] for key, e in cells.items()}
        f = (lambda s: 100 * np.log(s.mean())) if kind == "rel" else (lambda s: s.mean())
        L = {key: f(v) for key, v in m.items()}
        sig = noise.loc["V7", f"test_{k}"]
        n = 6
        prof = ((L["static", "mixed"] - L["zero", "mixed"]) + (L["static", "block"] - L["zero", "block"])) / 2
        mask = ((L["zero", "block"] - L["zero", "mixed"]) + (L["static", "block"] - L["static", "mixed"])) / 2
        inter = (L["static", "block"] - L["zero", "block"]) - (L["static", "mixed"] - L["zero", "mixed"])
        # main effect: 4 cell means with weights ±1/2 -> σ/√n; interaction: weights ±1 -> 2σ/√n
        se_main, se_int = sig / np.sqrt(n), sig * np.sqrt(4 / n)
        out.append({"metric": k, "profile": prof, "profile_z": prof / se_main, "masks": mask,
                    "masks_z": mask / se_main, "interaction": inter, "interaction_z": inter / se_int,
                    **{f"{p}_{mm}": np.exp(L[p, mm] / 100) if kind == "rel" else L[p, mm] for p, mm in cells}})
    return pd.DataFrame(out).set_index("metric")


def weather_survival(cmp):
    """How much of the perfect-forecast effect survives with weather observed at the origin."""
    t = cmp[cmp.label.str.startswith("weather extremes at target")].iloc[0]
    o = cmp[cmp.label.str.startswith("weather extremes at origin")].iloc[0]
    return {k: (o[f"{k}_d"], t[f"{k}_d"]) for k in HEAD}


def seed_effects(r7):
    """Per-seed deviation from each config mean (log %), and the static-ridge effect by seed."""
    out = {}
    for k in ("all", "trans", "ind"):
        col = f"test_{METRICS[k][0]}"
        y = 100 * (np.log(r7[col]) - np.log(r7[col]).groupby(r7.experiment).transform("mean"))
        out[f"{k}_seed_dev"] = y.groupby(r7.seed).mean()
    piv = r7.pivot(index="seed", columns="experiment", values="test_obs_ind_rmse")
    out["ind_static_minus_base"] = 100 * np.log(piv.v7_profile_static / piv.v7_base)
    out["ind_static_e150_minus_base_e150"] = 100 * np.log(piv.v7_profile_static_e150 / piv.v7_base_e150)
    out["ind_static_block_minus_block"] = 100 * np.log(piv.v7_static_block / piv.v7_block)
    return pd.DataFrame(out)


def static_by_seed(se):
    """Static-ridge held-out effect averaged per seed over its three paired settings (mixed / block /
    150 epochs): seeds are the independent unit, so the sign test is on 6 values, not 18."""
    eff = se[["ind_static_minus_base", "ind_static_block_minus_block", "ind_static_e150_minus_base_e150"]]
    per_seed = eff.mean(axis=1)
    return {"pairs_favouring": int((eff < 0).sum().sum()), "pairs": eff.size,
            "seeds_favouring": int((per_seed < 0).sum()), "mean": per_seed.mean(), "median": per_seed.median(),
            "mean_without_47": per_seed.drop(47).mean(),
            "sign_p": stats.binomtest(int((per_seed < 0).sum()), len(per_seed)).pvalue,
            "per_seed": per_seed.round(2).to_dict()}


def seed_pairing(r7):
    """Share of within-config variance that a seed main effect removes (v6: 15-20%)."""
    out = {}
    for k in ("all", "trans", "ind", "crps"):
        col = f"test_{METRICS[k][0]}"
        y = np.log(r7[col]) - np.log(r7[col]).groupby(r7.experiment).transform("mean")
        res = y - y.groupby(r7.seed).transform("mean")
        out[k] = 1 - (res ** 2).sum() / (y ** 2).sum()
    return out


def epoch_stats(r7, ep7):
    out = []
    best_val = ep7.groupby("run").obs_rmse.min()
    at100 = ep7[ep7.epoch <= 100].groupby("run").obs_rmse.min()
    for e, g in r7.groupby("experiment"):
        dev = g.best_epoch - g.best_epoch.mean()
        cap = ep7[ep7.run.isin(g.run)].groupby("run").max_epochs.first()
        row = {"experiment": e, "sel_epoch_mean": g.best_epoch.mean(), "sel_epoch_sd": g.best_epoch.std(),
               "trained_epochs_mean": g.epochs_run.mean(), "hit_cap": (g.epochs_run >= cap.reindex(g.run).values).mean(),
               "sel_gt_100": (g.best_epoch > 100).mean(),
               "val_all_gain_after_100": 100 * (best_val.reindex(g.run) / at100.reindex(g.run) - 1).mean()}
        out.append(row)
    t = pd.DataFrame(out).set_index("experiment")
    x = r7[~r7.experiment.str.endswith("_e150")]
    dev = x.best_epoch - x.groupby("experiment").best_epoch.transform("mean")
    pooled = np.sqrt((dev ** 2).sum() / (x.groupby("experiment").size() - 1).sum())
    return t, pooled


def epoch_confound(r7):
    """Within-config: later selected epoch -> lower RMSE? (v6 V6: transductive rho -0.32, -0.34%/10 ep)."""
    g = r7.copy()
    g["ep_dm"] = g.best_epoch - g.groupby("experiment").best_epoch.transform("mean")
    res = {"n": len(g)}
    for k in ("trans", "all", "ind"):
        col = f"test_{METRICS[k][0]}"
        y = 100 * (np.log(g[col]) - np.log(g[col]).groupby(g.experiment).transform("mean"))
        rho = stats.spearmanr(g.ep_dm, y)
        res[k] = (round(rho.statistic, 2), float(f"{rho.pvalue:.2g}"), round(10 * np.polyfit(g.ep_dm, y, 1)[0], 2))
    return res


def val_test_heldout(r7):
    """Does the validation-window held-out RMSE (same 24 stations) track the test window?"""
    a = stats.spearmanr(r7.val_obs_ind_rmse, r7.test_obs_ind_rmse)
    dm = lambda s: s - s.groupby(r7.experiment).transform("mean")  # noqa: E731
    w = stats.spearmanr(dm(r7.val_obs_ind_rmse), dm(r7.test_obs_ind_rmse))
    cm = r7.groupby("experiment")[["val_obs_ind_rmse", "test_obs_ind_rmse"]].mean()
    c = stats.spearmanr(cm.val_obs_ind_rmse, cm.test_obs_ind_rmse)
    return {"all runs": a, "within config": w, "config means": c}


# ------------------------------------------------------------------ figures
def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_noise(noise):
    keys = ["all", "trans", "ind", "peak", "so12", "crps"]
    fams = [("V6", "v6 V6 (3 seeds, 52 train)", GRAY), ("V7", "v7 (6 seeds, 57 train)", BLUE)]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    w, x = 0.36, np.arange(len(keys))
    for i, (f, lab, c) in enumerate(fams):
        vals = [noise.loc[f, f"test_{k}"] for k in keys]
        ax.bar(x + (i - 0.5) * w, vals, w * 0.92, color=c, label=lab)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.06, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(["all-node", "transductive", "held-out", "peak", "SO@12", "CRPS"])
    ax.set_ylabel("pooled seed s.d. (%)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "fig01_seed_noise.png")


def fig_static_seeds(r7, naive_ind):
    """Held-out RMSE per seed: zero vs static-ridge profile, at each of the three paired settings."""
    pairs = [("v7_base", "v7_profile_static", "100 epochs, mixed masks"),
             ("v7_block", "v7_static_block", "100 epochs, block masks"),
             ("v7_base_e150", "v7_profile_static_e150", "150 epochs, mixed masks")]
    piv = r7.pivot(index="seed", columns="experiment", values="test_obs_ind_rmse")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (a, b, title) in zip(axes, pairs):
        for s in piv.index:
            up = piv.loc[s, b] > piv.loc[s, a]
            ax.plot([0, 1], [piv.loc[s, a], piv.loc[s, b]], color=ORANGE if up else GRAY_LIGHT, lw=1.4, zorder=1)
        ly = piv[b].sort_values()
        pos = list(ly.values)
        for i in range(1, len(pos)):   # keep seed labels at least 0.35 apart
            pos[i] = max(pos[i], pos[i - 1] + 0.35)
        for s, yv in zip(ly.index, pos):
            ax.text(1.06, yv, str(s), fontsize=7, color=INK2, va="center")
        ax.scatter(np.zeros(6), piv[a], color=GRAY, s=22, zorder=3)
        ax.scatter(np.ones(6), piv[b], color=BLUE, s=22, zorder=3)
        ax.plot([0, 1], [piv[a].mean(), piv[b].mean()], color=BLUE, lw=2.4, zorder=2)
        ax.axhline(naive_ind, color=GRAY, ls="--", lw=0.8)
        d = 100 * (piv[b].mean() / piv[a].mean() - 1)
        ax.set_title(f"{title}\nmean {piv[a].mean():.2f} → {piv[b].mean():.2f} ({d:+.1f}%)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["zero profile", "static-ridge"])
        ax.set_xlim(-0.3, 1.35)
    axes[0].text(1.3, naive_ind + 0.15, "naive profile", ha="right", va="bottom", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("test held-out RMSE (24 stations)")
    fig.text(0.01, -0.03, "Thin lines: one seed (orange = static-ridge worse). Thick line: mean over seeds 42–47.",
             fontsize=7.5, color=INK2)
    save(fig, "fig02_static_ridge_seeds.png")


def fig_factorial(fac):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, k, t in zip(axes, ("ind", "all", "trans"), ("held-out RMSE", "all-node RMSE", "transductive RMSE")):
        for p, c, lab in (("zero", GRAY, "zero profile"), ("static", BLUE, "static-ridge")):
            y = [fac.loc[k, f"{p}_mixed"], fac.loc[k, f"{p}_block"]]
            ax.plot([0, 1], y, color=c, marker="o", ms=6, label=lab)
            ax.text(1.06, y[1], lab, color=INK2, fontsize=7.5, va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["mixed masks", "block masks"])
        ax.set_xlim(-0.2, 1.6)
        ax.set_title(f"{t}\ninteraction {fac.loc[k, 'interaction']:+.1f}% (z {fac.loc[k, 'interaction_z']:+.1f})")
    save(fig, "fig04_factorial.png")


def fig_epochs(ep7, r7):
    """Validation all-node RMSE per epoch: 100- vs 150-epoch budgets."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, (a, b, title) in zip(axes, (("v7_base", "v7_base_e150", "V6 base (zero profile)"),
                                        ("v7_profile_static", "v7_profile_static_e150", "static-ridge profile"))):
        for e, c, lab in ((a, GRAY, "100-epoch budget"), (b, BLUE, "150-epoch budget")):
            sub = ep7[ep7.run.isin(r7[r7.experiment == e].run)]
            for _, g in sub.groupby("run"):
                ax.plot(g.epoch, g.obs_rmse, color=c, lw=0.5, alpha=0.35)
            med = sub.groupby("epoch").obs_rmse.median()
            n = sub.groupby("epoch").run.nunique()
            med = med[n >= 4]
            ax.plot(med.index, med.values, color=c, lw=2.2, label=lab)
            sel = r7[r7.experiment == e].best_epoch
            ax.scatter(sel, np.full(len(sel), 20.2), marker="|", s=60, color=c)
        ax.axvline(100, color=GRAY, ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("epoch  (ticks at bottom: selected epoch per run)")
    axes[0].set_ylabel("validation all-node RMSE")
    axes[0].set_ylim(20, 30)
    axes[0].legend(loc="upper right")
    save(fig, "fig05_epochs.png")


def fig_gate(gate):
    rules = [(f"{lin}/{rule}", lin, rule) for lin in ("all", "v6_lineage", "p_family")
             for rule in ("random5", "stratified8")]
    lab = {"all": "all v6 GNN runs", "v6_lineage": "V6 lineage", "p_family": "P-family"}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(rules))[::-1]
    for yi, (key, lin, rule) in zip(y, rules):
        g = gate["rules"][key]
        c = BLUE if rule == "stratified8" else GRAY
        ax.plot([g["rho_p10"], g["rho_p90"]], [yi, yi], color=c, lw=3, alpha=0.35, solid_capstyle="round")
        ax.scatter(g["rho_median"], yi, color=c, s=28, zorder=3)
        ax.text(g["rho_p90"] + 0.02, yi, f"median {g['rho_median']:.2f}, ρ>0 in {100 * g['frac_positive']:.0f}%",
                va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.axvline(gate["threshold"], color=ORANGE, ls="--", lw=1)
    ax.text(gate["threshold"], y[0] + 0.6, "gate: median ≥ 0.3", color=ORANGE, fontsize=7.5, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab[lin]} · {rule}" for _, lin, rule in rules])
    ax.set_xlabel("Spearman ρ across runs, val-subset RMSE vs test RMSE on the rest (bar = p10–p90 of draws)")
    ax.set_xlim(-0.6, 1.05)
    ax.set_ylim(-0.6, len(rules) - 0.1)
    save(fig, "fig06_rec2_gate.png")


# --------------------------------------------------------------------- main
def main():
    r, r7 = load()
    ep7 = pd.read_csv(DATA / "epochs.csv")
    gate = json.load(open(DATA / "rec2_gate.json"))
    audit = json.load(open(DATA / "rec6_stop_list.json"))
    summ = summarise(r)
    noise = pooled_noise(r[r.stage != "v5"])
    cmp = compare(r, noise)
    fac = factorial(r7, noise)
    se = seed_effects(r7)
    et, pooled_ep = epoch_stats(r7, ep7)

    summ[summ.stage == "v7"].to_csv(DATA / "seed_summary.csv", float_format="%.5g")
    noise.to_csv(DATA / "seed_noise.csv", float_format="%.5g")
    cmp.to_csv(DATA / "seed_comparisons.csv", index=False, float_format="%.5g")
    fac.to_csv(DATA / "factorial.csv", float_format="%.4g")
    se.to_csv(DATA / "seed_effects.csv", float_format="%.4g")
    et.to_csv(DATA / "epoch_stats.csv", float_format="%.4g")
    print("wrote seed_summary, seed_noise, seed_comparisons, factorial, seed_effects, epoch_stats")

    naive_ind = r[r.experiment == "naive_profile"].test_obs_ind_rmse.iloc[0]
    s6.FIG = FIG   # s6.fig_forest saves through s6.save
    fig_noise(noise)
    fig_static_seeds(r7, naive_ind)
    s6.fig_forest(cmp, ["static", "block", "weather", "epochs"], "fig03_comparisons.png",
                  "v7 comparisons (6 v 6 seeds, 57 training stations)", metrics=("all", "trans", "ind", "so12", "crps"))
    fig_factorial(fac)
    fig_epochs(ep7, r7)
    fig_gate(gate)

    pd.set_option("display.width", 250)
    print("\nnoise\n", noise[[f"test_{k}" for k in HEAD + ["so96m"]] + [f"val_{k}" for k in ("all", "ind")]].round(2))
    print("\nseed pairing (share of within-config variance removed by seed)", {k: round(v, 2) for k, v in seed_pairing(r7).items()})
    print(f"\nselected-epoch pooled s.d. (100-epoch configs) {pooled_ep:.1f}")
    print(et.round(2).to_string())
    print("\nepoch confound (rho, p, %/10 ep)", epoch_confound(r7))
    print("\nval vs test held-out")
    for k, x in val_test_heldout(r7).items():
        print(f"  {k}: rho {x.statistic:.2f} p {x.pvalue:.2g}")
    print("\nweather survival (origin Δ, target Δ)", {k: tuple(round(v, 2) for v in x) for k, x in weather_survival(cmp).items()})
    print("\nfactorial\n", fac.round(2).to_string())
    print("\nseed effects\n", se.round(2).to_string())
    print("\nstatic-ridge by seed", static_by_seed(se))
    cv = r7.groupby("experiment").test_obs_ind_rmse.agg(lambda x: 100 * np.log(x).std())
    print("\nheld-out seed s.d. per config (log %)\n", cv.round(2).to_string())
    print("\nrec 2 gate:", gate["verdict"], "| rec 6 audit:", audit["verdict"], audit["n_configs"], "configs")
    cols = ["label"] + [f"{k}_{s}" for k in ("all", "trans", "ind", "so12", "crps") for s in ("d", "z", "verdict")]
    print("\ncomparisons\n", cmp[cols].round(2).to_string())


if __name__ == "__main__":
    main()
