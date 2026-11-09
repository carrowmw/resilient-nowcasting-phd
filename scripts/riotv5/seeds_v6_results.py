#!/usr/bin/env python3
"""
Seed analysis for the RIoT v6 stage (26 experiments x seeds 42 / 43 / 44).

Applies the protocol of the v5 seed report (docs/experiments/riot_v5/riot_v5_seed_report.md §1.2):
mean ± std over 3 seeds; z = Δ / sqrt(σ_a²/n_a + σ_b²/n_b) with σ the pooled seed std of each side's
family; real if |z| ≥ 2.45 and validation agrees, suggestive if |z| ≥ 1.5.

Families: V6 (V5F / V6 lineage incl. LSTM and seq48), V6P (P2B / P3B / P5B / E2.x / E3.x under the v6
protocol), and the same two for the validation-station checkpoint ("@ind" rows, from the
*_indckpt columns). v5 rows keep their v5 families (V4R / P1-P4 / V5).

Comparability with v5: held-out RMSE is scored on the same 24 test stations. Transductive and all-node
RMSE are not: v6 runs with validation stations ignore those 5 stations at test (src/train.py), so they
score 52 / 76 stations against v5's 57 / 81. Cross-stage comparisons are test-only (the v6 validation
split scores the 5 validation stations as "ind", v5 scores the 24 test stations), and are flagged.

Inputs:  docs/experiments/riot_v6/data/runs.csv, epochs.csv; logs/v6_stage/runs/*.log;
         logs/v6_stage/posthoc/*.json; docs/experiments/riot_v5/data/runs.csv
Outputs: docs/experiments/riot_v6/data/seed_summary.csv, seed_noise.csv, seed_comparisons.csv,
         val_station_epochs.csv, ensemble.csv, legacy.csv
         docs/experiments/riot_v6/figures/*.png

Usage: python scripts/riotv5/seeds_v6_results.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeds_v5_results as s5  # noqa: E402  (METRICS, pooled_noise, family, plot style)

plt = s5.plt
ROOT = s5.ROOT
DATA = ROOT / "docs" / "experiments" / "riot_v6" / "data"
FIG = ROOT / "docs" / "experiments" / "riot_v6" / "figures"
V5_DATA = s5.DATA
LOGS = ROOT / "logs" / "v6_stage"
METRICS = dict(s5.METRICS, peakq90=("peak_q90_rmse", "rel", 1), so1=("so_1_forecast_rmse", "rel", 2),
               so48=("so_48_forecast_rmse", "rel", 2), trans_mae=("obs_trans_mae", "rel", 2),
               ind_mae=("obs_ind_mae", "rel", 2))
s5.METRICS.clear()
s5.METRICS.update(METRICS)   # pooled_noise reads the module-level dict
Z_REAL, Z_SUGG = s5.Z_REAL, s5.Z_SUGG
V6_LINEAGE = ("v6p_v5f", "v6p_v5f_sched")


def family6(exp):
    return "V6" if exp.startswith("v6_") or exp in V6_LINEAGE else "V6P"


# --------------------------------------------------------------------- data
def load():
    r6 = pd.read_csv(DATA / "runs.csv").copy()
    r6 = r6.assign(stage="v6", fam=r6.experiment.map(family6))
    # Validation-station checkpoint as its own pseudo-experiment "<exp>@ind".
    ind = r6[r6.best_epoch_ind.notna()]
    cols = {c.replace("_indckpt_", "_"): ind[c] for c in ind.columns if "_indckpt_" in c}
    ind = ind.assign(**cols, best_epoch=ind.best_epoch_ind, experiment=ind.experiment + "@ind",
                     run=ind.run + "@ind", fam=ind.fam + "@ind")
    r5 = pd.read_csv(V5_DATA / "runs.csv").copy()
    r5 = r5.assign(stage="v5", fam=r5.group.map(s5.family))
    r = pd.concat([r5, r6, ind], ignore_index=True)
    return r, r6


def parse_val_station_epochs():
    """Per-epoch validation-station RMSE, which the collector does not keep (only the run logs do)."""
    rows = []
    for f in sorted((LOGS / "runs").glob("*.log")):
        ep = None
        for line in open(f):
            m = re.search(r"Epoch (\d+)/\d+ - ", line)
            if m:
                ep = int(m.group(1))
                continue
            m = re.search(r"Validation stations: indRMSE ([\d.]+) \(best [\d.]+ at epoch \d+, lr ([\d.e-]+)\)", line)
            if m:
                rows.append((f.stem, ep, float(m.group(1)), float(m.group(2))))
    if not rows and (DATA / "val_station_epochs.csv").exists():
        return pd.read_csv(DATA / "val_station_epochs.csv")
    v = pd.DataFrame(rows, columns=["run", "epoch", "val_ind_rmse", "lr"])
    v["experiment"] = v.run.str.rsplit("-", n=1).str[0]
    return v


def load_posthoc():
    ens, leg = [], []
    for f in sorted((LOGS / "posthoc").glob("ensemble_*.json")):
        d = json.load(open(f))
        seed = int(d["name"].rsplit("-s", 1)[1])
        ck = "ind" if d["checkpoint"] == "best_model_ind.pt" else "sel"
        for member, m in d["metrics"].items():
            role = "ensemble" if member == "ensemble" else member.rsplit("-", 1)[0]
            ens.append({"checkpoint": ck, "seed": seed, "role": role,
                        **{k: m.get(c) for k, (c, _, _) in METRICS.items() if c in m}})
    for f in sorted((LOGS / "posthoc").glob("legacy_*.json")):
        for run, d in json.load(open(f))["runs"].items():
            for bs in ("bs32", "bs128"):
                leg.append({"run": run, "experiment": run.rsplit("-", 1)[0], "train_bs": d["train_batch_size"],
                            "eval_bs": bs, "mean_of_batches": d[bs]["legacy_rmse_mean_of_batches"],
                            "pooled": d[bs]["pooled_rmse"], "n_batches": d[bs]["n_batches"]})
    if not ens and (DATA / "ensemble.csv").exists():
        return pd.read_csv(DATA / "ensemble.csv"), pd.read_csv(DATA / "legacy.csv")
    return pd.DataFrame(ens), pd.DataFrame(leg)


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
                                       best_epoch_std=("best_epoch", "std"), gpu_h=("duration_h", "sum"), **agg)


# -------------------------------------------------------------- comparisons
P = "v6p_"
COMPARISONS = [
    # rec 1: protocol (cross-stage = vs the v5 3-seed / 1-seed result of the same config)
    ("protocol", "V5F: cosine schedule only (57 train)", "v6p_v5f_sched", "v5_full"),
    ("protocol", "V5F: + 5 validation stations (52 train)", "v6p_v5f", "v6p_v5f_sched"),
    ("protocol", "V5F: full protocol vs v5", "v6p_v5f", "v5_full"),
    ("protocol", "P5B: full protocol vs v5", "v6p_p5b", "e4_4_anchored"),
    ("protocol", "P3B: full protocol vs v5", "v6p_p3b", "e2_7_mixed_030"),
    ("protocol", "P2B: full protocol vs v5", "v6p_p2b", "e1_6_complete_weather"),
    ("replicate", "E2.1 vs v5 (1 seed)", "v6p_e2_1", "e2_1_unify_iid"),
    ("replicate", "E2.1b vs v5 (1 seed)", "v6p_e2_1b", "e2_1b_iid_traffic"),
    ("replicate", "E2.2 block vs v5 (1 seed)", "v6p_e2_2_block", "e2_2_block"),
    ("replicate", "E3.3d profile-zero vs v5 (1 seed)", "v6p_e3_3d_profile_zero", "e3_3d_profile_inductive_zero"),
    ("replicate", "E3.2 log1p vs v5 (1 seed)", "v6p_e3_2_log1p", "e3_2_log1p"),
    ("replicate", "E3.7b lags vs v5 (1 seed)", "v6p_e3_7b_lags", "e3_7b_lags"),
    # rec 2: building blocks, all under the v6 protocol
    ("blocks", "E2.1 vs P2B", "v6p_e2_1", "v6p_p2b"),
    ("blocks", "E2.1b vs P2B", "v6p_e2_1b", "v6p_p2b"),
    ("blocks", "E2.2 block vs P2B", "v6p_e2_2_block", "v6p_p2b"),
    ("blocks", "P3B vs P2B", "v6p_p3b", "v6p_p2b"),
    ("blocks", "E2.1 vs E2.1b (observed-only restriction)", "v6p_e2_1", "v6p_e2_1b"),
    ("blocks", "E3.3d profile-zero vs P3B", "v6p_e3_3d_profile_zero", "v6p_p3b"),
    ("blocks", "E3.2 log1p vs P3B", "v6p_e3_2_log1p", "v6p_p3b"),
    ("blocks", "E3.7b lags vs P3B", "v6p_e3_7b_lags", "v6p_p3b"),
    ("blocks", "P5B vs P3B", "v6p_p5b", "v6p_p3b"),
    ("blocks", "V5F vs P5B", "v6p_v5f", "v6p_p5b"),
    # rec 3: V6 base and variants
    ("v6", "V6 vs V5F (both full protocol)", "v6_base", "v6p_v5f"),
    ("v6", "V6 vs V5F (v5, 57 train)", "v6_base", "v5_full"),
    ("v6", "V6 vs P5B (full protocol)", "v6_base", "v6p_p5b"),
    ("v6", "V6 vs E2.2 block (full protocol)", "v6_base", "v6p_e2_2_block"),
    ("v6", "+ GAT missing-source drop", "v6_gat_drop", "v6_base"),
    ("v6", "- sensor embedding", "v6_no_sensor_emb", "v6_base"),
    ("v6", "traffic-agnostic iid masks", "v6_iid_unify_masks", "v6_base"),
    ("v6", "+ outage head", "v6_outage_head", "v6_base"),
    # rec 4: held-out / kriging track
    ("heldout", "- kriging loss", "v6_no_kriging", "v6_base"),
    ("heldout", "kriging target on all slots", "v6_kriging_iid", "v6_base"),
    ("heldout", "static-ridge profile", "v6_profile_static", "v6_base"),
    ("heldout", "residual to k-NN profile", "v6_residual_neighbour", "v6_base"),
    # rec 5: borderline LOOs
    ("loo", "- seasonal profile", "v6_no_profile", "v6_base"),
    ("loo", "- calendar", "v6_no_calendar", "v6_base"),
    ("loo", "mask ratio 0.30", "v6_mask_030", "v6_base"),
    # rec 6: peaks / context
    ("peak", "+ weather extremes (perfect forecast)", "v6_weather_extremes", "v6_base"),
    ("peak", "V6-input LSTM", "v6_lstm", "v6_base"),
    ("peak", "seq_len 48", "v6_seq48", "v6_base"),
    # validation-station checkpoint vs the usual selection, same runs
    *[("ind_ckpt", f"{e}: val-station ckpt", f"{e}@ind", e) for e in
      ["v6p_v5f", "v6p_p5b", "v6p_p2b", "v6p_e2_1", "v6p_p3b", "v6p_e2_2_block", "v6p_e3_3d_profile_zero",
       "v6_base", "v6_no_kriging", "v6_profile_static", "v6_residual_neighbour", "v6_seq48", "v6_lstm"]],
    # naive profile (held-out comparable; trans / all are on different station sets for v6)
    *[("naive", f"{e} vs naive profile", e, "naive_profile") for e in
      ["v6_base", "v6p_v5f", "v6p_v5f_sched", "v6p_p5b", "v6p_p3b", "v6p_e2_2_block", "v6p_e3_2_log1p",
       "v6_no_kriging", "v6_profile_static", "v6_weather_extremes", "v6_residual_neighbour", "v6_seq48", "v6_lstm"]],
]


def compare(r, noise):
    rows = []
    by = {e: g for e, g in r.groupby("experiment")}
    for sec, label, a, b in COMPARISONS:
        ga, gb = by[a], by[b]
        na, nb = (0 if b == a else len(ga)), (0 if b in s5.NAIVE else len(gb))
        cross = ga.stage.iloc[0] != gb.stage.iloc[0]
        row = {"section": sec, "label": label, "exp": a, "ref": b, "n_exp": len(ga), "n_ref": nb, "cross": cross}
        for k, (c, kind, _) in METRICS.items():
            col = f"test_{c}"
            if col not in r or ga[col].isna().all() or gb[col].isna().all():
                continue
            ma, mb = ga[col].mean(), gb[col].mean()
            d = 100 * (ma / mb - 1) if kind == "rel" else ma - mb
            row[f"{k}_a"], row[f"{k}_b"], row[f"{k}_d"] = ma, mb, d
            row[f"{k}_sda"] = ga[col].std()
            sig = [noise.loc[f, f"test_{k}"] if f"test_{k}" in noise else np.nan for f in
                   (ga.fam.iloc[0], gb.fam.iloc[0])]
            se = np.sqrt(sig[0] ** 2 / na + (sig[1] ** 2 / nb if nb else 0))
            row[f"{k}_z"] = d / se if se > 0 else np.nan
            vcol = f"val_{c}"
            vd = np.nan
            if not cross and b not in s5.NAIVE and vcol in r and ga[vcol].notna().any() and gb[vcol].notna().any():
                va, vb = ga[vcol].mean(), gb[vcol].mean()
                vd = 100 * (va / vb - 1) if kind == "rel" else va - vb
            row[f"val_{k}_d"] = vd
            z = row[f"{k}_z"]
            agree = np.isnan(vd) or np.sign(vd) == np.sign(d)
            row[f"{k}_verdict"] = ("real" if abs(z) >= Z_REAL and agree else
                                   "suggestive" if abs(z) >= Z_SUGG else "noise")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ckpt(r6):
    """Validation-station checkpoint minus usual checkpoint, paired by run (same weights trajectory)."""
    x = r6[r6.best_epoch_ind.notna()]
    out = []
    for e, g in x.groupby("experiment"):
        row = {"experiment": e, "fam": g.fam.iloc[0], "ep_sel": g.best_epoch.mean(), "ep_ind": g.best_epoch_ind.mean(),
               "ep_ind_le5": int((g.best_epoch_ind <= 5).sum())}
        for k in ("all", "trans", "ind", "peak", "so12", "crps"):
            c = METRICS[k][0]
            dd = 100 * (g[f"test_indckpt_{c}"] / g[f"test_{c}"] - 1)
            row[f"{k}_d"], row[f"{k}_min"], row[f"{k}_max"] = dd.mean(), dd.min(), dd.max()
        out.append(row)
    return pd.DataFrame(out).set_index("experiment")


def val_station_validity(r6):
    """Does the 5-station validation RMSE predict test held-out RMSE (24 stations)?"""
    x = r6[r6.model_type == "gnn"].copy()
    res = {}
    for name, g in (("all runs", x), ("V6", x[x.fam == "V6"]), ("V6P", x[x.fam == "V6P"])):
        res[name] = stats.spearmanr(g.val_obs_ind_rmse, g.test_obs_ind_rmse)
        dm = lambda s: s - s.groupby(g.experiment).transform("mean")  # noqa: E731
        res[name + " within-config"] = stats.spearmanr(dm(g.val_obs_ind_rmse), dm(g.test_obs_ind_rmse))
        cm = g.groupby("experiment")[["val_obs_ind_rmse", "test_obs_ind_rmse"]].mean()
        res[name + " config means"] = stats.spearmanr(cm.val_obs_ind_rmse, cm.test_obs_ind_rmse)
    return res


def epoch_stats(r, ep6):
    """Selected-epoch spread and trained-epoch count per family, v5 vs v6."""
    x = r[(r.groupby("experiment").seed.transform("size") == 3) & (~r.experiment.isin(s5.NAIVE))]
    out = []
    last = ep6.groupby("run").epoch.max()
    for fam, g in x.groupby("fam"):
        dev = g.best_epoch - g.groupby("experiment").best_epoch.transform("mean")
        row = {"fam": fam, "configs": g.experiment.nunique(),
               "sel_epoch_pooled_sd": np.sqrt((dev ** 2).sum() / (2 * g.experiment.nunique())),
               "sel_epoch_mean": g.best_epoch.mean(), "sel_epoch_lt40": (g.best_epoch < 40).mean()}
        if g.stage.iloc[0] == "v6" and not fam.endswith("@ind"):
            le = last.reindex(g.run)
            row["trained_epochs_mean"], row["hit_cap"] = le.mean(), (le >= 100).mean()
        out.append(row)
    return pd.DataFrame(out).set_index("fam")


def epoch_confound(r6):
    """Within-config: later selected epoch -> lower transductive RMSE? (v5 §9: rho -0.79 in V5)."""
    out = {}
    for fam in ("V6", "V6P"):
        g = r6[(r6.fam == fam) & (r6.model_type == "gnn")].copy()
        g["ep_dm"] = g.best_epoch - g.groupby("experiment").best_epoch.transform("mean")
        res = {"n": len(g)}
        for k in ("trans", "all", "ind"):
            col = f"test_{METRICS[k][0]}"
            y = 100 * (np.log(g[col]) - np.log(g[col]).groupby(g.experiment).transform("mean"))
            rho = stats.spearmanr(g.ep_dm, y)
            res[k] = (round(rho.statistic, 2), float(f"{rho.pvalue:.2g}"), round(10 * np.polyfit(g.ep_dm, y, 1)[0], 2))
        out[fam] = res
    return out


# ------------------------------------------------------------------ figures
BLUE, ORANGE, AQUA, GRAY, GRAY_LIGHT, INK2, RED = s5.BLUE, s5.ORANGE, s5.AQUA, s5.GRAY, s5.GRAY_LIGHT, s5.INK2, s5.RED


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_val_station_curves(v):
    """Per-epoch validation-station RMSE: P-family runs diverge after warm-up, V6-lineage runs do not."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, fam, title in ((axes[0], "V6P", "P2B / P3B / P5B / E2.x / E3.x lineage"),
                           (axes[1], "V6", "V5F / V6 lineage (GNN, seq 12)")):
        sub = v[v.experiment.map(family6) == fam]
        if fam == "V6":
            sub = sub[~sub.experiment.isin(["v6_lstm", "v6_seq48"])]
        for _, g in sub.groupby("run"):
            ax.plot(g.epoch, g.val_ind_rmse, color=GRAY_LIGHT, lw=0.8)
        med = sub.groupby("epoch").val_ind_rmse.median()
        n = sub.groupby("epoch").run.nunique()
        med = med[n >= 0.5 * sub.run.nunique()]
        ax.plot(med.index, med.values, color=BLUE if fam == "V6" else ORANGE, lw=2.2, label="median over runs")
        ax.axvline(40, color=GRAY, ls=":", lw=1)
        ax.text(41, 38.5, "min_epochs", color=INK2, fontsize=7.5, va="top")
        ax.set_title(f"{title} ({sub.run.nunique()} runs)")
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("validation-station RMSE (3 of 5 stations have data)")
    axes[0].set_ylim(14, 40)
    axes[0].legend(loc="upper left")
    save(fig, "fig01_val_station_curves.png")


def fig_forest(cmp, sections, name, title, metrics=("all", "trans", "ind", "so12")):
    sub = cmp[cmp.section.isin(sections)].iloc[::-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.8 * len(metrics) + 2.6, 0.38 * len(sub) + 1.3), sharey=True)
    lab = {"all": "all-node RMSE", "trans": "transductive RMSE", "ind": "held-out RMSE", "so12": "SO@12 RMSE",
           "peak": "peak RMSE", "crps": "CRPS"}
    y = np.arange(len(sub))
    for ax, k in zip(axes, metrics):
        d, z = sub[f"{k}_d"].values, sub[f"{k}_z"].values
        thr = np.where(np.isfinite(z) & (z != 0), np.abs(d / z) * Z_REAL, np.nan)
        ax.barh(y, 2 * thr, left=-thr, height=0.58, color="#ebeae6", zorder=0)
        col = [BLUE if v == "real" else (ORANGE if v == "suggestive" else GRAY) for v in sub[f"{k}_verdict"]]
        ax.scatter(d, y, c=col, s=34, zorder=3)
        ax.axvline(0, color=GRAY, lw=0.8)
        ax.set_title(lab[k], fontsize=12, fontweight="bold")
        ax.set_xlabel("Δ vs reference (%)", fontsize=11)
        lo, hi = np.nanmin(np.r_[d, -thr, 0]), np.nanmax(np.r_[d, thr, 0])
        ax.set_xlim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub.label + np.where(sub.cross, "  †", ""), fontsize=10.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    handles = [plt.Line2D([], [], ls="", marker="o", color=c, label=l) for c, l in
               ((BLUE, f"real (|z| ≥ {Z_REAL})"), (ORANGE, f"suggestive (|z| ≥ {Z_SUGG})"), (GRAY, "noise"))]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#ebeae6", label="real-effect threshold"))
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=10)
    save(fig, name)


def fig_ind_ckpt(pc):
    pc = pc.sort_values("ind_d")
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(pc) + 1))
    y = np.arange(len(pc))
    for i, (e, rr) in enumerate(pc.iterrows()):
        ax.plot([rr.ind_min, rr.ind_max], [i, i], color=GRAY_LIGHT, lw=3, solid_capstyle="round")
    ax.scatter(pc.ind_d, y, color=[BLUE if f == "V6" else ORANGE for f in pc.fam], s=24, zorder=3)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}  (ep {rr.ep_ind:.0f} vs {rr.ep_sel:.0f})" for e, rr in pc.iterrows()])
    ax.set_xlabel("test held-out RMSE, val-station ckpt vs usual ckpt (%; dot = mean, bar = seed range)")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=c, label=l) for c, l in
                       ((BLUE, "V6 lineage"), (ORANGE, "P-family"))], loc="lower right")
    save(fig, "fig04_ind_checkpoint.png")


def fig_val_vs_test(r6):
    x = r6[r6.model_type == "gnn"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for fam, c in (("V6", BLUE), ("V6P", ORANGE)):
        g = x[x.fam == fam]
        ax.scatter(g.val_obs_ind_rmse, g.test_obs_ind_rmse, s=28, color=c, label=fam, alpha=0.85)
    ax.axhline(42.63, color=GRAY, ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], 42.4, "naive profile ", ha="right", va="top", fontsize=9.5, color=INK2)
    ax.set_xlabel("validation-station RMSE (3 stations with data, usual ckpt)", fontsize=10.5)
    ax.set_ylabel("test held-out RMSE (24 stations)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=10)
    save(fig, "fig05_val_vs_test_heldout.png")


def fig_noise(noise):
    keys = ["all", "trans", "ind", "peak", "so12", "crps"]
    fams = [("P1-P4", "v5 P1–P4", GRAY), ("V6P", "v6 P-family", ORANGE), ("V5", "v5 V5", GRAY_LIGHT),
            ("V6", "v6 V6", BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 3))
    w = 0.2
    x = np.arange(len(keys))
    for i, (f, lab, c) in enumerate(fams):
        ax.bar(x + (i - 1.5) * w, [noise.loc[f, f"test_{k}"] for k in keys], w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["all-node", "transductive", "held-out", "peak", "SO@12", "CRPS"])
    ax.set_ylabel("pooled seed s.d. (%)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "fig02_seed_noise.png")


def fig_bases(summ):
    order = ["naive_profile", "v5_full", "e4_4_anchored", "v6p_p2b", "v6p_e2_1", "v6p_p3b", "v6p_e2_2_block",
             "v6p_e3_2_log1p", "v6p_p5b", "v6p_v5f_sched", "v6p_v5f", "v6_base", "v6_seq48"]
    lab = {"naive_profile": "naive profile", "v5_full": "V5F (v5)", "e4_4_anchored": "P5B (v5)"}
    s = summ.reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    y = np.arange(len(s))[::-1]
    for ax, k, t in zip(axes, ("trans", "ind", "so12"), ("transductive RMSE *", "held-out RMSE", "SO@12 RMSE *")):
        m, sd = s[f"test_{k}_mean"].values, s[f"test_{k}_std"].fillna(0).values
        c = [GRAY if s.stage.iloc[i] == "v5" or s.index[i] == "naive_profile" else
             (BLUE if family6(s.index[i]) == "V6" else ORANGE) for i in range(len(s))]
        ax.errorbar(m, y, xerr=sd, fmt="none", ecolor=GRAY_LIGHT, elinewidth=3)
        ax.scatter(m, y, c=c, s=26, zorder=3)
        ax.axvline(s.loc["naive_profile", f"test_{k}_mean"], color=GRAY, ls="--", lw=0.8)
        ax.set_title(t)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab.get(e, e) for e in s.index])
    fig.text(0.01, -0.02, "* v6 runs with validation stations score 52 transductive stations (v5 and v6p_v5f_sched: 57).",
             fontsize=7.5, color=INK2)
    save(fig, "fig03_bases.png")


# --------------------------------------------------------------------- main
def main():
    r, r6 = load()
    v = parse_val_station_epochs()
    ens, leg = load_posthoc()
    ep6 = pd.read_csv(DATA / "epochs.csv")
    summ = summarise(r)
    noise = s5.pooled_noise(r)
    cmp = compare(r, noise)
    pc = paired_ckpt(r6)
    DATA.mkdir(parents=True, exist_ok=True)
    summ[summ.stage == "v6"].to_csv(DATA / "seed_summary.csv", float_format="%.5g")
    noise.to_csv(DATA / "seed_noise.csv", float_format="%.5g")
    cmp.to_csv(DATA / "seed_comparisons.csv", index=False, float_format="%.5g")
    v.to_csv(DATA / "val_station_epochs.csv", index=False)
    ens.to_csv(DATA / "ensemble.csv", index=False, float_format="%.5g")
    leg.to_csv(DATA / "legacy.csv", index=False, float_format="%.5g")
    pc.to_csv(DATA / "ind_checkpoint_paired.csv", float_format="%.4g")
    print("wrote seed_summary, seed_noise, seed_comparisons, val_station_epochs, ensemble, legacy, ind_checkpoint_paired")

    fig_val_station_curves(v)
    fig_noise(noise)
    fig_bases(summ)
    fig_ind_ckpt(pc)
    fig_val_vs_test(r6)
    fig_forest(cmp, ["blocks"], "fig06_blocks.png", "Building blocks under the v6 protocol (3 v 3 seeds)")
    fig_forest(cmp, ["v6", "heldout", "loo", "peak"], "fig07_v6_variants.png",
               "V6 base and variants (3 v 3 seeds; † = vs v5, test only)")
    fig_forest(cmp, ["protocol", "replicate"], "fig08_protocol_replication.png",
               "v6 protocol vs the v5 results of the same configs († = cross-stage, test only)")

    print("\nepoch stats\n", epoch_stats(r, ep6).round(2).to_string())
    print("\nepoch confound (rho, p, %/10 ep)", epoch_confound(r6))
    print("\nval-station validity")
    for k, x in val_station_validity(r6).items():
        print(f"  {k}: rho {x.statistic:.2f} p {x.pvalue:.2g}")


if __name__ == "__main__":
    main()
