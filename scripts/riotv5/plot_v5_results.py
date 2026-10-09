#!/usr/bin/env python3
"""
Figures for the RIoT v5 results report.

Inputs (written by collect_v5_results.py and the W&B pull):
  docs/experiments/riot_v5/data/runs.csv
  docs/experiments/riot_v5/data/epochs.csv
  docs/experiments/riot_v5/data/grad_cosine_history.csv

Output: docs/experiments/riot_v5/figures/figNN_*.png (static, light mode)

All metrics are test-split, seed 42, raw counts on observed targets unless stated.

Usage: python scripts/riotv5/plot_v5_results.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "experiments" / "riot_v5" / "data"
FIG = ROOT / "docs" / "experiments" / "riot_v5" / "figures"

# Categorical slots in fixed order; gray for context; diverging poles.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY = "#9a9893"
GRAY_LIGHT = "#d6d4cf"
MID = "#f0efec"
INK, INK2 = "#1f1f1f", "#5c5b57"
GOOD, BAD = BLUE, RED  # diverging poles: better (blue) / worse (red)

BASES = [("V4R", "v4r_full"), ("P2B", "e1_6_complete_weather"), ("P3B", "e2_7_mixed_030"),
         ("P4B", "e3_stack"), ("P5B", "e4_4_anchored"), ("V5F", "v5_full")]

# Group families for scatter plots (fold P4–P6 so at most 8 hues are used).
FAMILY = {"thesis_v4": "Thesis re-run (V4R)", "p1_hygiene": "P1 hygiene", "p2_masking": "P2 masking",
          "p3_inputs": "P3 inputs", "p4_likelihood": "P4–P6 screens", "p5_magnitude": "P4–P6 screens",
          "p6_mtl": "P4–P6 screens", "p7_v5_suite": "v5 suite", "loo": "LOO from V5F"}
FAMILY_COLOR = dict(zip(["Thesis re-run (V4R)", "P1 hygiene", "P2 masking", "P3 inputs",
                         "P4–P6 screens", "v5 suite", "LOO from V5F"], CAT))

NAIVE = {"naive_profile", "naive_last_obs"}
ORACLE = {"e3_3a_profile_transductive"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 2, "savefig.dpi": 160, "figure.dpi": 160,
    "text.color": INK,
})


def load():
    # Seed 42 only: the tier-3 seeds are analysed separately by seeds_v5_results.py.
    r = pd.read_csv(DATA / "runs.csv")
    r = r[r.seed == 42].copy()
    r["exp"] = r.run.str.replace("-s42", "", regex=False)
    r = r.set_index("exp")
    ep = pd.read_csv(DATA / "epochs.csv")
    ep = ep[ep.run.str.endswith("-s42")].copy()
    ep["exp"] = ep.run.str.replace("-s42", "", regex=False)
    gc = pd.read_csv(DATA / "grad_cosine_history.csv")
    gc = gc[gc.run.str.endswith("-s42")].copy()
    gc["exp"] = gc.run.str.replace("-s42", "", regex=False)
    return r, ep, gc


def pct(r, exp, base, col):
    return 100.0 * (r.loc[exp, col] / r.loc[base, col] - 1.0)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig01 chain
def fig_chain(r):
    names = [b for b, _ in BASES]
    runs = [e for _, e in BASES]
    panels = [
        ("test_obs_trans_rmse", "Transductive RMSE", "lower is better"),
        ("test_obs_ind_rmse", "Inductive (held-out) RMSE", "lower is better"),
        ("test_peak_rmse", "Peak RMSE (y ≥ 58)", "lower is better"),
        ("test_so_12_forecast_rmse", "SO@12 forecast RMSE", "lower is better"),
        ("test_so_96_mass_ratio", "SO@96 mass ratio", "1 = no mass lost"),
        ("test_collapse_rate", "Real-gap collapse rate", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    x = np.arange(len(runs))
    for ax, (col, title, sub) in zip(axes.flat, panels):
        y = r.loc[runs, col].values
        ax.plot(x, y, color=BLUE, marker="o", ms=5, zorder=3)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}" if yi < 5 else f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=INK2)
        if col in r.columns and not np.isnan(r.loc["naive_profile", col]) and "mass" not in col \
                and "collapse" not in col:
            ax.axhline(r.loc["naive_profile", col], color=GRAY, ls="--", lw=1)
            ax.annotate("naive profile", (0, r.loc["naive_profile", col]), xytext=(4, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=7, color=INK2)
        if "mass" in col:
            ax.axhline(1.0, color=GRAY, ls="--", lw=1)
        ax.set_xticks(x, names)
        ax.set_title(f"{title}\n", loc="left")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7, color=INK2)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0) if "mass" in col or "collapse" in col else lo, hi + 0.08 * (hi - lo))
    fig.suptitle("Pre-declared base chain, V4R → V5F (test, seed 42)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "fig01_base_chain.png")


# ------------------------------------------------------- fig02 outage length
def fig_outage(r):
    runs = [("v4r_full", "V4R"), ("e1_6_complete_weather", "P2B (hygiene)"),
            ("e2_1_unify_iid", "P2B + unified encoding"), ("v5_full", "V5F"),
            ("loo_no_unify", "V5F − unified encoding")]
    Ls = [1, 4, 12, 48, 96]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (exp, lab), c in zip(runs, CAT):
        axes[0].plot(Ls, [r.loc[exp, f"test_so_{L}_forecast_rmse"] for L in Ls], color=c, marker="o",
                     ms=4, label=lab)
        axes[1].plot(Ls, [r.loc[exp, f"test_so_{L}_mass_ratio"] for L in Ls], color=c, marker="o",
                     ms=4, label=lab)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(Ls, [str(L) for L in Ls])
        ax.minorticks_off()
        ax.set_xlabel("Synthetic outage length L (15-min steps)")
        ax.axvspan(12, 96, color=MID, zorder=0)
        ax.text(np.sqrt(12 * 96), 0.03, "L ≥ 12: same inputs hidden (seq_len 12)",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK2)
    axes[0].set_title("Forecast RMSE on outaged nodes", loc="left")
    axes[0].set_ylabel("RMSE (counts)")
    axes[1].set_title("Predicted / true mass on outaged nodes", loc="left")
    axes[1].axhline(1, color=GRAY, ls="--", lw=1)
    axes[1].set_ylim(-0.05, 1.18)
    axes[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    save(fig, "fig02_outage_length.png")


# ---------------------------------------------------------- fig03 replicates
def fig_replicates(r, ep):
    pairs = [(("v5_full", "v5_full"), ("loo_val_loss_selection", "loo_val_loss_selection")),
             (("v4r_full", "v4r_full"), ("v4r_bs128_steps4x", "v4r_bs128_steps4x (4× steps)"))]
    titles = ["V5F: identical training config, different checkpoint rule",
              "V4R: identical config except max epochs / patience"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ((a, la), (b, lb)), t in zip(axes, pairs, titles):
        for exp, lab, c in ((a, la, BLUE), (b, lb, ORANGE)):
            e = ep[ep.exp == exp].sort_values("epoch")
            ax.plot(e.epoch, e.obs_rmse, color=c, lw=1.2, label=lab)
            be = r.loc[exp, "best_epoch"]
            ax.axvline(be, color=c, lw=0.8, ls=":")
        ax.set_title(t, loc="left", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation obs RMSE")
        v = ep[ep.exp.isin([a, b])].obs_rmse
        ax.set_ylim(v.min() * 0.995, v.quantile(0.95))
        ax.legend(loc="upper right")
    fig.text(0.01, -0.02, "Dotted lines: selected checkpoint. Same seed; divergence is run-to-run "
             "non-determinism (≈1.4–1.9% per-epoch s.d.).", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "fig03_replicate_noise.png")


# ---------------------------------------------------------- fig04 best epoch
def fig_best_epoch(r):
    gnn = r[~r.index.isin(NAIVE | ORACLE) & (r.model_type != "lstm")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ctx = gnn[~gnn.group.isin(["p3_inputs", "p4_likelihood", "p5_magnitude"])]
    ax.scatter(ctx.best_epoch, ctx.test_obs_trans_rmse, s=22, color=GRAY_LIGHT, edgecolor=GRAY,
               lw=0.5, label="other groups", zorder=2)
    for g, lab, c in (("p3_inputs", "P3 inputs", YELLOW), ("p4_likelihood", "P4 likelihood", MAGENTA),
                      ("p5_magnitude", "P5 magnitude", VIOLET)):
        d = gnn[gnn.group == g]
        ax.scatter(d.best_epoch, d.test_obs_trans_rmse, s=34, color=c, edgecolor="white", lw=1,
                   label=lab, zorder=3)
    for exp, lab in (("e3_3b_profile_inductive_neighbour", "profile: neighbour / static"),
                     ("e3_4_runlength", "run-length"), ("e3_stack_seq48", "stack @ seq48"),
                     ("e3_stack", "P4B"),
                     ("e4_4_anchored", "P5B"), ("v5_full", "V5F"), ("e3_3d_profile_inductive_zero",
                                                                     "profile: zero")):
        ax.annotate(lab, (r.loc[exp, "best_epoch"], r.loc[exp, "test_obs_trans_rmse"]),
                    xytext=(6, 2), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel("Selected (best) epoch")
    ax.set_ylabel("Test transductive RMSE")
    ax.set_title("Early stopping confounds the P3–P5 screens\n"
                 "Spearman ρ = −0.90 within P3–P5 (n = 30), −0.67 over all GNN runs", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig04_best_epoch_confound.png")


# ----------------------------------------------------------------- fig05 LOO
def _tornado(ax, labels, vals, band, title, unit="%"):
    y = np.arange(len(labels))
    cols = [GOOD if v < -band else BAD if v > band else GRAY for v in vals]
    ax.axvspan(-band, band, color=MID, zorder=0)
    ax.barh(y, vals, color=cols, height=0.62, zorder=2)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=9)
    xmax = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), band) * 1.25
    ax.set_xlim(-xmax, xmax)
    for yi, v in zip(y, vals):
        if abs(v) > band:
            ax.text(v + (xmax * 0.03 if v > 0 else -xmax * 0.03), yi, f"{v:+.0f}" if unit == "%" else
                    f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=7, color=INK2)


def fig_loo(r):
    loo = r[r.group == "loo"].index.tolist()
    order = sorted(loo, key=lambda e: -pct(r, e, "v5_full", "test_so_12_forecast_rmse"))
    labels = [e.replace("loo_", "") for e in order]
    specs = [("test_obs_all_rmse", "Δ all-node RMSE (%)", 3, "%"),
             ("test_obs_ind_rmse", "Δ inductive RMSE (%)", 5, "%"),
             ("test_so_12_forecast_rmse", "Δ SO@12 RMSE (%)", 3, "%"),
             ("test_so_96_forecast_rmse", "Δ SO@96 RMSE (%)", 3, "%"),
             ("test_collapse_rate", "Δ real-gap collapse (abs.)", 0.02, "abs")]
    fig, axes = plt.subplots(1, len(specs), figsize=(13, 4.8), sharey=True)
    for ax, (col, title, band, unit) in zip(axes, specs):
        if unit == "%":
            vals = [pct(r, e, "v5_full", col) for e in order]
        else:
            vals = [r.loc[e, col] - r.loc["v5_full", col] for e in order]
        _tornado(ax, labels, vals, band, title, unit)
    fig.suptitle("Leave-one-out from V5F (positive = removing the component hurts). "
                 "Shaded = noise band; blue/red = beyond band", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig05_loo_tornado.png")


# ------------------------------------------------------ fig06 thesis ablation
def fig_thesis(r):
    thesis = {"no_impute": 7.56, "no_kriging": 3.25, "no_masks": 9.62, "no_pe": 4.48, "lstm": 22.42,
              "bs32": -7.29, "dropout03": 7.50}
    labs = list(thesis)
    v4 = [pct(r, f"v4r_{a}", "v4r_full", "test_obs_all_rmse") for a in labs]
    v5 = [pct(r, f"v5_{a}", "v5_full", "test_obs_all_rmse") for a in labs]
    y = np.arange(len(labs))
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.axvspan(-3, 3, color=MID, zorder=0)
    for k, (vals, lab, c) in enumerate(((list(thesis.values()), "Thesis (v4, contaminated RMSE)", GRAY),
                                        (v4, "V4R re-run (observed-only RMSE)", BLUE),
                                        (v5, "V5F (observed-only RMSE)", ORANGE))):
        ax.barh(y + (k - 1) * (h + 0.02), vals, height=h, color=c, label=lab, zorder=2)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, labs)
    ax.invert_yaxis()
    ax.set_xlabel("Δ test RMSE vs full model (%)  —  positive = ablation hurts")
    ax.set_title("Experiment 3 ablations: thesis vs v5 re-runs (shaded = ±3% noise band)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig06_thesis_ablations.png")


# -------------------------------------------------------- fig07 trans vs ind
def fig_trans_ind(r):
    d = r[~r.index.isin(ORACLE)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for fam, c in FAMILY_COLOR.items():
        s = d[(d.group.map(FAMILY) == fam) & ~d.index.isin(NAIVE)]
        ax.scatter(s.test_obs_trans_rmse, s.test_obs_ind_rmse, s=26, color=c, edgecolor="white", lw=0.8,
                   label=fam, zorder=3)
    n = d.loc[list(NAIVE)]
    ax.scatter(n.test_obs_trans_rmse, n.test_obs_ind_rmse, s=40, marker="s", color=INK2, label="naive",
               zorder=3)
    for exp, lab, off, ha in (("v4r_full", "V4R", (5, 3), "left"), ("v5_full", "V5F", (-5, 4), "right"),
                              ("e2_1_unify_iid", "e2_1 unify", (-6, -2), "right"),
                              ("e3_3d_profile_inductive_zero", "e3_3d profile zero", (6, -4), "left"),
                              ("e3_2_log1p", "e3_2 log1p", (6, 4), "left"),
                              ("naive_profile", "naive profile", (0, -12), "center"),
                              ("naive_last_obs", "naive last obs", (0, -12), "center"),
                              ("e3_3b_profile_inductive_neighbour", "e3_3b profile neighbour", (5, 3), "left"),
                              ("e3_4_runlength", "e3_4 run-length", (-5, 5), "right"),
                              ("v4r_lstm", "v4r LSTM", (5, 3), "left")):
        ax.annotate(lab, (d.loc[exp, "test_obs_trans_rmse"], d.loc[exp, "test_obs_ind_rmse"]),
                    xytext=off, textcoords="offset points", ha=ha, fontsize=7, color=INK2)
    ax.set_xlabel("Transductive RMSE (test)")
    ax.set_ylabel("Inductive (held-out) RMSE (test)")
    ax.set_title("Transductive gains do not transfer to held-out stations", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, "fig07_trans_vs_ind.png")


# -------------------------------------------------------------- fig08 bias
def fig_bias(r):
    runs = [("v4r_full", "V4R"), ("e1_6_complete_weather", "P2B"), ("e2_1_unify_iid", "e2_1 unify"),
            ("v5_full", "V5F"), ("v5_lstm", "v5 LSTM"), ("naive_profile", "naive profile")]
    cols = [f"test_bias_d{i}" for i in range(10)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (exp, lab), c in zip(runs, CAT[:5] + [GRAY]):
        ax.plot(range(1, 11), r.loc[exp, cols].values, color=c, marker="o", ms=3.5, lw=1.6, label=lab,
                ls="--" if exp == "naive_profile" else "-")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile of observed target (1 = lowest, 10 = highest)")
    ax.set_ylabel("Mean (prediction − truth), counts")
    ax.set_title("Bias by target decile: every model regresses to the mean at the top decile",
                 loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig08_bias_by_decile.png")


# --------------------------------------------------------------- fig09 PIT
def fig_pit(r):
    runs = [("v4r_full", "V4R (ZINB)"), ("e1_6_complete_weather", "P2B (NB)"),
            ("v5_no_outage_head", "V5F without outage head (NB)"), ("v5_full", "V5F (outage mixture)")]
    cols = [f"test_pit_b{i}" for i in range(10)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
    for ax, (exp, lab) in zip(axes, runs):
        v = r.loc[exp, cols].values.astype(float)
        v = v / v.sum() if v.sum() > 1.5 else v
        ax.bar(np.arange(10) + 0.5, v, width=0.92, color=BLUE, zorder=2)
        ax.axhline(0.1, color=GRAY, ls="--", lw=1)
        ax.set_title(f"{lab}\nCRPS {r.loc[exp, 'test_crps']:.2f}, q90 cov "
                     f"{r.loc[exp, 'test_q90_coverage']:.2f}", loc="left", fontsize=8)
        ax.set_xticks([0, 5, 10], ["0", "0.5", "1"])
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("Fraction")
    fig.suptitle("PIT histograms (flat = calibrated; mass piled at 1 = truth above the predictive "
                 "distribution, i.e. under-prediction)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig09_pit.png")


# ------------------------------------------------------- fig10 grad cosine
def fig_cosine(r, gc):
    m = gc.groupby("exp").cos.mean()
    order = ["thesis_v4", "p1_hygiene", "p2_masking", "p3_inputs", "p4_likelihood", "p5_magnitude",
             "p6_mtl", "loo", "p7_v5_suite"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        exps = [e for e in r.index[r.group == g] if e in m.index]
        vals = m.loc[exps]
        zero = [e for e in exps if r.loc[e, "model_type"] == "lstm" or "no_impute" in e]
        ok = vals.drop(zero)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(ok)), ok, s=16, color=BLUE, alpha=0.8, zorder=3)
        if zero:
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(zero)), vals.loc[zero], s=16, color=GRAY_LIGHT,
                       edgecolor=GRAY, lw=0.5, zorder=3)
        ax.hlines(ok.median(), i - 0.3, i + 0.3, color=INK, lw=1.5, zorder=4)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(len(order)), [g.split("_")[0] if g != "thesis_v4" else "thesis" for g in order])
    ax.set_ylabel("Mean cos(∇forecast, ∇impute) over training")
    ax.set_title("Forecast/impute gradients never conflict\n(bar = group median; gray = w_impute 0 / LSTM)",
                 loc="left", fontsize=9)
    ax = axes[1]
    for (lab, exp), c in zip(BASES, CAT):
        s = gc[gc.exp == exp].sort_values("epoch")
        ax.plot(s.epoch, s.cos.rolling(3, min_periods=1).mean(), color=c, lw=1.4, label=lab)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("cos (3-epoch rolling mean)")
    ax.set_title("Cosine over training for the six bases", loc="left", fontsize=9)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "fig10_grad_cosine.png")


# ------------------------------------------------------- fig11 degeneracy
def fig_degeneracy(r):
    runs = ["v4r_full", "e1_6_complete_weather", "e2_7_mixed_030", "e3_3d_profile_inductive_zero",
            "e3_stack", "e4_4_anchored", "v5_full", "v5_lstm", "naive_profile"]
    labs = ["V4R", "P2B", "P3B", "e3_3d profile zero", "P4B", "P5B", "V5F", "v5 LSTM", "naive profile"]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    y = np.arange(len(runs))
    pred = r.loc[runs, "test_ind_pred_pairwise_corr_mean"].values
    corr = r.loc[runs, "test_ind_corr_mean"].values
    true = r.loc["v5_full", "test_ind_true_pairwise_corr_mean"]
    ax.axvline(true, color=GRAY, ls="--", lw=1)
    ax.text(true, len(runs) - 0.6, f" truth pairwise corr {true:.2f}", fontsize=7, color=INK2,
            va="bottom")
    ax.scatter(pred, y, s=40, color=ORANGE, label="pairwise corr between held-out predictions", zorder=3)
    ax.scatter(corr, y, s=40, color=BLUE, marker="D", label="corr(prediction, truth) per station",
               zorder=3)
    for yi, p, c in zip(y, pred, corr):
        ax.plot([min(p, c), max(p, c)], [yi, yi], color=GRAY_LIGHT, lw=1, zorder=1)
    ax.set_yticks(y, labs)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("Mean Pearson correlation")
    ax.set_title("Inductive degeneracy: held-out predictions converge to a shared profile", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=1)
    fig.tight_layout()
    save(fig, "fig11_inductive_degeneracy.png")


# ------------------------------------------- fig12 accuracy vs resilience
def fig_tradeoff(r):
    d = r[~r.index.isin(ORACLE)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for fam, c in FAMILY_COLOR.items():
        s = d[(d.group.map(FAMILY) == fam) & ~d.index.isin(NAIVE)]
        ax.scatter(s.test_obs_all_rmse, s.test_so_12_forecast_rmse, s=26, color=c, edgecolor="white",
                   lw=0.8, label=fam, zorder=3)
    n = d.loc[list(NAIVE)]
    ax.scatter(n.test_obs_all_rmse, n.test_so_12_forecast_rmse, s=40, marker="s", color=INK2,
               label="naive", zorder=3)
    for exp, lab, off in (("v4r_full", "V4R", (5, 3)), ("v5_full", "V5F", (25, -18)),
                          ("e2_1_unify_iid", "e2_1 unify", (5, 3)), ("e3_7b_lags", "e3_7b lags", (-20, -18)),
                          ("e3_6_seq48", "e3_6 seq48", (5, 3)), ("loo_no_unify", "loo no_unify", (5, 3)),
                          ("naive_profile", "naive profile", (5, 6)), ("v5_lstm_plain", "v5 plain LSTM", (5, 3)),
                          ("e3_3d_profile_inductive_zero", "e3_3d profile zero", (5, 3)),
                          ("e3_4_runlength", "e3_4 run-length", (-5, 6))):
        xy = (d.loc[exp, "test_obs_all_rmse"], d.loc[exp, "test_so_12_forecast_rmse"])
        ax.annotate(lab, xy, xytext=off, textcoords="offset points", fontsize=7, color=INK2,
                    ha="right" if off[0] < 0 else "left",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6) if abs(off[1]) > 10 else None)
    ax.set_xlabel("All-node RMSE, normal conditions (test)")
    ax.set_ylabel("SO@12 forecast RMSE (outaged nodes)")
    ax.set_title("Normal accuracy vs outage resilience", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, "fig12_accuracy_vs_resilience.png")


# --------------------------------------------- fig13 masks × impute (Lim 9)
def fig_interaction(r):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)
    for ax, pre, title in ((axes[0], "v4r", "V4R"), (axes[1], "v5", "V5F")):
        full = f"{pre}_full"
        cells = {(1, 1): full, (0, 1): f"{pre}_no_masks", (1, 0): f"{pre}_no_impute",
                 (0, 0): f"{pre}_no_masks_no_impute"}
        for imp, c, lab in ((1, BLUE, "imputation loss on"), (0, ORANGE, "imputation loss off")):
            ys = [r.loc[cells[(m, imp)], "test_obs_all_rmse"] for m in (1, 0)]
            ax.plot([0, 1], ys, color=c, marker="o", ms=5, label=lab)
            for xi, yi in zip([0, 1], ys):
                ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(6, 0), textcoords="offset points", fontsize=7,
                            color=INK2, va="center")
        base = r.loc[full, "test_obs_all_rmse"]
        ax.axhspan(base * 0.97, base * 1.03, color=MID, zorder=0)
        ax.set_xticks([0, 1], ["mask channels on", "mask channels off"])
        ax.set_xlim(-0.3, 1.4)
        ax.set_title(f"{title}: masks × imputation (shaded = ±3% of full)", loc="left", fontsize=9)
        ax.set_ylabel("All-node RMSE (test)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig13_masks_x_impute.png")


# ----------------------------------------------------- fig14 phase screens
SCREEN_BASE = {"p1_hygiene": "v4r_full", "p2_masking": "e1_6_complete_weather",
               "p3_inputs": "e2_7_mixed_030", "p5_magnitude": "e4_4_anchored", "p6_mtl": "v5_full"}
P4_OLD = {"e4_1_ztnb", "e4_2_mixture_latent", "e4_3_mixture_missingness", "e4_3b_mixture_detached",
          "e4_4_anchored"}


def screen_base(r, exp):
    g = r.loc[exp, "group"]
    if g == "p4_likelihood":
        return "e3_stack" if exp in P4_OLD else "e4_4_anchored"
    if exp == "e5_8_full_plus_underpred":
        return "v5_full"
    return SCREEN_BASE[g]


def fig_screens(r):
    groups = ["p1_hygiene", "p2_masking", "p3_inputs", "p4_likelihood", "p5_magnitude", "p6_mtl"]
    rows, seps = [], []
    for g in groups:
        exps = [e for e in r.index[r.group == g] if e != screen_base(r, e)]
        rows += [(g, e, screen_base(r, e)) for e in sorted(exps)]
        seps.append(len(rows))
    base_name = {e: n for n, e in BASES}
    labels = [f"{e}  (vs {base_name[b]})" for _, e, b in rows]
    specs = [("test_obs_all_rmse", "Δ all-node RMSE (%)", 3), ("test_obs_ind_rmse", "Δ inductive RMSE (%)", 5),
             ("test_peak_rmse", "Δ peak RMSE (%)", 3), ("test_so_12_forecast_rmse", "Δ SO@12 RMSE (%)", 3)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 11), sharey=True)
    for ax, (col, title, band) in zip(axes, specs):
        vals = [pct(r, e, b, col) for _, e, b in rows]
        _tornado(ax, labels, vals, band, title)
        for s in seps[:-1]:
            ax.axhline(s - 0.5, color=GRAY, lw=0.6)
    for s0, s1, g in zip([0] + seps[:-1], seps, groups):
        axes[-1].text(1.02, (s0 + s1 - 1) / 2, g, transform=axes[-1].get_yaxis_transform(), va="center",
                      fontsize=8, color=INK2)
    fig.suptitle("One-factor screens vs their declared base (negative = better). Base shown in brackets; "
                 "e3_3a is an oracle", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig14_phase_screens.png")


# -------------------------------------------------- fig15 soft collapse (P1)
def fig_soft_collapse(r):
    runs = ["v4r_full", "e1_1_obs_forecast", "e1_2_obs_impute", "e1_3_obs_kriging", "e1_4_selection",
            "e1_5_nb", "e1_6_complete_weather", "e2_1_unify_iid", "e2_7_mixed_030", "v5_full"]
    labs = ["V4R", "E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6 P2B", "E2.1 unify", "P3B", "V5F"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(runs))
    for col, lab, c in (("test_collapse_rate", "collapse rate (point < 1)", RED),
                        ("test_real_gap_mass_ratio", "real-gap mass ratio", BLUE),
                        ("test_so_96_mass_ratio", "SO@96 mass ratio", AQUA)):
        ax.plot(x, r.loc[runs, col].values, color=c, marker="o", ms=4, label=lab)
    ax.axvspan(0.6, 6.4, color=MID, zorder=0)
    ax.text(3.5, 1.08, "P1 hygiene staircase: hard collapse gone, soft collapse remains",
            ha="center", fontsize=7, color=INK2)
    ax.axhline(1, color=GRAY, ls="--", lw=1)
    ax.set_xticks(x, labs, rotation=0, fontsize=7.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Rate / ratio")
    ax.set_title("From hard to soft collapse: the < 1 threshold hides lost mass", loc="left")
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "fig15_soft_collapse.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r, ep, gc = load()
    fig_chain(r)
    fig_outage(r)
    fig_replicates(r, ep)
    fig_best_epoch(r)
    fig_loo(r)
    fig_thesis(r)
    fig_trans_ind(r)
    fig_bias(r)
    fig_pit(r)
    fig_cosine(r, gc)
    fig_degeneracy(r)
    fig_tradeoff(r)
    fig_interaction(r)
    fig_screens(r)
    fig_soft_collapse(r)


if __name__ == "__main__":
    main()
