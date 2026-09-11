#!/usr/bin/env python3
"""Meta-analysis across the last 12 models (tough-dragon-7 → logical-surf-18).

Joins each run's outlier-analysis ``results.json`` (under ``outputs/runs/<id>/``)
with its training config from the run registry, compares outlier-event
performance across architectural decisions, and writes:

- ``outputs/meta_report.md``       — the meta report
- ``outputs/meta_results.json``    — the joined per-run table (machine-readable)
- ``outputs/figures/meta_*.png``   — comparison figures

Usage:  python compile_meta_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
RUNS_DIR = OUT_DIR / "runs"
REGISTRY = EXP_DIR / "cache" / "runs" / "run_registry.json"

# The 12 most recent models, newest first (registry generated_at order).
RUN_ORDER = [
    "logical-surf-18",
    "b1_kriging_w2_batch_32",
    "lyric-jazz-19",
    "valiant-waterfall-13",
    "misunderstood-totem-16",
    "whole-paper-14",
    "rich-dust-15",
    "atomic-morning-12",
    "cool-voice-11",
    "magic-snow-10",
    "copper-mountain-9",
    "tough-dragon-7",
]


def load_rows() -> list[dict]:
    registry = json.loads(REGISTRY.read_text())
    reg = {r["run_id"]: r for r in registry["runs"]}
    rows = []
    for run_id in RUN_ORDER:
        res_path = RUNS_DIR / run_id / "results.json"
        if not res_path.exists():
            print(f"[meta] WARNING: no results for {run_id}, skipping", file=sys.stderr)
            continue
        payload = json.loads(res_path.read_text())
        r = payload["results"]
        entry = reg[run_id]
        tc = entry["train_config"]
        loss = tc["loss"]
        model = tc["model"]
        training = tc["training"]
        ms = entry.get("metrics_summary", {})
        t1, t2, t3, t4 = r["test1"], r["test2"], r["test3"], r["test4"]
        t5, t6, t7 = r["test5"], r["test6"], r["test7"]
        rows.append({
            "run_id": run_id,
            # --- architectural / training decisions ---
            "w_kriging": loss["w_kriging"],
            "w_forecast": loss["w_forecast"],
            "w_impute": loss["w_impute"],
            "dropout": model["dropout"],
            "hidden_dim": model["hidden_dim"],
            "num_layers": model["num_layers"],
            "batch_size": training["batch_size"],
            "spatial_holdout": tc["data"]["spatial_holdout_prob"],
            "arch": f"{model['num_layers']}L/h{model['hidden_dim']}",
            # --- global validation quality (registry) ---
            "val_rmse": ms.get("val_rmse"),
            "val_mae": ms.get("val_mae"),
            "val_zero_acc": ms.get("val_zero_acc"),
            # --- outlier detection footprint (model-relative!) ---
            "n_events": t1["n_events"],
            "flagged_steps": t1["total_flagged_steps"],
            "pct_flagged": t1["mean_pct_flagged"],
            "n_network_wide": t1["class_counts"].get("network-wide", 0),
            # --- calibration during outliers (Test 2) ---
            "cov50_out": t2["coverage"]["50"]["outlier"],
            "cov90_out": t2["coverage"]["90"]["outlier"],
            "cov99_out": t2["coverage"]["99"]["outlier"],
            "cov99_base": t2["coverage"]["99"]["baseline"],
            "var_ratio": t2["variance_ratio"],
            "pi_ratio": t2["pi_ratio"],
            "outlier_mean_mu": t2["outlier"]["mean_mu"],
            "outlier_mean_theta": t2["outlier"]["mean_theta"],
            # --- propagation to zero-shot nodes (Test 3) ---
            "mu_shift": t3["mu_shift_median_abs"],
            "mu_shift_frac": t3["mu_shift_frac_gt_0p5z"],
            "att_shift": t3["attention_shift_median"],
            # --- early warning (Test 4) ---
            "pre_slope_mu": t4["pre_event_slopes_per_hour"]["mu"],
            "pre_slope_impute": t4["pre_event_slopes_per_hour"]["impute_res"],
            "mu_recovery_h": t4["mu_recovery_hours"],
            # --- kriging robustness under neighbour outliers (Test 5) ---
            "krig_rmse_normal": t5["kriging_rmse_normal"],
            "krig_rmse_exposed": t5["kriging_rmse_exposed"],
            "krig_degrade": t5["kriging_degradation_ratio"],
            "impute_degrade": t5["impute_degradation_ratio"],
            # --- tripwires / scale invariance ---
            "tripwires": t6["total"],
            "trip_outlier_frac": (t6["by_type"].get("PI_CLAMP") or {}).get(
                "frac_coincide_outlier"),
            "msle_ratio": t7["msle_ratio_high_over_low"],
            "two_extremes_degenerate": r["test8"].get("degenerate_export"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Scoring: which model handles outlier events best?
# --------------------------------------------------------------------------- #
# Rank-based composite over comparable, model-agnostic metrics.  Detection
# counts (n_events etc.) are EXCLUDED: flags are defined against each model's
# own residual sigma, so they measure the model's error distribution, not skill.
SCORE_METRICS = {
    # metric            higher_is_better
    "cov99_out": True,      # keeps honest uncertainty during outliers
    "cov90_out": True,
    "krig_degrade": False,  # robustness of held-out reconstruction to neighbour outliers
    "krig_rmse_exposed": False,  # absolute kriging error under outlier exposure
    "mu_shift": True,       # anomaly information reaches zero-shot nodes
    "pre_slope_mu": True,   # early-warning build-up signal
    "val_rmse": False,      # do not reward outlier skill bought with broken baseline accuracy
}


def add_scores(rows: list[dict]) -> None:
    n = len(rows)
    for m, hib in SCORE_METRICS.items():
        # None (metric undefined for that model, e.g. w_forecast=0) ranks worst.
        sentinel = -np.inf if hib else np.inf
        vals = [sentinel if r[m] is None else r[m] for r in rows]
        order = np.argsort([-v if hib else v for v in vals])  # best first
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        for r, rank in zip(rows, ranks):
            r[f"rank_{m}"] = n if r[m] is None else int(rank)
    for r in rows:
        r["mean_rank"] = round(float(np.mean([r[f"rank_{m}"] for m in SCORE_METRICS])), 2)


# --------------------------------------------------------------------------- #
# Group-by helper for the decision-axis tables
# --------------------------------------------------------------------------- #
AXES = [
    ("w_kriging", "Kriging loss weight"),
    ("w_forecast", "Forecast loss weight"),
    ("dropout", "Dropout"),
    ("batch_size", "Batch size"),
    ("spatial_holdout", "Spatial holdout prob"),
    ("arch", "Architecture (layers/hidden)"),
]
GROUP_METRICS = ["cov99_out", "krig_degrade", "krig_rmse_exposed", "mu_shift",
                 "pre_slope_mu", "var_ratio", "val_rmse", "mean_rank"]


def group_table(rows: list[dict], axis: str) -> list[dict]:
    out = []
    for val in sorted({r[axis] for r in rows}, key=str):
        grp = [r for r in rows if r[axis] == val]
        agg = {"value": val, "n": len(grp),
               "runs": ", ".join(r["run_id"] for r in grp)}
        for m in GROUP_METRICS:
            vals = [r[m] for r in grp if r[m] is not None]
            agg[m] = round(float(np.mean(vals)), 3) if vals else None
        out.append(agg)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_calibration_vs_kriging(rows):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        deg = r["w_forecast"] == 0
        ax.scatter(r["cov99_out"] * 100, r["krig_degrade"],
                   s=40 + 25 * r["w_kriging"],
                   c="tab:red" if deg else "tab:blue",
                   marker="s" if r["batch_size"] == 32 else "o",
                   alpha=0.85, edgecolors="k", linewidths=0.5)
        ax.annotate(r["run_id"].replace("_batch_32", ""), (r["cov99_out"] * 100,
                    r["krig_degrade"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("99% interval coverage during outliers (%) — higher is better")
    ax.set_ylabel("Kriging RMSE degradation ×(neighbour outlier) — lower is better")
    ax.set_title("Outlier calibration vs kriging robustness\n"
                 "(size ∝ w_kriging; square = batch 32; red = w_forecast 0)")
    fig.tight_layout()
    p = FIG_DIR / "meta_calibration_vs_kriging.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


def fig_metric_bars(rows):
    metrics = [("cov99_out", "99% coverage @ outliers", True),
               ("krig_degrade", "Kriging degradation ×", False),
               ("mu_shift", "Sim-node μ shift (σ)", True),
               ("pre_slope_mu", "Pre-event μ slope (/h)", True),
               ("var_ratio", "ZINB variance ratio ×", True),
               ("val_rmse", "Validation RMSE", False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    names = [r["run_id"].replace("_batch_32", "") for r in rows]
    for ax, (m, title, hib) in zip(axes.flat, metrics):
        vals = [np.nan if r[m] is None else r[m] for r in rows]
        best = (np.nanargmax if hib else np.nanargmin)(vals)
        colors = ["tab:green" if i == best else "tab:gray" for i in range(len(rows))]
        ax.barh(range(len(rows)), vals, color=colors)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title + (" ↑" if hib else " ↓"), fontsize=10)
    fig.suptitle("Per-run outlier-performance metrics (green = best)")
    fig.tight_layout()
    p = FIG_DIR / "meta_metric_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


def fig_axis_effects(rows):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (axis, label) in zip(axes.flat, AXES):
        vals = sorted({r[axis] for r in rows}, key=str)
        xs, cov, deg = [], [], []
        for i, v in enumerate(vals):
            grp = [r for r in rows if r[axis] == v]
            xs.append(i)
            cov.append(np.mean([r["cov99_out"] for r in grp]) * 100)
            deg.append(np.mean([r["krig_degrade"] for r in grp]))
        ax2 = ax.twinx()
        ax.bar([x - 0.17 for x in xs], cov, width=0.34, color="tab:blue",
               label="cov99 @out (%)")
        ax2.bar([x + 0.17 for x in xs], deg, width=0.34, color="tab:orange",
                label="krig degrade ×")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in vals])
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("cov99 @out (%)", color="tab:blue", fontsize=8)
        ax2.set_ylabel("krig degrade ×", color="tab:orange", fontsize=8)
    fig.suptitle("Mean outlier calibration (blue) and kriging degradation (orange) "
                 "by decision axis")
    fig.tight_layout()
    p = FIG_DIR / "meta_axis_effects.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def md_table(rows: list[dict], cols: list[tuple[str, str]], fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k)
            if v is None:
                cells.append("—")
            elif k in fmt:
                cells.append(fmt[k].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


FINDINGS = """\
## 4. Findings — which architectural decisions drive outlier-event performance

### 4.1 The forecast head is non-negotiable (`w_forecast`)
The two `w_forecast=0` runs (**lyric-jazz-19**, **atomic-morning-12**) cannot do
outlier work at all: the residual distribution degenerates (1 event flagged in 31
days vs 90–158 for every other model), the ZINB head shows **no** outlier response
(variance ratio ×0.9, π ratio ≈1.0, zero-accuracy ≈0) and validation RMSE roughly
doubles (≈46 vs 21–26). Their apparently excellent kriging robustness (×1.4–1.7 on
a low absolute error) is an artefact of the collapsed outlier definition, not
genuine skill. **Keep `w_forecast = 1`.**

### 4.2 Dropout is the single strongest calibration lever
99% interval coverage *during outliers* rises monotonically with dropout —
**0.1 → 52%, 0.3 → 73%, 0.5 → 93%** (group means; `misunderstood-totem-16` at
dropout 0.5 reaches 92.7% against a 98% baseline, close to eliminating the
over-confidence collapse every other model shows). It also posts the *lowest
absolute* exposed kriging RMSE among functional models (65.7). The cost is the
sharpness of the zero-inflation gate: zero-classification accuracy falls from
≈0.78 (dropout 0.1) to 0.40–0.50, and π saturation tripwires stay high. Dropout
regularisation appears to be doing the calibration work that the ZINB likelihood
alone does not.

### 4.3 Kriging weight: w=2 is the sweet spot; w=0.5 leaves the head fragile
Mean kriging degradation under neighbour outliers is **×1.64 at w=2** vs ×2.00 at
w=0.5 and ×1.93 at w=5, and the absolute exposed RMSE for w=0.5 runs is 2–4×
higher (mean 294 vs 123). Up-weighting to w=5 only pays off in *absolute* kriging
accuracy when combined with batch 32 (`logical-surf-18`: 47.7 normal / 81.4
exposed — the best absolute numbers of any functional run) but slightly hurts
calibration and propagation. **`w_kriging = 2` is the best default; 5 only if
absolute kriging accuracy is the priority.**

### 4.4 Small batch (32) beats large (128)
Batch-32 runs halve the absolute exposed kriging RMSE (mean 98 vs 233), carry
outlier information to zero-shot nodes more strongly (μ-shift 0.59σ vs 0.47σ) and
produced the two best all-round models (`b1_kriging_w2_batch_32`,
`logical-surf-18`). The b1-vs-`magic-snow-10` pair is a clean controlled contrast
(identical config except batch size): batch 32 wins on every headline metric
(val RMSE 21.5 vs 24.3, cov99 63.5% vs 62.0%, kriging ×1.55 vs ×1.80, μ-shift
0.68σ vs 0.33σ).

### 4.5 Spatial holdout above 0.2 damages outlier robustness
`valiant-waterfall-13` (0.4) and `whole-paper-14` (0.5) sit at the bottom on
calibration (44.8% / 41.9%) and kriging degradation (×2.29 / ×1.95, absolute
exposed RMSE 332 / 398), despite competitive validation RMSE. Training with a
sparser observed graph does force stronger cross-node signalling (highest μ-shift,
0.79σ) but the model that results is brittle exactly when neighbours misbehave.
**Keep `spatial_holdout_prob = 0.2`.**

### 4.6 Universal weaknesses — no configuration solves these
Every one of the 12 models shows: (i) **coverage collapse during outliers**
(even the best is 92.7% vs 99% nominal; most are 40–76%); (ii) **kriging
degradation ×1.4–2.3** under neighbour outliers — message passing amplifies
anomalies in all configurations; (iii) **no attention re-weighting** toward
outlier sources (median shift −0.03…+0.02 across all runs — propagation always
rides on message content, never on visible attention concentration); (iv) **μ
never recovers within 6h** of a major event (hysteresis in all runs); and (v) a
**degenerate two-extremes export** (gap ≡ 0 in all 12 runs — the export bug in
`export_paginated_payload.py` predates tough-dragon-7). These are architecture-
level limitations of the current GNN, not tuning artefacts, and match the
recommendations in the single-run report (calibration objective, event-aware
attention temperature, robust kriging aggregation).

### 4.7 Recommended next configuration
Combine the winning settings, none of which conflict:
`w_forecast=1, w_impute=1, w_kriging=2, batch_size=32, spatial_holdout=0.2,
dropout=0.3–0.5` (the only open trade-off: dropout 0.5 maximises outlier
calibration, 0.3 preserves more zero-gate sharpness — worth an A/B pair).
That is `b1_kriging_w2_batch_32`'s recipe plus the dropout lever from
`misunderstood-totem-16` — the two top-ranked models in §2.
"""


def main() -> int:
    rows = load_rows()
    if len(rows) < 2:
        print("[meta] need >=2 completed runs", file=sys.stderr)
        return 1
    add_scores(rows)
    ranked = sorted(rows, key=lambda r: r["mean_rank"])

    figs = [fig_metric_bars(ranked), fig_calibration_vs_kriging(rows),
            fig_axis_effects(rows)]

    (OUT_DIR / "meta_results.json").write_text(json.dumps({
        "runs": rows,
        "score_metrics": {k: ("higher" if v else "lower") + " is better"
                          for k, v in SCORE_METRICS.items()},
        "axis_tables": {axis: group_table(rows, axis) for axis, _ in AXES},
    }, indent=2))

    import datetime as _dt
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    cfg_cols = [("run_id", "Run"), ("w_kriging", "w_krig"), ("w_forecast", "w_fcst"),
                ("dropout", "drop"), ("batch_size", "batch"),
                ("spatial_holdout", "sp.hold"), ("arch", "arch"),
                ("val_rmse", "val RMSE"), ("val_zero_acc", "zero acc")]
    perf_cols = [("run_id", "Run"), ("mean_rank", "Mean rank"),
                 ("cov99_out", "cov99 @out"), ("cov90_out", "cov90 @out"),
                 ("krig_degrade", "krig ×"), ("krig_rmse_exposed", "krig RMSE exp"),
                 ("mu_shift", "μ-shift σ"), ("pre_slope_mu", "pre-slope μ/h"),
                 ("var_ratio", "var ×"), ("n_events", "events"),
                 ("pct_flagged", "% flagged")]
    fmt = {"val_rmse": "{:.1f}", "val_zero_acc": "{:.2f}", "cov99_out": "{:.1%}",
           "cov90_out": "{:.1%}", "krig_degrade": "{:.2f}",
           "krig_rmse_exposed": "{:.1f}", "mu_shift": "{:.2f}",
           "pre_slope_mu": "{:.2f}", "var_ratio": "{:.1f}", "pct_flagged": "{:.2f}"}

    parts = [
        "# Meta-Report — Outlier-Event Performance Across the Last 12 Models",
        f"**Models:** {rows[0]['run_id']} … {rows[-1]['run_id']}  |  "
        f"**Generated:** {ts}  |  Runs analysed: {len(rows)}/12",
        "> Every run re-analysed with the identical 8-test pipeline "
        "(`run_analysis.py --run-id <run>`); per-run artefacts in "
        "`outputs/runs/<run_id>/`. Aggregation is reproduced by "
        "`compile_meta_report.py`.",
        "",
        "## How to read this",
        "Outlier *flags* are defined against each model's own residual σ, so "
        "event counts measure the shape of a model's error distribution, not "
        "detection skill. Cross-model comparison therefore leans on metrics with "
        "an absolute meaning: interval **coverage during outliers** (test 2), "
        "**kriging degradation** under neighbour outliers (test 5), zero-shot "
        "**propagation strength** (test 3), **pre-event build-up slope** (test 4) "
        "and plain validation error. The composite is a mean rank over: "
        + ", ".join(f"`{m}`{'↑' if h else '↓'}" for m, h in SCORE_METRICS.items())
        + ".",
        "",
        "## 1. Model configurations",
        md_table(rows, cfg_cols, fmt),
        "",
        "## 2. Composite ranking (outlier-event performance)",
        md_table(ranked, perf_cols, fmt),
        "",
        f"![Per-run metrics](figures/{figs[0]})",
        f"![Calibration vs kriging robustness](figures/{figs[1]})",
        "",
        "## 3. Effect of each architectural decision",
    ]
    ax_cols = [("value", "Value"), ("n", "n"), ("cov99_out", "cov99 @out"),
               ("krig_degrade", "krig ×"), ("krig_rmse_exposed", "krig RMSE exp"),
               ("mu_shift", "μ-shift σ"), ("var_ratio", "var ×"),
               ("val_rmse", "val RMSE"), ("mean_rank", "mean rank"),
               ("runs", "runs")]
    ax_fmt = {"cov99_out": "{:.1%}"}
    for axis, label in AXES:
        parts += [f"### {label} (`{axis}`)",
                  md_table(group_table(rows, axis), ax_cols, ax_fmt), ""]
    parts += [f"![Axis effects](figures/{figs[2]})", "", FINDINGS]

    (OUT_DIR / "meta_report.md").write_text("\n".join(parts))
    print(f"[meta] wrote {OUT_DIR/'meta_report.md'} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
