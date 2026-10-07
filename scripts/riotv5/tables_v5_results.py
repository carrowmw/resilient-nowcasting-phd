#!/usr/bin/env python3
"""
Markdown tables for the RIoT v5 results report (test split, seed 42).

Input:  docs/experiments/riot_v5/data/runs.csv (collect_v5_results.py)
Output: docs/experiments/riot_v5/data/appendix_tables.md

Usage: python scripts/riotv5/tables_v5_results.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "experiments" / "riot_v5" / "data"

COLS = [("test_obs_all_rmse", "All", 2), ("test_obs_trans_rmse", "Trans", 2), ("test_obs_ind_rmse", "Ind", 2),
        ("test_obs_all_mae", "MAE", 2), ("test_peak_rmse", "Peak", 1), ("test_so_12_forecast_rmse", "SO12", 2),
        ("test_so_96_forecast_rmse", "SO96", 2), ("test_so_96_mass_ratio", "SO96 mass", 2),
        ("test_collapse_rate", "Collapse", 3), ("test_real_gap_mass_ratio", "Gap mass", 2),
        ("test_crps", "CRPS", 2), ("test_q90_coverage", "q90 cov", 2), ("test_impute_raw_rmse", "Impute", 1),
        ("test_kriging_raw_rmse", "Kriging", 1)]

GROUPS = [("thesis_v4", "Thesis Experiment 3 re-run on V4R"), ("p1_hygiene", "P1 hygiene staircase"),
          ("p2_masking", "P2 masking"), ("p3_inputs", "P3 inputs"), ("p4_likelihood", "P4 likelihood"),
          ("p5_magnitude", "P5 magnitude"), ("p6_mtl", "P6 MTL"), ("p7_v5_suite", "P7 v5 suite"),
          ("loo", "Leave-one-out from V5F")]


def fmt(v, d):
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def table(r, exps):
    head = "| Run | " + " | ".join(h for _, h, _ in COLS) + " | Best / run ep |"
    sep = "|---|" + "---:|" * (len(COLS) + 1)
    lines = [head, sep]
    for e in exps:
        row = r.loc[e]
        cells = [fmt(row[c], d) for c, _, d in COLS]
        lines.append(f"| `{e}` | " + " | ".join(cells) + f" | {int(row.best_epoch)} / {int(row.epochs_run)} |")
    return "\n".join(lines)


def main():
    r = pd.read_csv(DATA / "runs.csv").copy()
    r["exp"] = r.run.str.replace("-s42", "", regex=False)
    r = r.set_index("exp")
    out = ["# RIoT v5 — full result tables (test split, seed 42)", "",
           "All metrics are raw counts on observed targets. All / Trans / Ind = observed-target RMSE over all, "
           "transductive and held-out nodes; Peak = RMSE where y ≥ 58 (train p90); SO12 / SO96 = forecast RMSE "
           "on nodes with a synthetic outage of 12 / 96 steps; SO96 mass = predicted / true sum on those nodes; "
           "Collapse = P(point < 1 | target missing, own profile > 10); Gap mass = predicted / profile mass in real "
           "gaps; Impute / Kriging = raw-count RMSE of the imputation and kriging heads. "
           "Best / run ep = selected epoch / epochs trained.", ""]
    for g, title in GROUPS:
        exps = sorted(r.index[r.group == g])
        out += [f"## {title}", "", table(r, exps), ""]
    (DATA / "appendix_tables.md").write_text("\n".join(out))
    print("wrote", DATA / "appendix_tables.md")


if __name__ == "__main__":
    main()
