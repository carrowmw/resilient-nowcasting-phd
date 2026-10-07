#!/usr/bin/env python3
"""
Flatten the RIoT v5 suite outputs into analysis tables.

Inputs (all local copies of the GPU-server outputs):
  logs/v5_suite/status/<run>.json          run status, group, tier, duration
  models/resilient-iot-v5/<run>/results.json  metric dictionaries (val + test)
  logs/v5_suite/runs/<run>.log             per-epoch validation lines
  configs/train/v5/suite_manifest.yaml     descriptions

Outputs (docs/experiments/riot_v5/data/):
  runs.csv     one row per COMPLETED run; columns prefixed val_ / test_ / meta
  epochs.csv   one row per (run, epoch) parsed from the training log

Usage: python scripts/riotv5/collect_v5_results.py
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "experiments" / "riot_v5" / "data"

EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<max_epochs>\d+) - Train Loss: (?P<train_loss>[-\d.naninf]+) \| "
    r"Val Loss: (?P<val_loss>[-\d.naninf]+) \| Score\((?P<sel>[^)]+)\): (?P<score>[-\d.naninf]+) "
    r"\(obsRMSE: (?P<obs_rmse>[-\d.nan]+), peakRMSE: (?P<peak_rmse>[-\d.nan]+), "
    r"collapse: (?P<collapse>[-\d.nan]+), gapMass: (?P<gap_mass>[-\d.nan]+), "
    r"ImputeRMSE: (?P<impute_rmse>[-\d.nan]+)\), Kriging\(train=(?P<krig_train>[-\d.nan]+), "
    r"val=(?P<krig_val>[-\d.nan]+), RMSE=(?P<krig_rmse>[-\d.nan]+)\)"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load(open(ROOT / "configs/train/v5/suite_manifest.yaml"))
    desc = {j["name"]: j for j in manifest["jobs"]}

    rows, epochs = [], []
    for f in sorted(glob.glob(str(ROOT / "logs/v5_suite/status/*.json"))):
        st = json.load(open(f))
        if st["status"] != "COMPLETED" or st["mock"]:
            continue
        res_path = ROOT / "models/resilient-iot-v5" / st["name"] / "results.json"
        res = json.load(open(res_path))
        cfg = yaml.safe_load(open(ROOT / "models/resilient-iot-v5" / st["name"] / "config.yaml"))
        row = {
            "run": st["name"], "experiment": st["experiment"], "group": st["group"],
            "seed": st["seed"], "tier": st["tier"], "desc": st["desc"],
            "duration_h": st["duration_sec"] / 3600.0, "cost": desc[st["name"]]["cost"],
            "model_type": res["model_type"], "best_epoch": res["best_epoch"],
            "selection_metric": res["selection_metric"], "holdout_sha": res["holdout_manifest_sha"],
            "seq_len": cfg["data"]["seq_len"],
        }
        for k, v in res["v5_metrics"].items():
            row[f"val_{k}"] = v
        for k, v in res["v5_test_metrics"].items():
            row[f"test_{k}"] = v
        for k, v in res["test_metrics"].items():
            row[f"test_legacy_{k}"] = v
        rows.append(row)

        for line in open(ROOT / "logs/v5_suite/runs" / f"{st['name']}.log", errors="replace"):
            m = EPOCH_RE.search(line)
            if m:
                e = {k: (v if k == "sel" else float(v)) for k, v in m.groupdict().items()}
                e["run"] = st["name"]
                epochs.append(e)

    runs = pd.DataFrame(rows)
    ep = pd.DataFrame(epochs)
    n_ep = ep.groupby("run").epoch.max().rename("epochs_run")
    runs = runs.merge(n_ep, left_on="run", right_index=True, how="left")
    runs.to_csv(OUT / "runs.csv", index=False)
    ep.to_csv(OUT / "epochs.csv", index=False)
    print(f"{len(runs)} runs, {len(ep)} epoch rows -> {OUT}")


if __name__ == "__main__":
    main()
