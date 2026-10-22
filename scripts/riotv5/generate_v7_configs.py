#!/usr/bin/env python3
"""
Generate the v7 experiment stage: the next-stage recommendations of
docs/experiments/riot_v6/riot_v6_seed_report.md §9.2, as one unattended run.

Every run uses the v6 schedule (cosine LR, 3 warmup epochs, floor 0.02 x lr, 100-epoch
budget, early stopping after epoch 40 with patience 20) on the v5 57-station training
set, with no validation stations, unless stated otherwise. The test holdout (24
stations, inductive_holdout_seed 42) is unchanged, so held-out RMSE stays comparable
with v5 and v6. The v6 runs trained on 52 stations, so v7 does not reuse them as
controls: every comparison is inside v7.

Tiers (the runner schedules tier 1 first, longest job first within a tier):
  1  g1_static     rec. 1  V6 base vs static-ridge profile, seeds 42-47 (57 stations)
  2  g2_valid      rec. 2  8 stratified validation stations (47 train), seeds 42-44.
                           GATED: run_v7_stage.sh first checks on the v6 checkpoints
                           that validation-station RMSE tracks held-out RMSE
                           (posthoc_v7_eval.py valcheck/valgate) and skips the tier if not.
  3  g3_block      rec. 3  V6 + block masks, V6 + static-ridge + block masks, seeds 42-47.
                           With tier 1 this is a 2x2 (profile x masks) on shared seeds.
  4  g4_weather    rec. 4  weather extremes at the target step (perfect forecast) vs at
                           the forecast origin only, seeds 42-47 (control: v7_base)
  5  g5_epochs     rec. 5  V6 base and static-ridge at a 150-epoch budget, seeds 42-47
Rec. 6 (stop list) is enforced, not run: build_jobs() refuses any LSTM, seq_len != 12,
GAT-drop, outage-head or kriging-loss-variant configuration (STOP_LIST below).

Output:
  configs/train/v7/<group>/<name>__s<seed>.yaml
  configs/train/v7/suite_manifest.yaml   (logs_dir logs/v7_stage, W&B project resilient-iot-v7)

Usage:
  python scripts/riotv5/generate_v7_configs.py           # write configs + manifest
  python scripts/riotv5/generate_v7_configs.py --check   # also resolve + validate every config
  python scripts/riotv5/generate_v7_configs.py --audit docs/experiments/riot_v7/data/rec6_stop_list.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_v5_configs as gen  # noqa: E402
import generate_v6_configs as v6  # noqa: E402

OUT_DIR = gen.PROJECT_ROOT / "configs" / "train" / "v7"
LOGS_DIR = "logs/v7_stage"
WANDB_PROJECT = "resilient-iot-v7"
SEEDS = [42, 43, 44, 45, 46, 47]
SEEDS_GATED = [42, 43, 44]
GATED_TIER = 2

# V6 with the v6 schedule but no validation stations: 57 training stations.
V7 = gen.derive(v6.V6, {"data.val_station_count": 0})
STATIC = {"data.profile_prior": "inductive_static"}
BLOCK = {"data.mask_strategy": "block"}           # V6 already has the STRUCTURED options
E150 = {"training.epochs": 150, "training.min_epochs": 60, "training.early_stop_patience": 30}
# Rec. 2: 8 validation stations, one per train-level stratum, among stations with
# >= 25% of the validation window observed (v6: 2 of 5 had no data after early 2024).
VALID8 = {
    "data.val_station_count": 8,
    "data.val_station_seed": 0,
    "data.val_station_strategy": "stratified",
    "data.val_station_min_coverage": 0.25,
}

EXPERIMENTS = []


def E(group, tier, name, base, overrides, desc, rec, seeds=SEEDS):
    EXPERIMENTS.append(dict(group=group, tier=tier, name=name, base=base, overrides=overrides,
                            desc=desc, rec=rec, seeds=seeds))


# ---- tier 1: confirm the static-ridge profile (rec. 1) ----------------------------------
G = "g1_static"
E(G, 1, "v7_base", V7, {}, "V6 base, 57 training stations (control for every v7 arm)", 1)
E(G, 1, "v7_profile_static", V7, STATIC, "V6 + static-feature ridge profile at unseen stations", 1)

# ---- tier 2: fixed inductive validation (rec. 2, gated) ---------------------------------
G = "g2_valid"
E(G, 2, "v7v_base", V7, VALID8, "V6 base + 8 stratified validation stations (47 train)", 2, SEEDS_GATED)
E(G, 2, "v7v_profile_static", V7, {**STATIC, **VALID8}, "static-ridge + 8 stratified validation stations", 2,
  SEEDS_GATED)
E(G, 2, "v7v_e2_2_block", gen.P2B, {**gen.STRUCTURED, **BLOCK, **v6.SCHEDULE_ONLY, **VALID8},
  "E2.2 block (P-family, where the v6 checkpoint failed worst) + 8 stratified validation stations", 2, SEEDS_GATED)
E(G, 2, "v7v_p2b", gen.P2B, {**v6.SCHEDULE_ONLY, **VALID8}, "P2B + 8 stratified validation stations", 2,
  SEEDS_GATED)

# ---- tier 3: static-ridge x block masks (rec. 3) ----------------------------------------
G = "g3_block"
E(G, 3, "v7_block", V7, BLOCK, "V6 + E2.2 block masks (instead of mixed)", 3)
E(G, 3, "v7_static_block", V7, {**STATIC, **BLOCK}, "V6 + static-ridge profile + E2.2 block masks", 3)

# ---- tier 4: lagged weather extremes (rec. 4) -------------------------------------------
G = "g4_weather"
E(G, 4, "v7_weather_target", V7, {"data.weather_extreme_features": True},
  "V6 + weather extremes at the target step (perfect forecast, v6 replication)", 4)
E(G, 4, "v7_weather_origin", V7, {"data.weather_extreme_features": True, "data.weather_extreme_step": "origin"},
  "V6 + weather extremes observed at the forecast origin only", 4)

# ---- tier 5: 150-epoch budget (rec. 5) ---------------------------------------------------
G = "g5_epochs"
E(G, 5, "v7_base_e150", V7, E150, "V6 base, 150-epoch budget (min 60, patience 30)", 5)
E(G, 5, "v7_profile_static_e150", V7, {**STATIC, **E150}, "static-ridge, 150-epoch budget (min 60, patience 30)", 5)


# ---------------------------------------------------------------------------
# Rec. 6: the stop list. Each entry is (reason, predicate on the resolved config).
# ---------------------------------------------------------------------------
STOP_LIST = [
    ("LSTM / ensemble", lambda c: c["model_type"] != "gnn"),
    ("seq48 (longer context)", lambda c: c["data"]["seq_len"] != 12),
    ("GAT drop", lambda c: c["model"]["gat_drop_missing_sources"]),
    ("outage head", lambda c: c["loss"]["forecast_likelihood"] == "outage_mixture"
        or c["loss"]["w_artificial_outage"] > 0.0 or c["model"]["outage_head_input"] == "missingness"),
    ("kriging-loss variants", lambda c: c["loss"]["w_kriging"] != 0.5 or not c["loss"]["kriging_observed_only"]),
]


def stop_list_violations(cfg: dict):
    return [reason for reason, hit in STOP_LIST if hit(cfg)]


# ---------------------------------------------------------------------------
# GPU-hour estimate, calibrated on the v6 stage (status duration_sec, one job per GPU):
# v6_base 3.60 h, profile_static 3.56 h, weather_extremes 3.68 h, e2_2_block 4.25 h,
# p2b 3.09 h, all at 52 training stations and 100 epochs. Time scales with the
# number of trained stations (v6p_v5f_sched 57 st 2.39 h vs v6p_v5f 52 st 1.96 h) and
# with the epoch budget (67% of v6 runs hit the cap).
# ---------------------------------------------------------------------------
V6_HOURS_52 = {"v6": 3.6, "p2b": 3.1, "e2_2": 4.25}
STATION_SCALE = 1.15     # 57 vs 52 stations (v6p_v5f_sched / v6p_v5f gives 1.22)


def est_gpu_hours(e: dict, cfg: dict) -> float:
    if e["base"] is gen.P2B:
        hours = V6_HOURS_52["e2_2" if cfg["data"]["mask_strategy"] == "block" else "p2b"]
    else:
        hours = V6_HOURS_52["v6"] * (1.05 if cfg["data"]["weather_extreme_features"] else 1.0)
    n_val = cfg["data"]["val_station_count"]
    hours *= STATION_SCALE if n_val == 0 else 47 / 52
    hours *= cfg["training"]["epochs"] / 100 * (0.9 if cfg["training"]["epochs"] > 100 else 1.0)
    return round(hours, 2)


def build_jobs():
    names = [e["name"] for e in EXPERIMENTS]
    assert len(names) == len(set(names)), "duplicate experiment names"
    jobs = []
    for e in EXPERIMENTS:
        base_cfg = gen.derive(e["base"], e["overrides"])
        bad = stop_list_violations(base_cfg)
        assert not bad, f"rec. 6 stop list: {e['name']} is {bad}"
        assert (base_cfg["data"]["val_station_count"] > 0) == (e["tier"] == GATED_TIER), \
            f"{e['name']}: validation stations belong to the gated tier only"
        for seed in e["seeds"]:
            cfg = gen.derive(base_cfg, {"training.random_seed": seed})
            run_name = f"{e['name']}-s{seed}"
            cfg["wandb"]["project"] = WANDB_PROJECT
            cfg["wandb"]["collection"] = f"riot-v7-{e['group']}-{e['name']}"
            cfg["wandb"]["run_name"] = run_name
            path = OUT_DIR / e["group"] / f"{e['name']}__s{seed}.yaml"
            jobs.append(dict(
                name=run_name, experiment=e["name"], group=e["group"], seed=seed, tier=e["tier"],
                desc=f"[rec {e['rec']}] {e['desc']}", config=str(path.relative_to(gen.PROJECT_ROOT)),
                cost=est_gpu_hours(e, cfg), cfg=cfg,
            ))
    return jobs


def audit(out_path: Path) -> int:
    """Rec. 6 confirmation at the end of the stage: re-read every config the stage
    trained from disk and check it against the stop list. Writes a JSON record."""
    with open(OUT_DIR / "suite_manifest.yaml") as f:
        jobs = yaml.safe_load(f)["jobs"]
    violations = {}
    for j in jobs:
        with open(gen.PROJECT_ROOT / j["config"]) as f:
            bad = stop_list_violations(gen.resolve_config(yaml.safe_load(f)))
        if bad:
            violations[j["name"]] = bad
    record = {"rec": 6, "stopped": [r for r, _ in STOP_LIST], "n_configs": len(jobs), "violations": violations,
              "verdict": "PASS" if not violations else "FAIL"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"rec. 6 stop list: {record['verdict']} ({len(jobs)} configs, {len(violations)} violations) -> {out_path}")
    return 0 if not violations else 1


HEADER = """# ============================================================================
# RIoT v7 stage — {group} / {name} (seed {seed})
# {desc}
# GENERATED by scripts/riotv5/generate_v7_configs.py — edit the generator, not this file.
# ============================================================================
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="resolve_config every generated file")
    ap.add_argument("--audit", metavar="OUT_JSON", default=None,
                    help="rec. 6: check the written configs against the stop list (writes nothing else)")
    args = ap.parse_args()
    if args.audit:
        sys.exit(audit(Path(args.audit)))

    jobs = build_jobs()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    for j in jobs:
        path = gen.PROJECT_ROOT / j["config"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(HEADER.format(group=j["group"], name=j["experiment"], seed=j["seed"], desc=j["desc"]))
            yaml.safe_dump(j["cfg"], f, sort_keys=False, default_flow_style=None)
        if args.check:
            gen.resolve_config(j["cfg"])

    manifest = {
        "pipeline_version": gen.PIPELINE_VERSION,
        "wandb_project": WANDB_PROJECT,
        "logs_dir": LOGS_DIR,
        "cost_unit": "estimated GPU-hours",
        "jobs": [{k: v for k, v in j.items() if k != "cfg"} for j in jobs],
    }
    with open(OUT_DIR / "suite_manifest.yaml", "w") as f:
        f.write("# GENERATED by scripts/riotv5/generate_v7_configs.py\n")
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Wrote {len(jobs)} configs ({len(EXPERIMENTS)} experiments) to {OUT_DIR.relative_to(gen.PROJECT_ROOT)}")
    total = 0.0
    for t in sorted({j["tier"] for j in jobs}):
        sel = [j for j in jobs if j["tier"] == t]
        h = sum(j["cost"] for j in sel)
        total += h
        gate = "  (gated)" if t == GATED_TIER else ""
        print(f"  tier {t} {sel[0]['group']:<11} {len(sel):>3} runs  ~{h:6.1f} GPU-h  (cumulative ~{total:6.1f}){gate}")
    print(f"  total ~{total:.0f} GPU-h (~{total / 4:.0f} h wall on 4 GPUs) + rec. 2 pre-check (~5 GPU-h)")
    print(f"  rec. 6 stop list enforced: {', '.join(r for r, _ in STOP_LIST)}")
    if args.check:
        print("All configs resolve and validate.")


if __name__ == "__main__":
    main()
