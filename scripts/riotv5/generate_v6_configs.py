#!/usr/bin/env python3
"""
Generate the v6 experiment stage: the seven next-stage recommendations of
docs/experiments/riot_v5/riot_v5_seed_report.md §14.2, as one unattended run.

Every training run uses the v6 PROTOCOL (rec. 1):
  * cosine LR schedule, 3 warmup epochs, floor 0.02 x lr, fixed 100-epoch budget;
  * early stopping may only trigger after epoch 40, patience 20;
  * 5 inductive validation stations carved from the 57 training stations (fixed draw,
    val_station_seed 0, identical for every seed). Each run keeps two checkpoints:
    the usual selection (obs NLL + SO@12) and the best validation-station RMSE, and
    evaluates both on the test split (results.json v5_test_metrics / _ind_ckpt).
The test holdout (24 stations, inductive_holdout_seed 42) is unchanged, so held-out
RMSE stays comparable with v5; the models now learn from 52 stations, not 57.

An unattended launch cannot gate, so every base is pre-declared here (as in the v5
suite). V6 (rec. 3) is V5F with the nine components the v5 LOO found neutral removed,
and the P3 screen winners swapped in:
  - NB forecast head (outage head off), no quantile heads, no magnitude weighting,
    no Huber imputation mix, no missing-aware key padding, GAT keeps missing sources;
  - profile inductive_zero (not inductive_neighbour), lags 96 / 672, mask ratio 0.15;
  - kept: unified encoding, mixed structured masks, graph, calendar, run-length,
    sensor embedding, log1p, complete weather and the consistent night channel (the
    last two are leak fixes, not components).

Tiers (the runner schedules tier 1 first, longest job first within a tier):
  1  g1_protocol  rec. 1  protocol on V5F (with and without validation stations), P5B, P2B, E2.1
  2  g3_v6        rec. 3  V6 base + GAT drop / no sensor embedding / E2.1-style masks / outage head
  3  g2_blocks    rec. 2  P3B, E2.1b, E2.2 block, E3.3d profile-zero, E3.2 log1p, E3.7b lags
  4  g4_heldout   rec. 4  V6 kriging off / iid kriging target / static-ridge profile / residual-to-profile
  5  g5_loo_peak  rec. 5  V6 LOO (profile, calendar, mask 0.30) + rec. 6 weather extremes, V6 LSTM
  6  g6_seq48     rec. 6  V6 at seq_len 48 (the expensive arm: first to cut)
Rec. 7 (inference only) and the GNN + LSTM ensemble of rec. 6 run after training:
scripts/riotv5/posthoc_v6_eval.py, driven by scripts/riotv5/run_v6_stage.sh.

Output:
  configs/train/v6/<group>/<name>__s<seed>.yaml
  configs/train/v6/suite_manifest.yaml   (logs_dir logs/v6_stage, W&B project resilient-iot-v6)

Usage:
  python scripts/riotv5/generate_v6_configs.py           # write configs + manifest
  python scripts/riotv5/generate_v6_configs.py --check   # also resolve + validate every config
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_v5_configs as gen  # noqa: E402

OUT_DIR = gen.PROJECT_ROOT / "configs" / "train" / "v6"
LOGS_DIR = "logs/v6_stage"
WANDB_PROJECT = "resilient-iot-v6"
SEEDS = [42, 43, 44]

PROTOCOL = {
    "training.epochs": 100,
    "training.lr_schedule": "cosine",
    "training.warmup_epochs": 3,
    "training.lr_min_frac": 0.02,
    "training.min_epochs": 40,
    "training.early_stop_patience": 20,
    "data.val_station_count": 5,
    "data.val_station_seed": 0,
}
SCHEDULE_ONLY = {k: v for k, v in PROTOCOL.items() if not k.startswith("data.val_station")}

V6 = gen.derive(gen.V5F, {
    "loss.forecast_likelihood": "nb",
    "model.outage_head_input": "latent",
    "loss.w_artificial_outage": 0.0,
    "model.quantiles": [],
    "loss.magnitude_weight_alpha": 0.0,
    "loss.impute_huber_mix": 0.0,
    "model.temporal_key_padding": False,
    "model.gat_drop_missing_sources": False,
    "data.profile_prior": "inductive_zero",
    "data.lag_features": [96, 672],
    "data.mask_prob": 0.15,
    **PROTOCOL,
})

# V6-input LSTM (ensemble partner): same data options as V6 (so both members see identical
# batches), no graph, NB head, imputation head kept (as v5_lstm).
V6_LSTM = {
    "model_type": "lstm", "data.spatial_holdout_prob": 0.0, "model.embedding_dim": 16, "model.num_layers": 2,
    "training.batch_size": 64, "loss.w_kriging": 0.0, **gen.LSTM_RICH,
    "model.sensor_embedding_dim": 0, "model.sensor_embedding_dropout": 0.0,
}

EXPERIMENTS = []


def E(group, tier, name, base, overrides, desc, rec):
    EXPERIMENTS.append(dict(group=group, tier=tier, name=name, base=base, overrides=overrides, desc=desc, rec=rec))


# ---- tier 1: protocol (rec. 1) -------------------------------------------------------
G = "g1_protocol"
E(G, 1, "v6p_v5f_sched", gen.V5F, SCHEDULE_ONLY, "V5F + cosine/min-epoch protocol, no validation stations (57 train)", 1)
E(G, 1, "v6p_v5f", gen.V5F, PROTOCOL, "V5F + full protocol (5 validation stations, 52 train)", 1)
E(G, 1, "v6p_p5b", gen.P5B, PROTOCOL, "P5B + full protocol", 1)
E(G, 1, "v6p_p2b", gen.P2B, PROTOCOL, "P2B + full protocol", 1)
E(G, 1, "v6p_e2_1", gen.P2B, {"data.unify_missing_encoding": True, **PROTOCOL},
  "E2.1 (P2B + unified encoding, v4 iid masks) + full protocol", 1)

# ---- tier 2: V6 base and variants (rec. 3) ---------------------------------------------
G = "g3_v6"
E(G, 2, "v6_base", V6, {}, "V6 base: slim V5F + profile-zero, lags, mask 0.15, NB head", 3)
E(G, 2, "v6_gat_drop", V6, {"model.gat_drop_missing_sources": True}, "V6 + GAT ignores missing sources", 3)
E(G, 2, "v6_no_sensor_emb", V6, {"model.sensor_embedding_dim": 0, "model.sensor_embedding_dropout": 0.0},
  "V6 - sensor embedding", 3)
E(G, 2, "v6_iid_unify_masks", V6, {"data.mask_strategy": "iid", "data.mask_weather": True,
                                   "data.artificial_mask_observed_only": False},
  "V6 with E2.1-style traffic-agnostic iid masks (observed-only restriction lifted)", 3)
E(G, 2, "v6_outage_head", V6, gen.OUTAGE, "V6 + anchored outage mixture (diagnostic head)", 3)

# ---- tier 3: building blocks on the fixed protocol (rec. 2) -----------------------------
G = "g2_blocks"
E(G, 3, "v6p_p3b", gen.P3B, PROTOCOL, "P3B + full protocol (base for the P3 blocks)", 2)
E(G, 3, "v6p_e2_1b", gen.P2B, {**gen.STRUCTURED, "data.mask_strategy": "iid", **PROTOCOL},
  "E2.1b iid over observed traffic + unified + full protocol", 2)
E(G, 3, "v6p_e2_2_block", gen.P2B, {**gen.STRUCTURED, "data.mask_strategy": "block", **PROTOCOL},
  "E2.2 block gaps + full protocol", 2)
E(G, 3, "v6p_e3_3d_profile_zero", gen.P3B, {"data.profile_prior": "inductive_zero", **PROTOCOL},
  "E3.3d profile inductive_zero + full protocol", 2)
E(G, 3, "v6p_e3_2_log1p", gen.P3B, {"data.log1p_inputs": True, **PROTOCOL}, "E3.2 log1p inputs + full protocol", 2)
E(G, 3, "v6p_e3_7b_lags", gen.P3B, {"data.lag_features": [96, 672], **PROTOCOL}, "E3.7b lags 96/672 + full protocol", 2)

# ---- tier 4: held-out / kriging track on V6 (rec. 4) ------------------------------------
G = "g4_heldout"
E(G, 4, "v6_no_kriging", V6, {"loss.w_kriging": 0.0}, "V6 - kriging loss", 4)
E(G, 4, "v6_kriging_iid", V6, {"loss.kriging_observed_only": False}, "V6 kriging on all slots (not observed-only)", 4)
E(G, 4, "v6_profile_static", V6, {"data.profile_prior": "inductive_static"},
  "V6 with static-feature ridge profile at unseen stations (static level covariate)", 4)
E(G, 4, "v6_residual_neighbour", V6, {"data.profile_prior": "inductive_neighbour", "model.residual_to_profile": True},
  "V6 forecasting the residual to the k-NN donor profile (mu = (profile + 1) exp(head))", 4)

# ---- tier 5: borderline LOOs on V6 (rec. 5) + cheap peak arms (rec. 6) --------------------
G = "g5_loo_peak"
E(G, 5, "v6_no_profile", V6, {"data.profile_prior": "none"}, "V6 LOO - seasonal profile", 5)
E(G, 5, "v6_no_calendar", V6, {"data.calendar_features": False}, "V6 LOO - calendar features", 5)
E(G, 5, "v6_mask_030", V6, {"data.mask_prob": 0.30}, "V6 LOO mask ratio 0.15 -> 0.30", 5)
E(G, 5, "v6_weather_extremes", V6, {"data.weather_extreme_features": True},
  "V6 + weather-extreme indicators (perfect weather forecast at the target step)", 6)
E(G, 5, "v6_lstm", V6, V6_LSTM, "V6-input LSTM, NB head (ensemble partner of v6_base)", 6)

# ---- tier 6: longer context on V6 (rec. 6) ---------------------------------------------
G = "g6_seq48"
E(G, 6, "v6_seq48", V6, gen.SEQ48, "V6 at seq_len 48 (bs64 x2 accumulation)", 6)


# ---------------------------------------------------------------------------
# GPU-hour estimate, calibrated on the v5 suite (docs/experiments/riot_v5/data/runs.csv:
# median hours per epoch). With the cosine schedule most runs train close to the budget.
# ---------------------------------------------------------------------------
EXPECTED_EPOCH_FRAC = 0.9


def est_gpu_hours(cfg: dict) -> float:
    if cfg["model_type"] == "lstm":
        per_epoch = 0.015
    elif cfg["model"]["gat_drop_missing_sources"]:
        per_epoch = 0.025   # v5_full: dropping missing sources shrinks the attention graph
    else:
        per_epoch = 0.045   # P2B-P4B / loo_no_gat_drop range 0.037-0.055
    per_epoch *= (cfg["data"]["seq_len"] / 12.0) ** 1.1   # e3_6_seq48: 0.172 h / epoch
    return round(per_epoch * cfg["training"]["epochs"] * EXPECTED_EPOCH_FRAC, 2)


def build_jobs():
    names = [e["name"] for e in EXPERIMENTS]
    assert len(names) == len(set(names)), "duplicate experiment names"
    jobs = []
    for e in EXPERIMENTS:
        base_cfg = gen.derive(e["base"], e["overrides"])
        for seed in SEEDS:
            cfg = gen.derive(base_cfg, {"training.random_seed": seed})
            run_name = f"{e['name']}-s{seed}"
            cfg["wandb"]["project"] = WANDB_PROJECT
            cfg["wandb"]["collection"] = f"riot-v6-{e['group']}-{e['name']}"
            cfg["wandb"]["run_name"] = run_name
            path = OUT_DIR / e["group"] / f"{e['name']}__s{seed}.yaml"
            jobs.append(dict(
                name=run_name, experiment=e["name"], group=e["group"], seed=seed, tier=e["tier"],
                desc=f"[rec {e['rec']}] {e['desc']}", config=str(path.relative_to(gen.PROJECT_ROOT)),
                cost=est_gpu_hours(cfg), cfg=cfg,
            ))
    return jobs


HEADER = """# ============================================================================
# RIoT v6 stage — {group} / {name} (seed {seed})
# {desc}
# GENERATED by scripts/riotv5/generate_v6_configs.py — edit the generator, not this file.
# ============================================================================
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="resolve_config every generated file")
    args = ap.parse_args()

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
        f.write("# GENERATED by scripts/riotv5/generate_v6_configs.py\n")
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Wrote {len(jobs)} configs ({len(EXPERIMENTS)} experiments x {len(SEEDS)} seeds) to "
          f"{OUT_DIR.relative_to(gen.PROJECT_ROOT)}")
    total = 0.0
    for t in sorted({j["tier"] for j in jobs}):
        sel = [j for j in jobs if j["tier"] == t]
        h = sum(j["cost"] for j in sel)
        total += h
        print(f"  tier {t} {sel[0]['group']:<12} {len(sel):>3} runs  ~{h:6.1f} GPU-h  (cumulative ~{total:6.1f})")
    print(f"  total ~{total:.0f} GPU-h (~{total / 4:.0f} h wall on 4 GPUs) + post-hoc evaluation (~2 GPU-h)")
    if args.check:
        print("All configs resolve and validate.")


if __name__ == "__main__":
    main()
