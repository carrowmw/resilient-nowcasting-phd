#!/usr/bin/env python3
"""
Add missing scaling sweep runs to configs/train/v5_scaling/suite_manifest.yaml
without overwriting or removing any existing completed runs.

New experiments added:
  1. Extra seeds for N=57 (full density): seeds 44, 45, 46, 47 for scale_gnn_n57
     -> Tightens confidence intervals and variance estimation at the asymptotic anchor.
  2. Intermediate density N=51:
     -> scale_gnn_n51 (draws 0, 1, 2; seeds 42, 43) [6 runs]
     -> scale_naive_n51 (draws 0, 1, 2; seed 42)    [3 runs]
     -> Directly samples the critical transition regime between N=45 and N=57.
  3. Verification seed for N=25 draw 4:
     -> scale_gnn_n25_d4-s44 (seed 44)
     -> Verifies whether draw 4's spatial subset is difficult or seed 43 was an isolated fluke.

Usage:
  python3 scripts/riotv5/add_scaling_runs.py
"""
import shutil
import sys
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import generate_v5_configs as gen
import generate_scaling_configs as sc

OUT_DIR = PROJECT_ROOT / "configs" / "train" / "v5_scaling"
MANIFEST_PATH = OUT_DIR / "suite_manifest.yaml"


def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} does not exist. Run generate_scaling_configs.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    existing_jobs = manifest.get("jobs", [])
    existing_names = {j["name"] for j in existing_jobs}
    print(f"Loaded existing manifest with {len(existing_jobs)} jobs.")

    new_jobs = []

    # 1. N=57 GNN: seeds 44, 45, 46, 47
    base_gnn = sc.experiment_cfg("e3_3d_profile_inductive_zero")
    sc.check_gnn_stack(base_gnn)
    for seed in [44, 45, 46, 47]:
        exp = "scale_gnn_n57"
        name = f"{exp}-s{seed}"
        if name in existing_names:
            continue
        cfg = gen.derive(base_gnn, {
            "data.train_node_count": 57,
            "data.train_node_seed": 0,
            "training.random_seed": seed,
        })
        cfg["wandb"]["collection"] = f"riot-v5-scaling-{exp}"
        cfg["wandb"]["run_name"] = name
        path = OUT_DIR / f"{exp}__s{seed}.yaml"
        desc = f"clean inductive GNN (e3_3d stack), N_train=57 (draw 0) — seed {seed} replicate"
        new_jobs.append(dict(
            name=name, experiment=exp, group="scaling", seed=seed, tier=3, desc=desc,
            config=str(path.relative_to(PROJECT_ROOT)), cost=sc.node_cost(cfg, 57),
            arm="gnn", n_train=57, draw=0, source_experiment="e3_3d_profile_inductive_zero", cfg=cfg,
        ))

    # 2. N=51 Intermediate Density:
    # GNN: draws 0, 1, 2; seeds 42, 43
    for draw in [0, 1, 2]:
        exp = "scale_gnn_n51" + (f"_d{draw}" if draw else "")
        for seed in [42, 43]:
            name = f"{exp}-s{seed}"
            if name in existing_names:
                continue
            cfg = gen.derive(base_gnn, {
                "data.train_node_count": 51,
                "data.train_node_seed": draw,
                "training.random_seed": seed,
            })
            cfg["wandb"]["collection"] = f"riot-v5-scaling-{exp}"
            cfg["wandb"]["run_name"] = name
            path = OUT_DIR / f"{exp}__s{seed}.yaml"
            desc = f"clean inductive GNN (e3_3d stack), N_train=51 (draw {draw}) — crossover sweep"
            new_jobs.append(dict(
                name=name, experiment=exp, group="scaling", seed=seed, tier=1 if seed == 42 else 3,
                desc=desc, config=str(path.relative_to(PROJECT_ROOT)), cost=sc.node_cost(cfg, 51),
                arm="gnn", n_train=51, draw=draw, source_experiment="e3_3d_profile_inductive_zero", cfg=cfg,
            ))

    # Naive: draws 0, 1, 2; seed 42
    base_naive = sc.experiment_cfg("naive_profile")
    for draw in [0, 1, 2]:
        exp = "scale_naive_n51" + (f"_d{draw}" if draw else "")
        seed = 42
        name = f"{exp}-s{seed}"
        if name in existing_names:
            continue
        cfg = gen.derive(base_naive, {
            "data.train_node_count": 51,
            "data.train_node_seed": draw,
            "training.random_seed": seed,
        })
        cfg["wandb"]["collection"] = f"riot-v5-scaling-{exp}"
        cfg["wandb"]["run_name"] = name
        path = OUT_DIR / f"{exp}__s{seed}.yaml"
        desc = f"naive seasonal-profile donor baseline, N_train=51 (draw {draw})"
        new_jobs.append(dict(
            name=name, experiment=exp, group="scaling", seed=seed, tier=1, desc=desc,
            config=str(path.relative_to(PROJECT_ROOT)), cost=sc.node_cost(cfg, 51),
            arm="naive", n_train=51, draw=draw, source_experiment="naive_profile", cfg=cfg,
        ))

    # 3. N=25 draw 4 verification seed: seed 44
    exp = "scale_gnn_n25_d4"
    seed = 44
    name = f"{exp}-s{seed}"
    if name not in existing_names:
        cfg = gen.derive(base_gnn, {
            "data.train_node_count": 25,
            "data.train_node_seed": 4,
            "training.random_seed": seed,
        })
        cfg["wandb"]["collection"] = f"riot-v5-scaling-{exp}"
        cfg["wandb"]["run_name"] = name
        path = OUT_DIR / f"{exp}__s{seed}.yaml"
        desc = f"clean inductive GNN (e3_3d stack), N_train=25 (draw 4) — seed 44 robustness check"
        new_jobs.append(dict(
            name=name, experiment=exp, group="scaling", seed=seed, tier=3, desc=desc,
            config=str(path.relative_to(PROJECT_ROOT)), cost=sc.node_cost(cfg, 25),
            arm="gnn", n_train=25, draw=4, source_experiment="e3_3d_profile_inductive_zero", cfg=cfg,
        ))

    if not new_jobs:
        print("No new jobs to add (all jobs are already in manifest).")
        return

    # Write configs
    for j in new_jobs:
        gen.resolve_config(j["cfg"])
        cfg_out = PROJECT_ROOT / j["config"]
        with open(cfg_out, "w") as f:
            f.write(sc.HEADER.format(name=j["experiment"], seed=j["seed"], desc=j["desc"]))
            yaml.safe_dump(j["cfg"], f, sort_keys=False, default_flow_style=None)
        print(f"  + Created config: {j['config']}")

    # Append to manifest
    manifest_jobs_clean = [{k: v for k, v in j.items() if k != "cfg"} for j in new_jobs]
    manifest["jobs"].extend(manifest_jobs_clean)

    with open(MANIFEST_PATH, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"\nSuccessfully appended {len(new_jobs)} jobs to {MANIFEST_PATH.relative_to(PROJECT_ROOT)}.")
    print(f"Total jobs in manifest now: {len(manifest['jobs'])}.")
    print("\nNew jobs queued:")
    for j in new_jobs:
        print(f"  • {j['name']:<24} (arm={j['arm']:<5}, N={j['n_train']:<2}, draw={j['draw']}, tier={j['tier']})")


if __name__ == "__main__":
    main()
