#!/usr/bin/env python3
"""
Generate the spatial-sparsity scaling sweep: inductive zero-shot RMSE vs the number
of training stations, for the clean permutation-equivariant GNN and the naive
seasonal-profile donor baseline.

Design
  * Evaluation frozen: the 24 held-out stations of inductive_holdout_seed=42 (drawn
    over the full 81-station list), the same chronological test window, the same
    metric suite. Only obs_ind_rmse is comparable across densities — the trans
    metrics score a different station set at every N.
  * Density: data.train_node_count in N_TRAIN. The other non-holdout stations are
    removed from the dataset (no input, loss, graph node or profile donor), as if
    they were never installed. Subsets are nested (data.train_node_seed fixes the
    order), so each step up the sweep adds stations to the previous network.
  * GNN arm: v5 experiment e3_3d_profile_inductive_zero, unchanged — P3 base
    (observed-only targets, NB, complete weather, unified missing encoding, mixed
    gaps @ 0.30), no sensor-ID embedding, GAT keeps missing sources, own seasonal
    profile at training stations and zero + flag at held-out ones (no donors).
  * Baseline arm: v5 naive_profile, unchanged — held-out stations get the mean
    profile of their k=5 static-feature nearest donors, and donors can only be the
    N stations left in the network.
  * Anchor: at N=57 nothing is dropped, so scale_*_n57 replicate e3_3d-s42 (39.09)
    and naive_profile-s42 (42.63). The naive replicate is exact; the GNN one
    measures GPU non-determinism.

Output:
  configs/train/v5_scaling/<name>__s<seed>.yaml
  configs/train/v5_scaling/suite_manifest.yaml   (logs_dir: logs/v5_scaling)

Usage:
  python scripts/riotv5/generate_scaling_configs.py                       # 5 GNN + 5 naive
  python scripts/riotv5/generate_scaling_configs.py --seeds 42 43 44      # optimisation variance
  python scripts/riotv5/generate_scaling_configs.py --draws 0 1 2         # station-draw variance
  python scripts/riotv5/generate_scaling_configs.py --patience 20         # early stopping patience override
  ./scripts/riotv5/start_scaling_sweep.sh --gpus 0 1 2 3                  # launch
  python scripts/riotv5/analyse_scaling.py                                # fit + crossover
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_v5_configs as gen  # noqa: E402

OUT_DIR = gen.PROJECT_ROOT / "configs" / "train" / "v5_scaling"
LOGS_DIR = "logs/v5_scaling"
GROUP = "scaling"
N_TRAIN = [15, 25, 35, 45, 57]
FULL_TRAIN_POOL = 57   # 81 stations - 24 held out
ALL_NODES = 81

ARMS = {
    # arm: (v5 experiment it replicates, description)
    "gnn": ("e3_3d_profile_inductive_zero", "clean inductive GNN (e3_3d stack)"),
    "naive": ("naive_profile", "naive seasonal-profile donor baseline"),
}


def experiment_cfg(name: str) -> dict:
    e = next(e for e in gen.EXPERIMENTS if e["name"] == name)
    return gen.derive(e["base"], e["overrides"])


def check_gnn_stack(cfg: dict) -> None:
    """The architecture this sweep is about; fail loudly if the v5 base ever drifts."""
    d, m, l = cfg["data"], cfg["model"], cfg["loss"]
    assert cfg["model_type"] == "gnn" and cfg["eval_mode"] == "inductive"
    assert d["unify_missing_encoding"] and l["target_mask"] == "observed"
    assert l["impute_positions"] == "masked_observed" and l["kriging_observed_only"]
    assert m["sensor_embedding_dim"] == 0, "sensor-ID embeddings break permutation equivariance"
    assert not m["gat_drop_missing_sources"]
    assert d["profile_prior"] == "inductive_zero", "the GNN must not see donor profiles"
    assert d["inductive_holdout_seed"] == 42


def node_cost(cfg: dict, n_train: int) -> float:
    # GAT / attention work is ~linear in the number of graph nodes.
    return round(gen.relative_cost(cfg) * (n_train + ALL_NODES - FULL_TRAIN_POOL) / ALL_NODES, 2)


def build_jobs(counts, seeds, draws, patience: int = None):
    jobs = []
    for arm, (source, arm_desc) in ARMS.items():
        base = experiment_cfg(source)
        if arm == "gnn":
            check_gnn_stack(base)
        # The naive baseline is deterministic: one seed is exact.
        arm_seeds = seeds if arm == "gnn" else seeds[:1]
        for n in counts:
            assert 1 <= n <= FULL_TRAIN_POOL, f"N_train={n} outside [1, {FULL_TRAIN_POOL}]"
            for draw in draws:
                if n == FULL_TRAIN_POOL and draw != draws[0]:
                    continue  # every draw keeps all 57 stations
                exp = f"scale_{arm}_n{n:02d}" + (f"_d{draw}" if draw else "")
                for seed in arm_seeds:
                    overrides = {
                        "data.train_node_count": n,
                        "data.train_node_seed": draw,
                        "training.random_seed": seed,
                    }
                    if patience is not None:
                        overrides["training.early_stop_patience"] = patience
                    cfg = gen.derive(base, overrides)
                    run_name = f"{exp}-s{seed}"
                    cfg["wandb"]["collection"] = f"riot-v5-{GROUP}-{exp}"
                    cfg["wandb"]["run_name"] = run_name
                    path = OUT_DIR / f"{exp}__s{seed}.yaml"
                    desc = f"{arm_desc}, N_train={n} (draw {draw}) — replicates {source} at N=57"
                    jobs.append(dict(
                        name=run_name, experiment=exp, group=GROUP, seed=seed,
                        tier=1 if seed == arm_seeds[0] else 3, desc=desc,
                        config=str(path.relative_to(gen.PROJECT_ROOT)), cost=node_cost(cfg, n),
                        arm=arm, n_train=n, draw=draw, source_experiment=source, cfg=cfg,
                    ))
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)), "duplicate run names"
    return jobs


HEADER = """# ============================================================================
# RIoT pipeline v5 — sensor-density scaling sweep / {name} (seed {seed})
# {desc}
# GENERATED by scripts/riotv5/generate_scaling_configs.py — edit the generator, not this file.
# ============================================================================
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", nargs="+", type=int, default=N_TRAIN, help="N_train values")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="training seeds (GNN arm)")
    ap.add_argument("--draws", nargs="+", type=int, default=[0], help="train_node_seed values (station draws)")
    ap.add_argument("--patience", type=int, default=None, help="early stopping patience (default: inherited from base config)")
    args = ap.parse_args()

    jobs = build_jobs(sorted(args.counts), args.seeds, args.draws, patience=args.patience)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    for j in jobs:
        gen.resolve_config(j["cfg"])
        with open(gen.PROJECT_ROOT / j["config"], "w") as f:
            f.write(HEADER.format(name=j["experiment"], seed=j["seed"], desc=j["desc"]))
            yaml.safe_dump(j["cfg"], f, sort_keys=False, default_flow_style=None)

    manifest = {
        "pipeline_version": gen.PIPELINE_VERSION,
        "wandb_project": gen.WANDB_PROJECT,
        "logs_dir": LOGS_DIR,
        "jobs": [{k: v for k, v in j.items() if k != "cfg"} for j in jobs],
    }
    with open(OUT_DIR / "suite_manifest.yaml", "w") as f:
        f.write("# GENERATED by scripts/riotv5/generate_scaling_configs.py\n")
        yaml.safe_dump(manifest, f, sort_keys=False)

    total = sum(j["cost"] for j in jobs)
    print(f"Wrote {len(jobs)} configs to {OUT_DIR.relative_to(gen.PROJECT_ROOT)} "
          f"({sum(j['arm'] == 'gnn' for j in jobs)} GNN, {sum(j['arm'] == 'naive' for j in jobs)} naive)")
    for j in jobs:
        print(f"  {j['name']:<26} N_train={j['n_train']:<3} cost {j['cost']}")
    print(f"  total cost {total:.2f} v4-full-run units (~{total / 4:.2f} on 4 GPUs)")


if __name__ == "__main__":
    main()
