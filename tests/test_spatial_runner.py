# Tests for scripts/run_spatial_experiments.py — sha scoping, job expansion.
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "run_spatial_experiments",
    Path(__file__).parent.parent / "scripts" / "run_spatial_experiments.py")
runner = importlib.util.module_from_spec(_spec)
sys.modules["run_spatial_experiments"] = runner
_spec.loader.exec_module(runner)

BASE_CONFIG = yaml.safe_load("""
data_source: parquet
data: {parquet_path: x.parquet, start_date: 2024-01-01, end_date: 2024-02-01,
       veh_class: person, direction: all, freq: 15min}
nodes: {min_completeness: 0.5, edge_features_cache: cache.json,
        static_features_cache: static.parquet, min_nodes: 5}
temporal: {window: 12, horizons: [1, 2], stride: 1, train_frac: 0.7,
           val_frac: 0.15, purge_boundaries: true}
inputs: {mask_channel: true, time_features: [day]}
graph:
  variants:
    identity: {method: identity}
    gabriel: {method: gabriel, edge_attr: physical}
models:
  persistence: {type: persistence}
  gat_plain: {type: gat, temporal_encoder: none, gat_layers: 2, heads: 4,
              hidden_dim: 64, use_static_features: false, lr: 1.0e-3,
              batch_size: 32, epochs: 5, patience: 2, scheduler_step_size: 1,
              scheduler_gamma: 0.9, grad_clip: 1.0}
run:
  output_dir: out
  device: null
  matrix:
    - {tag: A, models: [persistence], graphs: [identity], seeds: [42]}
    - {tag: B, models: [gat_plain], graphs: [identity, gabriel], seeds: [42, 43]}
wandb: {enabled: false, project: p, mode: offline}
""")


def test_expand_jobs_deterministic_and_counts():
    jobs = runner.expand_jobs(BASE_CONFIG)
    assert len(jobs) == 1 + 2 * 2
    keys = [j.job_key for j in jobs]
    assert keys[0] == "persistence_identity_s42"
    assert "gat_plain_gabriel_s43" in keys
    assert runner.expand_jobs(BASE_CONFIG) == jobs   # deterministic


def test_expand_jobs_rejects_unknown_and_duplicates():
    bad = copy.deepcopy(BASE_CONFIG)
    bad["run"]["matrix"][0]["models"] = ["ghost"]
    with pytest.raises(AssertionError, match="unknown model"):
        runner.expand_jobs(bad)
    bad2 = copy.deepcopy(BASE_CONFIG)
    bad2["run"]["matrix"].append(bad2["run"]["matrix"][0])
    with pytest.raises(AssertionError, match="duplicate job keys"):
        runner.expand_jobs(bad2)
    bad3 = copy.deepcopy(BASE_CONFIG)
    bad3["run"]["matrix"][0]["graphs"] = ["delaunay"]
    with pytest.raises(AssertionError, match="unknown graph"):
        runner.expand_jobs(bad3)


def test_job_sha_scoping():
    sha = runner.job_sha(BASE_CONFIG, "gat_plain", "gabriel", 42)
    assert len(sha) == 16

    # Irrelevant edits do NOT invalidate: wandb, run.matrix, OTHER models,
    # OTHER graph variants.
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["wandb"]["project"] = "renamed"
    cfg["run"]["matrix"] = []
    cfg["models"]["persistence"] = {"type": "persistence", "note": 1}
    cfg["graph"]["variants"]["identity"] = {"method": "identity", "x": 1}
    assert runner.job_sha(cfg, "gat_plain", "gabriel", 42) == sha

    # Relevant edits DO invalidate.
    for mutate in (
        lambda c: c["models"]["gat_plain"].__setitem__("heads", 8),
        lambda c: c["graph"]["variants"]["gabriel"].__setitem__("edge_attr", "none"),
        lambda c: c["temporal"].__setitem__("window", 24),
        lambda c: c["data"].__setitem__("end_date", "2024-03-01"),
        lambda c: c["inputs"].__setitem__("mask_channel", False),
        lambda c: c["nodes"].__setitem__("min_completeness", 0.9),
    ):
        c = copy.deepcopy(BASE_CONFIG)
        mutate(c)
        assert runner.job_sha(c, "gat_plain", "gabriel", 42) != sha

    # Seed is part of the identity.
    assert runner.job_sha(BASE_CONFIG, "gat_plain", "gabriel", 43) != sha


def test_smoke_config_parses_and_expands():
    with open(Path(__file__).parent.parent / "configs/spatial/spatial_smoke.yaml") as f:
        cfg = yaml.safe_load(f)
    jobs = runner.expand_jobs(cfg)
    assert len(jobs) == 1 + 1 + 2 * 2
    for job in jobs:
        assert len(runner.job_sha(cfg, job.model_key, job.graph_variant, job.seed)) == 16


def test_full_config_parses_and_expands():
    with open(Path(__file__).parent.parent / "configs/spatial/spatial_full.yaml") as f:
        cfg = yaml.safe_load(f)
    jobs = runner.expand_jobs(cfg)
    # 4 baselines + 2 temporal×3 + 3 spatial×3 + 3 graphs×3 + learned×3
    # + 4 ablations×3 + no-edge-attr×3
    assert len(jobs) == 4 + 6 + 9 + 9 + 3 + 12 + 3
    keys = {j.job_key for j in jobs}
    assert "gat_transformer_gabriel_s42" in keys
    assert "gat_learned_identity_s44" in keys
