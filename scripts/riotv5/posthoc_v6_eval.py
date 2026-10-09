#!/usr/bin/env python3
"""
Inference-only evaluations for the v6 stage (docs/experiments/riot_v5/riot_v5_seed_report.md
§14.2, recommendations 6 and 7). No training; every pass uses saved checkpoints.

ensemble  GNN + LSTM ensemble on the test split. Both members see the SAME prepared
          batches (BatchPreparer of the first run, identical evaluation-mask seeds to
          train.py), and the members are scored alone in the same loop, so the three
          rows are paired. The ensemble forecast is the moment-matched NB of the
          equal-weight mixture:
              m = (m1 + m2) / 2,  v = mean(v_i + m_i^2) - m^2,  theta = m^2 / (v - m)
          Point = m; CRPS / PIT / q90 / peak_q90 use that NB.

legacy    The thesis legacy test RMSE (harness/evaluate.py: mean of per-batch RMSEs over
          every slot, zero-filled gaps included) recomputed at fixed evaluation batch
          sizes, next to the pooled RMSE sqrt(sum SE / sum n) of the same predictions.
          Confirms numerically that the bs32 "gain" is an evaluation-batch-size artefact.

Checkpoints: models/<project>/<run>/best_model.pt (or best_model_ind.pt with
--checkpoint ind). If a checkpoint is missing locally it is fetched from the run's W&B
model artifact (read-only).

Usage:
  python scripts/riotv5/posthoc_v6_eval.py ensemble --project resilient-iot-v6 \\
      --runs v6_base-s42 v6_lstm-s42 --name v6_ens-s42
  python scripts/riotv5/posthoc_v6_eval.py legacy --project resilient-iot-v5 \\
      --runs v4r_full-s42 v4r_bs32-s42 --batch-sizes 32 128
  (add --use_mock_data --models-dir models/local for a CPU smoke test)

Output: logs/v6_stage/posthoc/<kind>_<name>.json (and a printed summary).
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from harness.adapters import build_adapter  # noqa: E402
from harness.batch_prep import BatchPreparer  # noqa: E402
from harness.contract import Capabilities, Predictions  # noqa: E402
from harness.evaluate import evaluate_batch  # noqa: E402
from harness.evaluate_v5 import V5Evaluator  # noqa: E402
from losses import TrafficScale  # noqa: E402
from model_factory import build_model  # noqa: E402
from run_config import resolve_config  # noqa: E402
from train import _loader, _to_device, build_dataset, build_node_masks, chronological_purged_split  # noqa: E402

logger = logging.getLogger("posthoc_v6")
OUT_DIR = PROJECT_ROOT / "logs" / "v6_stage" / "posthoc"

# Data keys that may differ between ensemble members: they change the training-time
# holdout re-roll only, never the evaluation batch.
ENSEMBLE_DATA_KEYS_FREE = {"spatial_holdout_prob"}


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------
def run_dir(models_dir: Path, run: str) -> Path:
    """models/<project>/<run>/, or the newest models/local/<run>_<timestamp>/ (mock runs)."""
    direct = models_dir / run
    if direct.exists():
        return direct
    local = sorted(models_dir.glob(f"{run}_*"), key=lambda p: p.stat().st_mtime)
    assert local, f"CRITICAL: no run directory for {run} under {models_dir}"
    return local[-1]


def ensure_checkpoint(d: Path, run: str, project: str, filename: str) -> Path:
    path = d / filename
    if path.exists():
        return path
    logger.info(f"{path} missing locally — fetching the W&B model artifact of {run} (read-only)")
    import wandb
    api = wandb.Api(timeout=120)
    matches = [r for r in api.runs(f"{api.default_entity}/{project}", filters={"display_name": run})]
    assert matches, f"CRITICAL: W&B run {run} not found in {project}"
    for art in matches[-1].logged_artifacts():
        if art.type == "model" and filename in [f.name for f in art.files()]:
            art.get_entry(filename).download(root=str(d))
            assert path.exists(), f"CRITICAL: download of {filename} for {run} failed"
            return path
    raise FileNotFoundError(f"CRITICAL: no model artifact with {filename} for {run}")


def load_config(d: Path) -> dict:
    with open(d / "config.yaml") as f:
        return resolve_config(yaml.safe_load(f))


class Member:
    """One trained model bound to a shared dataset / preparer."""

    def __init__(self, name: str, config: dict, dataset, preparer: BatchPreparer, scale: TrafficScale,
                 checkpoint: Path, device: torch.device, use_mock_data: bool):
        self.name, self.config = name, config
        model = build_model(
            config, dataset, device, covariate_dim=preparer.covariates.dim,
            edge_features_cache_path=None if use_mock_data else config["model"].get("edge_features_cache_path"),
        )
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        self.model = model
        self.adapter = build_adapter(config["model_type"], model, config, scale=scale, covariates=preparer.covariates)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def nb_moment_match(preds: List[Predictions], weights: List[float]) -> Predictions:
    """Moment-matched NB of a mixture of NB members (the first member's auxiliary
    outputs — imputation, kriging — are carried along unchanged)."""
    for p in preds:
        assert p.nll_params is not None and p.nll_params["likelihood"] == "nb", \
            "CRITICAL: ensemble members must have an NB forecast head (loss.forecast_likelihood / baseline_head nb)"
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    mus = [p.nll_params["mu"].float() for p in preds]
    thetas = [p.nll_params["theta"].float() for p in preds]
    m = sum(wi * mu for wi, mu in zip(w, mus))
    second = sum(wi * (mu + mu ** 2 / th + mu ** 2) for wi, mu, th in zip(w, mus, thetas))
    var = second - m ** 2
    theta = m ** 2 / (var - m).clamp_min(1e-6)
    base = preds[0]
    return base._replace(
        mu=m, theta=theta, pi=None, point=m, p_zero=torch.pow(theta / (theta + m), theta),
        outage_q=None, quantiles=None, quantile_levels=(),
        nll_params={"likelihood": "nb", "mu": m, "theta": theta, "pi": None, "q": None},
    )


def setup(config: dict, use_mock_data: bool, device: torch.device, batch_size: Optional[int] = None):
    dataset = build_dataset(config, use_mock_data)
    _, _, test_ds = chronological_purged_split(dataset, config["train_split"], config["val_split"],
                                               purge_steps=config["data"]["seq_len"] + 1)
    assert test_ds is not None and len(test_ds) > 0, "CRITICAL: empty test split"
    loader = _loader(test_ds, batch_size or config["training"]["batch_size"],
                     config["training"]["num_workers"], train=False)
    location_ids = list(dataset.locations)
    masks = build_node_masks(config, location_ids, dataset.all_locations, device)
    eval_holdout = masks.unseen
    if config["eval_mode"] != "inductive" and config["data"]["spatial_holdout_prob"] > 0.0:
        raise NotImplementedError("post-hoc evaluation is implemented for eval_mode=inductive runs")
    preparer = BatchPreparer(
        config, dataset, device, inductive_mask=masks.unseen, loss_node_mask=None if masks.unseen is None else ~masks.unseen,
        eval_holdout_mask=eval_holdout, static_features=dataset.static_features.to(device), location_ids=location_ids,
    )
    scale = TrafficScale(torch.tensor(dataset.traffic_mean, device=device),
                         torch.tensor(dataset.traffic_std, device=device), config["data"]["log1p_inputs"])
    return dataset, loader, masks, preparer, scale


def cmd_ensemble(args) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = PROJECT_ROOT / args.models_dir / ("" if args.use_mock_data else args.project)
    dirs = [run_dir(models_dir, r) for r in args.runs]
    configs = [load_config(d) for d in dirs]
    ref = configs[0]
    for r, c in zip(args.runs[1:], configs[1:]):
        diff = {k for k in ref["data"] if ref["data"][k] != c["data"][k]} - ENSEMBLE_DATA_KEYS_FREE
        assert not diff, f"CRITICAL: {r} differs from {args.runs[0]} in data keys {sorted(diff)} — batches would differ"
        assert c["eval_mode"] == ref["eval_mode"] and c["evaluation"] == ref["evaluation"]
    filename = "best_model_ind.pt" if args.checkpoint == "ind" else "best_model.pt"
    ckpts = [ensure_checkpoint(d, r, args.project, filename) for d, r in zip(dirs, args.runs)]

    dataset, loader, masks, preparer, scale = setup(ref, args.use_mock_data, device)
    members = [Member(r, c, dataset, preparer, scale, p, device, args.use_mock_data)
               for r, c, p in zip(args.runs, configs, ckpts)]
    weights = args.weights or [1.0] * len(members)
    assert len(weights) == len(members)

    ind_mask = masks.test_holdout
    ignore = masks.val_stations
    caps = Capabilities(uncertainty=True, imputation=True, kriging=True)
    def evaluator(member_caps):
        return V5Evaluator(dataset, preparer.covariates, member_caps, ref, ind_mask, scale, device,
                           diagnostics=True, store_series=True, ignore_mask=ignore)

    evaluators = {m.name: evaluator(m.adapter.capabilities) for m in members}
    evaluators["ensemble"] = evaluator(caps)
    eval_seed = ref["evaluation"]["eval_mask_seed"]
    amp = ref["amp"] and device.type == "cuda"

    def run_pass(so_length: Optional[int]):
        gen = torch.Generator(device=device).manual_seed(eval_seed + (so_length or 0))
        for raw in loader:
            batch = preparer.prepare(_to_device(raw, device), train=False, gen=gen, so_length=so_length)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                preds = [m.adapter.predict(batch) for m in members]
            scored = [(m.name, p) for m, p in zip(members, preds)] + [("ensemble", nb_moment_match(preds, weights))]
            for name, p in scored:
                if so_length is None:
                    evaluators[name].update(p, batch)
                else:
                    evaluators[name].update_so(so_length, p, batch)

    logger.info(f"Ensemble {args.name}: {args.runs} ({filename}), weights {weights}")
    run_pass(None)
    for L in ref["evaluation"]["so_lengths"]:
        run_pass(L)
    return {"kind": "ensemble", "name": args.name, "runs": args.runs, "checkpoint": filename, "weights": weights,
            "metrics": {name: ev.summary() for name, ev in evaluators.items()}}


# ---------------------------------------------------------------------------
# Legacy metric at fixed evaluation batch sizes
# ---------------------------------------------------------------------------
def cmd_legacy(args) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = PROJECT_ROOT / args.models_dir / ("" if args.use_mock_data else args.project)
    out = {"kind": "legacy", "name": args.name, "runs": {}}
    cache = {}
    for run in args.runs:
        d = run_dir(models_dir, run)
        config = load_config(d)
        ckpt = ensure_checkpoint(d, run, args.project, "best_model.pt")
        key = json.dumps(config["data"], sort_keys=True, default=str) + str(args.use_mock_data and config["training"]["random_seed"])
        if key not in cache:
            cache[key] = setup(config, args.use_mock_data, device)
        dataset, _, masks, preparer, scale = cache[key]
        member = Member(run, config, dataset, preparer, scale, ckpt, device, args.use_mock_data)
        _, _, test_ds = chronological_purged_split(dataset, config["train_split"], config["val_split"],
                                                   purge_steps=config["data"]["seq_len"] + 1)
        res = {"train_batch_size": config["training"]["batch_size"]}
        for bs in args.batch_sizes:
            loader = _loader(test_ds, bs, config["training"]["num_workers"], train=False)
            gen = torch.Generator(device=device).manual_seed(config["evaluation"]["eval_mask_seed"])
            per_batch, sse, n = [], 0.0, 0
            for raw in loader:
                batch = preparer.prepare(_to_device(raw, device), train=False, gen=gen)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                     enabled=config["amp"] and device.type == "cuda"):
                    preds = member.adapter.predict(batch)
                m = evaluate_batch(member.adapter.capabilities, preds, batch, holdout_mask=preparer.eval_holdout_mask)
                per_batch.append(m["rmse"])
                point = preds.point if preds.point is not None else preds.mu
                err = (batch["target_forecast"].float() - point.float()) ** 2
                sse += float(err.sum())
                n += err.numel()
            res[f"bs{bs}"] = {"legacy_rmse_mean_of_batches": sum(per_batch) / len(per_batch),
                              "pooled_rmse": math.sqrt(sse / n), "n_batches": len(per_batch)}
        out["runs"][run] = res
        logger.info(f"{run}: " + ", ".join(
            f"bs{bs} legacy {res[f'bs{bs}']['legacy_rmse_mean_of_batches']:.3f} / pooled {res[f'bs{bs}']['pooled_rmse']:.3f}"
            for bs in args.batch_sizes))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ensemble", "legacy"):
        p = sub.add_parser(name)
        p.add_argument("--runs", nargs="+", required=True)
        p.add_argument("--name", required=True)
        p.add_argument("--project", default="resilient-iot-v6", help="W&B project = models/<project>/")
        p.add_argument("--models-dir", default="models")
        p.add_argument("--use_mock_data", action="store_true")
        p.add_argument("--out-dir", default=str(OUT_DIR))
    sub.choices["ensemble"].add_argument("--weights", nargs="+", type=float, default=None)
    sub.choices["ensemble"].add_argument("--checkpoint", choices=("best", "ind"), default="best")
    sub.choices["legacy"].add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128])
    args = ap.parse_args()

    result = cmd_ensemble(args) if args.cmd == "ensemble" else cmd_legacy(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.cmd}_{args.name}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=lambda v: None)
    if args.cmd == "ensemble":
        keys = ("obs_all_rmse", "obs_trans_rmse", "obs_ind_rmse", "peak_rmse", "peak_q90_rmse",
                "so_12_forecast_rmse", "crps", "q90_coverage")
        logger.info("member / ensemble: " + " | ".join(keys))
        for name, m in result["metrics"].items():
            logger.info(f"  {name:<28} " + " ".join(
                "--" if m.get(k) is None else f"{m[k]:.3f}" for k in keys))
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
