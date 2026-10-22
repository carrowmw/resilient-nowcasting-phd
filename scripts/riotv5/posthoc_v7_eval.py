#!/usr/bin/env python3
"""
Rec. 2 pre-check for the v7 stage (docs/experiments/riot_v6/riot_v6_seed_report.md §9.2):
does validation-station RMSE track held-out RMSE once the validation stations have
data in the validation window and span the volume range? v6 drew 5 stations at random:
2 had no data after early 2024 and the other 3 were quiet, and the score correlated
NEGATIVELY with held-out RMSE (Spearman -0.38). Inference only, no training.

valcheck  For every saved v6 GNN run (best_model.pt, fetched from W&B if missing),
          predict the validation and test splits and store, per held-out station, the
          squared-error sum and observed count in each window, with the station's
          train-slice level and validation / test coverage.
          -> logs/v7_stage/valcheck/<run>.json  (resumes: existing files are skipped)

valgate   Simulates validation-station selection on those predictions. The 24 held-out
          stations are the only stations no v6 run trained on, so they stand in for
          candidate validation stations: each draw picks a subset as "validation" and
          keeps the rest as "test". Per draw, Spearman rho across runs between
            validation-window RMSE on the subset   and   test-window RMSE on the rest.
          Rules:
            random5      5 stations uniformly from the held-out set (the v6 rule)
            stratified8  8 stations, one per train-level stratum, among stations with
                         >= --min-coverage of the validation window observed (the v7 rule,
                         harness.holdout.stratified_validation_station_ids)
          PASS if stratified8 has median rho >= --threshold and rho > 0 in >= --min-positive
          of the draws. Exit 0 = pass, 1 = fail, 2 = not enough data to decide.
          -> logs/v7_stage/rec2_gate.json

The held-out stations are used here only to validate the selection PROCEDURE on finished
runs. No v7 model is selected on them: the gated v7 runs validate on 8 TRAINING stations.

Usage:
  python scripts/riotv5/posthoc_v7_eval.py valcheck --shard 0 --num-shards 4
  python scripts/riotv5/posthoc_v7_eval.py valgate
  (add --use_mock_data --models-dir models/local --manifest <v7 manifest> for a CPU smoke test)
"""
import argparse
import gc
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_v6_eval as ph  # noqa: E402  (puts src/ on sys.path)

from train import _loader, _to_device, chronological_purged_split, split_bounds, station_window_stats  # noqa: E402

logger = logging.getLogger("posthoc_v7")
PROJECT_ROOT = ph.PROJECT_ROOT
V6_MANIFEST = PROJECT_ROOT / "configs" / "train" / "v6" / "suite_manifest.yaml"
OUT_DIR = PROJECT_ROOT / "logs" / "v7_stage"


# ---------------------------------------------------------------------------
# valcheck
# ---------------------------------------------------------------------------
def data_key(config: dict) -> str:
    return hashlib.sha1(json.dumps([config["data"], config["train_split"], config["val_split"]],
                                   sort_keys=True, default=str).encode()).hexdigest()[:12]


def shard_runs(items, num_shards: int):
    """Assign (run, key) items to shards whole data-key groups at a time (largest group
    first, to the least-loaded shard), so each shard builds every dataset once."""
    groups = {}
    for run, key in items:
        groups.setdefault(key, []).append(run)
    shards = [[] for _ in range(num_shards)]
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        min(shards, key=len).extend((run, key) for run in sorted(groups[key]))
    return shards


def station_errors(member, preparer, loader, config, device, n_nodes):
    """Per-node squared-error sum and observed-target count over one split."""
    se = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    n = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    gen = torch.Generator(device=device).manual_seed(config["evaluation"]["eval_mask_seed"])
    amp = config["amp"] and device.type == "cuda"
    for raw in loader:
        batch = preparer.prepare(_to_device(raw, device), train=False, gen=gen)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            preds = member.adapter.predict(batch)
        point = (preds.point if preds.point is not None else preds.mu).float()
        obs = batch["target_observed"]
        err = torch.where(obs, (point - batch["target_forecast"].float()) ** 2, torch.zeros_like(point))
        se += err.sum(0).double()
        n += obs.sum(0).double()
    return se.cpu().numpy(), n.cpu().numpy()


def cmd_valcheck(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = PROJECT_ROOT / args.models_dir / ("" if args.use_mock_data else args.project)
    out_dir = Path(args.out_dir) / "valcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest) as f:
        jobs = yaml.safe_load(f)["jobs"]
    if args.seeds:
        jobs = [j for j in jobs if j["seed"] in args.seeds]

    items = []
    for j in jobs:
        try:
            d = ph.run_dir(models_dir, j["name"])
        except AssertionError:
            logger.warning(f"{j['name']}: no run directory, skipped")
            continue
        config = ph.load_config(d)
        if config["model_type"] != "gnn" or config["eval_mode"] != "inductive":
            continue
        items.append((j["name"], data_key(config)))
    mine = shard_runs(items, args.num_shards)[args.shard]
    logger.info(f"valcheck shard {args.shard}/{args.num_shards}: {len(mine)} of {len(items)} GNN runs")

    cache_key, cache, failed = None, None, 0
    for run, key in mine:
        path = out_dir / f"{run}.json"
        if path.exists() and not args.force:
            continue
        try:
            d = ph.run_dir(models_dir, run)
            config = ph.load_config(d)
            ckpt = ph.ensure_checkpoint(d, run, args.project, "best_model.pt")
            if key != cache_key:
                # Free the previous dataset before building the next (one per process).
                cache = cache_key = dataset = masks = preparer = scale = loaders = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                dataset, _, masks, preparer, scale = ph.setup(config, args.use_mock_data, device, args.batch_size)
                _, val_ds, test_ds = chronological_purged_split(dataset, config["train_split"], config["val_split"],
                                                                purge_steps=config["data"]["seq_len"] + 1)
                bs = args.batch_size or config["training"]["batch_size"]
                loaders = {w: _loader(ds, bs, config["training"]["num_workers"], train=False)
                           for w, ds in (("val", val_ds), ("test", test_ds))}
                level, val_cov = station_window_stats(config, dataset)
                test_cov = test_coverage(config, dataset)
                cache, cache_key = (dataset, masks, preparer, scale, loaders, level, val_cov, test_cov), key
            dataset, masks, preparer, scale, loaders, level, val_cov, test_cov = cache
            member = ph.Member(run, config, dataset, preparer, scale, ckpt, device, args.use_mock_data)
            locs = list(dataset.locations)
            errs = {w: station_errors(member, preparer, loaders[w], config, device, len(locs)) for w in loaders}
            del member
            held = set(masks.held_ids)
            stations = {
                loc: {"level": level[loc], "val_coverage": val_cov[loc], "test_coverage": test_cov[loc],
                      "val_se": float(errs["val"][0][i]), "val_n": float(errs["val"][1][i]),
                      "test_se": float(errs["test"][0][i]), "test_n": float(errs["test"][1][i])}
                for i, loc in enumerate(locs) if loc in held
            }
            with open(path, "w") as f:
                json.dump({"run": run, "data_key": key, "experiment": run.rsplit("-s", 1)[0],
                           "held_ids": masks.held_ids, "val_ids": masks.val_ids, "stations": stations}, f, indent=1)
            logger.info(f"{run}: wrote {path.name}")
        except Exception as exc:  # one bad checkpoint must not stop the pre-check
            failed += 1
            logger.exception(f"{run}: valcheck FAILED ({exc})")
    return 1 if failed and failed == len(mine) else 0


def test_coverage(config: dict, dataset) -> dict:
    seq_len = config["data"]["seq_len"]
    _, _, _, test_start, test_end = split_bounds(len(dataset), config["train_split"], config["val_split"],
                                                 seq_len + 1)
    observed = dataset.observed[test_start + seq_len:test_end + seq_len]
    cov = observed.float().mean(0).tolist() if len(observed) else [0.0] * len(dataset.locations)
    return dict(zip(dataset.locations, cov))


# ---------------------------------------------------------------------------
# valgate
# ---------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pooled_rmse(se: np.ndarray, n: np.ndarray, cols) -> np.ndarray:
    s, c = se[:, cols].sum(1), n[:, cols].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, np.sqrt(s / np.maximum(c, 1)), np.nan)


def draw_random(rng, n_stations: int, count: int, **_):
    return rng.choice(n_stations, size=count, replace=False)


def draw_stratified(rng, n_stations: int, count: int, level, coverage, min_coverage, **_):
    pool = [i for i in range(n_stations) if coverage[i] >= min_coverage]
    pool.sort(key=lambda i: level[i])
    return np.array([s[rng.randint(len(s))] for s in np.array_split(np.array(pool), count)])


def simulate(rule, mats, stations, count, draws, seed, level, coverage, min_coverage):
    val_se, val_n, test_se, test_n = mats
    rng = np.random.RandomState(seed)
    rhos, n_val_empty = [], 0
    for _ in range(draws):
        sub = rule(rng, len(stations), count, level=level, coverage=coverage, min_coverage=min_coverage)
        rest = np.setdiff1d(np.arange(len(stations)), sub)
        v = pooled_rmse(val_se, val_n, sub)
        if not np.isfinite(v).any():
            n_val_empty += 1
        rhos.append(spearman(v, pooled_rmse(test_se, test_n, rest)))
    rhos = np.array(rhos)
    fin = rhos[np.isfinite(rhos)]
    return {
        "count": count, "draws": draws, "scorable_draws": int(len(fin)), "draws_no_val_data": n_val_empty,
        "rho_median": float(np.median(fin)) if len(fin) else None,
        "rho_p10": float(np.percentile(fin, 10)) if len(fin) else None,
        "rho_p90": float(np.percentile(fin, 90)) if len(fin) else None,
        "frac_positive": float((fin > 0).mean()) if len(fin) else None,
    }


def cmd_valgate(args) -> int:
    files = sorted((Path(args.out_dir) / "valcheck").glob("*.json"))
    recs = [json.load(open(f)) for f in files]
    out = {"kind": "rec2_gate", "n_runs": len(recs), "threshold": args.threshold,
           "min_positive": args.min_positive, "min_coverage": args.min_coverage}

    def finish(code: int, verdict: str) -> int:
        out["verdict"], out["exit_code"] = verdict, code
        path = Path(args.out_dir) / args.out_name
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"rec. 2 gate: {verdict} -> {path}")
        return code

    if len(recs) < args.min_runs:
        return finish(2, f"INSUFFICIENT: {len(recs)} runs with predictions (need {args.min_runs})")
    stations = sorted(set.intersection(*(set(r["stations"]) for r in recs)))
    ref = recs[0]["stations"]
    level = np.array([ref[s]["level"] for s in stations])
    coverage = np.array([ref[s]["val_coverage"] for s in stations])
    mats = [np.array([[r["stations"][s][k] for s in stations] for r in recs])
            for k in ("val_se", "val_n", "test_se", "test_n")]
    covered = int((coverage >= args.min_coverage).sum())
    out.update(n_stations=len(stations), n_covered=covered,
               stations={s: {"level": float(level[i]), "val_coverage": float(coverage[i]),
                             "test_coverage": float(ref[s]["test_coverage"])} for i, s in enumerate(stations)})
    count = min(args.count, covered - 2)
    if count < args.min_count:
        return finish(2, f"INSUFFICIENT: {covered} held-out stations with coverage >= {args.min_coverage}")

    lineages = {"all": np.arange(len(recs))}
    for name, prefix in (("v6_lineage", "v6_"), ("p_family", "v6p_")):
        idx = np.array([i for i, r in enumerate(recs) if r["experiment"].startswith(prefix)], dtype=int)
        if len(idx) >= args.min_runs // 2:
            lineages[name] = idx
    out["rules"] = {}
    for lin, idx in lineages.items():
        sub_mats = [m[idx] for m in mats]
        for rule_name, rule, k in (("random5", draw_random, min(5, len(stations) - 1)),
                                   ("stratified%d" % count, draw_stratified, count)):
            res = simulate(rule, sub_mats, stations, k, args.draws, args.seed, level, coverage, args.min_coverage)
            res["n_runs"] = int(len(idx))
            out["rules"][f"{lin}/{rule_name}"] = res
            logger.info(f"{lin:>10} {rule_name:<12} runs {len(idx):>3}  median rho {res['rho_median']}  "
                        f"p10 {res['rho_p10']}  rho>0 in {res['frac_positive']}")

    gate = out["rules"][f"all/stratified{count}"]
    if gate["scorable_draws"] < 0.5 * args.draws:
        return finish(2, "INSUFFICIENT: fewer than half the stratified draws are scorable")
    ok = gate["rho_median"] >= args.threshold and gate["frac_positive"] >= args.min_positive
    return finish(0 if ok else 1,
                  f"{'PASS' if ok else 'FAIL'}: stratified{count} median rho {gate['rho_median']:.2f} "
                  f"(need >= {args.threshold}), rho > 0 in {gate['frac_positive']:.0%} of draws "
                  f"(need >= {args.min_positive:.0%})")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("valcheck")
    p.add_argument("--manifest", default=str(V6_MANIFEST), help="runs to evaluate (GNN, inductive only)")
    p.add_argument("--project", default="resilient-iot-v6", help="W&B project = models/<project>/")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--use_mock_data", action="store_true")
    p = sub.add_parser("valgate")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--min-coverage", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--min-positive", type=float, default=0.8)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-runs", type=int, default=20)
    p.add_argument("--min-count", type=int, default=4, help="smallest stratified subset worth testing")
    p.add_argument("--out-name", default="rec2_gate.json")
    for p in sub.choices.values():
        p.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    sys.exit(cmd_valcheck(args) if args.cmd == "valcheck" else cmd_valgate(args))


if __name__ == "__main__":
    main()
