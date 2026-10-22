#!/usr/bin/env python3
"""
Aggregates spatial-track runs into thesis tables and figures.

Reads <output_dir>/runs/*/ produced by run_spatial_experiments.py; writes
<output_dir>/aggregated/:
  model_comparison.csv       per (model, graph, horizon): node-median metrics,
                             mean ± sd across seeds (test split)
  champion.json              val-selected champion + the rule applied
  dm_<focal>_vs_<baseline>_h<h>.csv / dm_summary.csv
  ablation_vs_champion.csv
  attention_entropy.csv / attention_distance.csv / attention_stability.csv
  figures/*.png

DM pairs are explicit config entries (analysis.dm_pairs) — never name-prefix
heuristics.

Usage:
  python scripts/aggregate_spatial_results.py --config configs/spatial/spatial_smoke.yaml
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from spatial.attention import (  # noqa: E402
    AttentionRecord,
    attention_entropy,
    attention_seed_agreement,
    attention_vs_distance,
    top_edges_table,
)
from spatial.graphs import load_distance_matrix  # noqa: E402
from spatial.stats import ablation_table, dm_by_node, dm_summary, seed_stability  # noqa: E402
from spatial import plotting  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aggregate_spatial")

CORE_METRICS = ("mae_raw", "rmse_raw", "mase", "smape_raw")


def load_runs(out_dir: Path):
    """(results long-frame, {job_key: manifest})."""
    runs = sorted((out_dir / "runs").glob("*/manifest.json"))
    assert runs, f"CRITICAL: no completed runs under {out_dir / 'runs'}"
    manifests, frames = {}, []
    for man_path in runs:
        with open(man_path) as f:
            man = json.load(f)
        manifests[man["job_key"]] = man
        df = pd.read_csv(man_path.parent / "results.csv")
        df["stage"] = man["stage"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True), manifests


def model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Node-median per (model, graph, seed, horizon), then seed mean ± sd."""
    test = results[results["split"] == "test"]
    per_seed = (test.groupby(["model", "graph_variant", "seed", "horizon"])
                [list(CORE_METRICS)].median().reset_index())
    agg = (per_seed.groupby(["model", "graph_variant", "horizon"])
           .agg(**{
               **{f"{m}_median": (m, "mean") for m in CORE_METRICS},
               **{f"{m}_median_sd": (m, "std") for m in CORE_METRICS},
               "n_seeds": ("seed", "count"),
           }).reset_index())
    return agg


def select_champion(results: pd.DataFrame, manifests: Dict, analysis: dict) -> dict:
    """Champion = best val node-median champion_metric (seed-averaged) at the
    first horizon, among jobs in analysis.champion_stages. Test never touched."""
    metric = analysis["champion_metric"]
    stages = analysis["champion_stages"]
    val = results[(results["split"] == "val") & results["stage"].isin(stages)]
    assert len(val), f"CRITICAL: no val results in champion stages {stages}"
    h0 = int(val["horizon"].min())
    val = val[val["horizon"] == h0]
    per_seed = (val.groupby(["model", "graph_variant", "seed"])[metric]
                .median().reset_index())
    scores = (per_seed.groupby(["model", "graph_variant"])[metric]
              .mean().reset_index().sort_values(metric))
    best = scores.iloc[0]
    return {
        "model": best["model"], "graph_variant": best["graph_variant"],
        "key": f"{best['model']}|{best['graph_variant']}",
        "val_" + metric: float(best[metric]),
        "rule": (f"min seed-mean of node-median val {metric} at h={h0} "
                 f"over stages {stages}"),
        "candidates": scores.to_dict("records"),
    }


def _load_predictions(out_dir: Path, model: str, graph: str) -> Dict[int, pd.DataFrame]:
    """{seed: predictions frame} for one (model, graph)."""
    out = {}
    for p in sorted((out_dir / "runs").glob(f"{model}_{graph}_s*/predictions.parquet")):
        df = pd.read_parquet(p)
        out[int(df["seed"].iloc[0])] = df
    return out


def run_dm_pairs(out_dir: Path, agg_dir: Path, analysis: dict,
                 horizons: List[int]) -> Optional[pd.DataFrame]:
    rows = []
    for pair in analysis.get("dm_pairs", []):
        focal, base = pair["focal"], pair["baseline"]
        preds_f = _load_predictions(out_dir, focal["model"], focal["graph"])
        preds_b = _load_predictions(out_dir, base["model"], base["graph"])
        seeds = sorted(set(preds_f) & set(preds_b))
        if not seeds:
            logger.warning(f"DM pair {pair} — no common seeds with predictions, skipped")
            continue
        tag = f"{focal['model']}-{focal['graph']}_vs_{base['model']}-{base['graph']}"
        for h in horizons:
            per_seed_frames = []
            for s in seeds:
                dm = dm_by_node(preds_f[s], preds_b[s], horizon=h)
                dm.insert(0, "seed", s)
                per_seed_frames.append(dm)
                summary = dm_summary(dm)
                rows.append({"pair": tag, "horizon": h, "seed": s, **summary})
            pd.concat(per_seed_frames).to_csv(
                agg_dir / f"dm_{tag}_h{h}.csv", index=False)
    if not rows:
        return None
    summary = pd.DataFrame(rows)
    summary.to_csv(agg_dir / "dm_summary.csv", index=False)
    return summary


def _attention_records(out_dir: Path, model: str, graph: str) -> List[AttentionRecord]:
    records = []
    for p in sorted((out_dir / "runs").glob(f"{model}_{graph}_s*/attention.parquet")):
        df = pd.read_parquet(p)
        head_cols = [c for c in df.columns if c.startswith("alpha_head")]
        seed = int(json.loads((p.parent / "manifest.json").read_text())["seed"])
        E = len(df)
        records.append(AttentionRecord(
            edge_index=np.vstack([df["src_idx"].to_numpy(), df["dst_idx"].to_numpy()]),
            alpha_mean=df[head_cols].to_numpy(dtype=np.float32),
            alpha_by_hour=np.full((24, E), np.nan, dtype=np.float32),
            n_windows=-1, seed=seed,
        ))
    return records


def run_attention_analysis(out_dir: Path, agg_dir: Path, fig_dir: Path,
                           config: dict, manifests: Dict) -> None:
    att_jobs = sorted({(m["model"], m["graph_variant"])
                       for m in manifests.values() if m.get("attention_written")})
    if not att_jobs:
        logger.info("No attention outputs — analysis skipped")
        return
    with open(out_dir / "context_manifest.json") as f:
        node_ids = json.load(f)["node_ids"]
    dist = load_distance_matrix(node_ids, config["nodes"]["edge_features_cache"])

    ent_frames, stab_rows = [], []
    for model, graph in att_jobs:
        records = _attention_records(out_dir, model, graph)
        if not records:
            continue
        src, dst = records[0].edge_index
        if not (src != dst).any():
            logger.info(f"Attention for {model}|{graph}: self-loops only "
                        f"(identity control) — edge analysis skipped")
            continue
        ent = attention_entropy(records[0], n_nodes=len(node_ids))
        ent.insert(0, "model", model)
        ent.insert(1, "graph_variant", graph)
        ent["node_id"] = [node_ids[i] for i in ent["node_idx"]]
        ent_frames.append(ent)
        dist_df = attention_vs_distance(records[0], dist, node_ids)
        dist_df.insert(0, "model", model)
        dist_df.insert(1, "graph_variant", graph)
        dist_df.to_csv(agg_dir / f"attention_distance_{model}_{graph}.csv", index=False)
        top_edges_table(records[0], node_ids).to_csv(
            agg_dir / f"attention_top_edges_{model}_{graph}.csv", index=False)
        if len(records) >= 2:
            is_learned = any(m.get("capabilities", {}).get("learns_topology", False)
                             for m in manifests.values()
                             if m["model"] == model and m["graph_variant"] == graph)
            if is_learned:
                logger.info(f"Attention for {model}|{graph}: learns topology — seed agreement skipped")
            else:
                stab_rows.append({"model": model, "graph_variant": graph,
                                  **attention_seed_agreement(records)})
        plotting.plot_attention_graph(
            records[0], dist, node_ids,
            str(fig_dir / f"attention_graph_{model}_{graph}.png"),
            title=f"{model} on {graph} — mean attention (seed {records[0].seed})")
        plotting.plot_attention_distance_decay(
            dist_df, str(fig_dir / f"attention_distance_{model}_{graph}.png"),
            title=f"{model} on {graph}")
    if ent_frames:
        pd.concat(ent_frames).to_csv(agg_dir / "attention_entropy.csv", index=False)
    if stab_rows:
        pd.DataFrame(stab_rows).to_csv(agg_dir / "attention_stability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["run"]["output_dir"])
    agg_dir = out_dir / "aggregated"
    fig_dir = agg_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results, manifests = load_runs(out_dir)
    horizons = sorted(results["horizon"].unique())
    analysis = config["analysis"]

    # ---- tables ----
    comparison = model_comparison(results)
    comparison.to_csv(agg_dir / "model_comparison.csv", index=False)

    champion = select_champion(results, manifests, analysis)
    with open(agg_dir / "champion.json", "w") as f:
        json.dump(champion, f, indent=2, default=str)
    logger.info(f"Champion (val-selected): {champion['key']}")

    tab = ablation_table(comparison, champion_key=champion["key"],
                         metric=f"{analysis['champion_metric']}_median")
    tab.to_csv(agg_dir / "ablation_vs_champion.csv", index=False)

    test = results[results["split"] == "test"].copy()
    stab = seed_stability(test, metric="mae_raw")
    stab.to_csv(agg_dir / "seed_stability.csv", index=False)

    dm = run_dm_pairs(out_dir, agg_dir, analysis, horizons)

    # ---- figures ----
    test["label"] = test["model"] + "|" + test["graph_variant"]
    plotting.plot_model_comparison(
        test, "mae_raw", str(fig_dir / "model_comparison_mae.png"),
        title="Per-node MAE (raw counts, test)")
    plotting.plot_model_comparison(
        test, "mase", str(fig_dir / "model_comparison_mase.png"),
        title="Per-node MASE (test)")
    plotting.plot_seed_stability(stab, "node-median MAE (raw)",
                                 str(fig_dir / "seed_stability.png"))
    champ_stab = stab[stab["model"] == champion["model"]]
    if champ_stab["graph_variant"].nunique() > 1:
        plotting.plot_graph_ablation(
            champ_stab, "node-median MAE (raw)",
            str(fig_dir / "graph_ablation.png"))

    run_attention_analysis(out_dir, agg_dir, fig_dir, config, manifests)

    logger.info(f"Aggregated {len(manifests)} runs → {agg_dir}")
    if dm is not None:
        for _, r in dm.iterrows():
            logger.info(
                f"DM {r['pair']} h={r['horizon']} seed={r['seed']}: "
                f"frac A better (FDR-sig) = {r['frac_a_better_significant']:.2f}")


if __name__ == "__main__":
    main()
