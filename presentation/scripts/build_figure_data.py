#!/usr/bin/env python3
"""Build cached real-data JSON for the presentation figures.

Pulls the default run's spatial topology and one representative day of model
predictions from the Cloudflare R2 payload bucket (the same source the live
digital-twin dashboard consumes) and distils them into two small, self-contained
files that the static reveal.js figures can load without network access:

  presentation/figures/data/topology.json  -> topology-graph.html
  presentation/figures/data/kriging.json    -> kriging-map.html

Re-run to refresh the cache:
  python3 presentation/scripts/build_figure_data.py
"""
from __future__ import annotations

import json
import math
import statistics
import urllib.request
from pathlib import Path

R2_BASE = Path(__file__).resolve().parents[2] / "frontend" / "public" / "data" / "runs"
# Representative day pulled for the snapshot (evaluation period).
SNAPSHOT_DATE = "2026-05-15"
# Edge budget per node — matched across both graphs for a fair side-by-side.
K = 4
# PIT diagram runs: underdispersion vs good calibration
PIT_RUNS = {
    "underdispersed": "dark-brook-37",
    "calibrated": "pretty-glitter-29",
}

OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "data"


def fetch_json(url: str | Path) -> dict:
    if isinstance(url, str) and url.startswith("http"):
        req = urllib.request.Request(
            url,
            headers={"Accept-Encoding": "gzip", "User-Agent": "Mozilla/5.0 (figure-cache)"},
        )
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip

                raw = gzip.decompress(raw)
            return json.loads(raw)
    else:
        # Load locally
        return json.loads(Path(url).read_text())


def haversine(a: dict, b: dict) -> float:
    """Great-circle distance in metres between two {lat, lon} nodes."""
    r = 6371000.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dphi = math.radians(b["lat"] - a["lat"])
    dlmb = math.radians(b["lon"] - a["lon"])
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def main() -> None:
    registry = fetch_json(f"{R2_BASE}/run_registry.json")
    run_id = registry["default_run_id"]
    run_meta = next(r for r in registry["runs"] if r["run_id"] == run_id)
    print(f"default run: {run_id}")

    spatial = fetch_json(f"{R2_BASE}/{run_id}/spatial.json")
    page = fetch_json(f"{R2_BASE}/{run_id}/{SNAPSHOT_DATE}/page.json")

    nodes = spatial["nodes"]
    by_id = {n["id"]: n for n in nodes}

    # ── Mean GAT attention per directed edge (real learned weights) ──
    attn: dict[tuple[int, int], float] = {}
    for e in page["edge_attention"]:
        s, t = e["source"], e["target"]
        if s == t:
            continue  # drop self-loops
        attn[(s, t)] = statistics.mean(e["attention_sequence"])

    # ── Learned graph: top-K outgoing attention edges per node ──
    out_edges: dict[int, list[tuple[int, float]]] = {}
    for (s, t), w in attn.items():
        out_edges.setdefault(s, []).append((t, w))
    learned_und: dict[tuple[int, int], float] = {}
    for s, lst in out_edges.items():
        for t, w in sorted(lst, key=lambda x: -x[1])[:K]:
            key = (min(s, t), max(s, t))
            learned_und[key] = max(learned_und.get(key, 0.0), w)

    # ── Euclidean graph: K nearest neighbours per node ──
    euclid_und: set[tuple[int, int]] = set()
    for n in nodes:
        dists = sorted(
            ((m["id"], haversine(n, m)) for m in nodes if m["id"] != n["id"]),
            key=lambda x: x[1],
        )
        for t, _ in dists[:K]:
            euclid_und.add((min(n["id"], t), max(n["id"], t)))

    # ── Attention centrality (sum of incident learned weights) for colouring ──
    centrality = {n["id"]: 0.0 for n in nodes}
    for (s, t), w in learned_und.items():
        centrality[s] += w
        centrality[t] += w

    # ── Non-local ratio: learned edges absent from the distance graph ──
    non_local = sum(1 for k in learned_und if k not in euclid_und)
    non_local_pct = round(100 * non_local / max(len(learned_und), 1))

    node_out = [
        {
            "id": n["id"],
            "name": n["location_name"],
            "lat": n["lat"],
            "lon": n["lon"],
            "sim": n["is_simulated"],
            "centrality": round(centrality[n["id"]], 4),
        }
        for n in nodes
    ]

    topology = {
        "meta": {
            "run_id": run_id,
            "wandb_run_id": spatial["metadata"].get("wandb_run_id"),
            "num_nodes": len(nodes),
            "num_real": spatial["metadata"]["num_real_nodes"],
            "num_sim": spatial["metadata"]["num_simulated_nodes"],
            "k": K,
        },
        "nodes": node_out,
        "euclidean_edges": [{"s": s, "t": t} for (s, t) in sorted(euclid_und)],
        "learned_edges": [
            {"s": s, "t": t, "w": round(w, 4)}
            for (s, t), w in sorted(learned_und.items())
        ],
        "stats": {
            "nodes": len(nodes),
            "euclidean_edges": len(euclid_und),
            "learned_edges": len(learned_und),
            "non_local_pct": non_local_pct,
        },
    }

    # ── Kriging snapshot: per-node mean prediction + uncertainty ──
    krig_nodes = []
    for n in nodes:
        ts = page["node_timeseries"].get(str(n["id"]), [])
        if not ts:
            continue
        mu = statistics.mean(e["prediction"] for e in ts)
        sigma = statistics.mean(e["uncertainty"] for e in ts)
        krig_nodes.append(
            {
                "id": n["id"],
                "name": n["location_name"],
                "lat": n["lat"],
                "lon": n["lon"],
                "sim": n["is_simulated"],
                "mu": round(mu, 2),
                "sigma": round(sigma, 2),
            }
        )

    # Learned edges that feed the simulated (kriging-target) nodes — these show
    # the graph context each out-of-sample node is reconstructed from.
    sim_ids = {n["id"] for n in nodes if n["is_simulated"]}
    krig_edges = [
        {"s": s, "t": t, "w": round(w, 4)}
        for (s, t), w in learned_und.items()
        if s in sim_ids or t in sim_ids
    ]

    kriging = {
        "meta": {
            "run_id": run_id,
            "date": SNAPSHOT_DATE,
            "num_real": spatial["metadata"]["num_real_nodes"],
            "num_sim": spatial["metadata"]["num_simulated_nodes"],
            "val_rmse": run_meta["metrics_summary"].get("val_rmse"),
            "val_mae": run_meta["metrics_summary"].get("val_mae"),
        },
        "nodes": krig_nodes,
        "kriging_edges": krig_edges,
    }

    # ── PIT diagrams: fetch validation metrics and PIT histograms ──
    pit_data = {}
    for label, run_id in PIT_RUNS.items():
        try:
            run_meta = next(
                (r for r in registry["runs"] if r["run_id"] == run_id), None
            )
            if not run_meta:
                print(f"warning: {run_id} not found in registry, skipping PIT for {label}")
                continue

            # Fetch PIT histogram from a representative day (same as topology/kriging)
            page = fetch_json(f"{R2_BASE}/{run_id}/{SNAPSHOT_DATE}/page.json")
            pit_histogram = page.get("pit_histogram", [])

            # Convert to density array matching the bin centers
            pit_densities = [bin_data["density"] for bin_data in pit_histogram]

            metrics = run_meta.get("metrics_summary", {})
            pit_data[label] = {
                "run_id": run_id,
                "wandb_run_id": run_meta.get("wandb_run_id"),
                "val_mae": metrics.get("val_mae"),
                "val_rmse": metrics.get("val_rmse"),
                "val_zero_acc": metrics.get("val_zero_acc"),
                "pit_histogram": pit_densities,
            }
            print(f"  {label}: {len(pit_densities)} PIT bins")
        except Exception as e:
            print(f"error fetching PIT data for {label} ({run_id}): {e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "topology.json").write_text(json.dumps(topology, separators=(",", ":")))
    (OUT_DIR / "kriging.json").write_text(json.dumps(kriging, separators=(",", ":")))
    (OUT_DIR / "pit.json").write_text(json.dumps(pit_data, separators=(",", ":")))

    print(f"topology.json: {topology['stats']}")
    print(
        f"kriging.json: {len(krig_nodes)} nodes, "
        f"{len(krig_edges)} kriging edges, date {SNAPSHOT_DATE}"
    )
    print(f"pit.json: {len(pit_data)} runs")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
