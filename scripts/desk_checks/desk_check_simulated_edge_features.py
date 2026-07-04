#!/usr/bin/env python3
"""
scripts/desk_check_simulated_edge_features.py

Hand-verifiable desk check for the dual-cache (real + simulated) merge in
DynamicGraphLearner. Uses 3 real sensors (R1, R2, R3) and 2 simulated nodes
(Simulated_Node_0, Simulated_Node_1) with manually chosen feature values so every
assertion is numerically verifiable without OSMnx.

Run inside the container:
    docker compose exec cpu-runner python scripts/desk_check_simulated_edge_features.py
"""
import json
import os
import sys
import tempfile

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from graph_learner import DynamicGraphLearner, _ROAD_KEYS

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _pair(sp, eucl, directness, roads):
    """Build a feature dict. `roads` is a 6-tuple summing to 1.0."""
    d = {"shortest_path_distance": sp, "euclidean_distance": eucl, "directness": directness}
    for k, v in zip(_ROAD_KEYS, roads):
        d[k] = v
    return d


# ---------------------------------------------------------------------------
# Hand-chosen cache values (all road fractions sum exactly to 1.0).
# ---------------------------------------------------------------------------

REAL_ROADS = (0.5, 0.3, 0.1, 0.1, 0.0, 0.0)  # sums to 1.0
SIM_ROADS  = (0.2, 0.4, 0.0, 0.2, 0.2, 0.0)  # sums to 1.0

# Real cache: R↔R pairs only.  Max sp = 8000.0 m.
REAL_CACHE = {
    "R1": {
        "R2": _pair(5000.0, 4500.0, 0.9, REAL_ROADS),
        "R3": _pair(8000.0, 7200.0, 0.9, REAL_ROADS),
    },
    "R2": {
        "R1": _pair(5000.0, 4500.0, 0.9, REAL_ROADS),
        "R3": _pair(3000.0, 2700.0, 0.9, REAL_ROADS),
    },
    "R3": {
        "R1": _pair(8000.0, 7200.0, 0.9, REAL_ROADS),
        "R2": _pair(3000.0, 2700.0, 0.9, REAL_ROADS),
    },
}

# Simulated cache: all pairs touching Simulated_Node_0 or Simulated_Node_1.
# Deliberately set max sp = 12000.0 m (> real-cache max) to verify max_distance update.
SIM_CACHE = {
    "R1": {
        "Simulated_Node_0": _pair(6000.0, 5400.0, 0.9, SIM_ROADS),
        "Simulated_Node_1": _pair(7000.0, 6300.0, 0.9, SIM_ROADS),
    },
    "R2": {
        "Simulated_Node_0": _pair(4000.0, 3600.0, 0.9, SIM_ROADS),
        "Simulated_Node_1": _pair(5500.0, 4950.0, 0.9, SIM_ROADS),
    },
    "R3": {
        "Simulated_Node_0": _pair(2000.0, 1800.0, 0.9, SIM_ROADS),
        "Simulated_Node_1": _pair(3500.0, 3150.0, 0.9, SIM_ROADS),
    },
    "Simulated_Node_0": {
        "R1": _pair(6000.0, 5400.0, 0.9, SIM_ROADS),
        "R2": _pair(4000.0, 3600.0, 0.9, SIM_ROADS),
        "R3": _pair(2000.0, 1800.0, 0.9, SIM_ROADS),
        "Simulated_Node_1": _pair(12000.0, 10800.0, 0.9, SIM_ROADS),  # global max
    },
    "Simulated_Node_1": {
        "R1": _pair(7000.0, 6300.0, 0.9, SIM_ROADS),
        "R2": _pair(5500.0, 4950.0, 0.9, SIM_ROADS),
        "R3": _pair(3500.0, 3150.0, 0.9, SIM_ROADS),
        "Simulated_Node_0": _pair(12000.0, 10800.0, 0.9, SIM_ROADS),
    },
}

EXPECTED_MAX_DISTANCE = 12000.0
LOCATION_IDS = ["R1", "R2", "R3", "Simulated_Node_0", "Simulated_Node_1"]


def check(label: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    msg = f"  [{status}] {label}"
    if detail and not ok:
        msg += f"\n         {detail}"
    print(msg)
    return ok


def main():
    torch.manual_seed(42)

    # Write caches to temp files.
    rf = tempfile.NamedTemporaryFile(mode='w', suffix='_real.json', delete=False)
    json.dump(REAL_CACHE, rf); rf.close()
    sf = tempfile.NamedTemporaryFile(mode='w', suffix='_sim.json', delete=False)
    json.dump(SIM_CACHE, sf); sf.close()

    all_passed = True
    try:
        learner = DynamicGraphLearner(
            static_feat_dim=4, embedding_dim=8, sparsity_threshold=0.0,
            edge_features_cache_path=rf.name,
            simulated_edge_features_cache_path=sf.name,
        )

        static = torch.ones(len(LOCATION_IDS), 4)
        edge_index, edge_weight, edge_attr = learner(static, location_ids=LOCATION_IDS)
        E = edge_index.shape[1]

        print("\n=== Desk Check: Simulated Node Edge Features ===\n")

        # 1. edge_attr shape
        ok = edge_attr.shape == (E, 10)
        all_passed &= check(f"edge_attr.shape == ({E}, 10)", ok, str(edge_attr.shape))

        # 2. No NaNs
        ok = not torch.isnan(edge_attr).any()
        all_passed &= check("No NaNs in edge_attr", ok)

        # 3. Road fractions (cols 3:9) sum to ~1.0 (connected) or ~0.0 (disconnected)
        road_sums = edge_attr[:, 3:9].sum(dim=1)
        connected = (road_sums - 1.0).abs() < 1e-3
        disconnected = road_sums.abs() < 1e-3
        ok = bool(torch.all(connected | disconnected))
        offenders = road_sums[~(connected | disconnected)]
        all_passed &= check(
            "Road fractions sum ∈ {~1.0, ~0.0} for every edge",
            ok, f"offending sums={offenders}"
        )

        # 4. max_distance = global max across merged cache (12000.0 from sim cache)
        ok = abs(learner.max_distance - EXPECTED_MAX_DISTANCE) < 1e-6
        all_passed &= check(
            f"max_distance == {EXPECTED_MAX_DISTANCE}",
            ok, f"got {learner.max_distance}"
        )

        # 5-7. Per-pair feature verification: build a lookup from edge_index.
        edge_lookup = {}
        for e in range(E):
            src_idx = edge_index[0, e].item()
            dst_idx = edge_index[1, e].item()
            edge_lookup[(src_idx, dst_idx)] = edge_attr[e]

        def verify_pair(src_name, dst_name, ref_pair, label):
            src_idx = LOCATION_IDS.index(src_name)
            dst_idx = LOCATION_IDS.index(dst_name)
            if (src_idx, dst_idx) not in edge_lookup:
                return check(label, False, f"Edge ({src_name}→{dst_name}) absent from edge_index (sparsity filtered it out)")
            ea = edge_lookup[(src_idx, dst_idx)]
            expected_dist_norm = ref_pair['shortest_path_distance'] / EXPECTED_MAX_DISTANCE
            ok_dist = abs(ea[0].item() - expected_dist_norm) < 1e-5
            expected_eucl_norm = ref_pair['euclidean_distance'] / EXPECTED_MAX_DISTANCE
            ok_eucl = abs(ea[1].item() - expected_eucl_norm) < 1e-5
            roads_ok = all(
                abs(ea[3 + k].item() - ref_pair[rk]) < 1e-5
                for k, rk in enumerate(_ROAD_KEYS)
            )
            ok = ok_dist and ok_eucl and roads_ok
            detail = (f"sp_norm={ea[0].item():.6f} (exp {expected_dist_norm:.6f}), "
                      f"eucl_norm={ea[1].item():.6f} (exp {expected_eucl_norm:.6f})")
            return check(label, ok, detail)

        all_passed &= verify_pair("R1", "Simulated_Node_0", SIM_CACHE["R1"]["Simulated_Node_0"],
                                  "Real→Sim features match sim cache (R1→Simulated_Node_0)")
        all_passed &= verify_pair("Simulated_Node_0", "R2", SIM_CACHE["Simulated_Node_0"]["R2"],
                                  "Sim→Real features match sim cache (Simulated_Node_0→R2)")
        all_passed &= verify_pair("Simulated_Node_0", "Simulated_Node_1",
                                  SIM_CACHE["Simulated_Node_0"]["Simulated_Node_1"],
                                  "Sim→Sim features match sim cache (Simulated_Node_0→Simulated_Node_1)")
        all_passed &= verify_pair("R1", "R2", REAL_CACHE["R1"]["R2"],
                                  "Real→Real features match real cache (R1→R2)")

        # 8. 10th column = learned edge_weight
        ok = bool(torch.allclose(edge_attr[:, 9], edge_weight))
        all_passed &= check("edge_attr[:, 9] == edge_weight (learned similarity)", ok)

        print()
        if all_passed:
            print("=== ALL CHECKS PASSED ===\n")
        else:
            print("=== SOME CHECKS FAILED — see above ===\n")
            sys.exit(1)

    finally:
        os.unlink(rf.name)
        os.unlink(sf.name)


if __name__ == '__main__':
    main()
