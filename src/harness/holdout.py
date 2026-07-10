"""
Fixed inductive holdout + run manifests.

The holdout manifest persists the *resolved* held-out location IDs (not just the
seed — node ordering could drift between runs), co-located with the checkpoint.
It is the bridge between training and inference: exporters load it to blank the
same nodes and tag them node_role="holdout", and cross-model comparison requires
identical manifest hashes.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional

import torch

from dataset import make_spatial_holdout_mask

HOLDOUT_MANIFEST_FILENAME = "holdout_manifest.json"
RESULTS_MANIFEST_FILENAME = "results.json"


def make_fixed_holdout_mask(
    num_nodes: int,
    holdout_frac: float,
    seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Deterministic inductive holdout mask. Seed is mandatory: an unseeded
    inductive split makes runs incomparable across models."""
    assert seed is not None, "CRITICAL: inductive holdout requires an explicit seed"
    mask = make_spatial_holdout_mask(num_nodes, holdout_frac, device=device, seed=seed)
    assert 0 < int(mask.sum()) < num_nodes, (
        f"CRITICAL: inductive holdout selected {int(mask.sum())}/{num_nodes} nodes; "
        f"need a non-trivial split (adjust holdout_frac/seed)"
    )
    return mask


def save_holdout_manifest(
    checkpoint_dir: str,
    location_ids: List[str],
    holdout_mask: torch.Tensor,
    seed: int,
    holdout_frac: float,
) -> str:
    assert len(location_ids) == holdout_mask.shape[0], (
        f"CRITICAL: {len(location_ids)} location_ids vs mask of {holdout_mask.shape[0]}"
    )
    manifest = {
        "seed": seed,
        "holdout_frac": holdout_frac,
        "num_nodes": len(location_ids),
        "holdout_location_ids": [
            loc for loc, held in zip(location_ids, holdout_mask.tolist()) if held
        ],
    }
    path = os.path.join(checkpoint_dir, HOLDOUT_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_holdout_manifest(checkpoint_dir: str) -> Optional[Dict]:
    """Returns the manifest dict, or None if the run had no inductive holdout."""
    path = os.path.join(checkpoint_dir, HOLDOUT_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def mask_from_manifest(
    manifest: Dict,
    location_ids: List[str],
    device: str = "cpu",
) -> torch.Tensor:
    """Rebuild the boolean holdout mask against the current node ordering.
    Gracefully handles manifest locations that are absent in the current inference window."""
    holdout = set(manifest["holdout_location_ids"])
    present_holdout = holdout & set(location_ids)
    missing = holdout - present_holdout
    if missing:
        import logging
        logging.getLogger("harness.holdout").warning(
            f"Graceful holdout: {len(missing)} manifest nodes absent from evaluation window: {sorted(missing)}"
        )
    mask = torch.tensor([loc in present_holdout for loc in location_ids], dtype=torch.bool, device=device)
    assert int(mask.sum()) == len(present_holdout)
    return mask


def manifest_sha(manifest: Optional[Dict]) -> Optional[str]:
    """Stable hash of the evaluation conditions. Identical sha across two runs
    means their metric rows are directly comparable."""
    if manifest is None:
        return None
    canonical = json.dumps(
        {
            "num_nodes": manifest["num_nodes"],
            "holdout_location_ids": sorted(manifest["holdout_location_ids"]),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def save_results_manifest(checkpoint_dir: str, payload: Dict) -> str:
    """Standardized cross-model results row. null means 'model cannot produce
    this metric' — never zero."""
    path = os.path.join(checkpoint_dir, RESULTS_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
