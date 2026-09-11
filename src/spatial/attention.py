"""
Attention extraction and quantitative analysis.

The legacy explainability work (docs/context/GAT/Analysis-2/4/5) inspected
attention PNGs qualitatively. Here attention claims must survive numbers:
per-node inbound entropy (uniform vs peaked), attention-vs-distance decay,
and — the credibility gate for any thesis interpretation — cross-seed
agreement of the edge ranking.
"""
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class AttentionRecord:
    """Mean GATv2 attention of the FINAL GAT layer over a window subsample."""
    edge_index: np.ndarray      # [2, E] (self-loops included)
    alpha_mean: np.ndarray      # [E, heads] — mean over sampled windows
    alpha_by_hour: np.ndarray   # [24, E] — head-averaged, per origin hour (NaN = hour unsampled)
    n_windows: int
    seed: int

    def __post_init__(self):
        E = self.edge_index.shape[1]
        assert self.alpha_mean.shape[0] == E, "CRITICAL: alpha/edge_index mismatch"
        assert self.alpha_by_hour.shape == (24, E)

    def alpha_flat(self) -> np.ndarray:
        """[E] head-averaged mean attention."""
        return self.alpha_mean.mean(axis=1)


def attention_entropy(record: AttentionRecord, n_nodes: int) -> pd.DataFrame:
    """
    Per-node inbound attention entropy, normalised by log(in-degree):
    1.0 = uniform over neighbours, 0.0 = all mass on one edge.
    Nodes with in-degree < 2 (self-loop only) are reported with NaN entropy.
    """
    alpha = record.alpha_flat()
    src, dst = record.edge_index
    rows = []
    for j in range(n_nodes):
        inbound = alpha[dst == j]
        k = len(inbound)
        if k < 2 or inbound.sum() <= 0:
            rows.append({"node_idx": j, "in_degree": k, "entropy_norm": np.nan})
            continue
        p = inbound / inbound.sum()
        ent = -np.sum(p * np.log(np.clip(p, 1e-12, None)))
        rows.append({"node_idx": j, "in_degree": k,
                     "entropy_norm": float(ent / np.log(k))})
    return pd.DataFrame(rows)


def attention_vs_distance(record: AttentionRecord, dist: np.ndarray,
                          node_ids: Sequence[str]) -> pd.DataFrame:
    """Edge-level table: distance (m) vs mean attention (non-loop edges)."""
    alpha = record.alpha_flat()
    src, dst = record.edge_index
    keep = src != dst
    return pd.DataFrame({
        "src": [node_ids[int(s)] for s in src[keep]],
        "dst": [node_ids[int(d)] for d in dst[keep]],
        "distance_m": dist[src[keep], dst[keep]],
        "alpha": alpha[keep],
    })


def attention_seed_agreement(records: List[AttentionRecord]) -> dict:
    """
    Pairwise Spearman rho of the edge-attention ranking across seeds. Low
    agreement means "which edges matter" is seed noise and must not be
    interpreted in the thesis.
    """
    assert len(records) >= 2, "CRITICAL: seed agreement needs >= 2 records"
    e0 = records[0].edge_index
    for r in records[1:]:
        assert r.edge_index.shape == e0.shape and (r.edge_index == e0).all(), \
            "CRITICAL: attention records disagree on the edge set"
    rhos = []
    for a in range(len(records)):
        for b in range(a + 1, len(records)):
            rho, _ = sps.spearmanr(records[a].alpha_flat(), records[b].alpha_flat())
            rhos.append(float(rho))
    return {"n_seeds": len(records), "pairwise_spearman": rhos,
            "mean_spearman": float(np.mean(rhos)), "min_spearman": float(np.min(rhos))}


def top_edges_table(record: AttentionRecord, node_ids: Sequence[str],
                    k: int = 20) -> pd.DataFrame:
    """Top-k non-loop edges by mean attention."""
    alpha = record.alpha_flat()
    src, dst = record.edge_index
    keep = np.where(src != dst)[0]
    order = keep[np.argsort(alpha[keep])[::-1][:k]]
    return pd.DataFrame({
        "src": [node_ids[int(s)] for s in src[order]],
        "dst": [node_ids[int(d)] for d in dst[order]],
        "alpha": alpha[order],
    })
