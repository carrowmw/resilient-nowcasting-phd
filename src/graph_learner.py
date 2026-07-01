import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
import logging

logger = logging.getLogger(__name__)

# PHASE 3: physical edge feature column layout for the [E, 10] edge_attr tensor.
# Cols 0-8 are static physical features from the OSM cache; col 9 is the learned
# similarity (edge_weight) supplied per forward pass.
#   0: shortest_path_distance  (normalized by global max_distance -> [0, 1])
#   1: euclidean_distance      (normalized by global max_distance -> [0, 1])
#   2: directness              (already [0, 1])
#   3: pedestrian fraction
#   4: residential fraction
#   5: tertiary fraction
#   6: secondary fraction
#   7: primary fraction
#   8: service fraction
#   9: learned similarity (edge_weight)
_ROAD_KEYS = ['pedestrian', 'residential', 'tertiary', 'secondary', 'primary', 'service']
_NUM_PHYSICAL = 9   # 2 normalized distances + directness + 6 road fractions
_EDGE_ATTR_DIM = 10  # physical + learned similarity


class DynamicGraphLearner(nn.Module):
    """
    DynamicGraphLearner: Infers graph topology from static node features.

    Architecture:
    1. Linear projection to latent space.
    2. Pairwise dot-product similarity to form adjacency matrix.
    3. ReLU and tunable sparsity thresholding.
    4. Sparse conversion to (edge_index, edge_weight).

    PHASE 3 (Layer 3): optionally enriches each edge with 9 physical features loaded
    from data/external/edge_features_cache.json + the learned similarity, producing a
    [E, 10] edge_attr tensor consumed by the GATv2 encoder.
    """
    def __init__(self,
                 static_feat_dim: int,
                 embedding_dim: int = 32,
                 sparsity_threshold: float = 0.5,
                 edge_features_cache_path: Optional[str] = None,
                 simulated_edge_features_cache_path: Optional[str] = None):
        super(DynamicGraphLearner, self).__init__()
        # Ensure random seeds are handled globally or at the top of the script

        self.static_feat_dim = static_feat_dim
        self.embedding_dim = embedding_dim

        # Parameterized projection layer
        self.lin = nn.Linear(static_feat_dim, embedding_dim)

        # Tunable sparsity threshold (can be a learned parameter if needed, but here a buffer/float)
        self.register_buffer('sparsity_threshold', torch.tensor(sparsity_threshold))

        # PHASE 3: load physical edge features cache (fail-fast on missing file).
        self.edge_features_cache = None
        self.max_distance = None

        if simulated_edge_features_cache_path is not None and edge_features_cache_path is None:
            raise ValueError(
                "CRITICAL: simulated_edge_features_cache_path requires edge_features_cache_path to also be set"
            )

        if edge_features_cache_path is not None:
            cache_path = Path(edge_features_cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"CRITICAL: Edge features cache not found: {cache_path}"
                )
            with open(cache_path, 'r') as f:
                self.edge_features_cache = json.load(f)

            if simulated_edge_features_cache_path is not None:
                sim_path = Path(simulated_edge_features_cache_path)
                if not sim_path.exists():
                    raise FileNotFoundError(
                        f"CRITICAL: Simulated edge features cache not found: {sim_path}"
                    )
                with open(sim_path, 'r') as f:
                    sim_cache = json.load(f)
                for src, dst_dict in sim_cache.items():
                    if src in self.edge_features_cache:
                        self.edge_features_cache[src].update(dst_dict)
                    else:
                        self.edge_features_cache[src] = dict(dst_dict)
                logger.info(
                    f"Merged simulated cache: {len(sim_cache)} additional sources from {sim_path}"
                )

            # Global max shortest_path_distance for normalizing BOTH distance channels.
            # Recomputed here so it covers the merged cache (real + simulated pairs).
            distances = [
                pair_data['shortest_path_distance']
                for pairs in self.edge_features_cache.values()
                for pair_data in pairs.values()
            ]
            self.max_distance = max(distances)
            assert self.max_distance > 0, \
                f"CRITICAL: max_distance must be positive, got {self.max_distance}"

            logger.info(
                f"Loaded edge features cache: {len(self.edge_features_cache)} sources, "
                f"max_distance={self.max_distance:.1f}m"
            )

        # Memoization slots for the dense [N, N, 9] physical feature tensor. Rebuilt
        # only when location_ids or device change (the physical features are static,
        # so this avoids reconstructing the tensor every forward pass).
        self._phys_dense = None            # torch.Tensor [N, N, 9] or None
        self._phys_location_ids = None     # tuple(location_ids) or None

    def _build_physical_dense(self, location_ids: List[str], device: torch.device) -> torch.Tensor:
        """
        Build the dense [N, N, 9] physical edge-feature tensor once for a given node
        ordering. Entry [i, j] holds the 9 physical features for the directed edge i->j.

        Diagonal entries (i == j) remain zero — self-pairs never survive sparsity and
        are absent from the cache.

        Fail-fast ValueError if any required (src, dst) pair is missing from the cache.
        """
        assert self.edge_features_cache is not None, \
            "CRITICAL: _build_physical_dense called without a loaded cache"
        N = len(location_ids)
        dense = np.zeros((N, N, _NUM_PHYSICAL), dtype=np.float32)

        for i, src_loc in enumerate(location_ids):
            if src_loc not in self.edge_features_cache:
                raise ValueError(f"CRITICAL: Missing cache for source {src_loc}")
            src_pairs = self.edge_features_cache[src_loc]
            for j, dst_loc in enumerate(location_ids):
                if i == j:
                    continue
                if dst_loc not in src_pairs:
                    raise ValueError(
                        f"CRITICAL: Missing cache for edge {src_loc} -> {dst_loc}"
                    )
                pair = src_pairs[dst_loc]
                dense[i, j, 0] = pair['shortest_path_distance'] / self.max_distance
                dense[i, j, 1] = pair['euclidean_distance'] / self.max_distance
                dense[i, j, 2] = pair['directness']
                for k, road_key in enumerate(_ROAD_KEYS):
                    dense[i, j, 3 + k] = pair[road_key]

        tensor = torch.from_numpy(dense).to(device)
        logger.info(
            f"Built dense physical edge features: shape={tuple(tensor.shape)}, device={device}"
        )
        return tensor

    def _build_edge_attr(self,
                         edge_index: torch.Tensor,
                         edge_weight: torch.Tensor,
                         location_ids: Optional[List[str]]) -> torch.Tensor:
        """
        Construct the [E, 10] edge attribute tensor: 9 physical features (vectorized
        gather from the memoized dense tensor) + the learned similarity (edge_weight).

        Falls back to [E, 1] (learned similarity only) when no cache is loaded.
        """
        E = edge_index.shape[1]

        # Fallback: learned similarity only.
        if self.edge_features_cache is None or location_ids is None:
            return edge_weight.unsqueeze(1)

        device = edge_weight.device

        # (Re)build the dense physical tensor if the node ordering or device changed.
        loc_key = tuple(location_ids)
        if (self._phys_dense is None
                or self._phys_location_ids != loc_key
                or self._phys_dense.device != device):
            self._phys_dense = self._build_physical_dense(location_ids, device)
            self._phys_location_ids = loc_key

        assert self._phys_dense.shape[0] == len(location_ids), \
            f"CRITICAL: dense node count {self._phys_dense.shape[0]} != location_ids {len(location_ids)}"

        # Vectorized gather of the 9 physical features for every edge.
        phys = self._phys_dense[edge_index[0], edge_index[1]]  # [E, 9]
        learned = edge_weight.unsqueeze(1)                     # [E, 1]
        edge_attr = torch.cat([phys, learned], dim=1)          # [E, 10]

        # OFFENSIVE ASSERTIONS
        assert edge_attr.shape == (E, _EDGE_ATTR_DIM), \
            f"CRITICAL: Expected ({E}, {_EDGE_ATTR_DIM}), got {tuple(edge_attr.shape)}"
        assert not torch.isnan(edge_attr).any(), "CRITICAL: NaN in edge_attr"

        # Road fractions (cols 3:9) must sum to ~1.0 for connected pairs OR ~0.0 for
        # disconnected pairs (no OSM path — 38 such pairs exist in the real cache).
        road_sums = edge_attr[:, 3:9].sum(dim=1)
        connected = (road_sums - 1.0).abs() < 1e-3
        disconnected = road_sums.abs() < 1e-3
        assert torch.all(connected | disconnected), (
            "CRITICAL: Road fractions must sum to ~1.0 (connected) or ~0.0 (disconnected); "
            f"offending sums={road_sums[~(connected | disconnected)]}"
        )

        return edge_attr

    def forward(self,
                static_features: torch.Tensor,
                location_ids: Optional[List[str]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        Args:
            static_features: [num_nodes, static_feat_dim]
            location_ids: optional list of node location identifiers (length num_nodes),
                used to look up physical edge features from the cache.
        Returns:
            edge_index: [2, E]
            edge_weight: [E]
            edge_attr: [E, 10] (physical + learned) or [E, 1] (learned-only fallback)
        """
        assert static_features.dim() == 2, f"Expected static_features shape [num_nodes, static_feat_dim], got {static_features.shape}"
        assert static_features.shape[1] == self.static_feat_dim, f"Feature dimension mismatch: {static_features.shape[1]} != {self.static_feat_dim}"

        num_nodes = static_features.shape[0]

        # 1. Project to embedding space
        # [num_nodes, embedding_dim]
        embeddings = self.lin(static_features)

        # 2. Pairwise dot-product similarity
        # [num_nodes, num_nodes]
        adj = torch.matmul(embeddings, embeddings.t())

        # 3. Apply ReLU to ensure non-negative weights
        adj = F.relu(adj)

        # 4. Sparsity: Subtract threshold and apply ReLU
        # This drops connections with weights <= sparsity_threshold
        adj = adj - self.sparsity_threshold
        adj = F.relu(adj)

        # 5. Convert to sparse format
        edge_index, edge_weight = dense_to_sparse(adj)

        # 6. PHASE 3: build enriched edge attributes (or learned-only fallback).
        edge_attr = self._build_edge_attr(edge_index, edge_weight, location_ids)

        return edge_index, edge_weight, edge_attr
