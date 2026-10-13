"""
The GAT forecaster family.

Architecture (one configurable class, so every ablation is a config variant):
  1. Temporal encode per node  — shared encoder classes from encoders.py,
     so 'gat_lstm vs per-node lstm' differs ONLY in message passing.
  2. Spatial mixing            — gat_layers × GATv2Conv (real depth axis;
     the legacy src/encoder.py had a single GAT layer regardless of config),
     ELU + residual + dropout, applied per window on the batched
     disjoint-union graph (offset arithmetic, the proven src/encoder.py
     pattern). One graph pass per window — not per timestep — because
     temporal encoding happens first (deliberate, ~T× cheaper).
  3. MLP decoder               — [F (+15 static)] → H horizons.

LearnedTopologyGATForecaster swaps the static GraphSpec for an internal
DynamicGraphLearner (imported from src/graph_learner.py) over the 15-dim
static features; the resolved topology is persisted via `resolved_graph()`.
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from graph_learner import DynamicGraphLearner
from spatial.attention import AttentionRecord
from spatial.contract import GraphSpec, GraphSplit, SpatialCapabilities
from spatial.models.base import TorchGraphForecaster
from spatial.models.encoders import build_temporal_encoder


def _batch_graph(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor],
                 batch_size: int, n_nodes: int):
    """Disjoint-union replication of one graph across a batch."""
    E = edge_index.shape[1]
    offsets = (torch.arange(batch_size, device=edge_index.device) * n_nodes)
    ei = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)      # [B, 2, E]
    ei = ei.permute(1, 0, 2).reshape(2, batch_size * E)
    ea = edge_attr.repeat(batch_size, 1) if edge_attr is not None else None
    return ei, ea


class _GATStack(nn.Module):
    """gat_layers × GATv2Conv with ELU + residual + dropout."""

    def __init__(self, hidden_dim: int, gat_layers: int, heads: int,
                 edge_dim: Optional[int], dropout: float):
        super().__init__()
        assert gat_layers >= 1, "CRITICAL: gat_layers must be >= 1"
        assert hidden_dim % heads == 0, \
            f"CRITICAL: hidden_dim {hidden_dim} not divisible by heads {heads}"
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                      dropout=dropout, edge_dim=edge_dim, add_self_loops=False)
            for _ in range(gat_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr, return_attention: bool = False):
        alpha = None
        for i, conv in enumerate(self.convs):
            is_last = i == len(self.convs) - 1
            if return_attention and is_last:
                out, (_, alpha) = conv(h, edge_index, edge_attr,
                                       return_attention_weights=True)
            else:
                out = conv(h, edge_index, edge_attr)
            h = self.dropout(F.elu(out)) + h                  # residual
        return h, alpha


class _GATForecastNet(nn.Module):
    def __init__(self, in_dim: int, window: int, n_horizons: int, spec: Dict,
                 edge_dim: Optional[int]):
        super().__init__()
        hidden = int(spec["hidden_dim"])
        self.encoder = build_temporal_encoder(
            spec["temporal_encoder"], in_dim, window, hidden, spec)
        self.gat = _GATStack(hidden, int(spec["gat_layers"]), int(spec["heads"]),
                             edge_dim, float(spec.get("dropout", 0.1)))
        self.use_static = bool(spec["use_static_features"])
        dec_in = hidden + (15 if self.use_static else 0)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden), nn.GELU(), nn.Linear(hidden, n_horizons))

    def _spatial(self, x, edge_index, edge_attr, return_attention):
        B, N, T, D = x.shape
        h = self.encoder(x.reshape(B * N, T, D))              # [B*N, F]
        ei, ea = _batch_graph(edge_index, edge_attr, B, N)
        h, alpha = self.gat(h, ei, ea, return_attention=return_attention)
        if alpha is not None:
            alpha = alpha.reshape(B, -1, alpha.shape[-1])     # [B, E, heads]
        return h.reshape(B, N, -1), alpha

    def _decode(self, h, static):
        if self.use_static:
            B = h.shape[0]
            h = torch.cat([h, static.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        return self.decoder(h)                                 # [B, N, H]

    def forward(self, x, edge_index, edge_attr, static):
        h, _ = self._spatial(x, edge_index, edge_attr, return_attention=False)
        return self._decode(h, static)

    def forward_attention(self, x, edge_index, edge_attr, static):
        """(pred, edge_index_used, alpha [B, E, heads]) — final GAT layer."""
        h, alpha = self._spatial(x, edge_index, edge_attr, return_attention=True)
        assert alpha is not None
        return self._decode(h, static), edge_index, alpha


class GATForecaster(TorchGraphForecaster):
    """Static-topology GATv2 forecaster (type: gat)."""

    name = "gat"

    def __init__(self, spec: Dict, seed: int, device: Optional[str], inputs_cfg: Dict):
        super().__init__(spec, seed, device, inputs_cfg)
        self.capabilities = SpatialCapabilities(
            trainable=True, uses_graph=True, emits_attention=True,
            uses_edge_attr=True,   # resolved per graph at fit time
        )

    def _build_module(self, in_dim, n_nodes, window, n_horizons, graph: GraphSpec):
        edge_dim = graph.edge_dim if graph.edge_attr is not None else None
        return _GATForecastNet(in_dim, window, n_horizons, self.spec, edge_dim)

    def predict_attention(self, split: GraphSplit, graph: GraphSpec,
                          max_windows: int = 256) -> AttentionRecord:
        assert self.module is not None, \
            f"CRITICAL: {self.name}.predict_attention() before fit()"
        ctx = split.context
        rng = np.random.default_rng(self.seed)
        n = len(split)
        take = rng.permutation(n)[: min(max_windows, n)]
        take.sort()

        from spatial.dataset import GraphWindowDataset, collate_graph
        ds = GraphWindowDataset(split, mask_channel=self.mask_channel,
                                time_features=self.time_features)
        static = torch.from_numpy(ctx.static_features).float().to(self.device)
        ei, ea = self._graph_tensors(graph)
        hours = split.origin_dt.hour.to_numpy()[take]

        self.module.eval()
        alpha_sum = alpha_cnt = None
        hour_sum = hour_cnt = None
        edge_index_used = None
        with torch.no_grad():
            for lo in range(0, len(take), self.batch_size):
                idx = take[lo: lo + self.batch_size]
                batch = collate_graph([ds[int(i)] for i in idx])
                _, ei_used, alpha = self.module.forward_attention(
                    batch["x"].to(self.device), ei, ea, static)
                a = alpha.cpu().numpy()                        # [b, E, heads]
                if alpha_sum is None:
                    E, heads = a.shape[1], a.shape[2]
                    alpha_sum = np.zeros((E, heads))
                    alpha_cnt = 0
                    hour_sum = np.zeros((24, E))
                    hour_cnt = np.zeros(24)
                    edge_index_used = ei_used.cpu().numpy()
                alpha_sum += a.sum(axis=0)
                alpha_cnt += a.shape[0]
                for b, h in enumerate(hours[lo: lo + len(idx)]):
                    hour_sum[h] += a[b].mean(axis=1)
                    hour_cnt[h] += 1
        assert alpha_cnt and alpha_cnt > 0
        with np.errstate(invalid="ignore"):
            by_hour = hour_sum / hour_cnt[:, None]             # NaN where unsampled
        return AttentionRecord(
            edge_index=edge_index_used,
            alpha_mean=(alpha_sum / alpha_cnt).astype(np.float32),
            alpha_by_hour=by_hour.astype(np.float32),
            n_windows=int(alpha_cnt), seed=self.seed,
        )


class _LearnedTopologyNet(_GATForecastNet):
    """GAT net whose topology comes from an internal DynamicGraphLearner."""

    def __init__(self, in_dim, window, n_horizons, spec, edge_dim, location_ids,
                 edge_features_cache_path):
        super().__init__(in_dim, window, n_horizons, spec, edge_dim)
        self.location_ids = list(location_ids)
        self.learner = DynamicGraphLearner(
            static_feat_dim=15,
            embedding_dim=int(spec["embedding_dim"]),
            sparsity_threshold=float(spec["sparsity_threshold"]),
            edge_features_cache_path=edge_features_cache_path,
        )

    def _learned_graph(self, static):
        edge_index, edge_weight, edge_attr = self.learner(static, self.location_ids)
        assert edge_index.shape[1] > 0, (
            "CRITICAL: DynamicGraphLearner produced zero edges — lower "
            "sparsity_threshold"
        )
        return edge_index, edge_attr

    def forward(self, x, edge_index, edge_attr, static):
        ei, ea = self._learned_graph(static)
        h, _ = self._spatial(x, ei, ea, return_attention=False)
        return self._decode(h, static)

    def forward_attention(self, x, edge_index, edge_attr, static):
        ei, ea = self._learned_graph(static)
        h, alpha = self._spatial(x, ei, ea, return_attention=True)
        assert alpha is not None
        return self._decode(h, static), ei, alpha


class LearnedTopologyGATForecaster(GATForecaster):
    """type: gat_learned — learns its own topology; receives (and asserts)
    the identity GraphSpec to keep the runner contract uniform."""

    name = "gat_learned"

    def __init__(self, spec: Dict, seed: int, device: Optional[str], inputs_cfg: Dict):
        super().__init__(spec, seed, device, inputs_cfg)
        mode = spec["edge_attr_mode"]
        assert mode in ("learned_only", "physical_plus_learned"), \
            f"CRITICAL: unknown edge_attr_mode {mode!r}"
        self.edge_attr_mode = mode
        self.capabilities = SpatialCapabilities(
            trainable=True, uses_graph=True, uses_edge_attr=True,
            learns_topology=True, emits_attention=True,
        )
        self._node_ids = None

    def fit(self, train, val, graph: GraphSpec):
        assert graph.method == "identity", (
            "CRITICAL: gat_learned ignores static topology and must be paired "
            f"with the identity GraphSpec, got {graph.method!r}"
        )
        self._node_ids = list(train.context.node_ids)
        return super().fit(train, val, graph)

    def _build_module(self, in_dim, n_nodes, window, n_horizons, graph: GraphSpec):
        cache = self.spec.get("edge_features_cache")
        if self.edge_attr_mode == "physical_plus_learned":
            assert cache, ("CRITICAL: edge_attr_mode=physical_plus_learned needs "
                           "models.<key>.edge_features_cache")
            edge_dim = 10
        else:
            cache = None
            edge_dim = 1
        return _LearnedTopologyNet(in_dim, window, n_horizons, self.spec,
                                   edge_dim, self._node_ids, cache)

    def resolved_graph(self, context) -> dict:
        """The learned edge set at eval time (for the job manifest)."""
        assert self.module is not None, "CRITICAL: resolved_graph before fit()"
        static = torch.from_numpy(context.static_features).float().to(self.device)
        self.module.eval()
        with torch.no_grad():
            ei, _ = self.module._learned_graph(static)
        ei = ei.cpu().numpy()
        return {
            "n_edges": int(ei.shape[1]),
            "edges": [[context.node_ids[int(s)], context.node_ids[int(d)]]
                      for s, d in ei.T],
        }
