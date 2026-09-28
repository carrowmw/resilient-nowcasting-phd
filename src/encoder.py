import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple, Union


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the temporal sequence dimension (F-C08 remediation).
    Maps positions t in [0, max_len) to harmonic embeddings of dimension hidden_dim.
    Input layout: [Seq_len, Batch, Dim].
    """
    def __init__(self, hidden_dim: int, max_len: int = 100):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register buffer so it moves across devices with the module
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Seq_len, Batch, Dim]
        Returns:
            Tensor of shape [Seq_len, Batch, Dim] with positional encoding added.
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len].unsqueeze(1)


class MaskedSpatiotemporalEncoder(nn.Module):
    """
    MaskedSpatiotemporalEncoder: Processes dynamic IoT data using GATv2 and Transformers.
    
    Architecture:
    1. Spatial: GATv2Conv layers applied independently across each time step.
    2. Temporal: Sinusoidal Positional Encoding + TransformerEncoder across the sequence.
    """
    def __init__(self, 
                 num_nodes: int, 
                 seq_len: int, 
                 time_series_dim: int, 
                 hidden_dim: int = 64, 
                 num_heads: int = 4, 
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 dim_feedforward: int = 256,
                 edge_dim: int = 1,
                 mask_dim: int = 2,
                 use_positional_encoding: bool = True):
        super(MaskedSpatiotemporalEncoder, self).__init__()

        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.time_series_dim = time_series_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding
        # PHASE 3: edge_dim=1 -> learned similarity only; edge_dim=10 -> enriched
        # physical + learned edge features from the cache.
        self.edge_dim = edge_dim

        # 1. Spatial Processing: GATv2Conv
        # We use a single layer for simplicity as per common IoT patterns,
        # or can be extended to multiple.
        self.gat = GATv2Conv(
            in_channels=time_series_dim + mask_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim  # PHASE 3: 1 (learned-only) or 10 (enriched edge_attr)
        )

        # 2. Temporal Processing: Positional Encoding + Transformer
        if use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=max(seq_len * 2, 64))
        else:
            self.pos_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # Standard Transformer [Seq, Batch, Dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        dynamic_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            dynamic_features: [batch_size, num_nodes, seq_len, time_series_dim]
            edge_index: [2, E]
            edge_attr: [E] (1-D learned similarity) or [E, edge_dim] (enriched).
                A 1-D tensor is unsqueezed to [E, 1] for backward compatibility.
            return_attention_weights: If True, also return the GAT attention
                coefficients alpha [batched_E, num_heads] for downstream analysis.
        Returns:
            latent_embeddings: [batch_size, num_nodes, seq_len, hidden_dim]
            alpha (optional): [batched_E, num_heads] — only when return_attention_weights=True.
        """
        # PHASE 3: accept either a 1-D edge_weight [E] or a 2-D edge_attr [E, D].
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        assert edge_attr.dim() == 2, f"Expected edge_attr [E, D], got {edge_attr.shape}"
        assert edge_attr.shape[1] == self.edge_dim, \
            f"edge_attr dim {edge_attr.shape[1]} != encoder edge_dim {self.edge_dim}"
        # --- Defensive Data Handling (Rule 2) ---
        assert dynamic_features.dim() == 4, f"Expected 4D features, got {dynamic_features.shape}"
        batch_size, num_nodes, seq_len, ts_dim = dynamic_features.shape
        assert num_nodes == self.num_nodes, f"Node mismatch: {num_nodes} != {self.num_nodes}"
        assert seq_len == self.seq_len, f"Seq mismatch: {seq_len} != {self.seq_len}"
        assert ts_dim == self.time_series_dim + self.mask_dim, f"Feature mismatch: {ts_dim} != {self.time_series_dim + self.mask_dim}"

        # --- Spatial Processing ---
        # Reshape to [B, T, N, D] then to [B*T*N, D]
        # We want to apply GAT to each (batch, time_step) independently.
        x = dynamic_features.permute(0, 2, 1, 3).contiguous() # [B, T, N, D]
        x = x.view(batch_size * seq_len * num_nodes, ts_dim)

        # Construct batched edge_index
        # Original edge_index is [2, E] for num_nodes.
        # We need to repeat it batch_size * seq_len times with offsets.
        num_graphs = batch_size * seq_len
        E = edge_index.shape[1]
        
        # [2, E * num_graphs]
        batched_edge_index = edge_index.repeat(1, num_graphs)
        offsets = torch.arange(num_graphs, device=edge_index.device).view(-1, 1) * num_nodes
        # offsets shape: [num_graphs, 1]. Repeat for each edge in the graph.
        offsets = offsets.repeat(1, E).view(-1) # [num_graphs * E]
        batched_edge_index = batched_edge_index + offsets

        # edge_attr also needs repeating: [E, D] -> [E * num_graphs, D]
        batched_edge_attr = edge_attr.repeat(num_graphs, 1)
        assert batched_edge_attr.shape[0] == batched_edge_index.shape[1], (
            f"batched edge_attr rows {batched_edge_attr.shape[0]} != "
            f"batched edges {batched_edge_index.shape[1]}"
        )

        # Apply GAT — optionally collect attention coefficients for analysis.
        if return_attention_weights:
            # PyG returns (out, (edge_index_used, alpha)) when return_attention_weights=True
            x_spatial, (_, alpha) = self.gat(
                x, batched_edge_index,
                edge_attr=batched_edge_attr,
                return_attention_weights=True,
            )
        else:
            x_spatial = self.gat(x, batched_edge_index, edge_attr=batched_edge_attr)
            alpha = None

        # --- Temporal Processing ---
        # Reshape back to [B, T, N, H]
        x_temporal = x_spatial.view(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        # Transformer expects [Seq, Batch, Dim]
        # We treat each node in each batch as an independent sequence.
        # Batch dimension for Transformer: B * N
        # Seq dimension: T
        x_temporal = x_temporal.permute(1, 0, 2, 3).contiguous() # [T, B, N, H]
        x_temporal = x_temporal.view(seq_len, batch_size * num_nodes, self.hidden_dim)

        # Inject harmonic positional encoding (F-C08 fix)
        if self.use_positional_encoding and self.pos_encoder is not None:
            x_temporal = self.pos_encoder(x_temporal)

        # Pass through Transformer
        # x_out: [T, B * N, H]
        x_out = self.transformer(x_temporal)

        # Reshape back to [B, N, T, H]
        x_out = x_out.view(seq_len, batch_size, num_nodes, self.hidden_dim)
        x_out = x_out.permute(1, 2, 0, 3).contiguous() # [B, N, T, H]

        # --- Tripwires (Rule 5) ---
        if torch.isnan(x_out).any():
            raise ValueError("NaN detected in MaskedSpatiotemporalEncoder output!")

        if return_attention_weights:
            return x_out, alpha
        return x_out


