import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional

class MaskedSpatiotemporalEncoder(nn.Module):
    """
    MaskedSpatiotemporalEncoder: Processes dynamic IoT data using GATv2 and Transformers.
    
    Architecture:
    1. Spatial: GATv2Conv layers applied independently across each time step.
    2. Temporal: TransformerEncoder to capture dependencies across the sequence.
    """
    def __init__(self, 
                 num_nodes: int, 
                 seq_len: int, 
                 time_series_dim: int, 
                 hidden_dim: int = 64, 
                 num_heads: int = 4, 
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(MaskedSpatiotemporalEncoder, self).__init__()
        
        # Reproducibility at initialization
        torch.manual_seed(42)
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.time_series_dim = time_series_dim
        self.hidden_dim = hidden_dim

        # 1. Spatial Processing: GATv2Conv
        # We use a single layer for simplicity as per common IoT patterns, 
        # or can be extended to multiple.
        self.gat = GATv2Conv(
            in_channels=time_series_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1 # For edge_weight
        )

        # 2. Temporal Processing: Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False # Standard Transformer [Seq, Batch, Dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, 
                dynamic_features: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dynamic_features: [batch_size, num_nodes, seq_len, time_series_dim]
            edge_index: [2, E]
            edge_weight: [E]
        Returns:
            latent_embeddings: [batch_size, num_nodes, seq_len, hidden_dim]
        """
        # --- Defensive Data Handling (Rule 2) ---
        assert dynamic_features.dim() == 4, f"Expected 4D features, got {dynamic_features.shape}"
        batch_size, num_nodes, seq_len, ts_dim = dynamic_features.shape
        assert num_nodes == self.num_nodes, f"Node mismatch: {num_nodes} != {self.num_nodes}"
        assert seq_len == self.seq_len, f"Seq mismatch: {seq_len} != {self.seq_len}"
        assert ts_dim == self.time_series_dim, f"Feature mismatch: {ts_dim} != {self.time_series_dim}"

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
        
        # edge_weight also needs repeating
        batched_edge_weight = edge_weight.repeat(num_graphs)

        # Apply GAT
        # x_spatial: [B * T * N, hidden_dim]
        x_spatial = self.gat(x, batched_edge_index, edge_attr=batched_edge_weight)

        # --- Temporal Processing ---
        # Reshape back to [B, T, N, H]
        x_temporal = x_spatial.view(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        # Transformer expects [Seq, Batch, Dim]
        # We treat each node in each batch as an independent sequence.
        # Batch dimension for Transformer: B * N
        # Seq dimension: T
        x_temporal = x_temporal.permute(1, 0, 2, 3).contiguous() # [T, B, N, H]
        x_temporal = x_temporal.view(seq_len, batch_size * num_nodes, self.hidden_dim)

        # Pass through Transformer
        # x_out: [T, B * N, H]
        x_out = self.transformer(x_temporal)

        # Reshape back to [B, N, T, H]
        x_out = x_out.view(seq_len, batch_size, num_nodes, self.hidden_dim)
        x_out = x_out.permute(1, 2, 0, 3).contiguous() # [B, N, T, H]

        # --- Tripwires (Rule 5) ---
        if torch.isnan(x_out).any():
            raise ValueError("NaN detected in MaskedSpatiotemporalEncoder output!")

        return x_out

if __name__ == "__main__":
    import numpy as np
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Starting MaskedSpatiotemporalEncoder Offensive Unit Test")

    # Parameters
    batch_size = 4
    num_nodes = 50
    seq_len = 12
    time_series_dim = 3
    hidden_dim = 64

    # 1. Instantiate module
    model = MaskedSpatiotemporalEncoder(
        num_nodes=num_nodes,
        seq_len=seq_len,
        time_series_dim=time_series_dim,
        hidden_dim=hidden_dim
    )

    # 2. Dummy Inputs
    dynamic_features = torch.randn(batch_size, num_nodes, seq_len, time_series_dim)
    
    # Mock graph: fully connected for stress test
    adj = torch.ones((num_nodes, num_nodes))
    from torch_geometric.utils import dense_to_sparse
    edge_index, edge_weight = dense_to_sparse(adj)

    # 3. Forward Pass
    logger.info(f"Input shape: {dynamic_features.shape}")
    output = model(dynamic_features, edge_index, edge_weight)
    logger.info(f"Output shape: {output.shape}")

    # 4. Shape Assertion
    expected_shape = (batch_size, num_nodes, seq_len, hidden_dim)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    logger.info("Shape Assertion PASSED")

    # 5. Gradient Assertion
    loss = output.sum()
    loss.backward()

    # Check GAT gradients
    gat_has_grad = False
    for name, param in model.gat.named_parameters():
        if param.grad is not None:
            gat_has_grad = True
            break
    assert gat_has_grad, "Gradient Assertion Failed: GAT layers have no gradient."

    # Check Transformer gradients
    transformer_has_grad = False
    for name, param in model.transformer.named_parameters():
        if param.grad is not None:
            transformer_has_grad = True
            break
    assert transformer_has_grad, "Gradient Assertion Failed: Transformer layers have no gradient."

    logger.info("Gradient Assertion PASSED (Spatial & Temporal paths are connected)")
    
    print("--- MaskedSpatiotemporalEncoder Offensive Unit Test PASSED ---")
