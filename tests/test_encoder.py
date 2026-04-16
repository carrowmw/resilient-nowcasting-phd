import torch
import pytest
from torch_geometric.utils import dense_to_sparse
from encoder import MaskedSpatiotemporalEncoder

def test_masked_spatiotemporal_encoder():
    torch.manual_seed(42)

    batch_size = 4
    num_nodes = 50
    seq_len = 12
    time_series_dim = 3
    hidden_dim = 64

    model = MaskedSpatiotemporalEncoder(
        num_nodes=num_nodes,
        seq_len=seq_len,
        time_series_dim=time_series_dim,
        hidden_dim=hidden_dim
    )

    dynamic_features = torch.randn(batch_size, num_nodes, seq_len, time_series_dim)
    adj = torch.ones((num_nodes, num_nodes))
    edge_index, edge_weight = dense_to_sparse(adj)

    output = model(dynamic_features, edge_index, edge_weight)

    expected_shape = (batch_size, num_nodes, seq_len, hidden_dim)
    assert output.shape == expected_shape

    loss = output.sum()
    loss.backward()

    gat_has_grad = any(param.grad is not None for _, param in model.gat.named_parameters())
    assert gat_has_grad

    transformer_has_grad = any(param.grad is not None for _, param in model.transformer.named_parameters())
    assert transformer_has_grad
