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

    # encoder input = dynamic channels + the 2 structural mask channels
    dynamic_features = torch.randn(batch_size, num_nodes, seq_len, time_series_dim + 2)
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


def test_masked_spatiotemporal_encoder_edge_dim_10():
    """PHASE 3: encoder consumes a [E, 10] enriched edge_attr when edge_dim=10."""
    torch.manual_seed(42)

    batch_size = 4
    num_nodes = 20
    seq_len = 12
    time_series_dim = 3
    hidden_dim = 64
    edge_dim = 10

    model = MaskedSpatiotemporalEncoder(
        num_nodes=num_nodes,
        seq_len=seq_len,
        time_series_dim=time_series_dim,
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
    )

    # encoder input = dynamic channels + the 2 structural mask channels
    dynamic_features = torch.randn(batch_size, num_nodes, seq_len, time_series_dim + 2)
    adj = torch.ones((num_nodes, num_nodes))
    edge_index, _ = dense_to_sparse(adj)
    E = edge_index.shape[1]
    edge_attr = torch.rand(E, edge_dim)

    output = model(dynamic_features, edge_index, edge_attr)
    assert output.shape == (batch_size, num_nodes, seq_len, hidden_dim)

    output.sum().backward()
    assert any(p.grad is not None for _, p in model.gat.named_parameters())


def test_encoder_edge_attr_dim_mismatch_raises():
    """PHASE 3: passing edge_attr with the wrong width fails fast."""
    model = MaskedSpatiotemporalEncoder(
        num_nodes=10, seq_len=12, time_series_dim=3, hidden_dim=64, edge_dim=10
    )
    dynamic_features = torch.randn(2, 10, 12, 3)
    adj = torch.ones((10, 10))
    edge_index, _ = dense_to_sparse(adj)
    bad_edge_attr = torch.rand(edge_index.shape[1], 4)  # width 4 != edge_dim 10
    with pytest.raises(AssertionError):
        model(dynamic_features, edge_index, bad_edge_attr)
