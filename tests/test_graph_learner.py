import torch
import pytest
from graph_learner import DynamicGraphLearner

def test_dynamic_graph_learner():
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)

    num_nodes = 50
    static_feat_dim = 10
    embedding_dim = 16
    sparsity_threshold = 0.5
    
    model = DynamicGraphLearner(
        static_feat_dim=static_feat_dim, 
        embedding_dim=embedding_dim, 
        sparsity_threshold=sparsity_threshold
    )

    static_features = torch.randn(num_nodes, static_feat_dim)
    edge_index, edge_weight = model(static_features)

    E = edge_weight.shape[0]
    assert edge_index.shape == (2, E), f"edge_index shape mismatch: {edge_index.shape}"
    assert edge_weight.shape == (E,), f"edge_weight shape mismatch: {edge_weight.shape}"

    assert E > 0, "Graph is completely disconnected (E=0)."
    assert E < (num_nodes * num_nodes), f"Graph is fully connected (E={E}), sparsity threshold not working."

    loss = edge_weight.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient."
