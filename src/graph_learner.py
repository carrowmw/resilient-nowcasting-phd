import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from typing import Tuple

class DynamicGraphLearner(nn.Module):
    """
    DynamicGraphLearner: Infers graph topology from static node features.
    
    Architecture:
    1. Linear projection to latent space.
    2. Pairwise dot-product similarity to form adjacency matrix.
    3. ReLU and tunable sparsity thresholding.
    4. Sparse conversion to (edge_index, edge_weight).
    """
    def __init__(self, static_feat_dim: int, embedding_dim: int = 32, sparsity_threshold: float = 0.5):
        super(DynamicGraphLearner, self).__init__()
        # Ensure random seeds are handled globally or at the top of the script
        
        self.static_feat_dim = static_feat_dim
        self.embedding_dim = embedding_dim
        
        # Parameterized projection layer
        self.lin = nn.Linear(static_feat_dim, embedding_dim)
        
        # Tunable sparsity threshold (can be a learned parameter if needed, but here a buffer/float)
        self.register_buffer('sparsity_threshold', torch.tensor(sparsity_threshold))

    def forward(self, static_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        Args:
            static_features: [num_nodes, static_feat_dim]
        Returns:
            edge_index: [2, E]
            edge_weight: [E]
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

        return edge_index, edge_weight

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)

    print("--- Starting DynamicGraphLearner Offensive Unit Test ---")

    # 1. Instantiate module
    num_nodes = 50
    static_feat_dim = 10
    embedding_dim = 16
    sparsity_threshold = 0.5
    
    model = DynamicGraphLearner(
        static_feat_dim=static_feat_dim, 
        embedding_dim=embedding_dim, 
        sparsity_threshold=sparsity_threshold
    )

    # 2. Dummy input
    static_features = torch.randn(num_nodes, static_feat_dim)

    # 3. Forward pass
    edge_index, edge_weight = model(static_features)

    # 4. Shape Assertions
    E = edge_weight.shape[0]
    print(f"Number of edges: {E}")
    assert edge_index.shape == (2, E), f"edge_index shape mismatch: {edge_index.shape}"
    assert edge_weight.shape == (E,), f"edge_weight shape mismatch: {edge_weight.shape}"

    # 5. Topology Assertions (The Burn Test)
    print(f"Max possible edges: {num_nodes * num_nodes}")
    assert E > 0, "The Burn Test Failed: Graph is completely disconnected (E=0)."
    assert E < (num_nodes * num_nodes), f"The Burn Test Failed: Graph is fully connected (E={E}), sparsity threshold not working."

    # 6. Gradient Assertion
    loss = edge_weight.sum()
    loss.backward()

    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient Assertion Failed: Parameter '{name}' has no gradient."
        print(f"Gradient for '{name}' verified (norm: {param.grad.norm().item():.4f})")

    print("--- DynamicGraphLearner Offensive Unit Test PASSED ---")
