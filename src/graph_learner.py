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


