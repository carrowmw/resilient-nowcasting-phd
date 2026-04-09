# Plan: Implement DynamicGraphLearner

## Objective
Implement the `DynamicGraphLearner` module in `src/graph_learner.py` to infer dynamic graph topology from static node features, as specified in `ARCHITECTURE.md`.

## Proposed Architecture
- **Module Name**: `DynamicGraphLearner`
- **Input**: `static_features` [num_nodes, static_feat_dim]
- **Mechanism**:
    1. **Embedding**: A Linear layer to project static features into a latent space (e.g., `embedding_dim=32`).
    2. **Similarity**: Compute pairwise similarity using a matrix product (A = E @ E.T).
    3. **Sparsity**: 
        - Apply `ReLU` to ensure non-negative weights.
        - Apply a `sparsity_threshold` (tunable parameter).
        - Subtract threshold and apply `ReLU` again or mask values below the threshold to zero.
    4. **Output**: Convert the dense adjacency matrix to PyG's `edge_index` and `edge_weight` using `torch_geometric.utils.dense_to_sparse`.

## Implementation Steps
1. **Dependency Management**:
    - Install `torch-geometric` and necessary dependencies.
2. **Coding**:
    - Create `src/graph_learner.py`.
    - Implement `DynamicGraphLearner` class with type hints and assertions.
    - Include an offensive unit test block in `if __name__ == "__main__":`.
3. **Verification**:
    - Run `python src/graph_learner.py`.
    - Ensure all assertions (shape, topology, gradient) pass.
4. **Finalization**:
    - Commit changes.
    - Update `EXPERIMENTS.md`.

## Unit Test Details (The Burn Test)
- **Input**: Dummy `static_features` (50 nodes, 10 features).
- **Assertions**:
    - `edge_index.shape == (2, E)`
    - `edge_weight.shape == (E,)`
    - `0 < E < (num_nodes * num_nodes)` (Validates sparsity and connectivity).
    - `loss.backward()` results in non-null gradients for module parameters.
