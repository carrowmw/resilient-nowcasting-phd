# Resilient IoT Spatiotemporal Pipeline

## Objective

Implement an end-to-end, joint spatiotemporal graph neural network in PyTorch and PyTorch Geometric (PyG). The model is designed for an urban IoT sensor network (specifically pedestrian counting every 15 minutes). It must be self-healing, capable of dynamic graph inference, missing feature imputation, and uncertainty-aware forecasting.

## Core Paradigm

The system abandons the I.I.D. assumption, leveraging Graph Attention Networks (GATs) to pass messages along dynamically learned topological connections, alongside Transformers for temporal sequence modelling.

### 1. Architectural Components

The pipeline (ResilientIoTPipeline) consists of three primary modules sharing a latent space:

#### Dynamic Graph Learner (Topology Inference):

- Takes static node features (e.g., coordinates, sensor types).
- Uses a multi-head GAT (or linear embeddings with pairwise similarity) to learn a dynamic adjacency matrix on the fly.
- Applies a ReLU and a sparsity threshold, converting the dense matrix to PyG's edge_index and edge_weight using dense_to_sparse.

#### Masked Spatiotemporal Encoder (Feature Extraction):

- Takes artificially masked dynamic time-series data and the inferred graph structure.

- Spatial: Uses PyG GATv2Conv layers (independently across time steps) to aggregate neighbourhood features.

Temporal: Flattens the spatial output and passes it through a standard PyTorch TransformerEncoder to capture temporal dependencies.

#### Decoupled Decoders (Multi-Task Learning):

Imputation Decoder: A linear layer applied across the entire sequence to reconstruct the artificially masked historical data (Task: Self-healing).

Uncertainty-Aware Forecasting Decoder: A linear layer taking the final time step's embedding. It outputs three parameters representing a Zero-Inflated Negative Binomial (ZINB) distribution: $\mu$ (mean), $\theta$ (dispersion), and $\pi$ (zero-inflation probability).

### 2. Data Pipeline Specifications

The PyTorch Dataset must yield a dictionary for each batch to handle the complex inputs and masking required for joint training. Expected dictionary keys:

static_features: [num_nodes, static_feat_dim] (Physical sensor attributes).

dynamic_features: [batch_size, num_nodes, seq_len, time_series_dim] (Historical sequences with ~15% artificial NaN masking applied for self-supervised imputation training).

target_forecast: [batch_size, num_nodes] (The ground-truth future pedestrian counts).

target_impute: [batch_size, num_nodes, seq_len, time_series_dim] (The unmasked historical sequence to calculate imputation loss).

node_mask: [batch_size, num_nodes] (Boolean mask to ignore genuinely offline sensors during loss calculation).

### 3. Joint Training & Loss Logic

The model must be trained simultaneously on all tasks using a central compute_loss method.

Forecasting Loss (ZINB NLL constraints - CRITICAL):
The ZINB distribution is highly volatile and prone to NaN gradients. The implementation must include these mathematical tripwires:

Apply torch.sigmoid to the $\pi$ raw logit.

Apply F.softplus to the $\mu$ and $\theta$ raw logits.

Tripwire: Immediately add eps = 1e-6 to $\mu$ and $\theta$ to mathematically guarantee they never equal exactly zero.

Use torch.lgamma (log-gamma) for factorial equivalents to prevent overflow.

Clamp the minimum value of any torch.log() operation to eps to avoid log(0).

Calculate the Negative Log-Likelihood (NLL) of target_forecast given $\mu$, $\theta$, $\pi$.

Imputation Loss:

Calculates Mean Squared Error (MSE) between the imputed_features and target_impute strictly on the artificially masked nodes.

Total Loss:

$\mathcal{L}_{total} = (w_{forecast} \times \mathcal{L}_{ZINB\_NLL}) + (w_{impute} \times \mathcal{L}_{impute\_MSE})$

Training Loop Safeguards:

To prevent exploding gradients from the joint task, gradient clipping (torch.nn.utils.clip*grad_norm*(model.parameters(), max_norm=1.0)) is mandatory immediately after loss.backward() and before optimizer.step().

The forward pass must include an assert not torch.isnan(loss).any() check that crashes the script intentionally and prints the min/max of $\mu$, $\theta$, and $\pi$ if triggered.
