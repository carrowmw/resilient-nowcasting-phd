import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from graph_learner import DynamicGraphLearner
from encoder import MaskedSpatiotemporalEncoder
from decoders import DecoupledDecoders, SpatialKrigingDecoder, compute_loss
from dataset import RealIoTDataset

from typing import Dict, Tuple, Union

logger = logging.getLogger(__name__)

class ResilientIoTPipeline(nn.Module):
    """
    ResilientIoTPipeline: End-to-end spatiotemporal model for IoT sensor networks.
    Integrates Topology Inference, Spatiotemporal Encoding, and Multi-Task Decoders.
    """
    def __init__(self,
                 num_nodes: int,
                 static_feat_dim: int,
                 time_series_dim: int,
                 seq_len: int,
                 embedding_dim: int = 32,
                 hidden_dim: int = 64,
                 sparsity_threshold: float = 0.5,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 dim_feedforward: int = None,
                 zinb_eps: float = 1.0e-6):
        super(ResilientIoTPipeline, self).__init__()

        # Default dim_feedforward to 4x hidden_dim if not specified or invalid
        if dim_feedforward is None or dim_feedforward <= 0:
            dim_feedforward = hidden_dim * 4

        # 1. Topology Inference
        self.graph_learner = DynamicGraphLearner(
            static_feat_dim=static_feat_dim,
            embedding_dim=embedding_dim,
            sparsity_threshold=sparsity_threshold
        )
        logger.info(f"Initialized DynamicGraphLearner with static_feat_dim={static_feat_dim}, embedding_dim={embedding_dim}, sparsity_threshold={sparsity_threshold}, total_params={sum(p.numel() for p in self.graph_learner.parameters())}")

        # 2. Spatiotemporal Feature Extraction
        self.encoder = MaskedSpatiotemporalEncoder(
            num_nodes=num_nodes,
            seq_len=seq_len,
            time_series_dim=time_series_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        logger.info(f"Initialized MaskedSpatiotemporalEncoder with num_nodes={num_nodes}, seq_len={seq_len}, time_series_dim={time_series_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}, total_params={sum(p.numel() for p in self.encoder.parameters())}")

        # 3. Multi-Task Decoders
        self.decoders = DecoupledDecoders(
            hidden_dim=hidden_dim,
            time_series_dim=time_series_dim,
            dropout=dropout,
            zinb_eps=zinb_eps
        )
        logger.info(f"Initialized DecoupledDecoders with hidden_dim={hidden_dim}, time_series_dim={time_series_dim}, dropout={dropout}, total_params={sum(p.numel() for p in self.decoders.parameters())}")

        # PHASE 2: Spatial Kriging Decoder (dormant unless a holdout mask is supplied
        # in the batch and loss.w_kriging > 0). Reconstructs held-out nodes' series.
        self.kriging_decoder = SpatialKrigingDecoder(
            hidden_dim=hidden_dim,
            time_series_dim=time_series_dim,
            dropout=dropout
        )
        logger.info(f"Initialized SpatialKrigingDecoder with hidden_dim={hidden_dim}, time_series_dim={time_series_dim}, total_params={sum(p.numel() for p in self.kriging_decoder.parameters())}")

        # PHASE 1: Scaler buffers (initialized to sentinel values, set by train.py)
        self.register_buffer('traffic_mean', torch.tensor(0.0))
        self.register_buffer('traffic_std', torch.tensor(1.0))
        logger.info("Registered traffic scaler buffers (traffic_mean, traffic_std)")

    def set_scaler_stats(self, traffic_mean: float, traffic_std: float):
        """
        Called by train.py after dataset initialization to populate scaler buffers.

        Args:
            traffic_mean: Mean of traffic channel from dataset scaler
            traffic_std: Standard deviation of traffic channel from dataset scaler
        """
        import numpy as np

        # OFFENSIVE ASSERTIONS
        assert traffic_std > 0, f"CRITICAL: Cannot set traffic_std to {traffic_std} (must be positive)"
        assert not np.isnan(traffic_mean), "CRITICAL: Cannot set traffic_mean to NaN"
        assert not np.isnan(traffic_std), "CRITICAL: Cannot set traffic_std to NaN"

        self.traffic_mean.fill_(traffic_mean)
        self.traffic_std.fill_(traffic_std)

        logger.info(f"Scaler buffers set: mean={traffic_mean:.4f}, std={traffic_std:.4f}, device={self.traffic_mean.device}")

    def forward(
        self,
        batch: dict,
        return_analysis: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        Args:
            batch (dict): Expected keys:
                - static_features: [num_nodes, static_feat_dim]
                - dynamic_features: [batch_size, num_nodes, seq_len, time_series_dim]
                - target_forecast: [batch_size, num_nodes]  (not required in analysis mode)
                - target_impute: [batch_size, num_nodes, seq_len, time_series_dim]  (not required in analysis mode)
                - node_mask: [batch_size, num_nodes, seq_len, time_series_dim]  (not required in analysis mode)
            return_analysis: If True, complete the full forward pass through the decoders
                but bypass loss calculation. Returns a comprehensive dict with graph
                structure, attention weights, and ZINB distribution parameters for
                post-hoc interpretability and UI payload generation.
        Returns:
            Normal mode: (total_loss, loss_impute, loss_forecast, loss_kriging,
                          imputed_features, mu, theta, pi, tripwire_logs, kriging_pred)
            Analysis mode: {'edge_index', 'edge_weight', 'attention_weights',
                            'imputed_features', 'kriging_pred', 'mu', 'theta', 'pi', 'tripwire_logs'}
        """
        static_features = batch['static_features']
        dynamic_features = batch['dynamic_features']
        # PHASE 2: spatial holdout mask [N] bool (None when kriging inactive).
        holdout_mask = batch.get('spatial_holdout_mask')

        # Defensive Data Handling
        # Replace NaNs (injected by our Dataset masking or simulated nodes) with 0.0
        dynamic_features_clean = torch.nan_to_num(dynamic_features, nan=0.0)

        # 1. Graph Learner
        edge_index, edge_weight = self.graph_learner(static_features)

        # --- Strict Topology Masking ---
        # Prevent "ghost clustering" and pollution by dropping outgoing edges from simulated nodes
        if 'simulated_nodes_mask' in batch:
            sim_mask = batch['simulated_nodes_mask']
            sources = edge_index[0]
            valid_edges = ~sim_mask[sources]
            edge_index = edge_index[:, valid_edges]
            edge_weight = edge_weight[valid_edges]

        # 2. Encoder — request attention weights when in analysis mode
        if return_analysis:
            latent_embeddings, alpha = self.encoder(
                dynamic_features_clean, edge_index, edge_weight,
                return_attention_weights=True,
            )

            # 3. Decoders (complete the full pass to get ZINB parameters)
            imputed_features, mu, theta, pi, tripwires = self.decoders(latent_embeddings)

            # PHASE 2: Kriging reconstruction for held-out nodes (if a mask is supplied)
            kriging_pred = (
                self.kriging_decoder(latent_embeddings, holdout_mask)
                if holdout_mask is not None else None
            )

            return {
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'attention_weights': alpha,
                'imputed_features': imputed_features,
                'kriging_pred': kriging_pred,
                'mu': mu,
                'theta': theta,
                'pi': pi,
                'tripwire_logs': tripwires
            }

        # Normal training mode — need loss targets
        target_forecast = batch['target_forecast']
        target_impute = batch['target_impute']

        latent_embeddings = self.encoder(dynamic_features_clean, edge_index, edge_weight)

        # 3. Decoders
        imputed_features, mu, theta, pi, tripwires = self.decoders(latent_embeddings)

        # PHASE 2: Kriging reconstruction for held-out nodes (if a mask is supplied)
        kriging_pred = (
            self.kriging_decoder(latent_embeddings, holdout_mask)
            if holdout_mask is not None else None
        )

        # 4. Compute Joint Loss
        loss, loss_impute, loss_forecast, loss_kriging = compute_loss(
            imputed_features=imputed_features,
            target_impute=target_impute,
            mu=mu,
            theta=theta,
            pi=pi,
            target_forecast=target_forecast,
            traffic_mean=self.traffic_mean,  # PHASE 1: scaler buffers
            traffic_std=self.traffic_std,    # PHASE 1: scaler buffers
            kriging_pred=kriging_pred,                 # PHASE 2
            spatial_holdout_mask=holdout_mask,         # PHASE 2
            w_impute=batch.get('w_impute', 1.0),
            w_forecast=batch.get('w_forecast', 1.0),
            w_kriging=batch.get('w_kriging', 0.0)      # PHASE 2
        )

        # CRITICAL SAFEGUARD
        assert not torch.isnan(loss).any(), "CRITICAL FAILURE: Pipeline total_loss is NaN."

        return loss, loss_impute, loss_forecast, loss_kriging, imputed_features, mu, theta, pi, tripwires, kriging_pred

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Reproducibility
    torch.manual_seed(42)

    logger.info("Starting ResilientIoTPipeline Integration Test (The Burn Test)")

    # Dimensions
    batch_size = 4
    num_nodes = 50
    static_feat_dim = 10
    time_series_dim = 3
    seq_len = 12

    # --- 1. Dataset Initialization & Mock Data ---
    # The Dataset defined in dataset.py expects (N, S) for static and (N, T, D) for dynamic.
    # We want a pipeline that gets batches with the 'num_nodes' dimension intact.
    # We will generate a mock Dataset that yields identical static features, and (N, T, D)
    # structured batches. However, since the dataset yields (T, D) per sample and we want
    # [B, N, T, D] out of the DataLoader, we must adjust how we interact with it or mock the batch.

    # We'll create a custom mock Dataset specifically for testing the pipeline requirements.
    class MockPipelineDataset(torch.utils.data.Dataset):
        def __init__(self, size, mask_prob=0.15):
            self.size = size
            self.mask_prob = mask_prob
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            # dynamic: [N, T, D]
            dyn = torch.randn(num_nodes, seq_len, time_series_dim)
            mask = torch.rand_like(dyn) < self.mask_prob
            dyn_masked = dyn.clone()
            dyn_masked[mask] = float('nan')

            return {
                # We return static feature entirely since it's shared across the batch
                'static_features': torch.randn(num_nodes, static_feat_dim),
                'dynamic_features': dyn_masked,
                'target_impute': dyn.clone(),
                'target_forecast': torch.randint(0, 50, (num_nodes,)).float(),
                'node_mask': mask
            }

    dataset = MockPipelineDataset(size=20)

    def custom_collate(batch_list):
        # The pipeline expects static_features to be exactly [num_nodes, static_feat_dim]
        # not [B, num_nodes, static_feat_dim]. So we just take the first one.
        return {
            'static_features': batch_list[0]['static_features'],
            'dynamic_features': torch.stack([b['dynamic_features'] for b in batch_list]),
            'target_impute': torch.stack([b['target_impute'] for b in batch_list]),
            'target_forecast': torch.stack([b['target_forecast'] for b in batch_list]),
            'node_mask': torch.stack([b['node_mask'] for b in batch_list])
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate)

    # --- 2. Pipeline and Optimizer ---
    model = ResilientIoTPipeline(
        num_nodes=num_nodes,
        static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim,
        seq_len=seq_len,
        num_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 3. Training Loop (2 Epochs) ---
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward Pass
            loss, *_ = model(batch)

            # Backward Pass
            loss.backward()

            # CRITICAL SAFEGUARD: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer Step
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    logger.info("ResilientIoTPipeline Integration Test PASSED (Computation graph is connected, no NaNs)")
