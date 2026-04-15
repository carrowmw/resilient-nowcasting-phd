import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from graph_learner import DynamicGraphLearner
from encoder import MaskedSpatiotemporalEncoder
from decoders import DecoupledDecoders, compute_loss
from dataset import ResilientIoTDataset

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
                 sparsity_threshold: float = 0.5):
        super(ResilientIoTPipeline, self).__init__()
        
        # 1. Topology Inference
        self.graph_learner = DynamicGraphLearner(
            static_feat_dim=static_feat_dim, 
            embedding_dim=embedding_dim, 
            sparsity_threshold=sparsity_threshold
        )
        
        # 2. Spatiotemporal Feature Extraction
        self.encoder = MaskedSpatiotemporalEncoder(
            num_nodes=num_nodes, 
            seq_len=seq_len, 
            time_series_dim=time_series_dim, 
            hidden_dim=hidden_dim
        )
        
        # 3. Multi-Task Decoders
        self.decoders = DecoupledDecoders(
            hidden_dim=hidden_dim, 
            time_series_dim=time_series_dim
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch (dict): Expected keys:
                - static_features: [num_nodes, static_feat_dim]
                - dynamic_features: [batch_size, num_nodes, seq_len, time_series_dim]
                - target_forecast: [batch_size, num_nodes]
                - target_impute: [batch_size, num_nodes, seq_len, time_series_dim]
                - node_mask: [batch_size, num_nodes, seq_len, time_series_dim] (or related)
        Returns:
            total_loss: Scalar torch.Tensor
        """
        static_features = batch['static_features']
        dynamic_features = batch['dynamic_features']
        target_forecast = batch['target_forecast']
        target_impute = batch['target_impute']
        
        # Defensive Data Handling
        # Replace NaNs (injected by our Dataset masking) with 0.0 to allow PyTorch ops
        dynamic_features_clean = torch.nan_to_num(dynamic_features, nan=0.0)
        
        # 1. Graph Learner
        edge_index, edge_weight = self.graph_learner(static_features)
        
        # 2. Encoder
        latent_embeddings = self.encoder(dynamic_features_clean, edge_index, edge_weight)
        
        # 3. Decoders
        imputed_features, mu, theta, pi = self.decoders(latent_embeddings)
        
        # 4. Compute Joint Loss
        loss = compute_loss(
            imputed_features=imputed_features,
            target_impute=target_impute,
            mu=mu,
            theta=theta,
            pi=pi,
            target_forecast=target_forecast
        )
        
        # CRITICAL SAFEGUARD
        assert not torch.isnan(loss).any(), "CRITICAL FAILURE: Pipeline total_loss is NaN."
        
        return loss

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
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            # dynamic: [N, T, D]
            dyn = torch.randn(num_nodes, seq_len, time_series_dim)
            mask = torch.rand_like(dyn) < 0.15
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
        seq_len=seq_len
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 3. Training Loop (2 Epochs) ---
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward Pass
            loss = model(batch)
            
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
