import torch
import pytest
from pipeline import ResilientIoTPipeline
from dataset import make_spatial_holdout_mask, apply_spatial_holdout
from torch.utils.data import DataLoader

def test_resilient_iot_pipeline():
    torch.manual_seed(42)

    batch_size = 4
    num_nodes = 50
    static_feat_dim = 10
    time_series_dim = 3
    seq_len = 12

    class MockPipelineDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            dyn = torch.randn(num_nodes, seq_len, time_series_dim)
            mask = torch.rand_like(dyn) < 0.15
            dyn_masked = dyn.clone()
            dyn_masked[mask] = float('nan')
            return {
                'static_features': torch.randn(num_nodes, static_feat_dim),
                'dynamic_features': dyn_masked,
                'target_impute': dyn.clone(),
                'target_forecast': torch.randint(0, 50, (num_nodes,)).float(),
                'node_mask': mask
            }
            
    dataset = MockPipelineDataset(size=8)
    
    def custom_collate(batch_list):
        return {
            'static_features': batch_list[0]['static_features'],
            'dynamic_features': torch.stack([b['dynamic_features'] for b in batch_list]),
            'target_impute': torch.stack([b['target_impute'] for b in batch_list]),
            'target_forecast': torch.stack([b['target_forecast'] for b in batch_list]),
            'node_mask': torch.stack([b['node_mask'] for b in batch_list])
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate)

    model = ResilientIoTPipeline(
        num_nodes=num_nodes,
        static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim,
        seq_len=seq_len
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for batch in dataloader:
        optimizer.zero_grad()
        # PHASE 2: forward now returns a 10-tuple (adds loss_kriging + kriging_pred).
        loss, _, _, loss_kriging, _, _, _, _, tripwires, kriging_pred = model(batch)
        assert not torch.isnan(loss).any()
        assert isinstance(tripwires, list)
        # No holdout mask supplied -> kriging path is dormant.
        assert loss_kriging.item() == 0.0
        assert kriging_pred is None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def test_pipeline_kriging_active():
    """PHASE 2: with a spatial holdout mask + w_kriging > 0, the kriging head runs,
    contributes a positive loss, and receives gradients."""
    torch.manual_seed(42)

    batch_size = 4
    num_nodes = 20
    static_feat_dim = 10
    time_series_dim = 3
    seq_len = 12

    model = ResilientIoTPipeline(
        num_nodes=num_nodes,
        static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim,
        seq_len=seq_len,
    )

    holdout_mask = make_spatial_holdout_mask(num_nodes, holdout_prob=0.25, seed=7)
    assert holdout_mask.sum() > 0

    dyn = torch.randn(batch_size, num_nodes, seq_len, time_series_dim)
    target_impute = dyn.clone()
    batch = {
        "static_features": torch.randn(num_nodes, static_feat_dim),
        "dynamic_features": apply_spatial_holdout(dyn, holdout_mask),
        "target_impute": target_impute,
        "target_forecast": torch.randint(0, 50, (batch_size, num_nodes)).float(),
        "node_mask": torch.zeros_like(dyn, dtype=torch.bool),
        "spatial_holdout_mask": holdout_mask,
        "w_kriging": 0.5,
    }

    loss, _, _, loss_kriging, _, _, _, _, _, kriging_pred = model(batch)

    assert kriging_pred is not None
    assert kriging_pred.shape == (batch_size, int(holdout_mask.sum()), seq_len, time_series_dim)
    assert loss_kriging.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert model.kriging_decoder.kriging_head[0].weight.grad is not None
    assert model.kriging_decoder.kriging_head[0].weight.grad.abs().sum().item() > 0.0
