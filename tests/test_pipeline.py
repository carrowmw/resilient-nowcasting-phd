import torch
import pytest
from pipeline import ResilientIoTPipeline
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
        loss, _, _, _, _, _, _, tripwires = model(batch)
        assert not torch.isnan(loss).any()
        assert isinstance(tripwires, list)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
