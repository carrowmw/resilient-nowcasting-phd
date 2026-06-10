import os
import torch
import pandas as pd
from src.dataset import RealIoTDataset

def test_dataset_places_enrichment():
    parquet_path = "data/processed/master_dataset_15T.parquet"
    
    if not os.path.exists(parquet_path):
        print(f"Parquet file {parquet_path} not found. Skipping test.")
        return

    # 1. Instantiate dataset with real parquet file
    # We use a small date range for the smoke test
    try:
        dataset = RealIoTDataset(
            parquet_path=parquet_path,
            start_date="2024-04-02",
            end_date="2024-04-03",
            seq_len=12,
            mask_prob=0.15,
            target_veh_class="person",
            target_dir="all"
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # 2. Assert correct static feature dimensionality
    print(f"Loaded Static Feature Shape: {dataset.static_features.shape}")
    print(f"Dataset static_feat_dim: {dataset.static_feat_dim}")
    
    # Static feature schema is 15-dim (Google Places 12 + OSMnx centrality 3),
    # per the 2026-05-21 / 2026-06-09 static-feature expansion.
    assert dataset.static_feat_dim == 15, f"Expected 15 features, got {dataset.static_feat_dim}"
    assert dataset.static_features.shape == (dataset.num_nodes, 15), "Static features shape mismatch!"
    
    # 3. Assert correct normalization (mean close to 0, std close to 1)
    mean = dataset.static_features.mean(dim=0)
    std = dataset.static_features.std(dim=0)
    print(f"Static Feature Means: {mean.tolist()}")
    print(f"Static Feature StdDevs: {std.tolist()}")
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1), "Static features mean is not normalized!"
    
    # 4. Assert absence of NaNs / Infs
    assert not torch.isnan(dataset.static_features).any(), "Found NaNs in static features!"
    assert not torch.isinf(dataset.static_features).any(), "Found Infs in static features!"
    
    print("✓ Dataset Smoke Test Passed Successfully!")

if __name__ == "__main__":
    test_dataset_places_enrichment()
