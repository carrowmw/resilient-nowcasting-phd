import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np

from pipeline import ResilientIoTPipeline
from dataset import RealIoTDataset, custom_collate_real
from logger import setup_logger

class DivergenceError(Exception):
    """Custom exception raised when model loss diverges (NaNs)."""
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResilientIoTPipeline")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--w_forecast", type=float, default=1.0, help="Weight for forecasting loss")
    parser.add_argument("--w_impute", type=float, default=1.0, help="Weight for imputation loss")
    
    # Dataset arguments
    parser.add_argument("--use_mock_data", action="store_true", help="Use mock data instead of real parquet data")
    parser.add_argument("--start_date", type=str, default="2023-03-02", help="Start date for real data (inclusive)")
    parser.add_argument("--end_date", type=str, default="2023-04-02", help="End date for real data (exclusive)")
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence length for temporal processing")
    
    return parser.parse_args()

def create_mock_dataloader(batch_size: int, num_nodes: int, seq_len: int, time_series_dim: int, static_feat_dim: int, dataset_size: int):
    """
    Creates a mock DataLoader yielding batches that match the pipeline's expected dimensional structure.
    """
    class MockPipelineDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
            self.static_feat_dim = static_feat_dim
            self.time_series_dim = time_series_dim
            self.num_nodes = num_nodes
            
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
            
    dataset = MockPipelineDataset(size=dataset_size)
    
    def custom_collate(batch_list):
        return {
            'static_features': batch_list[0]['static_features'], # Static across batch
            'dynamic_features': torch.stack([b['dynamic_features'] for b in batch_list]),
            'target_impute': torch.stack([b['target_impute'] for b in batch_list]),
            'target_forecast': torch.stack([b['target_forecast'] for b in batch_list]),
            'node_mask': torch.stack([b['node_mask'] for b in batch_list])
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate), dataset

def run_training(args, init_wandb=True) -> float:
    """
    Core execution logic for a single training run.
    Returns the final validation loss.
    """
    logger = logging.getLogger(__name__)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize W&B only if requested
    if init_wandb:
        wandb.init(project="resilient-iot", config=args)

    # Load Dataset
    if args.use_mock_data:
        logger.info("Using mock dataloader.")
        train_loader, train_dataset = create_mock_dataloader(args.batch_size, 50, args.seq_len, 3, 10, 20)
        val_loader, _ = create_mock_dataloader(args.batch_size, 50, args.seq_len, 3, 10, 8)
    else:
        logger.info(f"Loading RealIoTDataset from {args.start_date} to {args.end_date}")
        parquet_path = "data/processed/master_dataset_15T.parquet"
        try:
            # We must create disjoint datasets or just use a single one and split it
            # For simplicity in this step, let's load train and val identically if they overlap, 
            # or ideally we'd split. We'll use the same dataset object for test purposes or split it.
            full_dataset = RealIoTDataset(parquet_path, args.start_date, args.end_date, seq_len=args.seq_len)
            
            # Simple sequential split: 80% train, 20% val
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            if train_size == 0 or val_size == 0:
                 raise ValueError("Not enough data to split into train and val. Try a wider date range.")
                 
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            
            # The dimensions are stored on the parent dataset object
            train_dataset.num_nodes = full_dataset.num_nodes
            train_dataset.static_feat_dim = full_dataset.static_feat_dim
            train_dataset.time_series_dim = full_dataset.time_series_dim
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate_real, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_real)
        except Exception as e:
            logger.error(f"Failed to load RealIoTDataset: {e}")
            raise

    # Model
    model = ResilientIoTPipeline(
        num_nodes=train_dataset.num_nodes,
        static_feat_dim=train_dataset.static_feat_dim,
        time_series_dim=train_dataset.time_series_dim,
        seq_len=args.seq_len
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info("Starting training loop...")

    avg_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        
        epoch_loss_total = 0.0
        epoch_loss_forecast = 0.0
        epoch_loss_impute = 0.0
        
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss, loss_impute, loss_forecast = model(batch)
            loss.backward()
            
            # CRITICAL SAFEGUARD: Gradient Clipping (from ARCHITECTURE.md)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss_total += loss.item()
            epoch_loss_impute += loss_impute.item()
            epoch_loss_forecast += loss_forecast.item()
            
        avg_train_loss = epoch_loss_total / len(train_loader)
        avg_train_impute = epoch_loss_impute / len(train_loader)
        avg_train_forecast = epoch_loss_forecast / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss, _, _ = model(batch)
                val_loss_total += loss.item()
                
        avg_val_loss = val_loss_total / len(val_loader)
        
        # Offensive Assertion for Divergence
        if torch.isnan(torch.tensor(avg_val_loss)):
            raise DivergenceError("Validation loss is NaN! The Burn Test Failed.")
            
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} "
                    f"(Impute: {avg_train_impute:.4f}, Forecast: {avg_train_forecast:.4f}) | "
                    f"Val Loss: {avg_val_loss:.4f}")
                    
        # W&B Logging
        if init_wandb or wandb.run is not None:
            wandb.log({
                "train_loss_total": avg_train_loss,
                "train_loss_impute": avg_train_impute,
                "train_loss_forecast": avg_train_forecast,
                "val_loss_total": avg_val_loss,
                "epoch": epoch + 1
            })

    logger.info("Training complete.")
    if init_wandb:
        wandb.finish()
        
    return avg_val_loss

from logger import setup_logger

def main():
    logger = setup_logger("train_pipeline")
    args = parse_args()
    run_training(args, init_wandb=True)

if __name__ == "__main__":
    main()
