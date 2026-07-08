# Context for Architect: GPU Data Starvation Investigation

## Objective
Investigate why the GPU is idling between 0% and 100% every cycle (epoch/batch) and determine if data loading can be sped up or pre-loaded.

## Relevant File Paths
- `src/dataset.py` (Specifically `RealIoTDataset.__init__` and `__getitem__`)
- `src/train.py` (Specifically `DataLoader` instantiation and the training loop)
- `configs/infra.yaml` (Infrastructure configuration)

## High-Level Logic Summary & Findings

### 1. Data Loading Mechanism
- **In-Memory Dataset**: `RealIoTDataset` loads the *entire dataset into RAM* during `__init__`. 
- **Trivial `__getitem__`**: The `__getitem__` method performs extremely fast operations—mostly just slicing in-memory tensors (`self.dynamic_data[idx:...]` and `self.structural_masks[...]`) and permuting them.
- **Multiprocessing Overhead**: `configs/infra.yaml` configures `num_workers: 16`. When `num_workers > 0`, PyTorch's `DataLoader` relies on Python's `multiprocessing`. Because the data is already in RAM and `__getitem__` is trivial, the inter-process communication (IPC) overhead of shipping sliced tensors across 16 worker processes back to the main process heavily bottlenecks the pipeline. The workers are slowing down the data supply rather than speeding it up.

### 2. GPU Transfer & Pinning
- **Batch-by-Batch Pinning**: In `src/train.py`, the `DataLoader` is created with `pin_memory=torch.cuda.is_available()`. This means a background thread in the main process must dynamically copy every batch into page-locked (pinned) memory before moving it to the GPU via `.to(device, non_blocking=True)` in the training loop.
- **No Upfront Pinning**: The underlying tensors in `RealIoTDataset` (`dynamic_data`, `structural_masks`, `unscaled_traffic`) are *not* pinned in RAM during `__init__`.

### 3. Training Loop Mechanics
- The `for batch in train_loader:` loop transfers data to the GPU and subsequently applies dynamic `NaN` masking and structural holdouts on the GPU itself. Thus, the CPU is purely responsible for slicing and assembling the clean batches.

## Potential Bottlenecks (Tripwires)
1. **IPC Starvation**: `num_workers=16` for a fully in-memory dataset is a classic anti-pattern that creates artificial IPC overhead and starves the GPU.
2. **Dynamic Pinning Overhead**: The batch-by-batch pinning by the `DataLoader` thread might also be struggling to keep up with the GPU.

## Insights for Architect
- Evaluate if `num_workers` should be forced to `0` for `RealIoTDataset` to avoid IPC overhead.
- Evaluate if we can pin the tensors (`self.dynamic_data.pin_memory()`, etc.) directly in `RealIoTDataset.__init__`. If the tensors are already pinned, `num_workers=0` will yield batches that are immediately ready for non-blocking GPU transfer.
- Consider moving the entire dataset directly to the GPU in `__init__` if VRAM permits, effectively bypassing CPU-GPU transfer during the training loop completely.