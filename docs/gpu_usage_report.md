# GPU Usage, Multi-GPU Training, and Device Specification Report

This report provides an in-depth context-gathering analysis of how GPU resources are utilized, configured, and managed within the ML workspace. It is prepared from the perspective of the **Interrogator** to assist development, systems architecture, and deployment alignment.

---

## 1. Executive Summary

- **GPU Utilization:** Training runs (`src/train.py`) and Optuna hyperparameter sweeps (`src/tune.py`) dynamically offload workloads to a CUDA-compatible GPU if PyTorch detects one. All deep learning components (encoders, decoders, loss calculations) and model tensors are dynamically pushed to the active device. Conversely, the UI payload generation (`src/export_ui_payload.py`) is explicitly hardcoded to execute on the CPU.
- **Multi-GPU Training:** **Not currently implemented at the PyTorch model-level.** Single training runs execute on a single GPU using standard sequential processing. However, the container orchestration environment supports infrastructure-level parallel execution by exposing multiple GPUs to independent containers/runs.
- **GPU Specification:** The GPU environment is controlled using Docker Compose GPU device reservations (`device_ids: ["0", "1"]`). At the code level, PyTorch defaults to using the first visible device (`cuda:0` / standard `"cuda"` device index), with no code-level mechanism currently exposing selection controls or supporting PyTorch distributed training libraries.

---

## 2. Deep-Dive: How the GPU is Used

GPU integration in our MLOps system utilizes a hybrid approach: containerized environments for isolation and dynamic Torch device placement for execution.

### A. The Container Execution Layer
Training and sweeps on the remote server run inside a headless, optimized CUDA-enabled Docker container:
- **Base Image:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (defined in `Dockerfile.gpu`).
- **Accessing GPUs:** Built using the local GPU server configuration via `docker compose build gpu-runner` and run using:
  ```bash
  docker compose run --rm gpu-runner python src/train.py --epochs 50 --batch_size 128
  ```

### B. PyTorch Device Selection Logic
Within `src/train.py`, the active device is resolved dynamically at runtime:
```python
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
```
Once the device is identified:
1. **Model Placement:** The entire `ResilientIoTPipeline` architecture is pushed to the target device:
   ```python
   model = ResilientIoTPipeline(...).to(device)
   ```
2. **Tensor/Batch Transfer:** During both the training and validation epochs, loaded tensor batches are moved to the GPU dynamically inside the data loops:
   ```python
   batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
   ```

### C. Sweeps & Tuning
The Optuna sweep framework in `src/tune.py` executes trial objectives by calling `run_training(config=config, init_wandb=False)`. Consequently, all hyperparameter search trials run inside the `gpu-runner` context and utilize the active GPU device dynamically.

### D. CPU-Only Exceptions
The UI payload generator (`src/export_ui_payload.py`) explicitly targets the CPU for post-processing and export steps:
```python
device = torch.device("cpu")  # CPU runner as specified
logger.info(f"Device: {device}")
```
This is a deliberate architectural decision to ensure payload compilation does not require GPU resources or compete with training containers.

---

## 3. Multi-GPU Implementation Status

### A. Single-Model Multi-GPU: **Not Implemented**
The core training script (`src/train.py`) utilizes standard single-device execution. The code **lacks** any implementations of:
- `torch.nn.DataParallel` (DP)
- `torch.nn.parallel.DistributedDataParallel` (DDP)
- Fully Sharded Data Parallel (FSDP)
- High-level orchestration integrations (e.g., PyTorch Ignite, PyTorch Lightning, Hugging Face Accelerate).

As a result, a single model training process cannot split its batch size or model layers across multiple physical cards simultaneously.

### B. Infrastructure-Level Parallelism: **Supported**
Although single-model training is single-GPU, the environment supports running **multiple independent experiments concurrently** on different GPUs on the server. 

According to `NSTAR_MLOPS.md` (§1, Parallel Execution Rule):
> *"Because the GPU runner mounts the local directory (`.:/workspace`), you cannot run multiple branches from the same physical directory. To run experiments in parallel, clone the repository into separate directories on the server and execute `docker compose` independently in each."*

By isolating physical directories and running independent containers, multiple hyperparameter trials or architecture explorations can run simultaneously on different GPU indexes.

---

## 4. How the GPU to Use is Specified

Currently, the selection of which physical GPU to use is managed outside of the Python code itself, moving from Docker Compose to the operating system level.

### A. Docker Compose Resource Allocation
In `docker-compose.yml`, the `gpu-runner` service specifies which hardware GPUs are mapped into the container environment:

```yaml
  # Run on GPU server for training.
  gpu-runner:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    container_name: gpu-executor
    volumes:
      - .:/workspace
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
    # This block maps the physical server GPUs into the container
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1"]
              capabilities: [gpu]
```

This configuration reserves physical GPU cards with index `0` and `1` and exposes them to the container under `/dev/nvidia0` and `/dev/nvidia1`.

### B. Python/PyTorch Default Behavior
Inside the container, PyTorch sees both GPUs as `cuda:0` and `cuda:1`.
However, because `train.py` queries `torch.device("cuda")` without a specific index, PyTorch will automatically default to selecting **device 0** (`cuda:0`). 

### C. Targeting a Specific GPU
There is currently:
- **No** CLI flag in `train.py` or `tune.py` (e.g., `--device cuda:1`).
- **No** parameter in `configs/train.yaml` specifying target device indices.

To target a specific GPU (for example, to force a run onto the second card, `1`), you must specify the standard NVIDIA runtime environment variable `CUDA_VISIBLE_DEVICES` when launching the container command. For example:

```bash
docker compose run --rm -e CUDA_VISIBLE_DEVICES=1 gpu-runner python src/train.py
```
This forces the container to only expose physical GPU `1` to the training script, which PyTorch will then bind to as its default `cuda:0` device.

---

## 5. Architectural Recommendations (Next Steps)

For the next phases of development (as described in `NSTAR_PHASES.md`), the following enhancements are recommended to improve GPU efficiency and robustness:

1. **Add Code-Level Device Configuration:**
   Introduce a `device` key in `configs/train.yaml` (e.g., `device: "cuda:0"` or `device: "cuda:1"`) and support a `--device` CLI override in `src/train.py`. This will eliminate reliance on environment variables for targeting specific cards.
   
2. **Implement PyTorch DDP for Scaled Training:**
   If dataset size or sequences scale in Phase 2/3, implement `DistributedDataParallel` in `src/train.py`. DDP is highly efficient and standard for multi-GPU training.
   
3. **Add GPU Telemetry Tracking in W&B:**
   Add system telemetry metrics (GPU temperature, memory usage, and processor load) in W&B dashboard configurations. This helps identify memory bottlenecks (e.g., when scaling the dynamic feedforward dimension during Optuna sweeps).
