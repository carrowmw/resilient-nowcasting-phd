# W&B Artifacts Implementation Plan

This plan details the transition to using W&B as the primary model registry and artifact store, decoupling the secure training server from the local development environment.

## Overview: Local Storage vs. Cloud Access
To answer your question: **You do not need to permanently sync or store all models locally.** 

When you use a model for inference or exporting locally, PyTorch/TensorFlow inherently requires the file to be present on the machine's disk to load it into memory. However, W&B handles this seamlessly via a **temporary local cache**:
1. You query the model you want via code (e.g., "give me version 4 of the model").
2. W&B downloads *only that specific file* into a hidden cache directory (usually `~/.cache/wandb/artifacts/`).
3. You load the model directly from that cache into your script.
4. You can safely clear this cache at any time to free up local space, and your models will remain safe in the W&B cloud.

---

## Phase 1: Server-Side Pipeline (Uploading)
**Objective**: Push models and logs directly to W&B cloud during the training loop.

**Steps for the Implementer:**
1. Locate the logic in the training script where the final model checkpoints (e.g., `.pt` files) and logs are saved.
2. Initialize a W&B Artifact. It is highly recommended to only upload the "best" or "final" checkpoints to preserve the 200GB limit, rather than every single epoch.
3. Add the files to the artifact and log it to W&B.

**Example Implementation:**
```python
import wandb

# ... (training finishes, best_model.pt is saved to disk) ...

# 1. Create the Artifact object
model_artifact = wandb.Artifact(
    name="iot-resilient-model", # Name of the model
    type="model",
    description="Training checkpoints and logs"
)

# 2. Attach only the essential files to save cloud space
model_artifact.add_file("models/best_model.pt") 
# model_artifact.add_dir("logs/") # Uncomment if you also want to upload all logs

# 3. Log it to the cloud
wandb.log_artifact(model_artifact)

# W&B will automatically handle the upload in the background before the script exits.
```

## Phase 2: Local-Side Scripts (Downloading On-Demand)
**Objective**: Update local scripts (like `export_ui_payload.py` or evaluation scripts) to dynamically fetch the model from W&B instead of expecting it to exist in the local `models/` directory.

**Steps for the Implementer:**
1. Replace local file path references with the W&B API fetch logic.
2. Load the model directly from the returned W&B cache path.

**Example Implementation:**
```python
import wandb
import torch

def load_model_from_wandb(artifact_name="iot-resilient-model:latest"):
    # 1. Initialize the W&B API (does not start a new run)
    api = wandb.Api()
    
    # 2. Query the artifact. You can specify a version or 'latest'
    print(f"Fetching {artifact_name} from W&B...")
    artifact = api.artifact(f"your_wandb_entity/your_project_name/{artifact_name}")
    
    # 3. Download to local cache. If it's already cached, this happens instantly.
    artifact_dir = artifact.download()
    
    # 4. Load the model from the cache path
    model_path = f"{artifact_dir}/best_model.pt"
    model = torch.load(model_path)
    
    return model
```

```
