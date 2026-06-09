# Context for Architect: UI Payload Generation and W&B Model Ingestion

This document details the spatiotemporal IoT prediction pipeline's frontend static payload generation system, its ability to connect to Weights & Biases (W&B) for remote model checkpoints, and execution options.

---

## 1. System for Generating the UI Payload

The static data bridge between the PyTorch Backend and the Next.js Frontend is governed by the script [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py).

### Output Architecture
Rather than writing a single massive monolithic file, [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py) shards the data for performance:
1. **Master Index**: Written to [master_index.json](file:///Users/administrator/Code/ml-agent-workspace/frontend/public/data/master_index.json). It contains top-level metadata, PIT calibration distributions, zero-inflation reliability metrics, and active node arrays.
2. **Sharded Node Details**: Written to [nodes/node_{id}.json](file:///Users/administrator/Code/ml-agent-workspace/frontend/public/data/nodes/) (one per node ID). Each file contains:
   - Ground-truth series (length 96)
   - Imputed series (length 96)
   - ZINB parameters ($\mu$, $\theta$, $\sigma$, and $\pi$ zero-inflation probability sequence)
   - Ego-graph directed edge lists ranked by spatial attention.

---

## 2. Model Loading and W&B Integration

The exporter script can load model weights from two distinct sources:
1. **Local Filesystem Checkpoint**: Specified via `--checkpoint` (defaults to `models/resilient-iot/feasible-capybara-22/best_model.pt`).
2. **W&B Remote Artifact**: Specified via `--artifact`. When provided, this flag overrides the local checkpoint path.

### Execution Command with W&B
If the desired model is hosted on the W&B server and not available locally, you do **not** need to copy it manually. The script integrates with the W&B API to pull and cache it dynamically.

Run the script inside the Docker container using:
```bash
docker compose exec agent-workspace python src/export_ui_payload.py --artifact <entity>/<project>/<artifact_name>:<version>
```
*Example:*
```bash
docker compose exec agent-workspace python src/export_ui_payload.py --artifact my-team/resilient-iot/resilient-iot-model:latest
```

### High-Level Code Logic: `load_model_from_wandb`

The implementation in [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py#L126-L144) is structured as follows:

```python
def load_model_from_wandb(artifact_name: str) -> Tuple[str, str, str]:
    """
    Downloads a model artifact from W&B.
    Returns (checkpoint_path, config_path, wandb_run_id).
    """
    api = wandb.Api()
    logger.info(f"PHASE 2: Fetching artifact {artifact_name} from W&B...")
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    checkpoint_path = os.path.join(artifact_dir, "best_model.pt")
    config_path = os.path.join(artifact_dir, "config.yaml")

    # Get the run ID from the artifact's producer
    wandb_run_id = "unknown"
    if artifact.producer:
        wandb_run_id = artifact.producer.id

    return checkpoint_path, config_path, wandb_run_id
```

### Flow Walkthrough
1. **Instantiation**: `wandb.Api()` initializes a connection using the active shell's W&B credentials.
2. **Download**: `artifact.download()` fetches the remote artifact bundle (containing `best_model.pt` and `config.yaml`) and stores it locally under a default cache directory (typically `./artifacts/` or `~/.cache/wandb/`).
3. **Paths Resolution**: It extracts the absolute path to `best_model.pt` and the co-located `config.yaml` to ensure hyperparameters are dynamically matched during the exporter's initialization (Artifact Twinning).
4. **ID Tracking**: The producer run's ID is retrieved to inject `wandb_run_id` into the output JSON metadata for full lineage auditability.
