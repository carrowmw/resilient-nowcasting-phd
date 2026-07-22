import os
import yaml
import wandb
import torch
import datetime as dt
from harness.holdout import make_fixed_holdout_mask, save_holdout_manifest
from db_connector import TimescaleConnector, ParquetConnector
from dataset import RealIoTDataset

def patch_run_holdout(full_artifact_name: str):
    api = wandb.Api()

    print(f"\n--- Patching {full_artifact_name} ---")
    print(f"Downloading artifact {full_artifact_name}...")
    artifact = api.artifact(full_artifact_name)
    artifact_dir = artifact.download()

    producer_run = artifact.logged_by()
    if producer_run is None:
        raise ValueError(f"Artifact {full_artifact_name} has no producer run. Cannot patch.")

    wandb_run_id = producer_run.id
    run_name = producer_run.name
    entity = producer_run.entity
    project = producer_run.project

    print(f"Resolved run: {run_name} (ID: {wandb_run_id}, Entity: {entity}, Project: {project})")

    # 2. Parse the training and data configuration
    config_path = os.path.join(artifact_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_cfg = config["training"]

    # 3. Resolve the database / data connector
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        connector = TimescaleConnector(db_url)
    else:
        connector = ParquetConnector(data_cfg.get("parquet_path", "data/traffic_20230108_20260608.parquet"))

    # 4. Load the dataset to resolve the canonical location list
    print("Loading dataset to resolve canonical locations...")
    start_date = data_cfg["start_date"]
    end_date = data_cfg["end_date"]
    seq_len = data_cfg["seq_len"]
    veh_class = data_cfg["target_veh_class"]
    direction = data_cfg["target_dir"]

    dataset = RealIoTDataset(
        connector,
        start_date=(dt.date.fromisoformat(start_date) - dt.timedelta(days=1)).isoformat(),
        end_date=end_date,
        seq_len=seq_len,
        target_veh_class=veh_class,
        target_dir=direction,
    )

    location_ids = list(dataset.locations)
    num_nodes = len(location_ids)

    # 5. Deterministically regenerate the holdout manifest
    print("Generating holdout manifest...")
    inductive_seed = train_cfg["random_seed"]
    inductive_frac = data_cfg["inductive_holdout_frac"]

    inductive_holdout_mask = make_fixed_holdout_mask(
        num_nodes=num_nodes,
        holdout_frac=inductive_frac,
        seed=inductive_seed,
        device="cpu"
    )

    manifest_path = save_holdout_manifest(
        artifact_dir, location_ids, inductive_holdout_mask,
        seed=inductive_seed, holdout_frac=inductive_frac
    )
    print(f"Generated manifest at: {manifest_path}")

    # 6. Resume the W&B run and log a new version of the artifact
    print(f"Resuming W&B run {wandb_run_id} to log patched artifact...")
    run = wandb.init(
        entity=entity,
        project=project,
        id=wandb_run_id,
        resume="must"
    )

    artifact_collection_name = full_artifact_name.split("/")[-1].split(":")[0]

    new_artifact = wandb.Artifact(
        name=artifact_collection_name,
        type="model",
        description=f"Patched model with holdout manifest for run {run_name}",
        metadata=config
    )
    new_artifact.add_file(os.path.join(artifact_dir, "best_model.pt"), name="best_model.pt")
    new_artifact.add_file(config_path, name="config.yaml")
    new_artifact.add_file(manifest_path, name="holdout_manifest.json")

    run.log_artifact(new_artifact)
    run.finish()
    print(f"Successfully patched {run_name}!")

if __name__ == "__main__":
    # Update the last two runs
    patch_run_holdout("rappel-velvet9310-newcastle-university/resilient-iot-v2/riot-full-inductive:v2")
    patch_run_holdout("rappel-velvet9310-newcastle-university/resilient-iot-v2/riot-full-inductive:v1")