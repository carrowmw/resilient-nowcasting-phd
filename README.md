# ML Agent Workspace

## Build Instructions

To build the agent workspace container on your local machine, run the following command in the terminal:

```Bash
docker compose build agent-workspace
```

This will build the container that you will use for development on your Mac.

```Bash
docker compose build cpu-runner
```

When you are ready to execute your training script on the GPU server, you will need to build the CUDA image. Run the following command on the server:

```Bash
docker compose build gpu-runner
```

## Execution Instructions

### On Mac (Development):

You will boot the interactive terminal exactly as you have been doing to let the agent code:

```Bash
docker compose run --rm agent-workspace bash
```

### On the GPU Server (Execution):

Once you have pushed your main branch to GitHub and pulled it down to the server, you will run the GPU container to execute the W&B training script:

```Bash
docker compose run --rm gpu-runner python src/process_all.py
```

```Bash
# This builds the CUDA image and runs your training script
docker compose run --rm gpu-runner python src/train.py --epochs 50 --batch_size 128
```

```Bash
# This maps the GPU, mounts the repo, and runs exactly one month of data
docker compose run --rm gpu-runner python src/train.py \
  --epochs 2 \
  --batch_size 128 \
  --start_date "2023-03-02" \
  --end_date "2023-04-02"
```

## Unified Architecture Configuration

The training config (`configs/train/train.yaml`, mirrored at `epochs: 1` in
`configs/train/train_test.yaml`) exposes the knobs for the three unified-architecture layers. All
three are **active by default** in the production config.

| Key | Default | Meaning |
|-----|---------|---------|
| `data.spatial_holdout_prob` | `0.2` | Fraction of nodes spatially held out each step for the kriging head to reconstruct (IGNNK-style). `0.0` disables spatial holdout. |
| `loss.w_kriging` | `0.5` | Weight of the spatial-kriging reconstruction loss in the joint objective. `0.0` bypasses the kriging head entirely (no compute, no gradient). |
| `loss.w_impute` | `1.0` | Weight of the imputation loss (traffic MSLE + weather MSE). |
| `loss.w_forecast` | `1.0` | Weight of the ZINB forecasting NLL. |
| `model.edge_features_cache_path` | `data/external/edge_features_cache.json` | Path to the precomputed OSM edge-feature cache feeding the GATv2 `edge_attr` (9 physical + 1 learned → `edge_dim=10`). Empty/omitted → learned-only attention (`edge_dim=1`). |

**Loss note.** The traffic channel of both the imputation and kriging losses uses **TRUE MSLE on raw
counts** — the prediction is inverse-transformed from the `StandardScaler` domain back to raw counts
(via registered `traffic_mean`/`traffic_std` buffers) before `log1p`, rather than a clamped
"pseudo-MSLE" on scaled features. Weather channels use MSE on scaled values. Rationale and trade-offs:
[`docs/adr/0001-eliminate-pseudo-msle.md`](docs/adr/0001-eliminate-pseudo-msle.md).

Run a config directly (database mode):

```Bash
docker compose exec agent-workspace python src/train.py --use-database --config configs/train/train.yaml
```

## Hyperparameter Tuning (Optuna)

The tuning entry point is `src/tune.py`. It launches an Optuna study and logs trial metrics to Weights & Biases through `WeightsAndBiasesCallback`.

The search space is defined in `configs/sweep.yaml`:

- `lr` (log-uniform)
- `w_forecast` (uniform)
- `w_impute` (uniform)

### Local CPU Smoke Sweep (Mock Data)

Use this first to validate the full tuning loop quickly on your Mac:

```Bash
docker compose run --rm cpu-runner python src/tune.py --use_mock_data
```

If you want to keep runs local-only and avoid syncing to W&B cloud:

```Bash
docker compose run --rm cpu-runner bash -lc 'WANDB_MODE=offline python src/tune.py --use_mock_data'
```

### GPU Server Sweep (Real Data)

Run this on the GPU server after pulling the latest code and setting your `.env`:

```Bash
docker compose run --rm gpu-runner python src/tune.py
```

Recommended explicit run (select your exact data window and stream):

```Bash
docker compose run --rm gpu-runner python src/tune.py \
  --n_trials 2 \
  --epochs 2 \
  --batch_size 4 \
  --start_date 2023-10-01 \
  --end_date 2023-10-04 \
  --veh_class person \
  --dir all
```

### Logs and Outputs

- Sweep logs are written to `logs/optuna_sweep_YYYYMMDD_HHMMSS.log`
- Best parameters and best validation loss are printed at the end of the run

### Tune CLI Options

`src/tune.py` now supports CLI overrides so you do not need to edit source code for most sweep changes:

- `--n_trials`
- `--epochs`
- `--batch_size`
- `--start_date`
- `--end_date`
- `--seq_len`
- `--veh_class`
- `--dir`
- `--use_mock_data`

Show all options:

```Bash
docker compose run --rm cpu-runner python src/tune.py --help
```

### Troubleshooting: Empty Data Window / Filter Errors

If your selected `--start_date`, `--end_date`, `--veh_class`, or `--dir` produces no rows, tuning will fail fast with a clear error explaining which filter removed all rows and which values were available in the loaded data. Adjust the date window or stream arguments and rerun.

## W&B Local Storage Management

When using Weights & Biases Artifacts for model storage, W&B will download models to a local cache (usually `~/.cache/wandb/artifacts/`) when they are requested by inference or export scripts. To ensure your local environment doesn't run out of space over time, you can periodically run the following command to delete older cached models (this will *not* delete them from the cloud):

```Bash
# Keep only the last 5GB of downloaded artifacts locally
wandb artifact cache cleanup 5GB
```
