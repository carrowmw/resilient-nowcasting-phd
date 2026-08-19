#!/bin/bash

# ==============================================================================
# Train Execution Script
# ==============================================================================
# Usage:
#   ./scripts/run_execution/run_train.sh [-c <CONFIG_FILE>] [-g <GPUS>] [-d] [-m]
#
# Options:
#   -c  Path to config file (default: 'configs/train/train.yaml').
#   -g  GPUs to run on, comma-separated (optional, e.g. '0' or '0,1').
#   -d  Run in detached mode (background via docker -d).
#   -m  Use mock data instead of real parquet data.
#
# Examples:
#   1. Run training on GPU 0 in the foreground:
#      ./scripts/run_execution/run_train.sh -g 0
#
#   2. Run training on GPUs 0 and 1 in the background:
#      ./scripts/run_execution/run_train.sh -g 0,1 -d
# ==============================================================================

# Default values
CONFIG_FILE="configs/train/ablation/spatial_int/train_ablation_old_arch.yaml"
GPUS="3"
DETACH=true
USE_MOCK_DATA=false

# Parse command line arguments
while getopts "c:g:dmh" opt; do
  case ${opt} in
    c ) CONFIG_FILE=$OPTARG ;;
    g ) GPUS=$OPTARG ;;
    d ) DETACH=true ;;
    m ) USE_MOCK_DATA=true ;;
    h ) echo "Usage: $0 [-c config_file] [-g gpus] [-d] [-m]"
        echo "  -c  Config file path (default: configs/train/train.yaml)"
        echo "  -g  Comma-separated GPU indices (e.g. 0,1)"
        echo "  -d  Run in detached mode (background)"
        echo "  -m  Use mock data"
        exit 0 ;;
    \? ) echo "Usage: $0 [-c config_file] [-g gpus] [-d] [-m]"
         exit 1 ;;
  esac
done

echo "Config file: $CONFIG_FILE"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file $CONFIG_FILE does not exist. (Assuming it will be valid inside the container or mounted)"
fi

# Resolve configuration base name for container naming
CONFIG_BASE=$(basename "$CONFIG_FILE" | sed 's/\.yaml//' | sed 's/\.yml//' | tr '_' '-')
# Clean container name to only have safe alphanumeric/hyphen characters
CONTAINER_NAME="train-${CONFIG_BASE}-$(date +%Y%m%d-%H%M%S)"

# Construct the inner command to run inside the container
INNER_CMD="python src/train.py --config $CONFIG_FILE"
if [ "$USE_MOCK_DATA" = true ]; then
    INNER_CMD="$INNER_CMD --use_mock_data"
fi

DOCKER_ENV_OPTS=()
if [ -n "$GPUS" ]; then
    DOCKER_ENV_OPTS+=("-e" "CUDA_VISIBLE_DEVICES=$GPUS")
    echo "Target GPU(s): $GPUS"
fi

if [ "$DETACH" = true ]; then
    echo "Starting training inside GPU runner container in detached mode..."
    echo "Container name: $CONTAINER_NAME"

    # Run container in background (-d) and auto-remove (--rm) on stop/completion
    docker compose run "${DOCKER_ENV_OPTS[@]}" -d --rm --name "$CONTAINER_NAME" gpu-runner $INNER_CMD

    echo ""
    echo "Container started successfully in the background."
    echo "Opening logs (Press Ctrl+C to disconnect from logs; the training container will continue running)..."
    echo "=============================================================================="

    # Sleep briefly to let the container initialize
    sleep 1

    # Stream logs to the terminal
    docker logs -f "$CONTAINER_NAME"
else
    echo "Running training inside GPU runner container in foreground..."
    docker compose run "${DOCKER_ENV_OPTS[@]}" --rm --name "$CONTAINER_NAME" gpu-runner $INNER_CMD
fi
