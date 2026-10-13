#!/bin/bash

# ==============================================================================
# Experiment Execution Script (Spatial & Quality Tracks)
# ==============================================================================
# Usage:
#   ./scripts/run_execution/run_experiments.sh [-t <TRACK>] [-c <CONFIG_FILE>] [-g <GPUS>] [-r <RUNNER>] [-d|-f]
#
# Options:
#   -t  Track/experiment type: 'spatial' or 'quality' (default: auto-detected from config or 'spatial').
#   -c  Path to config file (default: configs/spatial/spatial_full.yaml or configs/quality/quality_full_v2.yaml).
#       Shortcuts: 'smoke', 'full'.
#   -g  GPUs to run on, comma-separated (optional, e.g. '0' or '0,1').
#   -r  Docker Compose service/runner (default: 'gpu-runner'; use 'cpu-runner' for local CPU execution).
#   -d  Run in detached mode (background via docker compose -d; DEFAULT).
#   -f  Run in foreground mode (attached).
#   -n  Dry run (print the exact docker command without executing).
#   -h  Show this help message and exit.
#
# Examples:
#   1. Run full Chapter 6 spatial experiment on GPU 0 in detached mode (default):
#      ./scripts/run_execution/run_experiments.sh -t spatial -g 0
#
#   2. Run spatial smoke test on CPU runner:
#      ./scripts/run_execution/run_experiments.sh -t spatial -c smoke -r cpu-runner
#
#   3. Run quality experiment on GPU 1:
#      ./scripts/run_execution/run_experiments.sh -t quality -c configs/quality/quality_full_v2.yaml -g 1
#
#   4. Run with foreground logging:
#      ./scripts/run_execution/run_experiments.sh -t spatial -f
#
#   5. Dry-run to inspect the docker command:
#      ./scripts/run_execution/run_experiments.sh -t spatial -g 0 -n
# ==============================================================================

set -e

# Default settings
TRACK=""
CONFIG_ARG=""
GPUS=""
RUNNER="gpu-runner"
DETACH=true
DRY_RUN=false

# Parse command line arguments
while getopts "t:c:g:r:dfnh" opt; do
  case ${opt} in
    t ) TRACK=$OPTARG ;;
    c ) CONFIG_ARG=$OPTARG ;;
    g ) GPUS=$OPTARG ;;
    r ) RUNNER=$OPTARG ;;
    d ) DETACH=true ;;
    f ) DETACH=false ;;
    n ) DRY_RUN=true ;;
    h )
        echo "Usage: $0 [-t spatial|quality] [-c config_file] [-g gpus] [-r runner] [-d|-f] [-n]"
        echo "  -t  Track: 'spatial' or 'quality' (default: auto-detect or 'spatial')"
        echo "  -c  Config file path or shortcut ('smoke', 'full')"
        echo "  -g  Comma-separated GPU indices (e.g. 0 or 0,1)"
        echo "  -r  Docker runner service (default: gpu-runner; alternative: cpu-runner)"
        echo "  -d  Run in detached mode (default: true)"
        echo "  -f  Run in foreground mode"
        echo "  -n  Dry run (print commands without executing)"
        exit 0 ;;
    \? )
        echo "Usage: $0 [-t spatial|quality] [-c config_file] [-g gpus] [-r runner] [-d|-f] [-n]"
        exit 1 ;;
  esac
done

# Auto-detect track if not explicitly passed
if [ -z "$TRACK" ]; then
    if [[ "$CONFIG_ARG" =~ quality ]]; then
        TRACK="quality"
    elif [[ "$CONFIG_ARG" =~ spatial ]]; then
        TRACK="spatial"
    else
        TRACK="spatial"
    fi
fi

TRACK=$(echo "$TRACK" | tr '[:upper:]' '[:lower:]')

# Resolve configuration file
CONFIG_FILE=""
case "$TRACK" in
    spatial)
        SCRIPT_PATH="scripts/run_spatial_experiments.py"
        case "$CONFIG_ARG" in
            ""|full)
                CONFIG_FILE="configs/spatial/spatial_full.yaml" ;;
            smoke)
                CONFIG_FILE="configs/spatial/spatial_smoke.yaml" ;;
            *)
                CONFIG_FILE="$CONFIG_ARG" ;;
        esac
        ;;
    quality)
        SCRIPT_PATH="scripts/run_quality_experiments.py"
        case "$CONFIG_ARG" in
            ""|full)
                CONFIG_FILE="configs/quality/quality_full_v2.yaml" ;;
            smoke)
                CONFIG_FILE="configs/quality/quality_smoke.yaml" ;;
            *)
                CONFIG_FILE="$CONFIG_ARG" ;;
        esac
        ;;
    *)
        echo "Error: Unknown track '$TRACK'. Must be 'spatial' or 'quality'."
        exit 1
        ;;
esac

echo "=============================================================================="
echo "Experiment Configuration"
echo "=============================================================================="
echo "Track:       $TRACK"
echo "Script:      $SCRIPT_PATH"
echo "Config file: $CONFIG_FILE"
echo "Runner:      $RUNNER"
echo "Detached:    $DETACH"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file '$CONFIG_FILE' does not exist on host (verifying it exists inside container)."
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Experiment script '$SCRIPT_PATH' does not exist."
    exit 1
fi

# Resolve configuration base name for container naming
CONFIG_BASE=$(basename "$CONFIG_FILE" | sed 's/\.yaml//' | sed 's/\.yml//' | tr '_' '-')
CONTAINER_NAME="${TRACK}-${CONFIG_BASE}-$(date +%Y%m%d-%H%M%S)"

# Construct the inner command to run inside the container
INNER_CMD="python $SCRIPT_PATH --config $CONFIG_FILE"

DOCKER_ENV_OPTS=()
if [ -n "$GPUS" ]; then
    DOCKER_ENV_OPTS+=("-e" "CUDA_VISIBLE_DEVICES=$GPUS")
    echo "Target GPU(s): $GPUS"
fi
echo "Container:   $CONTAINER_NAME"
echo "Inner cmd:   $INNER_CMD"
echo "=============================================================================="

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute:"
    if [ "$DETACH" = true ]; then
        echo "docker compose run ${DOCKER_ENV_OPTS[*]} -d --rm --name \"$CONTAINER_NAME\" \"$RUNNER\" $INNER_CMD"
        echo "sleep 1"
        echo "docker logs -f \"$CONTAINER_NAME\""
    else
        echo "docker compose run ${DOCKER_ENV_OPTS[*]} --rm --name \"$CONTAINER_NAME\" \"$RUNNER\" $INNER_CMD"
    fi
    exit 0
fi

if [ "$DETACH" = true ]; then
    echo "Starting $TRACK experiment inside $RUNNER in detached mode..."

    # Run container in background (-d) and auto-remove (--rm) on completion
    docker compose run "${DOCKER_ENV_OPTS[@]}" -d --rm --name "$CONTAINER_NAME" "$RUNNER" $INNER_CMD

    echo ""
    echo "Container started successfully in the background."
    echo "You can safely close this terminal session; the experiment will continue running."
    echo ""
    echo "To monitor or re-attach later, run:"
    echo "  docker logs -f $CONTAINER_NAME"
    echo ""
    echo "To view all running experiment containers:"
    echo "  docker ps --filter \"name=${TRACK}-\""
    echo ""
    echo "Opening logs now (Press Ctrl+C to detach; container will keep running)..."
    echo "=============================================================================="

    # Sleep briefly to let the container initialize
    sleep 1

    # Stream logs to the terminal
    docker logs -f "$CONTAINER_NAME"
else
    echo "Running $TRACK experiment inside $RUNNER in foreground..."
    docker compose run "${DOCKER_ENV_OPTS[@]}" --rm --name "$CONTAINER_NAME" "$RUNNER" $INNER_CMD
fi
