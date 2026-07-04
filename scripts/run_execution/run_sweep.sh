#!/bin/bash

# ==============================================================================
# Sweep Execution Script
# ==============================================================================
# Usage:
#   ./scripts/run_execution/run_sweep.sh -s <SWEEP_CONFIG> [-g <GPUS>] [-n <N_TRIALS>] [-c <BASE_CONFIG>] [-d]
#
# Options:
#   -s  Sweep config type or path (required). Shortcuts: '01', '02', '03', '04', 'arch', 'optim', 'reg', 'taskp'
#   -g  GPUs to run on, comma-separated (optional, e.g. '0' or '0,1').
#   -n  Number of Optuna trials to run (default: 20).
#   -c  Path to base config file (default: 'configs/train/train.yaml').
#   -d  Run in detached mode (background via nohup).
#       Logs will be saved to a timestamped file in the 'logs/' directory.
#
# Examples:
#   1. Run the architecture sweep on GPU 0 in the foreground:
#      ./scripts/run_execution/run_sweep.sh -s 01 -g 0
#
#   2. Run the optimization sweep on GPUs 0 and 1 in the background:
#      ./scripts/run_execution/run_sweep.sh -s 02 -g 0,1 -d -n 50
# ==============================================================================

# Default values
CONFIG_FILE="configs/train/train.yaml"
SWEEP_INPUT=""
GPUS=""
N_TRIALS=20
DETACH=false
STORAGE="sqlite:///optuna.db"
STUDY_NAME=""

# Parse command line arguments
while getopts "s:g:n:c:t:dh" opt; do
  case ${opt} in
    s ) SWEEP_INPUT=$OPTARG ;;
    g ) GPUS=$OPTARG ;;
    n ) N_TRIALS=$OPTARG ;;
    c ) CONFIG_FILE=$OPTARG ;;
    t ) STUDY_NAME=$OPTARG ;;
    d ) DETACH=true ;;
    h ) echo "Usage: $0 -s <sweep_config> [-g gpus] [-n n_trials] [-c base_config] [-t study_name] [-d]"
        echo "  -s  Sweep config name/shortcut or path (required)"
        echo "  -g  Comma-separated GPU indices (e.g. 0,1)"
        echo "  -n  Number of trials to run (default: 20)"
        echo "  -c  Base config file path (default: configs/train/train.yaml)"
        echo "  -t  Optuna study name (optional, defaults to sweep_<config_name>)"
        echo "  -d  Run in detached mode (background)"
        exit 0 ;;
    \? ) echo "Usage: $0 -s <sweep_config> [-g gpus] [-n n_trials] [-c base_config] [-t study_name] [-d]"
         exit 1 ;;
  esac
done

if [ -z "$SWEEP_INPUT" ]; then
    echo "Error: Sweep config specification (-s) is required."
    echo "Choose one of: 01 (arch), 02 (optim), 03 (reg), 04 (taskp), or a full path."
    exit 1
fi

# Resolve sweep config file shortcut or path
SWEEP_CONFIG=""
case "$SWEEP_INPUT" in
    01|arch|01_arch)
        SWEEP_CONFIG="configs/sweeps/01_arch_sweep.yaml"
        ;;
    02|optim|02_optim)
        SWEEP_CONFIG="configs/sweeps/02_optim_sweep.yaml"
        ;;
    03|reg|03_reg)
        SWEEP_CONFIG="configs/sweeps/03_reg_sweep.yaml"
        ;;
    04|taskp|04_taskp)
        SWEEP_CONFIG="configs/sweeps/04_taskp_sweep.yaml"
        ;;
    sweep)
        SWEEP_CONFIG="configs/sweeps/sweep.yaml"
        ;;
    *)
        # Treat as file path
        if [ -f "$SWEEP_INPUT" ]; then
            SWEEP_CONFIG="$SWEEP_INPUT"
        else
            # Try to see if it exists in configs/sweeps
            if [ -f "configs/sweeps/${SWEEP_INPUT}" ]; then
                SWEEP_CONFIG="configs/sweeps/${SWEEP_INPUT}"
            elif [ -f "configs/sweeps/${SWEEP_INPUT}_sweep.yaml" ]; then
                SWEEP_CONFIG="configs/sweeps/${SWEEP_INPUT}_sweep.yaml"
            else
                echo "Error: Could not resolve sweep config from: $SWEEP_INPUT"
                exit 1
            fi
        fi
        ;;
esac

echo "Resolved sweep configuration: $SWEEP_CONFIG"
echo "Base config: $CONFIG_FILE"
echo "Target trials: $N_TRIALS"

# Resolve configuration base name for logging and container naming
CONFIG_BASE=$(basename "$SWEEP_CONFIG" | sed 's/\.yaml//' | sed 's/\.yml//' | tr '_' '-')
# Clean container name to only have safe alphanumeric/hyphen characters
CONTAINER_NAME="sweep-${CONFIG_BASE}-${N_TRIALS}t-$(date +%Y%m%d-%H%M%S)"

# Construct default study name matching the stable naming convention (e.g. sweep_01_arch) if not provided
if [ -z "$STUDY_NAME" ]; then
    STUDY_NAME="sweep_${CONFIG_BASE//-/_}"
fi

# Construct the inner command to run inside the container
INNER_CMD="python src/tune.py --config $CONFIG_FILE --sweep-config $SWEEP_CONFIG --n-trials $N_TRIALS --storage $STORAGE --study-name $STUDY_NAME"

DOCKER_ENV_OPTS=()
if [ -n "$GPUS" ]; then
    DOCKER_ENV_OPTS+=("-e" "CUDA_VISIBLE_DEVICES=$GPUS")
    echo "Target GPU(s): $GPUS"
fi

if [ "$DETACH" = true ]; then
    echo "Starting sweep inside GPU runner container in detached mode..."
    echo "Container name: $CONTAINER_NAME"
    
    # Run container in background (-d) and auto-remove (--rm) on stop/completion
    docker compose run "${DOCKER_ENV_OPTS[@]}" -d --rm --name "$CONTAINER_NAME" gpu-runner $INNER_CMD
    
    echo ""
    echo "Container started successfully in the background."
    echo "Opening logs (Press Ctrl+C to disconnect from logs; the sweep container will continue running)..."
    echo "=============================================================================="
    
    # Sleep briefly to let the container initialize
    sleep 1
    
    # Stream logs to the terminal
    docker logs -f "$CONTAINER_NAME"
else
    echo "Running sweep inside GPU runner container in foreground..."
    docker compose run "${DOCKER_ENV_OPTS[@]}" --rm --name "$CONTAINER_NAME" gpu-runner $INNER_CMD
fi
