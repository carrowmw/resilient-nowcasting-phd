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

# Parse command line arguments
while getopts "s:g:n:c:dh" opt; do
  case ${opt} in
    s ) SWEEP_INPUT=$OPTARG ;;
    g ) GPUS=$OPTARG ;;
    n ) N_TRIALS=$OPTARG ;;
    c ) CONFIG_FILE=$OPTARG ;;
    d ) DETACH=true ;;
    h ) echo "Usage: $0 -s <sweep_config> [-g gpus] [-n n_trials] [-c base_config] [-d]"
        echo "  -s  Sweep config name/shortcut or path (required)"
        echo "  -g  Comma-separated GPU indices (e.g. 0,1)"
        echo "  -n  Number of trials to run (default: 20)"
        echo "  -c  Base config file path (default: configs/train/train.yaml)"
        echo "  -d  Run in detached mode (background)"
        exit 0 ;;
    \? ) echo "Usage: $0 -s <sweep_config> [-g gpus] [-n n_trials] [-c base_config] [-d]"
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

# Construct the run command
CMD="python src/tune.py --config $CONFIG_FILE --sweep-config $SWEEP_CONFIG --n-trials $N_TRIALS --storage $STORAGE"

if [ -n "$GPUS" ]; then
    CMD="$CMD --gpus $GPUS"
    echo "Target GPU(s): $GPUS"
fi

# Create a logs directory if it doesn't exist
mkdir -p logs

if [ "$DETACH" = true ]; then
    # Create a unique log file name based on sweep config name
    CONFIG_BASE=$(basename "$SWEEP_CONFIG" | sed 's/\.yaml//' | sed 's/\.yml//')
    LOG_FILE="logs/sweep_${CONFIG_BASE}_$(date +%Y%m%d_%H%M%S).log"
    echo "Running in detached mode. Logs will be saved to $LOG_FILE"
    
    # Execute the command in the background
    nohup $CMD > "$LOG_FILE" 2>&1 &
    
    echo "Sweep process started with PID: $!"
else
    echo "Running in foreground..."
    $CMD
fi
