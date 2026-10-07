#!/usr/bin/env bash
# ==============================================================================
# start_scaling_suite.sh
# Launches the 5-run Spatial Sparsity Scaling experiment suite inside the
# active Docker container in the background.
# ==============================================================================
set -euo pipefail

export PATH="${PATH}:/usr/local/bin:/opt/homebrew/bin"
CONTAINER_NAME="ml-agent-workspace-agent-workspace-run-4146bd3b7c63"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs/scaling configs/train/scaling

echo "========================================================================"
echo " Launching Spatial Sparsity Scaling Suite inside Docker (${CONTAINER_NAME})"
echo " Configs: configs/train/scaling/ (N = 15, 25, 35, 45, 57)"
echo " Log file: logs/scaling/suite.log"
echo "========================================================================"

docker exec -d -e PYTHONPATH=/workspace/src -e CUDA_VISIBLE_DEVICES="" "${CONTAINER_NAME}" \
    bash -c "python3 scripts/scaling/run_scaling_experiments.py > logs/scaling/suite.log 2>&1"

echo "[✓] Suite launched successfully in the background!"
echo "To monitor live logs, run:"
echo "  tail -f logs/scaling/suite.log"
