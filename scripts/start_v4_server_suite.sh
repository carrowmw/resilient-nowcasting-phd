#!/usr/bin/env bash
# ==============================================================================
# start_v4_server_suite.sh
#
# Launches the RIoT v4 multi-GPU batch execution suite inside the
# docker compose gpu-runner container in a detached background state,
# and immediately streams the live output.
#
# NOTE: Pressing Ctrl+C while viewing logs detaches safely.
#       The container continues executing unattended in the background.
#
# Usage:
#   ./scripts/start_v4_server_suite.sh                 # Default: runs on GPUs 0 and 1
#   ./scripts/start_v4_server_suite.sh --gpus 0        # Run on GPU 0 only
#   ./scripts/start_v4_server_suite.sh --jobs riot-v4-full-inductive
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

CONTAINER_NAME="riot-v4-suite-runner"

# Resolve docker compose binary
if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "ERROR: Neither 'docker compose' nor 'docker-compose' found on PATH." >&2
    exit 1
fi

# If already running, seamlessly attach to active logs
if [ -n "$(docker ps -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "==================================================================="
    echo " [!] Suite container '${CONTAINER_NAME}' is ALREADY RUNNING."
    echo " Attaching to live stream (Press Ctrl+C to detach safely)..."
    echo "==================================================================="
    exec docker logs -f "${CONTAINER_NAME}"
fi

# Clean up stopped/exited container from a previous completed run
if [ -n "$(docker ps -a -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "[*] Cleaning up exited container '${CONTAINER_NAME}' from previous run..."
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# Default to GPU 0 and 1 if no arguments provided
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
    ARGS=(--gpus 0 1)
fi

echo "==================================================================="
echo " Starting RIoT v4 Multi-GPU Suite in Detached Docker Container"
echo " Container Name: ${CONTAINER_NAME}"
echo " Service:        gpu-runner"
echo " Suite Args:     ${ARGS[*]}"
echo " Working Dir:    ${WORKSPACE_ROOT}"
echo "==================================================================="

# Launch gpu-runner in detached mode (-d)
${COMPOSE} run \
    -d \
    --name "${CONTAINER_NAME}" \
    gpu-runner \
    python3 scripts/run_v4_server_suite.py "${ARGS[@]}"

echo ""
echo "[✓] Container '${CONTAINER_NAME}' started successfully in background."
echo ""
echo "Helpful commands:"
echo "  • Re-attach to logs:  docker logs -f ${CONTAINER_NAME}"
echo "  • Companion viewer:   ./scripts/show_v4_logs.sh"
echo "  • Stop execution:     ./scripts/stop_v4_server_suite.sh"
echo "  • View summary CSV:   cat logs/v4_suite/v4_suite_summary.csv"
echo "  • View LaTeX Table:   cat logs/v4_suite/tab_exp3_ablation.tex"
echo ""
echo "-------------------------------------------------------------------"
echo " Streaming live logs below (Press Ctrl+C to detach without stopping)"
echo "-------------------------------------------------------------------"
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
