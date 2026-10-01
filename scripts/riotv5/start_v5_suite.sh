#!/usr/bin/env bash
# ==============================================================================
# start_v5_suite.sh
#
# Launches the RIoT pipeline-v5 experiment suite (configs/train/v5/, 209 runs)
# on GPUs 0,1,2,3 inside a detached docker compose gpu-runner container, then
# streams its output. Ctrl+C detaches from the stream; the suite keeps running.
#
# Re-running this script while the container is up re-attaches to it. Re-running
# after a stop / crash resumes: completed runs (logs/v5_suite/status/) are skipped.
#
# Usage:
#   ./scripts/riotv5/start_v5_suite.sh                         # everything, GPUs 0-3
#   ./scripts/riotv5/start_v5_suite.sh --gpus 0 1 2 3 --tiers 1  # headline runs only
#   ./scripts/riotv5/start_v5_suite.sh --gpus 0 1 2 3 --groups thesis_v4 p7_v5_suite
#   ./scripts/riotv5/start_v5_suite.sh --gpus 0 1 2 3 --dry-run
# (any argument list replaces the default "--gpus 0 1 2 3", so pass --gpus explicitly)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKSPACE_ROOT}"

CONTAINER_NAME="riot-v5-suite-runner"
MANIFEST="configs/train/v5/suite_manifest.yaml"

if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "ERROR: Neither 'docker compose' nor 'docker-compose' found on PATH." >&2
    exit 1
fi

if [ -n "$(docker ps -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "==================================================================="
    echo " [!] Suite container '${CONTAINER_NAME}' is ALREADY RUNNING."
    echo " Attaching to live stream (Press Ctrl+C to detach safely)..."
    echo "==================================================================="
    exec docker logs -f "${CONTAINER_NAME}"
fi

if [ -n "$(docker ps -a -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "[*] Removing exited container '${CONTAINER_NAME}' from a previous run..."
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: ${MANIFEST} not found. Generate the configs first:" >&2
    echo "  python scripts/riotv5/generate_v5_configs.py --check" >&2
    exit 1
fi

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
    ARGS=(--gpus 0 1 2 3)
fi

echo "==================================================================="
echo " Starting RIoT v5 experiment suite in a detached container"
echo " Container: ${CONTAINER_NAME}   Service: gpu-runner"
echo " Suite args: ${ARGS[*]}"
echo " Workspace:  ${WORKSPACE_ROOT}"
echo "==================================================================="

${COMPOSE} run \
    -d \
    --name "${CONTAINER_NAME}" \
    gpu-runner \
    python3 scripts/riotv5/run_v5_suite.py "${ARGS[@]}"

echo ""
echo "[✓] Container '${CONTAINER_NAME}' started in the background."
echo ""
echo "Helpful commands:"
echo "  • Re-attach to logs:     ./scripts/riotv5/show_v5_logs.sh"
echo "  • One run's log:         tail -f logs/v5_suite/runs/<run-name>.log"
echo "  • Stop the suite:        ./scripts/riotv5/stop_v5_suite.sh"
echo "  • Per-experiment CSV:    logs/v5_suite/v5_suite_summary.csv"
echo "  • LaTeX tables:          logs/v5_suite/tables/<group>.tex"
echo "  • Rebuild summaries:     python scripts/riotv5/run_v5_suite.py --summarise-only"
echo ""
echo "-------------------------------------------------------------------"
echo " Streaming live logs (Press Ctrl+C to detach without stopping)"
echo "-------------------------------------------------------------------"
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
