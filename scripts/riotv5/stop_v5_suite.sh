#!/usr/bin/env bash
# ==============================================================================
# stop_v5_suite.sh
#
# Stops the background riot-v5-suite-runner container. The dispatcher forwards
# SIGTERM to the active training processes. Runs in flight are lost; completed
# runs are kept and skipped when start_v5_suite.sh is launched again.
# ==============================================================================

set -euo pipefail

CONTAINER_NAME="${V5_CONTAINER_NAME:-riot-v5-suite-runner}"

if [ -z "$(docker ps -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "No running container named '${CONTAINER_NAME}' was found."
    exit 0
fi

echo "Stopping container '${CONTAINER_NAME}'..."
docker stop --time 60 "${CONTAINER_NAME}"
echo "[✓] Container '${CONTAINER_NAME}' stopped."
