#!/usr/bin/env bash
# ==============================================================================
# stop_v4_server_suite.sh
#
# Cleanly terminates the background riot-v4-suite-runner container.
# ==============================================================================

set -euo pipefail

CONTAINER_NAME="riot-v4-suite-runner"

if [ -z "$(docker ps -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "No active running container named '${CONTAINER_NAME}' was found."
    exit 0
fi

echo "Stopping container '${CONTAINER_NAME}'..."
docker stop "${CONTAINER_NAME}"
echo "[✓] Container '${CONTAINER_NAME}' stopped."
