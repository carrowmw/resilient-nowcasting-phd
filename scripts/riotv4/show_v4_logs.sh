#!/usr/bin/env bash
# ==============================================================================
# show_v4_logs.sh
#
# Attaches to and streams live logs from the running riot-v4-suite-runner container.
# Pressing Ctrl+C exits the stream without stopping the background job.
# ==============================================================================

set -euo pipefail

CONTAINER_NAME="riot-v4-suite-runner"

if [ -z "$(docker ps -a -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "No container named '${CONTAINER_NAME}' found."
    echo "Recent log files on disk (logs/v4_suite/):"
    ls -lht logs/v4_suite/ 2>/dev/null | head -n 10 || true
    exit 1
fi

echo "Attaching to logs for '${CONTAINER_NAME}' (Press Ctrl+C to detach safely)..."
exec docker logs -f "${CONTAINER_NAME}"
