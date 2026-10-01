#!/usr/bin/env bash
# ==============================================================================
# show_v5_logs.sh
#
# Streams the dispatcher output of the riot-v5-suite-runner container.
# Ctrl+C exits the stream without stopping the suite.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

CONTAINER_NAME="riot-v5-suite-runner"

if [ -z "$(docker ps -a -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "No container named '${CONTAINER_NAME}' found."
    echo "Most recent run logs on disk (logs/v5_suite/runs/):"
    ls -lht logs/v5_suite/runs/ 2>/dev/null | head -n 10 || true
    exit 1
fi

echo "Attaching to logs for '${CONTAINER_NAME}' (Press Ctrl+C to detach safely)..."
exec docker logs -f "${CONTAINER_NAME}"
