#!/bin/bash
set -e

# Validate environment variables
if [ -z "${GITHUB_REPOSITORY}" ]; then
  echo "Error: GITHUB_REPOSITORY environment variable is not set."
  exit 1
fi

if [ -z "${GITHUB_TOKEN}" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  exit 1
fi

# Determine organization or repo URL
RUNNER_URL="https://github.com/${GITHUB_REPOSITORY}"
RUNNER_NAME="${RUNNER_NAME:-gpu-docker-runner}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,gpu,linux,payload-builder}"

echo "Configuring GitHub Actions Runner for url=${RUNNER_URL}..."
./config.sh --url "${RUNNER_URL}" \
            --token "${GITHUB_TOKEN}" \
            --name "${RUNNER_NAME}" \
            --labels "${RUNNER_LABELS}" \
            --work _work \
            --unattended \
            --replace

# Clean up configuration on shutdown
cleanup() {
    echo "Removing runner..."
    ./config.sh remove --token "${GITHUB_TOKEN}"
}
trap cleanup SIGINT SIGQUIT SIGTERM

echo "Starting runner..."
./run.sh
