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

# Dynamic token resolution (exchange PAT for registration token if possible)
echo "Obtaining registration token from GitHub..."
API_RESPONSE=$(curl -sX POST \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runners/registration-token")

REGISTRATION_TOKEN=$(echo "${API_RESPONSE}" | jq -r '.token // empty')

if [ -n "${REGISTRATION_TOKEN}" ] && [ "${REGISTRATION_TOKEN}" != "null" ]; then
  echo "Successfully obtained registration token using GITHUB_TOKEN."
  TOKEN_TO_USE="${REGISTRATION_TOKEN}"
else
  echo "Could not obtain registration token via API (Response: ${API_RESPONSE})."
  echo "Assuming GITHUB_TOKEN is a direct registration token..."
  TOKEN_TO_USE="${GITHUB_TOKEN}"
fi

echo "Configuring GitHub Actions Runner for url=${RUNNER_URL}..."
./config.sh --url "${RUNNER_URL}" \
            --token "${TOKEN_TO_USE}" \
            --name "${RUNNER_NAME}" \
            --labels "${RUNNER_LABELS}" \
            --work _work \
            --unattended \
            --replace

# Clean up configuration on shutdown
cleanup() {
    echo "Removing runner..."
    # Try to fetch a removal token using the GITHUB_TOKEN
    REMOVE_RESPONSE=$(curl -sX POST \
      -H "Authorization: token ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runners/remove-token")
    REMOVE_TOKEN=$(echo "${REMOVE_RESPONSE}" | jq -r '.token // empty')

    if [ -n "${REMOVE_TOKEN}" ] && [ "${REMOVE_TOKEN}" != "null" ]; then
      ./config.sh remove --token "${REMOVE_TOKEN}"
    else
      echo "Failed to get removal token via API. Trying fallback..."
      ./config.sh remove --token "${TOKEN_TO_USE}"
    fi
}
trap cleanup SIGINT SIGQUIT SIGTERM

echo "Starting runner..."
./run.sh
