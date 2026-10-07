#!/usr/bin/env bash
# ==============================================================================
# start_scaling_sweep.sh
#
# Launches the sensor-density scaling sweep (configs/train/v5_scaling/, from
# scripts/riotv5/generate_scaling_configs.py) in its own detached gpu-runner
# container. Same behaviour as start_v5_suite.sh: re-running re-attaches or
# resumes; completed runs (logs/v5_scaling/status/) are skipped.
#
# Usage:
#   ./scripts/riotv5/start_scaling_sweep.sh                    # GPUs 0-3
#   ./scripts/riotv5/start_scaling_sweep.sh --gpus 0 1 --jobs-per-gpu 2
#   ./scripts/riotv5/start_scaling_sweep.sh --gpus 0 1 2 3 --dry-run
# Afterwards:
#   python scripts/riotv5/analyse_scaling.py
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ ! -f configs/train/v5_scaling/suite_manifest.yaml ]; then
    python3 scripts/riotv5/generate_scaling_configs.py
fi

export V5_CONTAINER_NAME="riot-v5-scaling-runner"
export V5_MANIFEST="configs/train/v5_scaling/suite_manifest.yaml"
export V5_LOGS_DIR="logs/v5_scaling"
exec "${SCRIPT_DIR}/start_v5_suite.sh" "$@"
