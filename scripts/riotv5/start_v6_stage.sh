#!/usr/bin/env bash
# ==============================================================================
# start_v6_stage.sh
#
# Launches the v6 experiment stage (docs/experiments/riot_v5/riot_v5_seed_report.md
# §14.2, recommendations 1-7; configs from scripts/riotv5/generate_v6_configs.py) in a
# detached gpu-runner container, then streams its output. Ctrl+C detaches; the stage
# keeps running.
#
#   1. generates configs/train/v6/ (78 runs = 26 experiments x seeds 42/43/44)
#   2. PREFLIGHT in a throwaway container: unit tests, config validation, a 2-epoch
#      mock-data smoke of every experiment and of both post-hoc evaluations (~10-20 min).
#      The stage is not launched unless this passes.
#   3. launches scripts/riotv5/run_v6_stage.sh in container riot-v6-stage-runner:
#      training tiers 1 -> 6 within the compute window, then (last 1.5 h) the GNN+LSTM
#      ensembles and the legacy-metric check, then collection into docs/experiments/riot_v6/data/.
#
# Re-running while the container is up re-attaches. Re-running after a stop / crash
# resumes: completed runs (logs/v6_stage/status/) are skipped.
#
# Usage:
#   ./scripts/riotv5/start_v6_stage.sh                          # GPUs 0-3, 75 h window
#   ./scripts/riotv5/start_v6_stage.sh --gpus 0 1 2 3 --hours 70
#   ./scripts/riotv5/start_v6_stage.sh --no-preflight           # skip the mock check (resume)
#   ./scripts/riotv5/start_v6_stage.sh --dry-run                # print the job plan only
#
# Budget (estimated): tiers 1-5 ~280 GPU-h, tier 6 (seq48) ~56 GPU-h. With --hours 75 on
# 4 GPUs (~300 GPU-h) the runner starts no job that would overrun the window, so the
# seq48 arm is what gets dropped if time runs short.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKSPACE_ROOT}"

CONTAINER_NAME="riot-v6-stage-runner"
MANIFEST="configs/train/v6/suite_manifest.yaml"
GPUS=(0 1 2 3)
HOURS=75
PREFLIGHT=1
DRY_RUN=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) shift; GPUS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do GPUS+=("$1"); shift; done ;;
        --hours) HOURS="$2"; shift 2 ;;
        --no-preflight) PREFLIGHT=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) EXTRA+=("$1"); shift ;;   # passed through to run_v6_stage.sh (--skip-legacy, --tiers ...)
    esac
done

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
    echo " [!] Stage container '${CONTAINER_NAME}' is ALREADY RUNNING."
    echo " Attaching to live stream (Press Ctrl+C to detach safely)..."
    echo "==================================================================="
    exec docker logs -f "${CONTAINER_NAME}"
fi

echo "[*] Generating v6 configs..."
${COMPOSE} run --rm gpu-runner python3 scripts/riotv5/generate_v6_configs.py --check

if [ "${DRY_RUN}" -eq 1 ]; then
    exec ${COMPOSE} run --rm gpu-runner python3 scripts/riotv5/run_v5_suite.py \
        --manifest "${MANIFEST}" --gpus "${GPUS[@]}" --dry-run
fi

if [ "${PREFLIGHT}" -eq 1 ]; then
    echo "==================================================================="
    echo " PREFLIGHT: mock-data check of every v6 experiment and post-hoc step"
    echo "==================================================================="
    if ! ${COMPOSE} run --rm gpu-runner bash scripts/riotv5/run_v6_stage.sh --preflight --gpus "${GPUS[0]}"; then
        echo "ERROR: preflight failed — the stage was NOT launched. See logs/v6_stage_mock/." >&2
        exit 1
    fi
fi

if [ -n "$(docker ps -a -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "[*] Removing exited container '${CONTAINER_NAME}' from a previous run..."
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "==================================================================="
echo " Starting RIoT v6 stage in a detached container"
echo " Container: ${CONTAINER_NAME}   GPUs: ${GPUS[*]}   Window: ${HOURS} h"
echo " Started: $(date '+%a %Y-%m-%d %H:%M')   Window ends: $(date -d "+${HOURS} hours" '+%a %Y-%m-%d %H:%M' 2>/dev/null || echo "+${HOURS} h")"
echo "==================================================================="

${COMPOSE} run -d --name "${CONTAINER_NAME}" gpu-runner \
    bash scripts/riotv5/run_v6_stage.sh --gpus "${GPUS[@]}" --hours "${HOURS}" ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "[✓] Container '${CONTAINER_NAME}' started in the background."
echo ""
echo "Helpful commands:"
echo "  • Re-attach to logs:   ./scripts/riotv5/start_v6_stage.sh   (or docker logs -f ${CONTAINER_NAME})"
echo "  • One run's log:       tail -f logs/v6_stage/runs/<run-name>.log"
echo "  • Progress table:      cat logs/v6_stage/v5_suite_summary.csv"
echo "  • Stop the stage:      V5_CONTAINER_NAME=${CONTAINER_NAME} ./scripts/riotv5/stop_v5_suite.sh"
echo "  • Resume after a stop: ./scripts/riotv5/start_v6_stage.sh --no-preflight --hours <remaining>"
echo ""
echo "-------------------------------------------------------------------"
echo " Streaming live logs (Press Ctrl+C to detach without stopping)"
echo "-------------------------------------------------------------------"
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
