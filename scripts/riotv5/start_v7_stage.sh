#!/usr/bin/env bash
# ==============================================================================
# start_v7_stage.sh
#
# Launches the v7 experiment stage (docs/experiments/riot_v6/riot_v6_seed_report.md
# §9.2, recommendations 1-6; configs from scripts/riotv5/generate_v7_configs.py) in a
# detached gpu-runner container, then streams its output. Ctrl+C detaches; the stage
# keeps running.
#
#   1. generates configs/train/v7/ (60 runs, 12 experiments)
#   2. PREFLIGHT in a throwaway container: unit tests, config validation, a 2-epoch
#      mock-data smoke of every experiment, the rec 2 pre-check and the rec 6 audit on
#      the mock runs (~15-25 min). The stage is not launched unless this passes.
#   3. launches scripts/riotv5/run_v7_stage.sh in container riot-v7-stage-runner:
#      rec 1 -> rec 2 pre-check / gate -> rec 2-5 training -> rec 6 audit -> collection
#      into docs/experiments/riot_v7/data/.
#
# Re-running while the container is up re-attaches. Re-running after a stop / crash
# resumes: completed runs (logs/v7_stage/status/) are skipped.
#
# Usage:
#   ./scripts/riotv5/start_v7_stage.sh                          # GPUs 0-3, 118 h window
#   ./scripts/riotv5/start_v7_stage.sh --gpus 0 1 2 3 --hours 110
#   ./scripts/riotv5/start_v7_stage.sh --rec2-gate run          # train tier 2 regardless of the pre-check
#   ./scripts/riotv5/start_v7_stage.sh --no-preflight           # skip the mock check (resume)
#   ./scripts/riotv5/start_v7_stage.sh --dry-run                # print the job plan only
#
# Budget (estimated from the v6 run times): ~258 GPU-h if the rec 2 gate passes (~219
# without tier 2), about 65-80 h wall on 4 GPUs with tier barriers, plus up to 4 h for
# the rec 2 pre-check. The 118 h window leaves ~35 h of slack for slower-than-estimated
# runs (block masks, 150-epoch budget); the runner starts no job that would overrun it.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKSPACE_ROOT}"

CONTAINER_NAME="riot-v7-stage-runner"
MANIFEST="configs/train/v7/suite_manifest.yaml"
GPUS=(0 1 2 3)
HOURS=118
PREFLIGHT=1
DRY_RUN=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) shift; GPUS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do GPUS+=("$1"); shift; done ;;
        --hours) HOURS="$2"; shift 2 ;;
        --no-preflight) PREFLIGHT=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) EXTRA+=("$1"); shift ;;   # passed through to run_v7_stage.sh (--rec2-gate, --skip-train)
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

echo "[*] Generating v7 configs..."
${COMPOSE} run --rm gpu-runner python3 scripts/riotv5/generate_v7_configs.py --check

if [ "${DRY_RUN}" -eq 1 ]; then
    exec ${COMPOSE} run --rm gpu-runner python3 scripts/riotv5/run_v5_suite.py \
        --manifest "${MANIFEST}" --gpus "${GPUS[@]}" --dry-run
fi

if [ "${PREFLIGHT}" -eq 1 ]; then
    echo "==================================================================="
    echo " PREFLIGHT: mock-data check of every v7 experiment, rec 2 pre-check, rec 6 audit"
    echo "==================================================================="
    if ! ${COMPOSE} run --rm gpu-runner bash scripts/riotv5/run_v7_stage.sh --preflight --gpus "${GPUS[0]}"; then
        echo "ERROR: preflight failed — the stage was NOT launched. See logs/v7_stage_mock/." >&2
        exit 1
    fi
fi

if [ -n "$(docker ps -a -q -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "[*] Removing exited container '${CONTAINER_NAME}' from a previous run..."
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "==================================================================="
echo " Starting RIoT v7 stage in a detached container"
echo " Container: ${CONTAINER_NAME}   GPUs: ${GPUS[*]}   Window: ${HOURS} h"
echo " Started: $(date '+%a %Y-%m-%d %H:%M')   Window ends: $(date -d "+${HOURS} hours" '+%a %Y-%m-%d %H:%M' 2>/dev/null || echo "+${HOURS} h")"
echo "==================================================================="

${COMPOSE} run -d --name "${CONTAINER_NAME}" gpu-runner \
    bash scripts/riotv5/run_v7_stage.sh --gpus "${GPUS[@]}" --hours "${HOURS}" ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "[✓] Container '${CONTAINER_NAME}' started in the background."
echo ""
echo "Helpful commands:"
echo "  • Re-attach to logs:   ./scripts/riotv5/start_v7_stage.sh   (or docker logs -f ${CONTAINER_NAME})"
echo "  • One run's log:       tail -f logs/v7_stage/runs/<run-name>.log"
echo "  • Progress table:      cat logs/v7_stage/v5_suite_summary.csv"
echo "  • Stop the stage:      V5_CONTAINER_NAME=${CONTAINER_NAME} ./scripts/riotv5/stop_v5_suite.sh"
echo "  • Resume after a stop: ./scripts/riotv5/start_v7_stage.sh --no-preflight --hours <remaining>"
echo ""
echo "-------------------------------------------------------------------"
echo " Streaming live logs (Press Ctrl+C to detach without stopping)"
echo "-------------------------------------------------------------------"
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
