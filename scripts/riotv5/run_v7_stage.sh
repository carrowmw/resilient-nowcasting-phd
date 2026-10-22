#!/usr/bin/env bash
# ==============================================================================
# run_v7_stage.sh
#
# Driver for the v7 experiment stage (docs/experiments/riot_v6/riot_v6_seed_report.md
# §9.2, recommendations 1-6). Runs INSIDE the gpu-runner container (start_v7_stage.sh
# launches it); safe to run on a bare GPU host too. Steps, in recommendation order:
#
#   rec 1  tier 1: V6 base vs static-ridge profile, 6 seeds, 57 training stations
#   rec 2  pre-check on the saved v6 checkpoints (posthoc_v7_eval.py valcheck, one
#          process per GPU, then valgate). Tier 2 (8 stratified validation stations)
#          is trained only if validation-station RMSE tracks held-out RMSE.
#   rec 3  tier 3: V6 + block masks, V6 + static-ridge + block masks
#   rec 4  tier 4: weather extremes at the target step vs at the forecast origin
#   rec 5  tier 5: V6 base and static-ridge at a 150-epoch budget
#          (tiers [2] 3 4 5 run in one suite call; no job starts that would overrun)
#   rec 6  stop-list audit of every trained config, then collection into
#          docs/experiments/riot_v7/data/
#
# A failed step is logged and the driver moves on. Re-running resumes: completed
# training runs and existing valcheck files are skipped.
#
# Usage:
#   bash scripts/riotv5/run_v7_stage.sh --gpus 0 1 2 3 --hours 118
#   bash scripts/riotv5/run_v7_stage.sh --preflight --gpus 0      # mock-data check only
#   options: --rec2-gate auto|run|skip   (auto = run tier 2 only if the pre-check passes)
#            --skip-train
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

MANIFEST="configs/train/v7/suite_manifest.yaml"
LOGS="logs/v7_stage"
DATA_OUT="docs/experiments/riot_v7/data"
GATE_JSON="${LOGS}/rec2_gate.json"
# Time reserved at the end of the window for the rec 6 audit and collection (hours).
RESERVE_HOURS=0.5
# Upper bound on the rec 2 pre-check (W&B downloads + inference of ~72 v6 runs).
VALCHECK_HOURS=4

GPUS=(0 1 2 3)
HOURS=""
PREFLIGHT=0
SKIP_TRAIN=0
REC2_GATE="auto"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) shift; GPUS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do GPUS+=("$1"); shift; done ;;
        --hours) HOURS="$2"; shift 2 ;;
        --preflight) PREFLIGHT=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --rec2-gate) REC2_GATE="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "${REC2_GATE}" in auto|run|skip) ;; *) echo "--rec2-gate must be auto, run or skip" >&2; exit 2 ;; esac

mkdir -p "${LOGS}"
STAGE_START=$(date +%s)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# As the container's PID 1, bash ignores SIGTERM by default: forward it to the children
# (run_v5_suite.py then stops its training processes cleanly) and exit.
CHILDREN=()
on_term() {
    log "termination signal — stopping child processes"
    for c in ${CHILDREN[@]+"${CHILDREN[@]}"}; do kill -TERM "${c}" 2>/dev/null || true; done
    wait
    exit 143
}
trap on_term TERM INT
# Run a command as a tracked child and wait for it (keeps the trap responsive).
run_child() { "$@" & local pid=$!; CHILDREN+=("${pid}"); wait "${pid}"; }
# Hours left for training: window - reserve - elapsed (empty when no --hours).
hours_left() {
    [ -n "${HOURS}" ] || return 0
    python3 -c "print(round(max(0.1, ${HOURS} - ${RESERVE_HOURS} - ($(date +%s) - ${STAGE_START}) / 3600), 2))"
}
# Run the rec 2 pre-check: valcheck on every GPU in parallel, then valgate. Returns
# the valgate exit code (0 pass, 1 fail, 2 undecided).
rec2_precheck() {
    local out_dir="$1"; shift
    local -a extra=("$@")
    local -a pids=()
    local n=${#GPUS[@]} i
    for i in $(seq 0 $((n - 1))); do
        CUDA_VISIBLE_DEVICES="${GPUS[$i]}" timeout --signal=TERM "${VALCHECK_HOURS}h" \
            python3 scripts/riotv5/posthoc_v7_eval.py valcheck --shard "${i}" --num-shards "${n}" \
            --out-dir "${out_dir}" ${VALCHECK_ARGS[@]+"${VALCHECK_ARGS[@]}"} \
            > "${out_dir}/valcheck_shard${i}.log" 2>&1 &
        pids+=($!); CHILDREN+=($!)
    done
    for p in "${pids[@]}"; do wait "${p}" || log "rec 2: a valcheck shard exited non-zero (see ${out_dir}/valcheck_shard*.log)"; done
    python3 scripts/riotv5/posthoc_v7_eval.py valgate --out-dir "${out_dir}" ${extra[@]+"${extra[@]}"}
}

# ------------------------------------------------------------------------------
# Preflight: every code path of the stage on the synthetic city (minutes, no DB).
# ------------------------------------------------------------------------------
if [ "${PREFLIGHT}" -eq 1 ]; then
    set -e
    export WANDB_MODE=disabled
    log "preflight 1/5: unit tests"
    python3 -m pytest -q tests/test_v5.py tests/test_v6.py tests/test_v7.py
    log "preflight 2/5: generate + validate every v7 config"
    python3 scripts/riotv5/generate_v7_configs.py --check
    log "preflight 3/5: mock smoke of every v7 experiment (seed 42, 2 epochs, 12 nodes)"
    python3 scripts/riotv5/run_v5_suite.py --manifest "${MANIFEST}" --use_mock_data --smoke --seeds 42 \
        --gpus "${GPUS[0]}" --stagger 0 --force
    python3 - <<'EOF'
import json, glob, sys
bad = [json.load(open(f))["name"] for f in glob.glob("logs/v7_stage_mock/status/*-s42.json")
       if json.load(open(f))["status"] != "COMPLETED"]
n = len(glob.glob("logs/v7_stage_mock/status/*-s42.json"))
print(f"mock smoke: {n - len(bad)}/{n} completed" + (f" — failed: {bad}" if bad else ""))
sys.exit(1 if bad or n == 0 else 0)
EOF
    log "preflight 4/5: mock rec 2 pre-check (valcheck on the mock v7 runs + valgate)"
    mkdir -p "${LOGS}_mock"
    rm -rf "${LOGS}_mock/valcheck"
    GPUS=("${GPUS[0]}")
    VALCHECK_ARGS=(--use_mock_data --models-dir models/local --manifest "${MANIFEST}" --seeds 42)
    set +e
    rec2_precheck "${LOGS}_mock" --min-runs 4 --min-count 1 --count 2 --min-coverage 0.0 --draws 50
    rc=$?
    set -e
    n_val=$(ls "${LOGS}_mock/valcheck/"*.json 2>/dev/null | wc -l)
    log "mock valcheck wrote ${n_val} run files; valgate exit ${rc}"
    [ "${n_val}" -ge 4 ] && [ "${rc}" -le 2 ] || { log "preflight: rec 2 pre-check FAILED"; exit 1; }
    log "preflight 5/5: rec 6 stop-list audit"
    python3 scripts/riotv5/generate_v7_configs.py --audit "${LOGS}_mock/rec6_stop_list.json"
    log "preflight PASSED"
    exit 0
fi

python3 scripts/riotv5/generate_v7_configs.py || { log "config generation failed"; exit 1; }
VALCHECK_ARGS=()

# ------------------------------------------------------------------------------
# rec 1: static-ridge confirmation (tier 1)
# ------------------------------------------------------------------------------
suite() {
    local -a args=(--manifest "${MANIFEST}" --gpus "${GPUS[@]}" --tiers "$@")
    local left; left=$(hours_left)
    [ -n "${left}" ] && args+=(--stop-after-hours "${left}")
    log "training — run_v5_suite.py ${args[*]}"
    run_child python3 scripts/riotv5/run_v5_suite.py "${args[@]}" \
        || log "suite exited non-zero (see ${LOGS}/suite_*.log)"
}
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    log "rec 1: V6 base vs static-ridge profile, seeds 42-47, 57 training stations"
    suite 1
fi

# ------------------------------------------------------------------------------
# rec 2: validation-station pre-check on the v6 checkpoints, then the gate
# ------------------------------------------------------------------------------
RUN_TIER2=0
case "${REC2_GATE}" in
    run)  log "rec 2: gate overridden (--rec2-gate run) — tier 2 will be trained"; RUN_TIER2=1 ;;
    skip) log "rec 2: skipped (--rec2-gate skip)" ;;
    auto)
        log "rec 2: pre-check on the saved v6 checkpoints (logs -> ${LOGS}/valcheck_shard*.log)"
        rec2_precheck "${LOGS}"
        rc=$?
        if [ "${rc}" -eq 0 ]; then
            log "rec 2: gate PASSED — training the stratified-validation tier"; RUN_TIER2=1
        elif [ "${rc}" -eq 1 ]; then
            log "rec 2: gate FAILED — validation stations dropped; select on the usual criterion"
        else
            log "rec 2: gate UNDECIDED (exit ${rc}) — tier 2 skipped (see ${GATE_JSON}; --rec2-gate run forces it)"
        fi
        ;;
esac

# ------------------------------------------------------------------------------
# rec 3-5: block masks, lagged weather, 150-epoch budget (tiers [2] 3 4 5)
# ------------------------------------------------------------------------------
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    if [ "${RUN_TIER2}" -eq 1 ]; then
        log "rec 2-5: tiers 2 3 4 5"
        suite 2 3 4 5
    else
        log "rec 3-5: tiers 3 4 5"
        suite 3 4 5
    fi
fi

# ------------------------------------------------------------------------------
# rec 6: stop-list audit, then collect
# ------------------------------------------------------------------------------
mkdir -p "${DATA_OUT}"
python3 scripts/riotv5/generate_v7_configs.py --audit "${DATA_OUT}/rec6_stop_list.json" \
    || log "rec 6: stop-list audit FAILED (see ${DATA_OUT}/rec6_stop_list.json)"
[ -f "${GATE_JSON}" ] && cp "${GATE_JSON}" "${DATA_OUT}/"
python3 scripts/riotv5/run_v5_suite.py --manifest "${MANIFEST}" --summarise-only >/dev/null 2>&1 \
    || log "collect: summarise failed"
python3 scripts/riotv5/collect_v5_results.py --manifest "${MANIFEST}" --out "${DATA_OUT}" \
    || log "collect: collect_v5_results failed"
ELAPSED=$(python3 -c "print(round(($(date +%s) - ${STAGE_START}) / 3600, 2))")
log "v7 stage finished in ${ELAPSED} h. Results: ${LOGS}/v5_suite_summary.csv, ${DATA_OUT}/ (runs.csv, rec2_gate.json, rec6_stop_list.json)"
