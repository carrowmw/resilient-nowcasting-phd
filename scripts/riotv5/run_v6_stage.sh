#!/usr/bin/env bash
# ==============================================================================
# run_v6_stage.sh
#
# Driver for the v6 experiment stage (seed report §14.2, recommendations 1-7). Runs
# INSIDE the gpu-runner container (start_v6_stage.sh launches it); safe to run on a
# bare GPU host too. Steps:
#
#   1. rec 1-6 training: configs/train/v6/suite_manifest.yaml through run_v5_suite.py,
#             tiers 1 -> 6, no job started that would overrun the compute window
#   2. post-hoc, in parallel, one process per GPU (in the reserved last RESERVE_HOURS):
#             rec 6  GNN + LSTM ensembles (v6_base + v6_lstm, per seed, both checkpoints)
#             rec 7  legacy-metric check on saved v5 checkpoints (v4r_full / v4r_bs32, 3 seeds)
#             (after training, not alongside it: every process holds a copy of the dataset
#             and train.py aborts a run above 90 % host RAM)
#   3. collect: docs/experiments/riot_v6/data/{runs,epochs}.csv + logs/v6_stage summaries
#
# A failed step is logged and the driver moves on (a weekend run should not die on a
# post-hoc evaluation). Re-running resumes: completed training runs are skipped.
#
# Usage:
#   bash scripts/riotv5/run_v6_stage.sh --gpus 0 1 2 3 --hours 75
#   bash scripts/riotv5/run_v6_stage.sh --preflight --gpus 0      # CPU/GPU mock-data check only
#   options: --skip-legacy --skip-train --skip-ensemble --tiers 1 2 ...
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

MANIFEST="configs/train/v6/suite_manifest.yaml"
LOGS="logs/v6_stage"
POSTHOC="${LOGS}/posthoc"
DATA_OUT="docs/experiments/riot_v6/data"
SEEDS=(42 43 44)
# Time reserved at the end of the window for the ensembles and collection (hours).
RESERVE_HOURS=1.5

GPUS=(0 1 2 3)
HOURS=""
TIERS=()
PREFLIGHT=0
SKIP_LEGACY=0
SKIP_TRAIN=0
SKIP_ENSEMBLE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) shift; GPUS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do GPUS+=("$1"); shift; done ;;
        --tiers) shift; TIERS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do TIERS+=("$1"); shift; done ;;
        --hours) HOURS="$2"; shift 2 ;;
        --preflight) PREFLIGHT=1; shift ;;
        --skip-legacy) SKIP_LEGACY=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --skip-ensemble) SKIP_ENSEMBLE=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${LOGS}" "${POSTHOC}"
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
status_ok() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$1'))['status']=='COMPLETED' else 1)" 2>/dev/null; }

# ------------------------------------------------------------------------------
# Preflight: every code path of the stage on the synthetic city (minutes, no DB).
# ------------------------------------------------------------------------------
if [ "${PREFLIGHT}" -eq 1 ]; then
    set -e
    export WANDB_MODE=disabled
    log "preflight 1/5: unit tests"
    python3 -m pytest -q tests/test_v5.py tests/test_v6.py
    log "preflight 2/5: generate + validate every v6 config"
    python3 scripts/riotv5/generate_v6_configs.py --check
    log "preflight 3/5: mock smoke of every v6 experiment (seed 42, 2 epochs, 12 nodes)"
    python3 scripts/riotv5/run_v5_suite.py --manifest "${MANIFEST}" --use_mock_data --smoke --seeds 42 \
        --gpus "${GPUS[0]}" --stagger 0 --force
    python3 - <<'EOF'
import json, glob, sys
bad = [json.load(open(f))["name"] for f in glob.glob("logs/v6_stage_mock/status/*-s42.json")
       if json.load(open(f))["status"] != "COMPLETED"]
n = len(glob.glob("logs/v6_stage_mock/status/*-s42.json"))
print(f"mock smoke: {n - len(bad)}/{n} completed")
sys.exit(1 if bad or n == 0 else 0)
EOF
    log "preflight 4/5: mock ensemble (v6_base + v6_lstm)"
    python3 scripts/riotv5/posthoc_v6_eval.py ensemble --use_mock_data --models-dir models/local \
        --runs v6_base-s42 v6_lstm-s42 --name preflight_ens --out-dir "${LOGS}_mock/posthoc"
    python3 scripts/riotv5/posthoc_v6_eval.py ensemble --use_mock_data --models-dir models/local \
        --runs v6_base-s42 v6_lstm-s42 --name preflight_ens_ind --checkpoint ind --out-dir "${LOGS}_mock/posthoc"
    log "preflight 5/5: mock legacy-metric check (v4r_full / v4r_bs32 smoke runs)"
    python3 scripts/riotv5/run_v5_suite.py --use_mock_data --smoke --jobs v4r_full-s42 v4r_bs32-s42 \
        --gpus "${GPUS[0]}" --stagger 0 --force
    python3 scripts/riotv5/posthoc_v6_eval.py legacy --use_mock_data --models-dir models/local \
        --runs v4r_full-s42 v4r_bs32-s42 --name preflight_legacy --out-dir "${LOGS}_mock/posthoc"
    log "preflight PASSED"
    exit 0
fi

python3 scripts/riotv5/generate_v6_configs.py || { log "config generation failed"; exit 1; }

# ------------------------------------------------------------------------------
# 1. training (rec 1-6)
# ------------------------------------------------------------------------------
if [ "${SKIP_TRAIN}" -eq 0 ]; then
    SUITE_ARGS=(--manifest "${MANIFEST}" --gpus "${GPUS[@]}")
    [ ${#TIERS[@]} -gt 0 ] && SUITE_ARGS+=(--tiers "${TIERS[@]}")
    if [ -n "${HOURS}" ]; then
        TRAIN_HOURS=$(python3 -c "print(max(0.1, ${HOURS} - ${RESERVE_HOURS} - ($(date +%s) - ${STAGE_START}) / 3600))")
        SUITE_ARGS+=(--stop-after-hours "${TRAIN_HOURS}")
    fi
    log "step 1: training — run_v5_suite.py ${SUITE_ARGS[*]}"
    run_child python3 scripts/riotv5/run_v5_suite.py "${SUITE_ARGS[@]}" \
        || log "step 1: suite exited non-zero (see ${LOGS}/suite_*.log)"
fi

# ------------------------------------------------------------------------------
# 2. post-hoc: ensembles (rec 6) and the legacy-metric check (rec 7), one GPU each
# ------------------------------------------------------------------------------
PIDS=()
i=0
if [ "${SKIP_LEGACY}" -eq 0 ] && [ ! -f "${POSTHOC}/legacy_v4r_bs32_vs_full.json" ]; then
    gpu="${GPUS[$((${#GPUS[@]} - 1))]}"
    log "step 2: legacy-metric check on GPU ${gpu} -> ${POSTHOC}/legacy.log"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/riotv5/posthoc_v6_eval.py legacy \
        --project resilient-iot-v5 --name v4r_bs32_vs_full --batch-sizes 32 128 \
        --runs v4r_full-s42 v4r_full-s43 v4r_full-s44 v4r_bs32-s42 v4r_bs32-s43 v4r_bs32-s44 \
        > "${POSTHOC}/legacy.log" 2>&1 &
    PIDS+=($!)
    CHILDREN+=($!)
fi
if [ "${SKIP_ENSEMBLE}" -eq 0 ]; then
    for seed in "${SEEDS[@]}"; do
        a="v6_base-s${seed}"; b="v6_lstm-s${seed}"
        if status_ok "${LOGS}/status/${a}.json" && status_ok "${LOGS}/status/${b}.json"; then
            gpu="${GPUS[$((i % ${#GPUS[@]}))]}"; i=$((i + 1))
            log "step 2: ensemble ${a} + ${b} on GPU ${gpu}"
            (
                for ck in best ind; do
                    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/riotv5/posthoc_v6_eval.py ensemble \
                        --project resilient-iot-v6 --runs "${a}" "${b}" --name "v6_ens_${ck}-s${seed}" --checkpoint "${ck}" \
                        || echo "ensemble ${ck} s${seed} FAILED"
                done
            ) > "${POSTHOC}/ensemble-s${seed}.log" 2>&1 &
            PIDS+=($!)
            CHILDREN+=($!)
        else
            log "step 2: skip seed ${seed} (v6_base / v6_lstm not both completed)"
        fi
    done
fi
for p in ${PIDS[@]+"${PIDS[@]}"}; do wait "${p}" || log "step 2: a post-hoc process failed (see ${POSTHOC}/*.log)"; done
log "step 2: post-hoc evaluations done (${POSTHOC}/)"

# ------------------------------------------------------------------------------
# 3. collect
# ------------------------------------------------------------------------------
python3 scripts/riotv5/run_v5_suite.py --manifest "${MANIFEST}" --summarise-only >/dev/null 2>&1 \
    || log "step 3: summarise failed"
python3 scripts/riotv5/collect_v5_results.py --manifest "${MANIFEST}" --out "${DATA_OUT}" \
    || log "step 3: collect failed"
ELAPSED=$(python3 -c "print(round(($(date +%s) - ${STAGE_START}) / 3600, 2))")
log "v6 stage finished in ${ELAPSED} h. Results: ${LOGS}/v5_suite_summary.csv, ${DATA_OUT}/runs.csv, ${POSTHOC}/"
