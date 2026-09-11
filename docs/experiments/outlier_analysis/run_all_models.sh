#!/usr/bin/env bash
# Batch outlier analysis across the last 12 models (back to tough-dragon-7).
# logical-surf-18 is already complete and snapshotted under outputs/runs/.
set -uo pipefail
cd "$(dirname "$0")"

RUNS=(
  b1_kriging_w2_batch_32
  lyric-jazz-19
  valiant-waterfall-13
  misunderstood-totem-16
  whole-paper-14
  rich-dust-15
  atomic-morning-12
  cool-voice-11
  magic-snow-10
  copper-mountain-9
  tough-dragon-7
)

for run in "${RUNS[@]}"; do
  if [ -f "outputs/runs/${run}/results.json" ]; then
    echo "=== SKIP ${run} (already snapshotted) ==="
    continue
  fi
  echo "=== START ${run} $(date -u +%H:%M:%S) ==="
  if python run_analysis.py --run-id "${run}" --max-workers 6; then
    mkdir -p "outputs/runs/${run}"
    cp -r outputs/report.md outputs/results.json outputs/figures outputs/tables \
      "outputs/runs/${run}/"
    echo "=== DONE ${run} $(date -u +%H:%M:%S) ==="
  else
    echo "=== FAILED ${run} $(date -u +%H:%M:%S) ==="
  fi
done
echo "=== ALL COMPLETE ==="
