#!/bin/bash
# generate_all_payloads.sh — RIoT v5 batch payload generation.
#
# Generates paginated UI payloads for the top-k v5 runs (the v5_full-s42 reference
# + the best TOP_K-1 by validation RMSE, see export_paginated_payload.py), then
# builds the run registry. Set RUNS="run-a run-b" to export an explicit list.
#     bash scripts/generate_all_payloads.sh
#
# Serving path (DEC-1, Phase 3): generated runs/ are git-ignored and uploaded to
# Cloudflare R2; the frontend fetches from R2_PUBLIC_URL. Set R2_UPLOAD=0 to skip upload.
set -euo pipefail

CONFIG="${CONFIG:-configs/analysis/analysis.yaml}"     # D-1: user's existing config
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/payload_staging/runs}"
R2_UPLOAD="${R2_UPLOAD:-1}"
TOP_K="${TOP_K:-10}"
RUNS="${RUNS:-}"

echo "[generate_all_payloads] config=$CONFIG output_dir=$OUTPUT_DIR top_k=$TOP_K runs=${RUNS:-<top-k>} r2_upload=$R2_UPLOAD"

SELECT=(--top_k "$TOP_K")
if [ -n "$RUNS" ]; then
    # shellcheck disable=SC2206
    SELECT=(--runs $RUNS)
fi

python src/export_paginated_payload.py \
    "${SELECT[@]}" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

python scripts/build_run_registry.py "$OUTPUT_DIR"

echo "[generate_all_payloads] Local generation done."
du -sh "$OUTPUT_DIR"

if [ "$R2_UPLOAD" = "1" ]; then
    echo "[generate_all_payloads] Uploading to Cloudflare R2 …"
    python scripts/upload_to_r2.py --runs-dir "$OUTPUT_DIR" --prefix runs
fi

echo "[generate_all_payloads] Done."
