#!/bin/bash
# generate_all_payloads.sh — Phase 1 batch payload generation (Option B).
#
# Generates paginated UI payloads for ALL W&B model artifacts, then builds the
# run registry. Run inside the agent-workspace container:
#     bash scripts/generate_all_payloads.sh
#
# Serving path (DEC-1, Phase 3): generated runs/ are git-ignored and uploaded to
# Cloudflare R2; the frontend fetches from R2_PUBLIC_URL. Set R2_UPLOAD=0 to skip upload.
set -euo pipefail

CONFIG="${CONFIG:-configs/analysis/analysis.yaml}"     # D-1: user's existing config
OUTPUT_DIR="${OUTPUT_DIR:-frontend/public/data/runs}"
R2_UPLOAD="${R2_UPLOAD:-1}"

echo "[generate_all_payloads] config=$CONFIG output_dir=$OUTPUT_DIR r2_upload=$R2_UPLOAD"

python src/export_paginated_payload.py \
    --all_artifacts \
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
