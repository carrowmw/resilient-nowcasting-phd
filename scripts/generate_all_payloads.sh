#!/bin/bash
# generate_all_payloads.sh — Phase 1 batch payload generation (Option B).
#
# Generates paginated UI payloads for ALL W&B model artifacts, then builds the
# run registry. Run inside the agent-workspace container:
#     bash scripts/generate_all_payloads.sh
#
# Git is the serving path (DEC-1): generated runs/ are committed so Vercel serves
# them statically. W&B --upload is opt-in/archival and is NOT enabled here.
set -euo pipefail

CONFIG="${CONFIG:-configs/analysis/analysis.yaml}"     # D-1: user's existing config
OUTPUT_DIR="${OUTPUT_DIR:-frontend/public/data/runs}"

echo "[generate_all_payloads] config=$CONFIG output_dir=$OUTPUT_DIR"

python src/export_paginated_payload.py \
    --all_artifacts \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

python scripts/build_run_registry.py "$OUTPUT_DIR"

echo "[generate_all_payloads] Done."
du -sh "$OUTPUT_DIR"
