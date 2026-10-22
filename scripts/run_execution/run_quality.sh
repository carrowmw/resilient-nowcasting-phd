#!/bin/bash
# Convenience wrapper for running quality experiments
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_experiments.sh" -t quality "$@"
