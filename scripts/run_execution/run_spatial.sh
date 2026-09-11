#!/bin/bash
# Convenience wrapper for running Chapter 6 spatial experiments
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_experiments.sh" -t spatial "$@"
