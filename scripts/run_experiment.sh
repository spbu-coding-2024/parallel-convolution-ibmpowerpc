#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_PATH="${1:-}"
OUTPUT_CSV="${2:-$PROJECT_ROOT/benchmark-results/benchmark.csv}"

cd "$PROJECT_ROOT"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: ./scripts/run_experiment.sh <input.bmp> [benchmark-results/results.csv]" >&2
  exit 1
fi

mkdir -p "$(dirname "$INPUT_PATH")"
mkdir -p "$(dirname "$OUTPUT_CSV")"
mkdir -p "$PROJECT_ROOT/plots"

./gradlew benchmarkConvolution --args="$INPUT_PATH $OUTPUT_CSV"
python3 scripts/plot_benchmarks.py "$OUTPUT_CSV" "$PROJECT_ROOT/plots"
