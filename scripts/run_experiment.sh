#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="${1:-}"
OUTPUT_CSV="${2:-$PROJECT_ROOT/benchmark-results/convolution-benchmark.csv}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: scripts/run_experiment.sh <input.bmp-or-dir> [output.csv]" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"
mkdir -p "$PROJECT_ROOT/plots"

(
  cd "$PROJECT_ROOT"
  ./gradlew benchmarkConvolution --args="$INPUT_PATH --output-csv $OUTPUT_CSV"
)

python3 "$PROJECT_ROOT/scripts/plot_benchmarks.py" "$OUTPUT_CSV" "$PROJECT_ROOT/plots"

echo "CSV: $OUTPUT_CSV"
echo "Plots: $PROJECT_ROOT/plots"
