#!/bin/bash
#
# Run full benchmark suite comparing baseline vs optimized pipelines
#
# Usage:
#   ./scripts/run_benchmarks.sh [--quick]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/results"

# Default config path
CONFIG="${CONFIG:-configs/longlive_inference.yaml}"

# Parse args
QUICK_FLAG=""
if [[ "$1" == "--quick" ]]; then
    QUICK_FLAG="--quick"
    echo "Running in QUICK mode (reduced samples)"
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "LongLive Latency Benchmark Suite"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

# Run comparison benchmark
echo "Running comparison benchmark (baseline vs optimized)..."
python "$PROJECT_DIR/benchmarks/benchmark_suite.py" \
    --config "$CONFIG" \
    --compare \
    --preset balanced \
    --output "$OUTPUT_DIR" \
    $QUICK_FLAG

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files yet)"
