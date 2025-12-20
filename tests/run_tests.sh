#!/bin/bash
# Run all tests for LongLive-Optimized
#
# Usage:
#   ./tests/run_tests.sh          # Run CPU tests only
#   ./tests/run_tests.sh --gpu    # Include GPU tests
#   ./tests/run_tests.sh --all    # Run everything

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "LongLive-Optimized Test Suite"
echo "========================================"
echo ""

# Parse arguments
RUN_GPU=false
RUN_ALL=false

for arg in "$@"; do
    case $arg in
        --gpu)
            RUN_GPU=true
            shift
            ;;
        --all)
            RUN_ALL=true
            RUN_GPU=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--gpu] [--all]"
            echo ""
            echo "Options:"
            echo "  --gpu   Include GPU integration tests"
            echo "  --all   Run all tests including slow ones"
            exit 0
            ;;
    esac
done

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

echo "Python: $(python3 --version)"
echo ""

# Check pytest
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "Installing pytest..."
    pip install pytest
fi

# Run CPU integration tests
echo "========================================"
echo "Running CPU Integration Tests"
echo "========================================"
python3 -m pytest tests/test_integration.py -v --tb=short

# Run GPU tests if requested
if [ "$RUN_GPU" = true ]; then
    echo ""
    echo "========================================"
    echo "Running GPU Integration Tests"
    echo "========================================"

    # Check CUDA availability
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        python3 -m pytest tests/test_gpu_integration.py -v --tb=short
    else
        echo "CUDA not available - skipping GPU tests"
    fi
fi

# Run all tests including slow ones
if [ "$RUN_ALL" = true ]; then
    echo ""
    echo "========================================"
    echo "Running All Tests"
    echo "========================================"
    python3 -m pytest tests/ -v --tb=short
fi

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
