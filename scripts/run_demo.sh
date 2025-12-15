#!/bin/bash
# Run the interactive WebRTC demo with LongLive optimizations
#
# Prerequisites:
# 1. Models downloaded (see setup_h100.sh)
# 2. scope-optimized repository cloned
# 3. Optimization preset selected
#
# Usage:
#   ./scripts/run_demo.sh                    # Default balanced preset
#   ./scripts/run_demo.sh speed              # Maximum speed preset
#   ./scripts/run_demo.sh quality            # Maximum quality preset
#   ./scripts/run_demo.sh balanced --profile # With latency profiling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings
PRESET="${1:-balanced}"
PROFILE=""
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        quality|balanced|speed)
            PRESET="$1"
            shift
            ;;
        --profile)
            PROFILE="--profile"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [quality|balanced|speed] [--profile] [--port PORT]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "LongLive Optimized Demo"
echo "=========================================="
echo "Preset: $PRESET"
echo "Port: $PORT"
echo "Profile: ${PROFILE:-disabled}"
echo ""

# Check for scope integration
SCOPE_DIR="$PROJECT_ROOT/demo/scope_integration"
if [ ! -d "$SCOPE_DIR" ]; then
    echo "Error: Scope integration not found at $SCOPE_DIR"
    echo "Please ensure the demo/scope_integration directory exists."
    exit 1
fi

# Check for model weights
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/longlive}"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Warning: Model directory not found at $MODEL_DIR"
    echo "Set MODEL_DIR environment variable or run setup_h100.sh first."
fi

# Activate virtual environment if present
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Export optimization preset
export LONGLIVE_PRESET="$PRESET"
export LONGLIVE_PROFILE="${PROFILE:+1}"

echo "Starting demo server..."
echo ""
echo "Access the demo at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

# Run the demo
# Option 1: If using scope server
if command -v scope &> /dev/null; then
    scope server \
        --pipeline longlive-optimized \
        --port $PORT \
        --optimization-preset "$PRESET" \
        $PROFILE
# Option 2: Standalone Python server
else
    cd "$PROJECT_ROOT"
    python -m demo.scope_integration.server \
        --port $PORT \
        --preset "$PRESET" \
        $PROFILE
fi
