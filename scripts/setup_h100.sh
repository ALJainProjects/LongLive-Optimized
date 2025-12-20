#!/bin/bash
#
# Lambda Labs H100 Setup Script for LongLive-Optimized
#
# This script sets up the environment on a Lambda Labs H100 instance:
# 1. Uses Lambda Stack's pre-installed PyTorch/CUDA (via --system-site-packages)
# 2. Installs additional requirements
# 3. Downloads model weights
# 4. Verifies GPU setup
#
# Based on Lambda Labs documentation:
# https://docs.lambdalabs.com/on-demand-cloud/managing-your-system-environment
#
# Usage:
#   chmod +x scripts/setup_h100.sh
#   ./scripts/setup_h100.sh
#

set -e  # Exit on error

echo "========================================"
echo "LongLive-Optimized Setup Script"
echo "Lambda Labs H100"
echo "========================================"

# Prevent system from sleeping/suspending during long jobs
echo ""
echo "Preventing system sleep/suspend..."
sudo systemctl mask hibernate.target hybrid-sleep.target \
    suspend-then-hibernate.target sleep.target suspend.target

# Check if running on Lambda Labs with H100
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! nvidia-smi | grep -q "H100"; then
    echo "Warning: This script is optimized for H100 GPUs"
fi

# Show Lambda Stack info
echo ""
echo "Lambda Stack Python/PyTorch:"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" 2>/dev/null || echo "PyTorch not found in system Python"

# Check CUDA version
echo ""
echo "CUDA Information:"
nvcc --version 2>/dev/null || echo "nvcc not in PATH"

# Install minimal system dependencies (most are already on Lambda Stack)
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6

# Create virtual environment with --system-site-packages
# This inherits Lambda Stack's PyTorch, CUDA, and other preinstalled packages
echo ""
echo "Creating Python virtual environment (inheriting Lambda Stack packages)..."
VENV_DIR="${HOME}/.venv/longlive-optimized"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "Activating existing environment..."
else
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "Created new virtual environment at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Upgrade pip and install compatible setuptools/packaging versions
# Note: flash-attn build requires specific versions to avoid canonicalize_version error
pip install --upgrade pip wheel
pip install 'setuptools==70.0.0' 'packaging==23.2' ninja

# Verify PyTorch CUDA (inherited from Lambda Stack)
echo ""
echo "Verifying Lambda Stack PyTorch..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Install flash-attention (if not already installed)
echo ""
echo "Checking flash-attention..."
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "flash-attention already installed"
else
    echo "Installing flash-attention (this may take a few minutes)..."
    pip install flash-attn --no-build-isolation
fi

# Install LongLive requirements (only what's not in Lambda Stack)
echo ""
echo "Installing additional requirements..."
pip install \
    omegaconf \
    einops \
    timm \
    lpips \
    ftfy \
    regex \
    imageio \
    imageio-ffmpeg \
    pyyaml \
    peft

# Install huggingface_hub for model downloads
pip install huggingface_hub

# Clone LongLive-Optimized if not in repo directory
REPO_DIR="${HOME}/LongLive-Optimized"
if [ ! -f "inference.py" ] && [ ! -f "$REPO_DIR/inference.py" ]; then
    echo ""
    echo "Cloning LongLive-Optimized repository..."
    git clone https://github.com/ALJainProjects/LongLive-Optimized.git "$REPO_DIR"
    cd "$REPO_DIR"
elif [ -f "$REPO_DIR/inference.py" ]; then
    cd "$REPO_DIR"
    echo "Repository already exists at $REPO_DIR"
else
    echo "Already in repository directory"
fi

# Download model weights
echo ""
echo "Downloading model weights..."
echo "Note: You may need to authenticate with HuggingFace for some models"

# Create weights directory
mkdir -p checkpoints

# Download Wan2.1-T2V-1.3B (base model)
echo ""
echo "Checking Wan2.1-T2V-1.3B..."
if [ ! -d "checkpoints/Wan2.1-T2V-1.3B" ]; then
    echo "Downloading Wan2.1-T2V-1.3B..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-T2V-1.3B',
    local_dir='checkpoints/Wan2.1-T2V-1.3B',
    local_dir_use_symlinks=False,
)
print('Wan2.1-T2V-1.3B downloaded successfully')
"
else
    echo "Wan2.1-T2V-1.3B already exists"
fi

# Download LongLive checkpoint
echo ""
echo "Checking LongLive-1.3B..."
if [ ! -d "checkpoints/LongLive-1.3B" ]; then
    echo "Downloading LongLive-1.3B..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='NVlabs/LongLive-1.3B',
    local_dir='checkpoints/LongLive-1.3B',
    local_dir_use_symlinks=False,
)
print('LongLive-1.3B downloaded successfully')
"
else
    echo "LongLive-1.3B already exists"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import flash_attn
print('Verification Results:')
print(f'  PyTorch: {torch.__version__}')
print(f'  Flash Attention: {flash_attn.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  CUDA Version: {torch.version.cuda}')
"

# Run a quick test
echo ""
echo "Running quick sanity check..."
python3 -c "
import torch
import sys
sys.path.insert(0, '.')
from optimizations import OptimizationConfig, LatencyProfiler

config = OptimizationConfig.preset_balanced()
profiler = LatencyProfiler()

print('Optimizations module loaded successfully!')
print(f'  torch.compile: {config.use_torch_compile}')
print(f'  Compile mode: {config.compile_mode}')
print(f'  Static KV: {config.use_static_kv}')
print(f'  Async VAE: {config.use_async_vae}')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run optimized inference:"
echo "  python inference.py --config configs/longlive_inference.yaml --optimized"
echo ""
echo "To run benchmarks:"
echo "  python benchmarks/benchmark_suite.py --config configs/longlive_inference.yaml --compare"
echo ""
