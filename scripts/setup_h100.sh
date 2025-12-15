#!/bin/bash
#
# Lambda Labs H100 Setup Script for LongLive-Optimized
#
# This script sets up the environment on a Lambda Labs H100 instance:
# 1. Installs system dependencies
# 2. Sets up Python environment with PyTorch 2.x
# 3. Installs flash-attention and other requirements
# 4. Downloads model weights
# 5. Verifies GPU setup
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

# Check if running on Lambda Labs
if ! nvidia-smi | grep -q "H100"; then
    echo "Warning: This script is optimized for H100 GPUs"
fi

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Update system
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6

# Check CUDA version
echo ""
echo "CUDA Information:"
nvcc --version

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv ~/.venv/longlive
source ~/.venv/longlive/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.x
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install flash-attention
echo ""
echo "Installing flash-attention (this may take a few minutes)..."
pip install flash-attn --no-build-isolation

# Install LongLive requirements
echo ""
echo "Installing LongLive requirements..."
pip install \
    accelerate \
    diffusers \
    transformers \
    omegaconf \
    einops \
    timm \
    lpips \
    clip \
    opencv-python \
    imageio \
    imageio-ffmpeg \
    wandb \
    tensorboard \
    pyyaml \
    tqdm \
    ftfy \
    regex

# Install additional optimization requirements
pip install \
    numpy \
    scipy \
    pillow \
    matplotlib

# Clone LongLive if not in repo directory
if [ ! -f "inference.py" ]; then
    echo ""
    echo "Cloning LongLive repository..."
    cd ..
    git clone https://github.com/NVlabs/LongLive.git
    cd LongLive
fi

# Download model weights
echo ""
echo "Downloading model weights..."
echo "Note: You may need to manually download from HuggingFace if authentication is required"

# Create weights directory
mkdir -p checkpoints

# Download Wan2.1-T2V-1.3B (base model)
echo "Downloading Wan2.1-T2V-1.3B..."
if [ ! -d "checkpoints/Wan2.1-T2V-1.3B" ]; then
    pip install huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-T2V-1.3B',
    local_dir='checkpoints/Wan2.1-T2V-1.3B',
    local_dir_use_symlinks=False,
)
print('Wan2.1-T2V-1.3B downloaded successfully')
"
fi

# Download LongLive checkpoint
echo "Downloading LongLive checkpoint..."
if [ ! -d "checkpoints/LongLive-1.3B" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvlabs/LongLive-1.3B',
    local_dir='checkpoints/LongLive-1.3B',
    local_dir_use_symlinks=False,
)
print('LongLive-1.3B downloaded successfully')
"
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
sys.path.append('.')
from optimizations import OptimizationConfig, LatencyProfiler

config = OptimizationConfig.preset_balanced()
profiler = LatencyProfiler()

print('Optimizations module loaded successfully!')
print(f'Config: {config}')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source ~/.venv/longlive/bin/activate"
echo ""
echo "To run benchmarks:"
echo "  python benchmarks/benchmark_suite.py --config configs/longlive_inference.yaml"
echo ""
echo "To compare baseline vs optimized:"
echo "  python benchmarks/benchmark_suite.py --config configs/longlive_inference.yaml --compare"
echo ""
