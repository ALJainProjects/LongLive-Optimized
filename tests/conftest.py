"""
Pytest configuration and shared fixtures for LongLive-Optimized tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip CUDA tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def cuda_device():
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def dtype():
    """Default dtype for tests."""
    return torch.bfloat16


@pytest.fixture
def sample_latents(cuda_device, dtype):
    """Create sample latent tensor."""
    return torch.randn(
        1, 16, 1, 60, 104,
        device=cuda_device,
        dtype=dtype
    )


@pytest.fixture
def sample_prompt_embeddings(cuda_device, dtype):
    """Create sample prompt embeddings."""
    return torch.randn(
        1, 77, 4096,
        device=cuda_device,
        dtype=dtype
    )


@pytest.fixture
def optimization_config():
    """Create default optimization config."""
    from optimizations.config import OptimizationConfig
    return OptimizationConfig.preset_balanced()


@pytest.fixture(autouse=True)
def reset_cuda_memory():
    """Reset CUDA memory stats before each test."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_text_encoder():
    """Create mock text encoder."""
    class MockEncoder:
        def encode(self, prompts):
            return torch.randn(len(prompts), 77, 4096)

        def __call__(self, prompts):
            return self.encode(prompts)

    return MockEncoder()


@pytest.fixture
def mock_vae():
    """Create mock VAE."""
    class MockVAE:
        def decode(self, latents):
            b, c, f, h, w = latents.shape
            return torch.randn(b, 3, f, h * 8, w * 8, device=latents.device)

        def encode(self, frames):
            b, c, f, h, w = frames.shape
            return torch.randn(b, 16, f, h // 8, w // 8, device=frames.device)

    return MockVAE()


@pytest.fixture
def mock_generator():
    """Create mock diffusion generator."""
    class MockGenerator:
        def __init__(self):
            self.num_layers = 30
            self.num_heads = 12
            self.head_dim = 128

        def __call__(self, latents, timestep, prompt_embeds, **kwargs):
            # Just return slightly modified latents
            return latents * 0.99 + torch.randn_like(latents) * 0.01

    return MockGenerator()
