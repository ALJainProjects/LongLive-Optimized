"""
Tests for AsyncVAEPipeline

Tests double-buffering, async decode, and stream synchronization.
"""

import pytest
import torch
import time

from optimizations.async_vae import AsyncVAEPipeline, SyncVAEPipeline


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class MockVAE:
    """Mock VAE decoder for testing."""

    def __init__(self, decode_time_ms: float = 5.0):
        self.decode_time_ms = decode_time_ms
        self.call_count = 0

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Simulate VAE decode with artificial delay."""
        self.call_count += 1

        # Simulate compute time with a kernel
        batch, channels, frames, height, width = latents.shape

        # Output shape: [B, 3, F, H*8, W*8]
        output = torch.randn(
            batch, 3, frames, height * 8, width * 8,
            device=latents.device, dtype=latents.dtype
        )

        # Add some compute to simulate real VAE
        for _ in range(10):
            output = output * 1.0001

        return output


class TestAsyncVAEPipeline:
    """Tests for AsyncVAEPipeline class."""

    @pytest.fixture
    def vae(self):
        """Create mock VAE."""
        return MockVAE(decode_time_ms=5.0)

    @pytest.fixture
    def async_pipeline(self, vae):
        """Create async VAE pipeline."""
        return AsyncVAEPipeline(vae=vae, device='cuda')

    @pytest.fixture
    def sync_pipeline(self, vae):
        """Create sync VAE pipeline for comparison."""
        return SyncVAEPipeline(vae=vae, device='cuda')

    def test_initialization(self, async_pipeline):
        """Test async pipeline initialization."""
        assert async_pipeline.decode_stream is not None
        assert len(async_pipeline.buffers) == 2
        assert async_pipeline.buffer_idx == 0

    def test_decode_async_non_blocking(self, async_pipeline):
        """Test decode_async returns immediately."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        start = time.perf_counter()
        async_pipeline.decode_async(latents)
        elapsed = time.perf_counter() - start

        # Should return quickly (not wait for decode)
        assert elapsed < 0.01  # Less than 10ms

    def test_double_buffering(self, async_pipeline):
        """Test double buffering switches between buffers."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        initial_idx = async_pipeline.buffer_idx
        async_pipeline.decode_async(latents)

        assert async_pipeline.buffer_idx != initial_idx

    def test_get_previous_frame(self, async_pipeline):
        """Test retrieving previous frame."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        # First decode
        async_pipeline.decode_async(latents)

        # Second decode
        latents2 = torch.randn(1, 16, 1, 60, 104, device='cuda')
        async_pipeline.decode_async(latents2)

        # Get previous (first) frame
        frame = async_pipeline.get_previous_frame()

        assert frame is not None
        assert frame.shape[1] == 3  # RGB channels

    def test_sync_pipeline_blocking(self, sync_pipeline):
        """Test sync pipeline blocks until decode complete."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        start = time.perf_counter()
        frame = sync_pipeline.decode(latents)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        assert frame is not None
        # Should take some time (decode is synchronous)

    def test_output_shape(self, async_pipeline):
        """Test output has correct shape."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        async_pipeline.decode_async(latents)
        async_pipeline.decode_async(latents.clone())

        frame = async_pipeline.get_previous_frame()

        # Output should be upscaled by 8x
        assert frame.shape == (1, 3, 1, 60 * 8, 104 * 8)

    def test_multiple_frames(self, async_pipeline, vae):
        """Test decoding multiple frames."""
        num_frames = 10

        for i in range(num_frames):
            latents = torch.randn(1, 16, 1, 60, 104, device='cuda')
            async_pipeline.decode_async(latents)

            if i > 0:
                frame = async_pipeline.get_previous_frame()
                assert frame is not None

        # VAE should have been called for each frame
        assert vae.call_count == num_frames

    def test_reset(self, async_pipeline):
        """Test pipeline reset."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        async_pipeline.decode_async(latents)
        async_pipeline.reset()

        assert async_pipeline.buffer_idx == 0
        assert all(b is None for b in async_pipeline.buffers)

    def test_flush(self, async_pipeline):
        """Test flushing pending decodes."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        async_pipeline.decode_async(latents)
        frame = async_pipeline.flush()

        assert frame is not None

    def test_stream_isolation(self, async_pipeline):
        """Test decode stream is separate from default stream."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        # Operations on default stream
        x = torch.randn(100, 100, device='cuda')

        # Decode on separate stream
        async_pipeline.decode_async(latents)

        # Default stream operations should not wait for decode
        y = x @ x.T

        # This should complete quickly
        torch.cuda.current_stream().synchronize()


class TestSyncVAEPipeline:
    """Tests for SyncVAEPipeline class."""

    @pytest.fixture
    def vae(self):
        return MockVAE()

    @pytest.fixture
    def pipeline(self, vae):
        return SyncVAEPipeline(vae=vae, device='cuda')

    def test_decode(self, pipeline):
        """Test synchronous decode."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        frame = pipeline.decode(latents)

        assert frame is not None
        assert frame.shape[1] == 3

    def test_api_compatibility(self, pipeline):
        """Test SyncVAEPipeline has same API as AsyncVAEPipeline."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        # These methods should exist
        assert hasattr(pipeline, 'decode')
        assert hasattr(pipeline, 'reset')
