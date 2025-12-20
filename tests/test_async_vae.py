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

    def decode_to_pixel(self, latents: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Simulate VAE decode with artificial delay."""
        self.call_count += 1

        # Handle different input shapes
        if latents.dim() == 5:  # [B, C, F, H, W] or [B, F, C, H, W]
            batch = latents.shape[0]
            # Assume [B, C, F, H, W] format
            channels, frames, height, width = latents.shape[1:]
        elif latents.dim() == 4:  # [C, F, H, W]
            batch = 1
            channels, frames, height, width = latents.shape
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape}")

        # Output shape: [B, F, C, H*8, W*8] - video format
        output = torch.randn(
            batch, frames, 3, height * 8, width * 8,
            device=latents.device, dtype=latents.dtype
        )

        # Add some compute to simulate real VAE
        for _ in range(5):
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
        return AsyncVAEPipeline(
            vae=vae,
            device=torch.device('cuda'),
            height=480,
            width=832,
        )

    @pytest.fixture
    def sync_pipeline(self, vae):
        """Create sync VAE pipeline for comparison."""
        return SyncVAEPipeline(
            vae=vae,
            device=torch.device('cuda'),
            height=480,
            width=832,
        )

    def test_initialization(self, async_pipeline):
        """Test async pipeline initialization."""
        assert async_pipeline.vae_stream is not None
        assert len(async_pipeline._output_buffers) == 2  # default num_buffers
        assert async_pipeline._write_idx == 0

    def test_decode_async_non_blocking(self, async_pipeline):
        """Test decode_async returns immediately."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        start = time.perf_counter()
        async_pipeline.decode_async(latents)
        elapsed = time.perf_counter() - start

        # Should return quickly (not wait for decode)
        assert elapsed < 0.1  # Less than 100ms (generous for CI)

    def test_double_buffering(self, async_pipeline):
        """Test double buffering switches between buffers."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        initial_idx = async_pipeline._write_idx
        async_pipeline.decode_async(latents)

        assert async_pipeline._write_idx != initial_idx

    def test_get_completed_frame(self, async_pipeline):
        """Test retrieving completed frame."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        # First decode
        async_pipeline.decode_async(latents)

        # Get completed frame
        frame = async_pipeline.get_completed_frame(block=True)

        assert frame is not None
        assert frame.dim() == 3  # [C, H, W]
        assert frame.shape[0] == 3  # RGB channels

    def test_get_previous_frame(self, async_pipeline):
        """Test get_previous_frame waits and returns."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        # First decode
        async_pipeline.decode_async(latents)

        # Get previous frame (should wait)
        frame = async_pipeline.get_previous_frame()

        assert frame is not None
        assert frame.shape[0] == 3  # RGB channels

    def test_multiple_frames(self, async_pipeline, vae):
        """Test decoding multiple frames."""
        num_frames = 5

        for i in range(num_frames):
            latents = torch.randn(1, 16, 1, 60, 104, device='cuda')
            async_pipeline.decode_async(latents)

        # Flush all pending
        frames = async_pipeline.flush()

        # VAE should have been called for each frame
        assert vae.call_count == num_frames

    def test_reset(self, async_pipeline):
        """Test pipeline reset."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        async_pipeline.decode_async(latents)
        async_pipeline.reset()

        assert async_pipeline._write_idx == 0
        assert async_pipeline._read_idx == 0
        assert async_pipeline._in_flight == 0

    def test_flush(self, async_pipeline):
        """Test flushing pending decodes."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        async_pipeline.decode_async(latents)
        frames = async_pipeline.flush()

        assert len(frames) == 1
        assert frames[0] is not None

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

    def test_has_pending(self, async_pipeline):
        """Test has_pending property."""
        assert not async_pipeline.has_pending

        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')
        async_pipeline.decode_async(latents)

        assert async_pipeline.has_pending

        async_pipeline.flush()
        assert not async_pipeline.has_pending

    def test_synchronize(self, async_pipeline):
        """Test synchronize waits for VAE stream."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        async_pipeline.decode_async(latents)
        async_pipeline.synchronize()

        # After sync, decode should be complete
        # (though frame is still in buffer)


class TestSyncVAEPipeline:
    """Tests for SyncVAEPipeline class."""

    @pytest.fixture
    def vae(self):
        return MockVAE()

    @pytest.fixture
    def pipeline(self, vae):
        return SyncVAEPipeline(
            vae=vae,
            device=torch.device('cuda'),
            height=480,
            width=832,
        )

    def test_decode_async(self, pipeline):
        """Test synchronous decode via decode_async."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        pipeline.decode_async(latents)
        frame = pipeline.get_completed_frame()

        assert frame is not None
        assert frame.shape[0] == 3  # RGB

    def test_api_compatibility(self, pipeline):
        """Test SyncVAEPipeline has same API as AsyncVAEPipeline."""
        # These methods should exist
        assert hasattr(pipeline, 'decode_async')
        assert hasattr(pipeline, 'get_completed_frame')
        assert hasattr(pipeline, 'get_previous_frame')
        assert hasattr(pipeline, 'flush')
        assert hasattr(pipeline, 'reset')
        assert hasattr(pipeline, 'synchronize')
        assert hasattr(pipeline, 'has_pending')

    def test_get_previous_frame(self, pipeline):
        """Test get_previous_frame returns last decoded."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        pipeline.decode_async(latents)
        frame = pipeline.get_previous_frame()

        assert frame is not None

    def test_flush(self, pipeline):
        """Test flush returns last frame."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        pipeline.decode_async(latents)
        frames = pipeline.flush()

        assert len(frames) == 1

    def test_reset(self, pipeline):
        """Test reset clears state."""
        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')

        pipeline.decode_async(latents)
        pipeline.reset()

        assert pipeline._last_frame is None

    def test_has_pending_always_false(self, pipeline):
        """Test sync pipeline never has pending."""
        assert not pipeline.has_pending

        latents = torch.randn(1, 16, 1, 60, 104, device='cuda')
        pipeline.decode_async(latents)

        # Still no pending (sync completes immediately)
        assert not pipeline.has_pending
