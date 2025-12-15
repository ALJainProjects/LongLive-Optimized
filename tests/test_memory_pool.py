"""
Tests for FixedShapeMemoryPool

Tests pre-allocation, buffer management, and memory efficiency.
"""

import pytest
import torch

from optimizations.memory_pool import FixedShapeMemoryPool, LongLiveMemoryPool, BufferSpec


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestFixedShapeMemoryPool:
    """Tests for FixedShapeMemoryPool class."""

    @pytest.fixture
    def pool(self):
        """Create memory pool with test buffers."""
        specs = {
            'small': BufferSpec(shape=(1, 64, 64), dtype=torch.float32),
            'medium': BufferSpec(shape=(1, 256, 256), dtype=torch.float32),
            'large': BufferSpec(shape=(1, 1024, 1024), dtype=torch.bfloat16),
        }
        return FixedShapeMemoryPool(specs, device='cuda')

    def test_initialization(self, pool):
        """Test pool initialization."""
        assert 'small' in pool
        assert 'medium' in pool
        assert 'large' in pool

    def test_get_buffer(self, pool):
        """Test getting buffer from pool."""
        small = pool.get('small')

        assert small is not None
        assert small.shape == (1, 64, 64)
        assert small.dtype == torch.float32
        assert small.device.type == 'cuda'

    def test_buffer_reuse(self, pool):
        """Test same buffer is returned on multiple gets."""
        buffer1 = pool.get('small')
        buffer2 = pool.get('small')

        assert buffer1.data_ptr() == buffer2.data_ptr()

    def test_buffer_dtype(self, pool):
        """Test buffers have correct dtype."""
        large = pool.get('large')

        assert large.dtype == torch.bfloat16

    def test_unknown_buffer(self, pool):
        """Test error on unknown buffer name."""
        with pytest.raises(KeyError):
            pool.get('nonexistent')

    def test_no_allocation_on_get(self, pool):
        """Test getting buffer doesn't allocate memory."""
        # Warm up
        _ = pool.get('small')
        torch.cuda.synchronize()

        initial_memory = torch.cuda.memory_allocated()

        # Get buffer multiple times
        for _ in range(100):
            _ = pool.get('small')

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # No new allocations
        assert final_memory == initial_memory

    def test_total_memory(self, pool):
        """Test total memory calculation."""
        total = pool.total_memory_bytes()

        # Calculate expected
        expected = (
            1 * 64 * 64 * 4 +      # small: float32 = 4 bytes
            1 * 256 * 256 * 4 +    # medium: float32 = 4 bytes
            1 * 1024 * 1024 * 2    # large: bfloat16 = 2 bytes
        )

        assert total == expected

    def test_clear(self, pool):
        """Test clearing pool."""
        pool.clear()

        # Pool should be empty
        assert len(pool) == 0

    def test_add_buffer(self, pool):
        """Test adding buffer dynamically."""
        pool.add('extra', BufferSpec(shape=(10, 10), dtype=torch.float32))

        assert 'extra' in pool
        extra = pool.get('extra')
        assert extra.shape == (10, 10)

    def test_contains(self, pool):
        """Test __contains__ method."""
        assert 'small' in pool
        assert 'nonexistent' not in pool

    def test_len(self, pool):
        """Test __len__ method."""
        assert len(pool) == 3


class TestLongLiveMemoryPool:
    """Tests for LongLive-specific memory pool."""

    @pytest.fixture
    def pool(self):
        """Create LongLive memory pool."""
        return LongLiveMemoryPool(
            batch_size=1,
            num_frames=12,
            latent_channels=16,
            latent_height=60,
            latent_width=104,
            dtype=torch.bfloat16,
            device='cuda',
        )

    def test_longlive_buffers_exist(self, pool):
        """Test LongLive-specific buffers are created."""
        expected_buffers = [
            'noise',
            'latents',
            'denoised',
        ]

        for name in expected_buffers:
            assert name in pool

    def test_noise_shape(self, pool):
        """Test noise buffer has correct shape."""
        noise = pool.get('noise')

        # [B, C, F, H, W]
        assert noise.shape == (1, 16, 1, 60, 104)

    def test_latents_shape(self, pool):
        """Test latents buffer has correct shape."""
        latents = pool.get('latents')

        assert latents.shape == (1, 16, 1, 60, 104)

    def test_memory_efficiency(self, pool):
        """Test memory is allocated upfront."""
        # Get all buffers
        _ = pool.get('noise')
        _ = pool.get('latents')
        _ = pool.get('denoised')

        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Use buffers
        for _ in range(10):
            noise = pool.get('noise')
            noise.normal_()
            latents = pool.get('latents')
            latents.copy_(noise)

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow
        assert final_memory <= initial_memory * 1.01  # Allow 1% tolerance


class TestBufferSpec:
    """Tests for BufferSpec dataclass."""

    def test_buffer_spec_creation(self):
        """Test creating BufferSpec."""
        spec = BufferSpec(
            shape=(1, 3, 224, 224),
            dtype=torch.float32,
        )

        assert spec.shape == (1, 3, 224, 224)
        assert spec.dtype == torch.float32

    def test_buffer_spec_bytes(self):
        """Test buffer size calculation."""
        spec = BufferSpec(
            shape=(1, 3, 224, 224),
            dtype=torch.float32,
        )

        expected = 1 * 3 * 224 * 224 * 4  # float32 = 4 bytes
        assert spec.num_bytes() == expected

    def test_buffer_spec_bfloat16(self):
        """Test buffer size with bfloat16."""
        spec = BufferSpec(
            shape=(1, 16, 60, 104),
            dtype=torch.bfloat16,
        )

        expected = 1 * 16 * 60 * 104 * 2  # bfloat16 = 2 bytes
        assert spec.num_bytes() == expected
