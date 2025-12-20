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
        device = torch.device('cuda')
        specs = {
            'small': BufferSpec('small', (1, 64, 64), torch.float32, device),
            'medium': BufferSpec('medium', (1, 256, 256), torch.float32, device),
            'large': BufferSpec('large', (1, 1024, 1024), torch.bfloat16, device),
        }
        return FixedShapeMemoryPool(specs, device=device)

    def test_initialization(self, pool):
        """Test pool initialization."""
        assert pool.has_buffer('small')
        assert pool.has_buffer('medium')
        assert pool.has_buffer('large')

    def test_get_buffer(self, pool):
        """Test getting buffer from pool."""
        small = pool.get_buffer('small')
        pool.return_buffer('small')

        assert small is not None
        assert small.shape == (1, 64, 64)
        assert small.dtype == torch.float32
        assert small.device.type == 'cuda'

    def test_buffer_reuse(self, pool):
        """Test same buffer is returned on multiple gets."""
        buffer1 = pool.get_buffer('small')
        ptr1 = buffer1.data_ptr()
        pool.return_buffer('small')

        buffer2 = pool.get_buffer('small')
        ptr2 = buffer2.data_ptr()
        pool.return_buffer('small')

        assert ptr1 == ptr2

    def test_buffer_dtype(self, pool):
        """Test buffers have correct dtype."""
        large = pool.get_buffer('large')
        pool.return_buffer('large')

        assert large.dtype == torch.bfloat16

    def test_unknown_buffer(self, pool):
        """Test error on unknown buffer name."""
        with pytest.raises(KeyError):
            pool.get_buffer('nonexistent')

    def test_no_allocation_on_get(self, pool):
        """Test getting buffer doesn't allocate memory."""
        # Warm up
        buf = pool.get_buffer('small')
        pool.return_buffer('small')
        torch.cuda.synchronize()

        initial_memory = torch.cuda.memory_allocated()

        # Get buffer multiple times
        for _ in range(100):
            buf = pool.get_buffer('small')
            pool.return_buffer('small')

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # No new allocations
        assert final_memory == initial_memory

    def test_memory_usage(self, pool):
        """Test memory usage calculation."""
        usage = pool.memory_usage()

        # Calculate expected
        expected = (
            1 * 64 * 64 * 4 +      # small: float32 = 4 bytes
            1 * 256 * 256 * 4 +    # medium: float32 = 4 bytes
            1 * 1024 * 1024 * 2    # large: bfloat16 = 2 bytes
        )

        assert usage['total_bytes'] == expected

    def test_clear(self, pool):
        """Test clearing pool."""
        pool.clear()

        # Pool should be empty
        assert not pool.has_buffer('small')

    def test_add_buffer(self, pool):
        """Test adding buffer dynamically."""
        pool.add_buffer('extra', (10, 10), dtype=torch.float32)

        assert pool.has_buffer('extra')
        extra = pool.get_buffer('extra')
        pool.return_buffer('extra')
        assert extra.shape == (10, 10)

    def test_has_buffer(self, pool):
        """Test has_buffer method."""
        assert pool.has_buffer('small')
        assert not pool.has_buffer('nonexistent')


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
            device=torch.device('cuda'),
        )

    def test_longlive_buffers_exist(self, pool):
        """Test LongLive-specific buffers are created."""
        expected_buffers = [
            'noise',
            'latents',
            'denoised',
        ]

        for name in expected_buffers:
            assert pool.has_buffer(name)

    def test_noise_shape(self, pool):
        """Test noise buffer has correct shape."""
        noise = pool.get_buffer('noise')
        pool.return_buffer('noise')

        # [B, C, F, H, W]
        assert noise.shape == (1, 16, 1, 60, 104)

    def test_latents_shape(self, pool):
        """Test latents buffer has correct shape."""
        latents = pool.get_buffer('latents')
        pool.return_buffer('latents')

        assert latents.shape == (1, 16, 1, 60, 104)

    def test_memory_efficiency(self, pool):
        """Test memory is allocated upfront."""
        # Get all buffers
        noise = pool.get_buffer('noise')
        pool.return_buffer('noise')
        latents = pool.get_buffer('latents')
        pool.return_buffer('latents')
        denoised = pool.get_buffer('denoised')
        pool.return_buffer('denoised')

        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Use buffers
        for _ in range(10):
            noise = pool.get_buffer('noise')
            noise.normal_()
            pool.return_buffer('noise')
            latents = pool.get_buffer('latents')
            latents.copy_(noise)
            pool.return_buffer('latents')

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow
        assert final_memory <= initial_memory * 1.01  # Allow 1% tolerance


class TestBufferSpec:
    """Tests for BufferSpec dataclass."""

    def test_buffer_spec_creation(self):
        """Test creating BufferSpec."""
        spec = BufferSpec(
            name='test',
            shape=(1, 3, 224, 224),
            dtype=torch.float32,
            device=torch.device('cuda'),
        )

        assert spec.name == 'test'
        assert spec.shape == (1, 3, 224, 224)
        assert spec.dtype == torch.float32

    def test_buffer_spec_with_pinned(self):
        """Test BufferSpec with pinned memory."""
        spec = BufferSpec(
            name='pinned',
            shape=(1, 3, 224, 224),
            dtype=torch.float32,
            device=torch.device('cpu'),
            pinned=True,
        )

        assert spec.pinned is True

    def test_buffer_spec_bfloat16(self):
        """Test BufferSpec with bfloat16."""
        spec = BufferSpec(
            name='bf16',
            shape=(1, 16, 60, 104),
            dtype=torch.bfloat16,
            device=torch.device('cuda'),
        )

        assert spec.dtype == torch.bfloat16
