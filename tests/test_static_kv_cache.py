"""
Tests for StaticKVCache

Tests ring buffer operations, memory pre-allocation, and cache management.
"""

import pytest
import torch

from optimizations.static_kv_cache import StaticKVCache


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestStaticKVCache:
    """Tests for StaticKVCache class."""

    @pytest.fixture
    def cache_config(self):
        """Default cache configuration for tests."""
        return {
            'num_layers': 4,
            'num_heads': 8,
            'head_dim': 64,
            'local_window': 12,
            'sink_size': 3,
            'dtype': torch.bfloat16,
            'device': 'cuda',
        }

    @pytest.fixture
    def cache(self, cache_config):
        """Create a cache instance for testing."""
        return StaticKVCache(**cache_config)

    def test_initialization(self, cache, cache_config):
        """Test cache is initialized with correct shapes."""
        assert cache.num_layers == cache_config['num_layers']
        assert cache.num_heads == cache_config['num_heads']
        assert cache.local_window == cache_config['local_window']

        # Check buffers are allocated
        assert cache.local_k is not None
        assert cache.local_v is not None
        assert cache.sink_k is not None
        assert cache.sink_v is not None

    def test_buffer_shapes(self, cache, cache_config):
        """Test buffer shapes are correct."""
        expected_local_shape = (
            cache_config['num_layers'],
            1,  # batch
            cache_config['num_heads'],
            cache_config['local_window'],
            cache_config['head_dim'],
        )

        expected_sink_shape = (
            cache_config['num_layers'],
            1,  # batch
            cache_config['num_heads'],
            cache_config['sink_size'],
            cache_config['head_dim'],
        )

        assert cache.local_k.shape == expected_local_shape
        assert cache.local_v.shape == expected_local_shape
        assert cache.sink_k.shape == expected_sink_shape
        assert cache.sink_v.shape == expected_sink_shape

    def test_update_single_frame(self, cache):
        """Test updating cache with a single frame."""
        # Create dummy KV tensors
        new_k = torch.randn(
            cache.num_layers, 1, cache.num_heads, 1, cache.head_dim,
            device='cuda', dtype=torch.bfloat16
        )
        new_v = torch.randn_like(new_k)

        initial_write_idx = cache.write_idx
        cache.update(new_k, new_v)

        # Write index should advance
        assert cache.write_idx == (initial_write_idx + 1) % cache.local_window

    def test_ring_buffer_wrap(self, cache):
        """Test ring buffer wraps correctly."""
        # Fill entire buffer
        for i in range(cache.local_window + 5):
            new_k = torch.randn(
                cache.num_layers, 1, cache.num_heads, 1, cache.head_dim,
                device='cuda', dtype=torch.bfloat16
            )
            new_v = torch.randn_like(new_k)
            cache.update(new_k, new_v)

        # Should have wrapped
        assert cache.write_idx == 5

    def test_get_cache_ordering(self, cache):
        """Test get_cache returns correctly ordered KV pairs."""
        # Write known values
        for i in range(cache.local_window):
            new_k = torch.full(
                (cache.num_layers, 1, cache.num_heads, 1, cache.head_dim),
                fill_value=float(i),
                device='cuda', dtype=torch.bfloat16
            )
            new_v = torch.full_like(new_k, fill_value=float(i + 100))
            cache.update(new_k, new_v)

        k, v = cache.get_cache()

        # Values should be in order 0, 1, 2, ...
        for i in range(cache.local_window):
            assert k[0, 0, 0, i, 0].item() == pytest.approx(float(i), abs=0.1)

    def test_reset(self, cache):
        """Test cache reset."""
        # Add some data
        for _ in range(5):
            new_k = torch.randn(
                cache.num_layers, 1, cache.num_heads, 1, cache.head_dim,
                device='cuda', dtype=torch.bfloat16
            )
            new_v = torch.randn_like(new_k)
            cache.update(new_k, new_v)

        cache.reset()

        assert cache.write_idx == 0
        assert cache.num_frames == 0

    def test_step_increments_frame_count(self, cache):
        """Test step() increments frame count."""
        initial_frames = cache.num_frames

        cache.step()

        assert cache.num_frames == initial_frames + 1

    def test_no_allocation_during_update(self, cache):
        """Test that update doesn't allocate new memory."""
        torch.cuda.reset_peak_memory_stats()

        # Warm up
        for _ in range(3):
            new_k = torch.randn(
                cache.num_layers, 1, cache.num_heads, 1, cache.head_dim,
                device='cuda', dtype=torch.bfloat16
            )
            cache.update(new_k, new_k.clone())

        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Do many updates
        for _ in range(100):
            new_k = torch.randn(
                cache.num_layers, 1, cache.num_heads, 1, cache.head_dim,
                device='cuda', dtype=torch.bfloat16
            )
            cache.update(new_k, new_k.clone())

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow significantly (allow for small fluctuations)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1024 * 1024  # Less than 1MB growth

    def test_sink_frames_initialization(self, cache):
        """Test sink frames are set correctly."""
        # Create sink frames
        sink_k = torch.randn(
            cache.num_layers, 1, cache.num_heads, cache.sink_size, cache.head_dim,
            device='cuda', dtype=torch.bfloat16
        )
        sink_v = torch.randn_like(sink_k)

        cache.set_sink(sink_k, sink_v)

        # Check sink frames are set
        assert torch.allclose(cache.sink_k, sink_k)
        assert torch.allclose(cache.sink_v, sink_v)

    def test_get_full_cache_with_sink(self, cache):
        """Test get_full_cache combines sink and local correctly."""
        # Set sink
        sink_k = torch.ones(
            cache.num_layers, 1, cache.num_heads, cache.sink_size, cache.head_dim,
            device='cuda', dtype=torch.bfloat16
        )
        cache.set_sink(sink_k, sink_k.clone())

        # Add local frames
        for i in range(cache.local_window):
            new_k = torch.full(
                (cache.num_layers, 1, cache.num_heads, 1, cache.head_dim),
                fill_value=float(i + 10),
                device='cuda', dtype=torch.bfloat16
            )
            cache.update(new_k, new_k.clone())

        k, v = cache.get_full_cache()

        # Should have sink + local frames
        expected_length = cache.sink_size + cache.local_window
        assert k.shape[3] == expected_length

        # Sink frames should be first
        assert k[0, 0, 0, 0, 0].item() == pytest.approx(1.0, abs=0.1)
