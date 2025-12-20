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
            'local_window_frames': 12,
            'sink_frames': 3,
            'frame_seq_length': 100,  # tokens per frame
            'batch_size': 1,
            'dtype': torch.bfloat16,
            'device': torch.device('cuda'),
        }

    @pytest.fixture
    def cache(self, cache_config):
        """Create a cache instance for testing."""
        return StaticKVCache(**cache_config)

    def test_initialization(self, cache, cache_config):
        """Test cache is initialized with correct shapes."""
        assert cache.num_layers == cache_config['num_layers']
        assert cache.num_heads == cache_config['num_heads']
        assert cache.local_window_frames == cache_config['local_window_frames']

    def test_buffer_allocation(self, cache):
        """Test buffers are allocated for each layer."""
        for layer_idx in range(cache.num_layers):
            sink_k, sink_v = cache.get_sink_kv(layer_idx)
            local_k, local_v = cache.get_local_kv(layer_idx)

            assert sink_k is not None
            assert sink_v is not None
            assert local_k is not None
            assert local_v is not None

    def test_sink_kv_shapes(self, cache, cache_config):
        """Test sink KV cache shapes are correct."""
        sink_k, sink_v = cache.get_sink_kv(0)

        expected_sink_size = cache_config['sink_frames'] * cache_config['frame_seq_length']
        expected_shape = (
            cache_config['batch_size'],
            expected_sink_size,
            cache_config['num_heads'],
            cache_config['head_dim'],
        )

        assert sink_k.shape == expected_shape
        assert sink_v.shape == expected_shape

    def test_local_kv_shapes(self, cache, cache_config):
        """Test local KV cache shapes are correct."""
        local_k, local_v = cache.get_local_kv(0)

        expected_local_size = cache_config['local_window_frames'] * cache_config['frame_seq_length']
        expected_shape = (
            cache_config['batch_size'],
            expected_local_size,
            cache_config['num_heads'],
            cache_config['head_dim'],
        )

        assert local_k.shape == expected_shape
        assert local_v.shape == expected_shape

    def test_update_single_token(self, cache):
        """Test updating cache with tokens."""
        # Create dummy KV tensors for one token
        new_k = torch.randn(
            cache.batch_size, 1, cache.num_heads, cache.head_dim,
            device=cache.device, dtype=cache.dtype
        )
        new_v = torch.randn_like(new_k)

        layer_idx = 0
        initial_valid = cache.valid_lengths[layer_idx].item()

        cache.update(layer_idx, new_k, new_v)

        # Valid length should increase
        assert cache.valid_lengths[layer_idx].item() == initial_valid + 1

    def test_get_full_kv_empty(self, cache):
        """Test get_full_kv with no local cache."""
        k, v = cache.get_full_kv(0)

        # Should return only sink cache
        assert k.shape[1] == cache.sink_size

    def test_get_full_kv_with_data(self, cache):
        """Test get_full_kv after adding some tokens."""
        layer_idx = 0
        num_tokens = 50

        for _ in range(num_tokens):
            new_k = torch.randn(
                cache.batch_size, 1, cache.num_heads, cache.head_dim,
                device=cache.device, dtype=cache.dtype
            )
            new_v = torch.randn_like(new_k)
            cache.update(layer_idx, new_k, new_v)

        k, v = cache.get_full_kv(layer_idx)

        # Should have sink + local tokens
        expected_len = cache.sink_size + num_tokens
        assert k.shape[1] == expected_len

    def test_reset(self, cache):
        """Test cache reset."""
        layer_idx = 0

        # Add some data
        for _ in range(10):
            new_k = torch.randn(
                cache.batch_size, 1, cache.num_heads, cache.head_dim,
                device=cache.device, dtype=cache.dtype
            )
            new_v = torch.randn_like(new_k)
            cache.update(layer_idx, new_k, new_v)

        cache.reset()

        # All counters should be zero
        assert cache.write_indices[layer_idx].item() == 0
        assert cache.valid_lengths[layer_idx].item() == 0

    def test_no_allocation_during_update(self, cache):
        """Test that update doesn't allocate new memory."""
        layer_idx = 0
        torch.cuda.reset_peak_memory_stats()

        # Warm up
        for _ in range(3):
            new_k = torch.randn(
                cache.batch_size, 1, cache.num_heads, cache.head_dim,
                device=cache.device, dtype=cache.dtype
            )
            cache.update(layer_idx, new_k, new_k.clone())

        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Do many updates
        for _ in range(100):
            new_k = torch.randn(
                cache.batch_size, 1, cache.num_heads, cache.head_dim,
                device=cache.device, dtype=cache.dtype
            )
            cache.update(layer_idx, new_k, new_k.clone())

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow significantly (allow for small fluctuations)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1024 * 1024  # Less than 1MB growth

    def test_dtype_preserved(self, cache):
        """Test that cache preserves dtype."""
        sink_k, _ = cache.get_sink_kv(0)
        local_k, _ = cache.get_local_kv(0)

        assert sink_k.dtype == cache.dtype
        assert local_k.dtype == cache.dtype

    def test_device_preserved(self, cache):
        """Test that cache is on correct device."""
        sink_k, _ = cache.get_sink_kv(0)
        local_k, _ = cache.get_local_kv(0)

        assert sink_k.device.type == 'cuda'
        assert local_k.device.type == 'cuda'
