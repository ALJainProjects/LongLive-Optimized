"""
Tests for QuantizedKVCache

Tests INT8/FP8 quantization with ring buffer support.
"""

import pytest
import torch

from optimizations.quantized_kv import (
    quantize_int8,
    dequantize_int8,
    quantize_fp8,
    dequantize_fp8,
    QuantizedKVCache,
    QuantizationConfig,
)


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestINT8Quantization:
    """Tests for INT8 quantization functions."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return torch.randn(
            1, 100, 8, 64,
            dtype=torch.bfloat16, device='cuda'
        )

    def test_quantize_dequantize_roundtrip(self, sample_tensor):
        """Test that quantize/dequantize preserves values approximately."""
        quantized, scales = quantize_int8(sample_tensor, per_token=True)
        dequantized = dequantize_int8(quantized, scales)

        # Check shapes
        assert quantized.shape == sample_tensor.shape
        assert dequantized.shape == sample_tensor.shape

        # Check approximate equality
        error = (dequantized.float() - sample_tensor.float()).abs().mean()
        max_val = sample_tensor.float().abs().max()

        # Error should be small relative to data range (< 5% for INT8)
        assert error < max_val * 0.05

    def test_quantized_dtype(self, sample_tensor):
        """Test that quantized tensor has correct dtype."""
        quantized, _ = quantize_int8(sample_tensor)
        assert quantized.dtype == torch.int8

    def test_scales_shape_per_token(self, sample_tensor):
        """Test that scales have correct shape for per-token quantization."""
        quantized, scales = quantize_int8(sample_tensor, per_token=True)

        # Scales should be [batch, seq, 1, 1]
        expected_shape = (
            sample_tensor.shape[0],
            sample_tensor.shape[1],
            1, 1
        )
        assert scales.shape == expected_shape

    def test_symmetric_quantization(self, sample_tensor):
        """Test symmetric quantization properties."""
        quantized, scales = quantize_int8(sample_tensor, symmetric=True)

        # Values should be in [-128, 127]
        assert quantized.min() >= -128
        assert quantized.max() <= 127


class TestFP8Quantization:
    """Tests for FP8 quantization functions."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return torch.randn(
            1, 100, 8, 64,
            dtype=torch.bfloat16, device='cuda'
        )

    @pytest.mark.skipif(
        not hasattr(torch, 'float8_e4m3fn'),
        reason="FP8 not supported"
    )
    def test_fp8_quantize_dequantize(self, sample_tensor):
        """Test FP8 quantize/dequantize roundtrip."""
        quantized, scale = quantize_fp8(sample_tensor, format="e4m3")
        dequantized = dequantize_fp8(quantized, scale)

        # Check shapes
        assert dequantized.shape == sample_tensor.shape

        # Check approximate equality
        error = (dequantized.float() - sample_tensor.float()).abs().mean()
        max_val = sample_tensor.float().abs().max()

        # FP8 should have reasonable precision
        assert error < max_val * 0.1

    def test_fp8_fallback(self, sample_tensor):
        """Test that FP8 falls back gracefully when not supported."""
        quantized, scale = quantize_fp8(sample_tensor)

        # Should return some tensor
        assert quantized is not None
        assert scale is not None


class TestQuantizedKVCache:
    """Tests for QuantizedKVCache class."""

    @pytest.fixture
    def cache_config(self):
        """Default cache configuration."""
        return {
            'num_layers': 4,
            'num_heads': 8,
            'head_dim': 64,
            'local_window_frames': 12,
            'sink_frames': 3,
            'frame_seq_length': 100,
            'batch_size': 1,
            'quantization': 'int8',
            'output_dtype': torch.bfloat16,
            'device': torch.device('cuda'),
        }

    @pytest.fixture
    def cache(self, cache_config):
        """Create a cache instance."""
        return QuantizedKVCache(**cache_config)

    def test_initialization(self, cache, cache_config):
        """Test cache initializes correctly."""
        assert cache.num_layers == cache_config['num_layers']
        assert cache.num_heads == cache_config['num_heads']
        assert cache.quantization == 'int8'

    def test_buffer_allocation(self, cache):
        """Test that buffers are allocated for each layer."""
        for layer_idx in range(cache.num_layers):
            # Check sink buffers exist
            assert hasattr(cache, f'sink_k_{layer_idx}')
            assert hasattr(cache, f'sink_v_{layer_idx}')
            assert hasattr(cache, f'sink_k_scale_{layer_idx}')
            assert hasattr(cache, f'sink_v_scale_{layer_idx}')

            # Check local buffers exist
            assert hasattr(cache, f'local_k_{layer_idx}')
            assert hasattr(cache, f'local_v_{layer_idx}')
            assert hasattr(cache, f'local_k_scale_{layer_idx}')
            assert hasattr(cache, f'local_v_scale_{layer_idx}')

    def test_update_sink(self, cache):
        """Test updating sink cache."""
        layer_idx = 0

        new_k = torch.randn(
            cache.batch_size, cache.frame_seq_length, cache.num_heads, cache.head_dim,
            dtype=cache.output_dtype, device=cache.device
        )
        new_v = torch.randn_like(new_k)

        # Initial state
        assert cache.sink_filled[layer_idx] == False

        # Update sink
        cache.update(layer_idx, new_k, new_v, is_sink=True)

        # Valid length should increase
        assert cache.valid_lengths[layer_idx].item() == cache.frame_seq_length

    def test_update_local_ring(self, cache):
        """Test updating local ring buffer."""
        layer_idx = 0

        # First fill sink
        sink_tokens = cache.sink_size
        for _ in range(sink_tokens // 10):
            new_k = torch.randn(
                cache.batch_size, 10, cache.num_heads, cache.head_dim,
                dtype=cache.output_dtype, device=cache.device
            )
            cache.update(layer_idx, new_k, new_k.clone(), is_sink=True)

        # Now update local
        cache.update(
            layer_idx,
            torch.randn(1, 50, cache.num_heads, cache.head_dim,
                       dtype=cache.output_dtype, device=cache.device),
            torch.randn(1, 50, cache.num_heads, cache.head_dim,
                       dtype=cache.output_dtype, device=cache.device),
            is_sink=False
        )

    def test_get_full_kv(self, cache):
        """Test getting full KV cache."""
        layer_idx = 0

        # Add some data to sink
        new_k = torch.randn(
            cache.batch_size, cache.frame_seq_length, cache.num_heads, cache.head_dim,
            dtype=cache.output_dtype, device=cache.device
        )
        cache.update(layer_idx, new_k, new_k.clone(), is_sink=True)

        # Get full KV
        k, v = cache.get_full_kv(layer_idx)

        # Check shapes
        assert k.shape == (cache.batch_size, cache.sink_size, cache.num_heads, cache.head_dim)
        assert k.dtype == cache.output_dtype

    def test_reset(self, cache):
        """Test cache reset."""
        layer_idx = 0

        # Add some data
        cache.update(
            layer_idx,
            torch.randn(1, 50, cache.num_heads, cache.head_dim,
                       dtype=cache.output_dtype, device=cache.device),
            torch.randn(1, 50, cache.num_heads, cache.head_dim,
                       dtype=cache.output_dtype, device=cache.device),
            is_sink=True
        )

        cache.reset()

        # All counters should be zero
        assert cache.write_indices[layer_idx].item() == 0
        assert cache.valid_lengths[layer_idx].item() == 0
        assert cache.sink_filled[layer_idx] == False

    def test_memory_usage(self, cache):
        """Test memory usage reporting."""
        usage = cache.memory_usage()

        assert 'quantized_mb' in usage
        assert 'unquantized_mb' in usage
        assert 'savings_ratio' in usage

        # Quantized should be smaller
        assert usage['quantized_mb'] < usage['unquantized_mb']

        # Savings ratio should be > 1 (approximately 2x for INT8 vs BF16)
        assert usage['savings_ratio'] > 1.5

    def test_quantized_dtype_is_int8(self, cache):
        """Test that quantized tensors use INT8."""
        sink_k = getattr(cache, 'sink_k_0')
        local_k = getattr(cache, 'local_k_0')

        assert sink_k.dtype == torch.int8
        assert local_k.dtype == torch.int8

    def test_scales_are_float32(self, cache):
        """Test that scales are stored in float32."""
        sink_k_scale = getattr(cache, 'sink_k_scale_0')
        local_k_scale = getattr(cache, 'local_k_scale_0')

        assert sink_k_scale.dtype == torch.float32
        assert local_k_scale.dtype == torch.float32


class TestMemorySavings:
    """Integration tests for memory savings."""

    def test_realistic_dimensions(self):
        """Test with LongLive-like dimensions."""
        cache = QuantizedKVCache(
            num_layers=28,
            num_heads=12,
            head_dim=128,
            local_window_frames=12,
            sink_frames=3,
            frame_seq_length=1560,
            batch_size=1,
            quantization='int8',
        )

        usage = cache.memory_usage()

        print(f"\nMemory usage for LongLive-like config:")
        print(f"  Quantized: {usage['quantized_mb']:.2f} MB")
        print(f"  Unquantized: {usage['unquantized_mb']:.2f} MB")
        print(f"  Savings: {usage['savings_ratio']:.2f}x")

        # Should achieve good savings
        assert usage['savings_ratio'] > 1.5
