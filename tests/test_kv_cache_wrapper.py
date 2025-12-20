"""
Tests for QuantizedKVCacheWrapper

Tests transparent INT8 quantization/dequantization for KV caches.
"""

import pytest
import torch

from optimizations.kv_cache_wrapper import (
    QuantizedKVCacheDict,
    QuantizedKVCacheList,
    wrap_kv_cache_with_quantization,
    create_quantized_kv_cache,
)


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestQuantizedKVCacheDict:
    """Tests for QuantizedKVCacheDict class."""

    @pytest.fixture
    def cache_dict(self):
        """Create a quantized cache dict for testing."""
        return QuantizedKVCacheDict(
            quantize_kv=True,
            target_dtype=torch.bfloat16,
        )

    @pytest.fixture
    def kv_tensors(self):
        """Create sample K and V tensors."""
        batch_size = 1
        seq_len = 100
        num_heads = 8
        head_dim = 64

        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=torch.bfloat16, device='cuda'
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            dtype=torch.bfloat16, device='cuda'
        )
        return k, v

    def test_basic_setget(self, cache_dict, kv_tensors):
        """Test basic set and get operations."""
        k, v = kv_tensors

        cache_dict['k'] = k
        cache_dict['v'] = v

        k_out = cache_dict['k']
        v_out = cache_dict['v']

        # Output should have same shape
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_quantization_reduces_memory(self, cache_dict, kv_tensors):
        """Test that quantization reduces memory usage."""
        k, v = kv_tensors

        # Original memory: BF16 = 2 bytes per element
        original_bytes = k.numel() * 2 + v.numel() * 2

        cache_dict['k'] = k
        cache_dict['v'] = v

        # Quantized memory: INT8 = 1 byte per element + scales
        quantized_bytes = cache_dict.memory_usage_bytes()

        # Quantized should be significantly smaller (roughly 50%)
        assert quantized_bytes < original_bytes * 0.7

    def test_quantization_preserves_values_approximately(self, cache_dict, kv_tensors):
        """Test that dequantized values are close to original."""
        k, v = kv_tensors

        cache_dict['k'] = k
        cache_dict['v'] = v

        k_out = cache_dict['k']
        v_out = cache_dict['v']

        # Should be approximately equal (within quantization error)
        # Allow for some error due to INT8 quantization
        k_error = (k_out.float() - k.float()).abs().mean()
        v_error = (v_out.float() - v.float()).abs().mean()

        # Error should be small relative to data range
        k_range = k.float().abs().max()
        v_range = v.float().abs().max()

        assert k_error < k_range * 0.1  # Less than 10% error
        assert v_error < v_range * 0.1

    def test_dtype_output(self, cache_dict, kv_tensors):
        """Test that output has target dtype."""
        k, v = kv_tensors

        cache_dict['k'] = k
        cache_dict['v'] = v

        k_out = cache_dict['k']
        v_out = cache_dict['v']

        assert k_out.dtype == torch.bfloat16
        assert v_out.dtype == torch.bfloat16

    def test_non_kv_keys_stored_directly(self, cache_dict):
        """Test that non-K/V keys are stored without quantization."""
        idx_tensor = torch.tensor([42], dtype=torch.long, device='cuda')
        cache_dict['global_end_index'] = idx_tensor

        out = cache_dict['global_end_index']
        assert torch.equal(out, idx_tensor)

    def test_quantization_disabled(self):
        """Test that quantization can be disabled."""
        cache_dict = QuantizedKVCacheDict(
            quantize_kv=False,
            target_dtype=torch.bfloat16,
        )

        k = torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda')
        cache_dict['k'] = k

        k_out = cache_dict['k']

        # Should be exactly equal when not quantized
        assert torch.equal(k_out, k)


class TestQuantizedKVCacheList:
    """Tests for QuantizedKVCacheList class."""

    @pytest.fixture
    def cache_list(self):
        """Create a quantized cache list for testing."""
        return QuantizedKVCacheList(
            quantize_kv=True,
            target_dtype=torch.bfloat16,
        )

    def test_append(self, cache_list):
        """Test appending layer caches."""
        layer_cache = {
            'k': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
            'v': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
            'global_end_index': torch.tensor([0], dtype=torch.long, device='cuda'),
        }

        cache_list.append(layer_cache)

        assert len(cache_list) == 1
        assert isinstance(cache_list[0], QuantizedKVCacheDict)

    def test_multi_layer(self, cache_list):
        """Test multiple layers."""
        num_layers = 4

        for _ in range(num_layers):
            layer_cache = {
                'k': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
                'v': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
            }
            cache_list.append(layer_cache)

        assert len(cache_list) == num_layers

    def test_total_memory(self, cache_list):
        """Test total memory calculation."""
        num_layers = 4

        for _ in range(num_layers):
            layer_cache = {
                'k': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
                'v': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
            }
            cache_list.append(layer_cache)

        total_memory = cache_list.total_memory_bytes()

        # Should be non-zero
        assert total_memory > 0

        # Should be less than unquantized (roughly 50%)
        unquantized_bytes = num_layers * 2 * 100 * 8 * 64 * 2  # 2 tensors, BF16
        assert total_memory < unquantized_bytes * 0.7


class TestCreateQuantizedKVCache:
    """Tests for create_quantized_kv_cache factory function."""

    def test_create_basic(self):
        """Test basic cache creation."""
        cache = create_quantized_kv_cache(
            num_layers=4,
            batch_size=1,
            kv_cache_size=1000,
            num_heads=8,
            head_dim=64,
            device=torch.device('cuda'),
            dtype=torch.bfloat16,
            quantize=True,
        )

        assert len(cache) == 4
        assert isinstance(cache, QuantizedKVCacheList)

    def test_layer_structure(self):
        """Test that each layer has correct structure."""
        cache = create_quantized_kv_cache(
            num_layers=4,
            batch_size=1,
            kv_cache_size=1000,
            num_heads=8,
            head_dim=64,
        )

        for layer_cache in cache:
            # Should have K and V
            k = layer_cache['k']
            v = layer_cache['v']

            assert k.shape == (1, 1000, 8, 64)
            assert v.shape == (1, 1000, 8, 64)


class TestWrapKVCache:
    """Tests for wrap_kv_cache_with_quantization function."""

    def test_wrap_existing(self):
        """Test wrapping an existing KV cache."""
        # Create standard KV cache
        num_layers = 4
        original_cache = []

        for _ in range(num_layers):
            layer_cache = {
                'k': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
                'v': torch.randn(1, 100, 8, 64, dtype=torch.bfloat16, device='cuda'),
                'global_end_index': torch.tensor([50], dtype=torch.long, device='cuda'),
            }
            original_cache.append(layer_cache)

        # Wrap it
        wrapped = wrap_kv_cache_with_quantization(
            original_cache,
            quantize=True,
            target_dtype=torch.bfloat16,
        )

        assert len(wrapped) == num_layers
        assert isinstance(wrapped, QuantizedKVCacheList)

        # Check that data is accessible
        for layer_idx, layer_cache in enumerate(wrapped):
            k = layer_cache['k']
            v = layer_cache['v']

            assert k.shape == original_cache[layer_idx]['k'].shape
            assert v.shape == original_cache[layer_idx]['v'].shape


class TestMemoryReduction:
    """Integration tests for memory reduction."""

    def test_realistic_cache_size(self):
        """Test with realistic cache dimensions."""
        # Simulate LongLive-like cache
        num_layers = 28  # Similar to original
        batch_size = 1
        kv_cache_size = 18720  # Based on ablation study output
        num_heads = 12
        head_dim = 128

        # Calculate original memory
        original_bytes = (
            num_layers * batch_size * kv_cache_size *
            num_heads * head_dim * 2 * 2  # K+V, BF16
        )

        # Create quantized cache
        cache = create_quantized_kv_cache(
            num_layers=num_layers,
            batch_size=batch_size,
            kv_cache_size=kv_cache_size,
            num_heads=num_heads,
            head_dim=head_dim,
            quantize=True,
        )

        quantized_bytes = cache.total_memory_bytes()

        # Should save about 50% memory
        reduction = 1 - (quantized_bytes / original_bytes)
        assert reduction > 0.4  # At least 40% reduction

        print(f"\nMemory reduction test:")
        print(f"  Original: {original_bytes / 1e9:.2f} GB")
        print(f"  Quantized: {quantized_bytes / 1e9:.2f} GB")
        print(f"  Reduction: {reduction*100:.1f}%")
