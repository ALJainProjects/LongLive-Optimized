"""
Tests for IntegratedKVCache - Ring Buffer + INT8 Quantization.
"""

import pytest
import torch

from optimizations.integrated_kv_cache import (
    IntegratedKVCache,
    IntegratedKVCacheLayer,
    IntegratedKVConfig,
    create_integrated_kv_cache,
    LazyTensor,
    convert_model_to_fp16,
    convert_model_to_bf16,
)


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestLazyTensor:
    """Tests for LazyTensor wrapper."""

    def test_clone_unquantized(self):
        """Test clone returns correct tensor for unquantized."""
        data = torch.randn(1, 100, 8, 64, device='cuda', dtype=torch.bfloat16)
        lazy = LazyTensor(data, is_quantized=False)

        cloned = lazy.clone()
        assert cloned.shape == data.shape
        assert cloned.dtype == data.dtype
        assert not torch.equal(cloned.data_ptr(), data.data_ptr())  # Different memory

    def test_clone_quantized(self):
        """Test clone dequantizes for quantized tensor."""
        # Create INT8 tensor with scale
        data = torch.randint(-128, 127, (1, 100, 8, 64), device='cuda', dtype=torch.int8)
        scale = torch.ones(1, 100, 1, 1, device='cuda', dtype=torch.float32) * 0.1

        lazy = LazyTensor(data, scale=scale, is_quantized=True, target_dtype=torch.bfloat16)

        cloned = lazy.clone()
        assert cloned.dtype == torch.bfloat16
        assert cloned.shape == data.shape

    def test_getitem(self):
        """Test indexing through LazyTensor."""
        data = torch.randn(1, 100, 8, 64, device='cuda', dtype=torch.bfloat16)
        lazy = LazyTensor(data, is_quantized=False)

        sliced = lazy[:, :50]
        assert sliced.shape == (1, 50, 8, 64)


class TestIntegratedKVCacheLayer:
    """Tests for single layer cache."""

    @pytest.fixture
    def config(self):
        """Default config for testing."""
        return IntegratedKVConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            local_window_frames=12,
            sink_frames=3,
            frame_seq_length=100,
            batch_size=1,
            use_ring_buffer=True,
            use_quantization=False,
        )

    def test_dict_interface(self, config):
        """Test layer cache provides dict interface."""
        layer = IntegratedKVCacheLayer(0, config)

        # Should have expected keys
        assert 'k' in layer
        assert 'v' in layer
        assert 'global_end_index' in layer
        assert 'local_end_index' in layer

    def test_update_cache(self, config):
        """Test cache update."""
        layer = IntegratedKVCacheLayer(0, config)

        # Create new KV
        new_k = torch.randn(1, 50, 8, 64, device='cuda', dtype=torch.bfloat16)
        new_v = torch.randn(1, 50, 8, 64, device='cuda', dtype=torch.bfloat16)

        layer.update_cache(new_k, new_v, global_end=50, local_end=50)

        # Check update happened
        assert layer._valid_len == 50
        assert layer._global_end == 50

    def test_memory_usage(self, config):
        """Test memory usage calculation."""
        layer = IntegratedKVCacheLayer(0, config)
        memory = layer.memory_usage_bytes()
        assert memory > 0


class TestIntegratedKVCacheLayerQuantized:
    """Tests for quantized layer cache."""

    @pytest.fixture
    def config(self):
        """Config with quantization enabled."""
        return IntegratedKVConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            local_window_frames=12,
            sink_frames=3,
            frame_seq_length=100,
            batch_size=1,
            use_ring_buffer=True,
            use_quantization=True,
            quantization_method="int8",
        )

    def test_quantized_storage(self, config):
        """Test that storage uses INT8."""
        layer = IntegratedKVCacheLayer(0, config)
        assert layer._k_buffer.dtype == torch.int8
        assert layer._v_buffer.dtype == torch.int8

    def test_quantized_update(self, config):
        """Test update quantizes data."""
        layer = IntegratedKVCacheLayer(0, config)

        new_k = torch.randn(1, 50, 8, 64, device='cuda', dtype=torch.bfloat16)
        new_v = torch.randn(1, 50, 8, 64, device='cuda', dtype=torch.bfloat16)

        layer.update_cache(new_k, new_v, global_end=50, local_end=50)

        # Buffer should still be INT8
        assert layer._k_buffer.dtype == torch.int8

        # Scale should be updated
        assert layer._k_scale is not None

    def test_memory_savings(self, config):
        """Test that quantization saves memory."""
        layer_q = IntegratedKVCacheLayer(0, config)

        # Create unquantized config
        config_uq = IntegratedKVConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            local_window_frames=12,
            sink_frames=3,
            frame_seq_length=100,
            batch_size=1,
            use_ring_buffer=True,
            use_quantization=False,
        )
        layer_uq = IntegratedKVCacheLayer(0, config_uq)

        # Quantized should use less memory
        assert layer_q.memory_usage_bytes() < layer_uq.memory_usage_bytes()


class TestIntegratedKVCache:
    """Tests for full multi-layer cache."""

    def test_create_factory(self):
        """Test factory function."""
        cache = create_integrated_kv_cache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            use_ring_buffer=True,
            use_quantization=False,
        )

        assert len(cache) == 4
        assert isinstance(cache, IntegratedKVCache)

    def test_multi_layer_structure(self):
        """Test all layers are created."""
        cache = create_integrated_kv_cache(num_layers=28)

        assert len(cache) == 28
        for layer in cache:
            assert isinstance(layer, IntegratedKVCacheLayer)

    def test_reset(self):
        """Test reset clears all layers."""
        cache = create_integrated_kv_cache(num_layers=4)

        # Update first layer
        cache[0].update_cache(
            torch.randn(1, 50, 12, 128, device='cuda', dtype=torch.bfloat16),
            torch.randn(1, 50, 12, 128, device='cuda', dtype=torch.bfloat16),
            50, 50
        )

        cache.reset()

        for layer in cache:
            assert layer._valid_len == 0

    def test_memory_savings_quantized(self):
        """Test total memory savings with quantization."""
        cache = create_integrated_kv_cache(
            num_layers=28,
            num_heads=12,
            head_dim=128,
            use_quantization=True,
        )

        savings = cache.memory_savings()
        # Should save at least 40% with INT8
        assert savings > 0.4


class TestPrecisionConversion:
    """Tests for model precision conversion."""

    def test_convert_to_fp16(self):
        """Test FP16 conversion."""
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        ).cuda()

        # Convert
        model = convert_model_to_fp16(model)

        # Check all params
        for param in model.parameters():
            assert param.dtype == torch.float16

    def test_convert_to_bf16(self):
        """Test BF16 conversion."""
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        ).cuda()

        model = convert_model_to_bf16(model)

        for param in model.parameters():
            assert param.dtype == torch.bfloat16

    def test_fp16_bias_conversion(self):
        """Test that biases are also converted."""
        model = torch.nn.Linear(64, 64, bias=True).cuda()

        model = convert_model_to_fp16(model)

        assert model.weight.dtype == torch.float16
        assert model.bias.dtype == torch.float16


class TestRingBufferBehavior:
    """Tests for ring buffer wraparound."""

    @pytest.fixture
    def cache(self):
        """Small cache for testing wraparound."""
        return create_integrated_kv_cache(
            num_layers=1,
            num_heads=4,
            head_dim=32,
            local_window_frames=2,  # Small window
            sink_frames=1,
            frame_seq_length=10,  # Small for testing
            use_ring_buffer=True,
            use_quantization=False,
        )

    def test_no_wraparound(self, cache):
        """Test before buffer fills."""
        layer = cache[0]

        # Add small update
        new_k = torch.randn(1, 10, 4, 32, device='cuda', dtype=torch.bfloat16)
        layer.update_cache(new_k, new_k.clone(), 10, 10)

        assert layer._valid_len == 10
        assert layer._write_idx == 10

    def test_wraparound(self, cache):
        """Test ring buffer wraparound."""
        layer = cache[0]
        total_size = layer.total_size  # sink + local

        # Fill beyond capacity
        for i in range(5):
            new_k = torch.randn(1, 10, 4, 32, device='cuda', dtype=torch.bfloat16)
            layer.update_cache(new_k, new_k.clone(), (i+1)*10, (i+1)*10)

        # Valid length should be capped
        assert layer._valid_len <= total_size

        # Write index should have wrapped
        assert layer._write_idx < total_size
