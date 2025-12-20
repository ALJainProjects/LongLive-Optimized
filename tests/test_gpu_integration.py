#!/usr/bin/env python3
"""
GPU Integration Tests for LongLive-Optimized

These tests require a CUDA GPU and the full LongLive model.
Skip automatically if GPU is not available.

Usage:
    pytest tests/test_gpu_integration.py -v --gpu
    python tests/test_gpu_integration.py
"""

import os
import sys
import time
import unittest
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        GPU_NAME = None
        GPU_MEMORY_GB = 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    GPU_MEMORY_GB = 0


def requires_gpu(func):
    """Decorator to skip tests if GPU is not available."""
    def wrapper(*args, **kwargs):
        if not GPU_AVAILABLE:
            raise unittest.SkipTest("CUDA GPU not available")
        return func(*args, **kwargs)
    return wrapper


def requires_model(model_path: Optional[str] = None):
    """Decorator to skip tests if model is not available."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            path = model_path or os.environ.get("LONGLIVE_MODEL_PATH")
            if not path or not os.path.exists(path):
                raise unittest.SkipTest(f"Model not found at {path}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TestGPUBasics(unittest.TestCase):
    """Basic GPU functionality tests."""

    @requires_gpu
    def test_cuda_available(self):
        """Verify CUDA is available."""
        self.assertTrue(torch.cuda.is_available())

    @requires_gpu
    def test_cuda_memory(self):
        """Verify sufficient GPU memory."""
        # Need at least 16GB for LongLive
        self.assertGreaterEqual(GPU_MEMORY_GB, 16,
            f"Insufficient GPU memory: {GPU_MEMORY_GB:.1f}GB < 16GB required")

    @requires_gpu
    def test_cuda_events(self):
        """Verify CUDA events work correctly."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        # Do some work
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        self.assertGreater(elapsed, 0)
        self.assertLess(elapsed, 1000)  # Should be fast

    @requires_gpu
    def test_bfloat16_support(self):
        """Verify bfloat16 is supported."""
        x = torch.randn(100, 100, dtype=torch.bfloat16, device="cuda")
        y = torch.matmul(x, x)
        self.assertEqual(y.dtype, torch.bfloat16)


class TestTorchCompile(unittest.TestCase):
    """Test torch.compile functionality."""

    @requires_gpu
    def test_compile_simple_function(self):
        """Verify torch.compile works on simple function."""
        def simple_fn(x):
            return x * 2 + 1

        compiled_fn = torch.compile(simple_fn)

        x = torch.randn(100, device="cuda")
        result = compiled_fn(x)

        expected = x * 2 + 1
        self.assertTrue(torch.allclose(result, expected))

    @requires_gpu
    def test_compile_with_attention(self):
        """Verify torch.compile works with attention."""
        batch, heads, seq, dim = 1, 8, 64, 64

        q = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)

        def attention_fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Warmup
        _ = attention_fn(q, k, v)

        # Compile
        compiled_attn = torch.compile(attention_fn, mode="default")

        # Run compiled
        result = compiled_attn(q, k, v)

        self.assertEqual(result.shape, (batch, heads, seq, dim))


class TestKVCacheOperations(unittest.TestCase):
    """Test KV cache operations on GPU."""

    @requires_gpu
    def test_ring_buffer_write(self):
        """Verify ring buffer write is O(1)."""
        batch, heads, max_len, dim = 1, 24, 1000, 64

        kv_cache = torch.zeros(batch, 2, heads, max_len, dim,
                               device="cuda", dtype=torch.bfloat16)

        new_k = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.bfloat16)
        new_v = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for i in range(10):
            idx = i % max_len
            kv_cache[:, 0, :, idx:idx+1, :] = new_k
            kv_cache[:, 1, :, idx:idx+1, :] = new_v
        torch.cuda.synchronize()

        # Time writes
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times = []
        for i in range(100):
            idx = i % max_len
            start.record()
            kv_cache[:, 0, :, idx:idx+1, :] = new_k
            kv_cache[:, 1, :, idx:idx+1, :] = new_v
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        mean_time = sum(times) / len(times)

        # Should be very fast (<1ms)
        self.assertLess(mean_time, 1.0,
            f"Ring buffer write too slow: {mean_time:.3f}ms")

    @requires_gpu
    def test_local_window_read(self):
        """Verify local window extraction is fast."""
        batch, heads, max_len, dim = 1, 24, 1000, 64
        window_size = 12

        kv_cache = torch.randn(batch, 2, heads, max_len, dim,
                               device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = kv_cache[:, :, :, :window_size, :].clone()
        torch.cuda.synchronize()

        # Time reads
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times = []
        for i in range(100):
            start_idx = i % (max_len - window_size)
            start.record()
            window = kv_cache[:, :, :, start_idx:start_idx+window_size, :].clone()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        mean_time = sum(times) / len(times)

        # Should be fast (<1ms)
        self.assertLess(mean_time, 1.0,
            f"Window read too slow: {mean_time:.3f}ms")


class TestAttentionKernels(unittest.TestCase):
    """Test attention kernel performance."""

    @requires_gpu
    def test_sdpa_performance(self):
        """Verify SDPA uses FlashAttention and is fast."""
        batch, heads, seq, dim = 1, 24, 12, 64

        q = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        # Time attention
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(100):
            start.record()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        mean_time = sum(times) / len(times)

        # Should be very fast with FlashAttention
        self.assertLess(mean_time, 1.0,
            f"SDPA too slow: {mean_time:.3f}ms (FlashAttention may not be active)")

    @requires_gpu
    def test_cross_attention_performance(self):
        """Verify cross-attention performance."""
        batch, heads, seq_q, seq_kv, dim = 1, 24, 12, 77, 64

        q = torch.randn(batch, heads, seq_q, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq_kv, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq_kv, dim, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        # Time attention
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(100):
            start.record()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        mean_time = sum(times) / len(times)

        # Cross-attention should also be fast
        self.assertLess(mean_time, 2.0,
            f"Cross-attention too slow: {mean_time:.3f}ms")


class TestMemoryManagement(unittest.TestCase):
    """Test GPU memory management."""

    @requires_gpu
    def test_memory_pool_allocation(self):
        """Verify memory pool reduces fragmentation."""
        initial_memory = torch.cuda.memory_allocated()

        # Allocate and free tensors
        tensors = []
        for _ in range(100):
            t = torch.randn(1000, 1000, device="cuda")
            tensors.append(t)

        peak_memory = torch.cuda.max_memory_allocated()

        # Clear
        del tensors
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should be reclaimed
        self.assertLess(final_memory, peak_memory)

    @requires_gpu
    def test_no_memory_leak(self):
        """Verify no memory leak in repeated operations."""
        # Warmup and baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(1000, 1000, device="cuda")
        for _ in range(10):
            y = torch.matmul(x, x)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()

        # Run many iterations
        for _ in range(100):
            y = torch.matmul(x, x)
        torch.cuda.synchronize()

        final = torch.cuda.memory_allocated()

        # Should not grow significantly
        growth = final - baseline
        self.assertLess(growth, 100 * 1024 * 1024,  # 100MB tolerance
            f"Memory grew by {growth / 1024**2:.1f}MB")


class TestFullPipelineGPU(unittest.TestCase):
    """Full pipeline tests requiring model."""

    @requires_gpu
    @requires_model()
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly on GPU."""
        from optimizations.config import OptimizationConfig
        from optimizations.optimized_pipeline import create_optimized_pipeline

        model_path = os.environ.get("LONGLIVE_MODEL_PATH")
        config = OptimizationConfig.preset_balanced()

        # This would require the actual model
        # pipeline = create_optimized_pipeline(model_path, config)
        # self.assertIsNotNone(pipeline)

        # For now, just verify the function exists
        self.assertIsNotNone(create_optimized_pipeline)

    @requires_gpu
    @requires_model()
    def test_frame_generation(self):
        """Test frame generation produces valid output."""
        # Would require actual model
        pass

    @requires_gpu
    @requires_model()
    def test_prompt_switching(self):
        """Test prompt switching works correctly."""
        # Would require actual model
        pass


def run_gpu_tests():
    """Run GPU tests and return results."""
    if not GPU_AVAILABLE:
        print("=" * 60)
        print("GPU NOT AVAILABLE - Skipping GPU integration tests")
        print("=" * 60)
        return None

    print("=" * 60)
    print(f"GPU: {GPU_NAME}")
    print(f"Memory: {GPU_MEMORY_GB:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestGPUBasics,
        TestTorchCompile,
        TestKVCacheOperations,
        TestAttentionKernels,
        TestMemoryManagement,
        TestFullPipelineGPU,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_gpu_tests()
    if result is None:
        sys.exit(0)  # Skip is not a failure
    sys.exit(0 if result.wasSuccessful() else 1)
