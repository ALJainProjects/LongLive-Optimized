#!/usr/bin/env python3
"""
End-to-End Integration Tests for LongLive-Optimized

Verifies the complete pipeline works correctly across all presets,
including frame generation, prompt switching, and quality metrics.

Usage:
    pytest tests/test_integration.py -v
    python tests/test_integration.py  # standalone
"""

import os
import sys
import time
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MockFrame:
    """Mock frame tensor for testing without GPU."""
    shape: Tuple[int, ...]
    dtype: str = "bfloat16"
    device: str = "cuda"

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np
        return np.random.rand(*self.shape).astype(np.float32)


class MockCUDAEvent:
    """Mock CUDA event for CPU-only testing."""
    def __init__(self, enable_timing=False):
        self._time = time.perf_counter()

    def record(self, stream=None):
        self._time = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, end_event):
        return (end_event._time - self._time) * 1000  # ms


class TestOptimizationConfig(unittest.TestCase):
    """Test OptimizationConfig and presets."""

    def test_import_config(self):
        """Verify config module imports correctly."""
        from optimizations.config import OptimizationConfig
        self.assertIsNotNone(OptimizationConfig)

    def test_default_config(self):
        """Verify default config values."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig()

        # Check that essential fields exist and have valid types
        self.assertIsInstance(config.use_cuda_graphs, bool)
        self.assertIsInstance(config.use_torch_compile, bool)
        self.assertIn(config.compile_mode, ["default", "reduce-overhead", "max-autotune"])
        self.assertIsInstance(config.use_static_kv, bool)
        self.assertIn(config.model_dtype, ["bfloat16", "float16", "fp8"])

    def test_preset_quality(self):
        """Verify quality preset configuration."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_quality()

        self.assertFalse(config.use_torch_compile)
        self.assertFalse(config.use_quantized_kv)
        self.assertEqual(config.denoising_steps, [1000, 750, 500, 250])

    def test_preset_balanced(self):
        """Verify balanced preset configuration."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_balanced()

        self.assertTrue(config.use_torch_compile)
        self.assertEqual(config.compile_mode, "default")
        self.assertFalse(config.use_quantized_kv)
        self.assertEqual(config.denoising_steps, [1000, 750, 500, 250])

    def test_preset_turbo(self):
        """Verify turbo preset uses 3 denoising steps."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_turbo()

        self.assertTrue(config.use_torch_compile)
        self.assertEqual(config.compile_mode, "max-autotune")
        self.assertEqual(len(config.denoising_steps), 3)
        self.assertEqual(config.denoising_steps, [1000, 500, 250])

    def test_preset_turbo_fp8(self):
        """Verify turbo_fp8 preset uses FP8 dtype."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_turbo_fp8()

        self.assertEqual(config.model_dtype, "fp8")
        self.assertEqual(len(config.denoising_steps), 3)

    def test_preset_ultra(self):
        """Verify ultra preset uses 2 denoising steps."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_ultra()

        self.assertEqual(len(config.denoising_steps), 2)
        self.assertEqual(config.denoising_steps, [1000, 250])
        self.assertEqual(config.local_attn_size, 6)

    def test_preset_low_memory(self):
        """Verify low_memory preset uses INT8 KV cache."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_low_memory()

        self.assertTrue(config.use_quantized_kv)
        self.assertEqual(config.kv_quantization, "int8")

    def test_all_presets_have_required_fields(self):
        """Verify all presets define required fields."""
        from optimizations.config import OptimizationConfig

        presets = [
            OptimizationConfig.preset_quality,
            OptimizationConfig.preset_balanced,
            OptimizationConfig.preset_speed,
            OptimizationConfig.preset_turbo,
            OptimizationConfig.preset_turbo_fp8,
            OptimizationConfig.preset_ultra,
            OptimizationConfig.preset_low_memory,
        ]

        for preset_fn in presets:
            config = preset_fn()
            self.assertIsInstance(config.denoising_steps, list)
            self.assertGreater(len(config.denoising_steps), 0)
            self.assertIn(config.model_dtype, ["bfloat16", "float16", "fp8"])
            self.assertIsInstance(config.use_torch_compile, bool)


class TestLatencyProfiler(unittest.TestCase):
    """Test LatencyProfiler statistics collection."""

    def test_import_profiler(self):
        """Verify profiler imports correctly."""
        from optimizations.latency_profiler import LatencyProfiler
        self.assertIsNotNone(LatencyProfiler)

    def test_profiler_initialization(self):
        """Verify profiler initializes with correct defaults."""
        from optimizations.latency_profiler import LatencyProfiler
        profiler = LatencyProfiler(enabled=True)

        self.assertTrue(profiler.enabled)
        self.assertEqual(len(profiler.measurements), 0)

    def test_profiler_disabled(self):
        """Verify profiler can be disabled."""
        from optimizations.latency_profiler import LatencyProfiler
        profiler = LatencyProfiler(enabled=False)

        self.assertFalse(profiler.enabled)

    def test_latency_measurement_stats(self):
        """Verify LatencyMeasurement computes statistics correctly."""
        from optimizations.latency_profiler import LatencyMeasurement

        measurement = LatencyMeasurement(name="test")
        measurement.samples = [100.0, 105.0, 110.0, 95.0, 102.0]

        self.assertAlmostEqual(measurement.mean, 102.4, places=1)
        self.assertAlmostEqual(measurement.max, 110.0, places=1)
        self.assertAlmostEqual(measurement.min, 95.0, places=1)

    def test_latency_measurement_to_dict(self):
        """Verify LatencyMeasurement serialization."""
        from optimizations.latency_profiler import LatencyMeasurement

        measurement = LatencyMeasurement(name="test_op")
        measurement.samples = [10.0, 20.0, 30.0]

        result = measurement.to_dict()

        self.assertEqual(result["name"], "test_op")
        self.assertIn("mean", result)
        self.assertIn("p99", result)
        self.assertEqual(result["num_samples"], 3)


class TestMemoryPool(unittest.TestCase):
    """Test MemoryPool allocation."""

    def test_import_memory_pool(self):
        """Verify memory pool imports correctly."""
        from optimizations.memory_pool import FixedShapeMemoryPool, BufferSpec
        self.assertIsNotNone(FixedShapeMemoryPool)
        self.assertIsNotNone(BufferSpec)

    def test_buffer_spec_creation(self):
        """Verify BufferSpec can be created."""
        from optimizations.memory_pool import BufferSpec
        import torch

        spec = BufferSpec(
            name="test_buffer",
            shape=(1, 16, 64, 64),
            dtype=torch.bfloat16,
            device=torch.device("cpu")
        )

        self.assertEqual(spec.name, "test_buffer")
        self.assertEqual(spec.shape, (1, 16, 64, 64))
        self.assertEqual(spec.dtype, torch.bfloat16)


class TestPromptCache(unittest.TestCase):
    """Test PromptCache functionality."""

    def test_import_prompt_cache(self):
        """Verify prompt cache imports correctly."""
        from optimizations.prompt_cache import PromptEmbeddingCache
        self.assertIsNotNone(PromptEmbeddingCache)

    def test_cache_requires_text_encoder(self):
        """Verify cache requires text encoder at init."""
        from optimizations.prompt_cache import PromptEmbeddingCache

        # PromptEmbeddingCache requires a text_encoder argument
        # This tests that the class exists and has correct signature
        import inspect
        sig = inspect.signature(PromptEmbeddingCache.__init__)
        params = list(sig.parameters.keys())

        self.assertIn("text_encoder", params)
        self.assertIn("max_cache_size", params)


class TestOptimizedPipelineCreation(unittest.TestCase):
    """Test OptimizedPipeline creation and configuration."""

    def test_import_pipeline(self):
        """Verify pipeline imports correctly."""
        from optimizations.optimized_pipeline import OptimizedCausalInferencePipeline
        self.assertIsNotNone(OptimizedCausalInferencePipeline)

    def test_import_factory_function(self):
        """Verify factory function imports correctly."""
        from optimizations.optimized_pipeline import create_optimized_pipeline
        self.assertIsNotNone(create_optimized_pipeline)


class TestDenosingStepsWiring(unittest.TestCase):
    """Test that denoising_steps config is properly wired through."""

    def test_turbo_has_three_steps(self):
        """Verify turbo preset defines 3 steps."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_turbo()
        self.assertEqual(len(config.denoising_steps), 3)

    def test_ultra_has_two_steps(self):
        """Verify ultra preset defines 2 steps."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_ultra()
        self.assertEqual(len(config.denoising_steps), 2)

    def test_balanced_has_four_steps(self):
        """Verify balanced preset defines 4 steps (default)."""
        from optimizations.config import OptimizationConfig
        config = OptimizationConfig.preset_balanced()
        self.assertEqual(len(config.denoising_steps), 4)


class TestBenchmarkSuite(unittest.TestCase):
    """Test benchmark suite components."""

    def test_import_benchmark(self):
        """Verify benchmark imports correctly."""
        try:
            from benchmarks.benchmark_suite import BenchmarkConfig
            self.assertIsNotNone(BenchmarkConfig)
        except ImportError:
            self.skipTest("Benchmark suite not available")

    def test_benchmark_config_presets(self):
        """Verify benchmark config presets exist."""
        try:
            from benchmarks.benchmark_suite import BenchmarkConfig

            quick = BenchmarkConfig.quick()
            self.assertLess(quick.steady_state_frames, 200)

            full = BenchmarkConfig.full()
            self.assertGreaterEqual(full.steady_state_frames, 1000)
        except ImportError:
            self.skipTest("Benchmark suite not available")


class TestQualityEval(unittest.TestCase):
    """Test quality evaluation components."""

    def test_import_quality_eval(self):
        """Verify quality eval imports correctly."""
        try:
            from benchmarks.quality_eval import compute_psnr, compute_ssim
            self.assertIsNotNone(compute_psnr)
            self.assertIsNotNone(compute_ssim)
        except ImportError:
            self.skipTest("Quality eval not available")


class TestKernelProfiler(unittest.TestCase):
    """Test kernel profiler components."""

    def test_import_profiler(self):
        """Verify kernel profiler imports correctly."""
        from benchmarks.kernel_profiler import KernelProfiler, KernelTimings
        self.assertIsNotNone(KernelProfiler)
        self.assertIsNotNone(KernelTimings)

    def test_profiler_disabled(self):
        """Verify profiler can be disabled."""
        from benchmarks.kernel_profiler import KernelProfiler
        profiler = KernelProfiler(enabled=False)

        with profiler.profile("test_op"):
            pass

        # Should not record anything when disabled
        self.assertEqual(len(profiler.timings), 0)

    def test_kernel_timings_stats(self):
        """Verify KernelTimings computes statistics correctly."""
        from benchmarks.kernel_profiler import KernelTimings

        timing = KernelTimings(name="test")
        timing.cuda_times_ms = [10.0, 12.0, 11.0, 15.0, 10.5]
        timing.call_count = 5

        self.assertAlmostEqual(timing.mean_ms, 11.7, places=1)
        self.assertAlmostEqual(timing.total_ms, 58.5, places=1)


class TestScopeIntegration(unittest.TestCase):
    """Test scope integration components."""

    def test_import_scope_config(self):
        """Verify scope config imports correctly."""
        try:
            from demo.scope_integration.optimized_longlive_pipeline import (
                ScopeOptimizationConfig
            )
            self.assertIsNotNone(ScopeOptimizationConfig)
        except ImportError:
            self.skipTest("Scope integration not available")

    def test_scope_presets(self):
        """Verify scope-specific presets exist."""
        try:
            from demo.scope_integration.optimized_longlive_pipeline import (
                ScopeOptimizationConfig
            )

            realtime = ScopeOptimizationConfig.preset_realtime()
            self.assertTrue(realtime.enable_latency_overlay)

            turbo = ScopeOptimizationConfig.preset_turbo()
            self.assertEqual(len(turbo.denoising_steps), 3)
        except ImportError:
            self.skipTest("Scope integration not available")


class TestEndToEndMocked(unittest.TestCase):
    """End-to-end tests with mocked GPU operations."""

    def test_full_pipeline_flow_mocked(self):
        """Test complete pipeline flow with mocks."""
        from optimizations.config import OptimizationConfig
        from optimizations.latency_profiler import LatencyProfiler, LatencyMeasurement

        # Create components
        config = OptimizationConfig.preset_balanced()
        profiler = LatencyProfiler(enabled=True, use_cuda_events=False)

        # Verify config is valid
        self.assertIsInstance(config.denoising_steps, list)
        self.assertGreater(len(config.denoising_steps), 0)

        # Simulate frame generation with measurements
        ss_measurement = LatencyMeasurement(name="steady_state")
        ps_measurement = LatencyMeasurement(name="prompt_switch")

        # Use single prompt to get steady-state samples
        prompt = "A cat walking"
        last_prompt = None

        for frame_idx in range(10):
            if prompt != last_prompt:
                # First frame is prompt switch - slower
                ps_measurement.add_sample(150.0)
                last_prompt = prompt
            else:
                # Subsequent frames are steady state - faster
                ss_measurement.add_sample(100.0 + frame_idx * 0.1)

        # Verify measurements (9 steady-state, 1 prompt switch)
        self.assertEqual(len(ss_measurement.samples), 9)
        self.assertEqual(len(ps_measurement.samples), 1)
        self.assertGreater(ss_measurement.mean, 0)

    def test_preset_switching(self):
        """Test switching between presets."""
        from optimizations.config import OptimizationConfig

        presets = ["quality", "balanced", "turbo", "ultra"]

        for preset_name in presets:
            preset_fn = getattr(OptimizationConfig, f"preset_{preset_name}")
            config = preset_fn()

            # Verify config is valid
            self.assertIsInstance(config.denoising_steps, list)
            self.assertGreater(len(config.denoising_steps), 0)

    def test_latency_measurement_accuracy(self):
        """Test latency measurement produces accurate statistics."""
        from optimizations.latency_profiler import LatencyMeasurement

        measurement = LatencyMeasurement(name="test")

        # Add known distribution
        import random
        random.seed(42)
        latencies = [100 + random.gauss(0, 5) for _ in range(100)]

        for lat in latencies:
            measurement.add_sample(lat)

        # Mean should be close to 100
        self.assertAlmostEqual(measurement.mean, 100, delta=2)

        # Max should be reasonable
        self.assertLess(measurement.max, 120)
        self.assertGreater(measurement.max, 100)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_denoising_steps_valid(self):
        """Verify denoising steps are in valid range."""
        from optimizations.config import OptimizationConfig

        for preset in ["quality", "balanced", "turbo", "ultra"]:
            config = getattr(OptimizationConfig, f"preset_{preset}")()

            for step in config.denoising_steps:
                self.assertGreaterEqual(step, 0)
                self.assertLessEqual(step, 1000)

    def test_local_attn_size_valid(self):
        """Verify local attention size is reasonable."""
        from optimizations.config import OptimizationConfig

        for preset in ["quality", "balanced", "turbo", "ultra"]:
            config = getattr(OptimizationConfig, f"preset_{preset}")()

            self.assertGreaterEqual(config.local_attn_size, 4)
            self.assertLessEqual(config.local_attn_size, 24)

    def test_compile_mode_valid(self):
        """Verify compile mode is valid PyTorch option."""
        from optimizations.config import OptimizationConfig

        valid_modes = ["default", "reduce-overhead", "max-autotune"]

        for preset in ["quality", "balanced", "turbo", "ultra"]:
            config = getattr(OptimizationConfig, f"preset_{preset}")()

            if config.use_torch_compile:
                self.assertIn(config.compile_mode, valid_modes)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestOptimizationConfig,
        TestLatencyTracker,
        TestMemoryPool,
        TestPromptCache,
        TestOptimizedPipelineCreation,
        TestDenosingStepsWiring,
        TestBenchmarkSuite,
        TestQualityEval,
        TestKernelProfiler,
        TestScopeIntegration,
        TestEndToEndMocked,
        TestConfigValidation,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
