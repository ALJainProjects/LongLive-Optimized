"""
Tests for LatencyProfiler

Tests CUDA event timing, measurement collection, and statistics.
"""

import pytest
import torch
import time

from optimizations.latency_profiler import LatencyProfiler, ProfilingContext, LatencyMeasurement


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestLatencyProfiler:
    """Tests for LatencyProfiler class."""

    @pytest.fixture
    def profiler(self):
        """Create profiler instance."""
        return LatencyProfiler()

    def test_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.measurements == {}
        assert profiler.frame_count == 0

    def test_measure_context(self, profiler):
        """Test measurement context manager."""
        with profiler.measure("test_operation"):
            # Simulate some GPU work
            x = torch.randn(1000, 1000, device='cuda')
            y = x @ x.T
            torch.cuda.synchronize()

        assert "test_operation" in profiler.measurements
        assert len(profiler.measurements["test_operation"]) == 1

    def test_nested_measurements(self, profiler):
        """Test nested measurement contexts."""
        with profiler.measure("outer"):
            x = torch.randn(100, 100, device='cuda')

            with profiler.measure("inner"):
                y = x @ x.T
                torch.cuda.synchronize()

            z = y + x

        assert "outer" in profiler.measurements
        assert "inner" in profiler.measurements

    def test_multiple_measurements(self, profiler):
        """Test collecting multiple measurements."""
        for i in range(5):
            with profiler.measure("repeated"):
                x = torch.randn(100, 100, device='cuda')
                torch.cuda.synchronize()

        assert len(profiler.measurements["repeated"]) == 5

    def test_get_stats(self, profiler):
        """Test statistics calculation."""
        # Add some measurements
        for _ in range(10):
            with profiler.measure("test"):
                x = torch.randn(500, 500, device='cuda')
                y = x @ x.T
                torch.cuda.synchronize()

        stats = profiler.get_stats("test")

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p50" in stats
        assert "p99" in stats
        assert stats["count"] == 10

    def test_reset(self, profiler):
        """Test profiler reset."""
        with profiler.measure("test"):
            x = torch.randn(100, 100, device='cuda')

        profiler.reset()

        assert profiler.measurements == {}
        assert profiler.frame_count == 0

    def test_frame_timing(self, profiler):
        """Test frame-level timing."""
        profiler.start_frame()

        # Simulate frame generation
        x = torch.randn(100, 100, device='cuda')
        y = x @ x.T
        torch.cuda.synchronize()

        profiler.end_frame()

        assert profiler.frame_count == 1
        assert len(profiler.frame_times) == 1

    def test_inter_frame_latency(self, profiler):
        """Test inter-frame latency calculation."""
        # Generate multiple frames
        for i in range(5):
            profiler.start_frame()
            x = torch.randn(100, 100, device='cuda')
            torch.cuda.synchronize()
            profiler.end_frame()

        latencies = profiler.get_inter_frame_latencies()

        assert len(latencies) == 4  # N-1 gaps between N frames

    def test_report_generation(self, profiler):
        """Test report generation."""
        # Add some measurements
        for _ in range(5):
            with profiler.measure("denoise"):
                x = torch.randn(100, 100, device='cuda')
                torch.cuda.synchronize()

            with profiler.measure("vae"):
                y = torch.randn(50, 50, device='cuda')
                torch.cuda.synchronize()

        report = profiler.generate_report()

        assert "denoise" in report
        assert "vae" in report
        assert "mean" in report

    def test_cuda_event_accuracy(self, profiler):
        """Test CUDA event timing is accurate."""
        # Time a known-duration operation
        with profiler.measure("sleep"):
            # Use GPU operation with known cost
            x = torch.randn(2000, 2000, device='cuda')
            for _ in range(10):
                x = x @ x.T
            torch.cuda.synchronize()

        stats = profiler.get_stats("sleep")

        # Should measure some non-zero time
        assert stats["mean"] > 0
        assert stats["max"] >= stats["mean"]
        assert stats["min"] <= stats["mean"]


class TestProfilingContext:
    """Tests for ProfilingContext class."""

    def test_context_creation(self):
        """Test creating profiling context."""
        ctx = ProfilingContext("test")

        assert ctx.name == "test"
        assert ctx.start_event is not None
        assert ctx.end_event is not None

    def test_context_timing(self):
        """Test context measures time correctly."""
        ctx = ProfilingContext("test")

        with ctx:
            x = torch.randn(500, 500, device='cuda')
            y = x @ x.T
            torch.cuda.synchronize()

        elapsed = ctx.elapsed_ms()

        assert elapsed > 0

    def test_context_reusable(self):
        """Test context can be reused."""
        ctx = ProfilingContext("test")

        times = []
        for _ in range(3):
            with ctx:
                x = torch.randn(100, 100, device='cuda')
                torch.cuda.synchronize()
            times.append(ctx.elapsed_ms())

        assert len(times) == 3
        assert all(t > 0 for t in times)


class TestLatencyMeasurement:
    """Tests for LatencyMeasurement dataclass."""

    def test_measurement_creation(self):
        """Test creating measurement."""
        m = LatencyMeasurement(
            name="test",
            start_time_ms=0.0,
            end_time_ms=10.0,
            frame_idx=0,
        )

        assert m.name == "test"
        assert m.duration_ms == 10.0

    def test_measurement_duration(self):
        """Test duration calculation."""
        m = LatencyMeasurement(
            name="test",
            start_time_ms=5.0,
            end_time_ms=15.0,
            frame_idx=0,
        )

        assert m.duration_ms == 10.0

    def test_measurement_to_dict(self):
        """Test conversion to dictionary."""
        m = LatencyMeasurement(
            name="test",
            start_time_ms=0.0,
            end_time_ms=10.0,
            frame_idx=5,
        )

        d = m.to_dict()

        assert d["name"] == "test"
        assert d["duration_ms"] == 10.0
        assert d["frame_idx"] == 5
