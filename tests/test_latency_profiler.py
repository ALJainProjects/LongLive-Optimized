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
        assert profiler.current_frame_idx == 0

    def test_measure_context(self, profiler):
        """Test measurement context manager."""
        with profiler.measure("test_operation"):
            # Simulate some GPU work
            x = torch.randn(1000, 1000, device='cuda')
            y = x @ x.T
            torch.cuda.synchronize()

        assert "test_operation" in profiler.measurements
        assert len(profiler.measurements["test_operation"].samples) == 1

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

        assert len(profiler.measurements["repeated"].samples) == 5

    def test_get_summary(self, profiler):
        """Test statistics calculation via get_summary."""
        # Add some measurements
        for _ in range(10):
            with profiler.measure("test"):
                x = torch.randn(500, 500, device='cuda')
                y = x @ x.T
                torch.cuda.synchronize()

        summary = profiler.get_summary()

        assert "components" in summary
        assert "test" in summary["components"]
        stats = summary["components"]["test"]

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p50" in stats
        assert "p99" in stats
        assert stats["num_samples"] == 10

    def test_reset(self, profiler):
        """Test profiler reset."""
        with profiler.measure("test"):
            x = torch.randn(100, 100, device='cuda')

        profiler.reset()

        assert profiler.measurements == {}
        assert profiler.current_frame_idx == 0

    def test_frame_timing(self, profiler):
        """Test frame-level timing."""
        profiler.start_frame()

        # Simulate frame generation
        x = torch.randn(100, 100, device='cuda')
        y = x @ x.T
        torch.cuda.synchronize()

        profiler.end_frame()

        assert profiler.current_frame_idx == 1
        assert len(profiler.frame_end_events) == 1

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

    def test_print_report(self, profiler):
        """Test report generation via print_report."""
        # Add some measurements
        for _ in range(5):
            with profiler.measure("denoise"):
                x = torch.randn(100, 100, device='cuda')
                torch.cuda.synchronize()

            with profiler.measure("vae"):
                y = torch.randn(50, 50, device='cuda')
                torch.cuda.synchronize()

        # print_report should not raise
        profiler.print_report()

        # Verify measurements exist
        assert "denoise" in profiler.measurements
        assert "vae" in profiler.measurements

    def test_cuda_event_accuracy(self, profiler):
        """Test CUDA event timing is accurate."""
        # Time a known-duration operation
        with profiler.measure("compute"):
            # Use GPU operation with known cost
            x = torch.randn(2000, 2000, device='cuda')
            for _ in range(10):
                x = x @ x.T
            torch.cuda.synchronize()

        measurement = profiler.measurements["compute"]

        # Should measure some non-zero time
        assert measurement.mean > 0
        assert measurement.max >= measurement.mean
        assert measurement.min <= measurement.mean


class TestProfilingContext:
    """Tests for ProfilingContext class."""

    def test_context_creation(self):
        """Test creating profiling context."""
        ctx = ProfilingContext("test")

        assert ctx.name == "test"
        assert ctx.profiler is not None

    def test_context_timing(self):
        """Test context measures memory correctly."""
        with ProfilingContext("test") as ctx:
            x = torch.randn(500, 500, device='cuda')
            y = x @ x.T
            torch.cuda.synchronize()

        # Peak memory should be recorded
        assert ctx.peak_memory is not None
        assert ctx.peak_memory > 0

    def test_context_profiler_access(self):
        """Test profiler is accessible in context."""
        with ProfilingContext("test") as ctx:
            # Use profiler within context
            with ctx.profiler.measure("inner_op"):
                x = torch.randn(100, 100, device='cuda')
                torch.cuda.synchronize()

        assert "inner_op" in ctx.profiler.measurements


class TestLatencyMeasurement:
    """Tests for LatencyMeasurement dataclass."""

    def test_measurement_creation(self):
        """Test creating measurement."""
        m = LatencyMeasurement(name="test")

        assert m.name == "test"
        assert m.samples == []

    def test_measurement_add_sample(self):
        """Test adding samples."""
        m = LatencyMeasurement(name="test")
        m.add_sample(10.0)
        m.add_sample(20.0)

        assert len(m.samples) == 2
        assert m.mean == 15.0

    def test_measurement_statistics(self):
        """Test computed statistics."""
        m = LatencyMeasurement(name="test")
        for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
            m.add_sample(val)

        assert m.mean == 30.0
        assert m.min == 10.0
        assert m.max == 50.0
        assert m.p50 == 30.0

    def test_measurement_to_dict(self):
        """Test conversion to dictionary."""
        m = LatencyMeasurement(name="test")
        m.add_sample(10.0)
        m.add_sample(20.0)

        d = m.to_dict()

        assert d["name"] == "test"
        assert d["mean"] == 15.0
        assert d["num_samples"] == 2

    def test_measurement_empty(self):
        """Test statistics with no samples."""
        m = LatencyMeasurement(name="empty")

        assert m.mean == 0.0
        assert m.min == 0.0
        assert m.max == 0.0
        assert m.std == 0.0
