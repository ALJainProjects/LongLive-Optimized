"""
Latency Profiler for LongLive inference.

This module provides detailed latency profiling using CUDA events to measure:
1. Steady-state inter-frame latency
2. Prompt-switch latency
3. Component-level breakdown (denoising, VAE, KV ops, etc.)

All measurements use GPU-accurate CUDA events rather than wall clock time.
"""

import torch
import torch.cuda
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class LatencyMeasurement:
    """Container for a single latency measurement with statistics."""
    name: str
    samples: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if len(self.samples) > 1 else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.samples)) if self.samples else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0

    @property
    def p50(self) -> float:
        return float(np.percentile(self.samples, 50)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.samples, 99)) if self.samples else 0.0

    def add_sample(self, value: float):
        self.samples.append(value)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'p50': self.p50,
            'p95': self.p95,
            'p99': self.p99,
            'num_samples': len(self.samples),
        }


class LatencyProfiler:
    """
    Profiler for measuring LongLive inference latency with CUDA events.

    Usage:
        profiler = LatencyProfiler()

        # Profile a section
        with profiler.measure("denoising_step_0"):
            model.denoise(...)

        # Get results
        profiler.print_report()
        results = profiler.get_summary()
    """

    def __init__(self, enabled: bool = True, use_cuda_events: bool = True):
        """
        Initialize the profiler.

        Args:
            enabled: If False, profiling is a no-op (zero overhead)
            use_cuda_events: Use CUDA events for GPU timing (recommended)
        """
        self.enabled = enabled
        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()

        # Storage for measurements
        self.measurements: Dict[str, LatencyMeasurement] = {}

        # Current context stack for nested measurements
        self._context_stack: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        # Frame-level tracking
        self.frame_start_events: List[torch.cuda.Event] = []
        self.frame_end_events: List[torch.cuda.Event] = []
        self.current_frame_idx: int = 0

        # Prompt switch tracking
        self.prompt_switch_start: Optional[torch.cuda.Event] = None
        self.prompt_switch_end: Optional[torch.cuda.Event] = None

        # Device-host sync tracking
        self._sync_time_ms = 0.0
        self._sync_count = 0

    def measure_sync(self):
        """Measure a device↔host synchronization."""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda.synchronize()
            end.record()
            end.synchronize()  # Wait for end event
            self._sync_time_ms += start.elapsed_time(end)
            self._sync_count += 1

    def get_sync_stats(self) -> Dict:
        """Get synchronization statistics."""
        return {
            'total_sync_time_ms': self._sync_time_ms,
            'sync_count': self._sync_count,
            'avg_sync_ms': self._sync_time_ms / max(1, self._sync_count),
        }

    def reset(self):
        """Clear all measurements and reset state."""
        self.measurements.clear()
        self._context_stack.clear()
        self.frame_start_events.clear()
        self.frame_end_events.clear()
        self.current_frame_idx = 0
        self.prompt_switch_start = None
        self.prompt_switch_end = None
        self._sync_time_ms = 0.0
        self._sync_count = 0

    @contextmanager
    def measure(self, name: str):
        """
        Context manager for measuring a section of code.

        Args:
            name: Identifier for this measurement (e.g., "denoising_step_0")

        Usage:
            with profiler.measure("vae_decode"):
                decoded = vae.decode(latents)
        """
        if not self.enabled:
            yield
            return

        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            self._context_stack.append((name, start, end))
            try:
                yield
            finally:
                end.record()
                self._context_stack.pop()

                # Synchronize to get accurate timing
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)  # milliseconds

                if name not in self.measurements:
                    self.measurements[name] = LatencyMeasurement(name=name)
                self.measurements[name].add_sample(elapsed)
        else:
            import time
            start_time = time.perf_counter()
            try:
                yield
            finally:
                elapsed = (time.perf_counter() - start_time) * 1000  # ms
                if name not in self.measurements:
                    self.measurements[name] = LatencyMeasurement(name=name)
                self.measurements[name].add_sample(elapsed)

    def start_frame(self):
        """Mark the start of a new frame generation."""
        if not self.enabled:
            return

        if self.use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.frame_start_events.append(event)

    def end_frame(self):
        """Mark the end of frame generation."""
        if not self.enabled:
            return

        if self.use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.frame_end_events.append(event)
            self.current_frame_idx += 1

    def start_prompt_switch(self):
        """Mark the start of a prompt switch operation."""
        if not self.enabled:
            return

        if self.use_cuda_events:
            self.prompt_switch_start = torch.cuda.Event(enable_timing=True)
            self.prompt_switch_start.record()

    def end_prompt_switch(self):
        """Mark the end of a prompt switch operation."""
        if not self.enabled:
            return

        if self.use_cuda_events:
            self.prompt_switch_end = torch.cuda.Event(enable_timing=True)
            self.prompt_switch_end.record()

    def get_inter_frame_latencies(self) -> List[float]:
        """
        Calculate inter-frame latencies.

        Returns:
            List of latencies in ms between consecutive frame completions.
        """
        if not self.enabled or not self.use_cuda_events:
            return []

        torch.cuda.synchronize()

        latencies = []
        for i in range(len(self.frame_end_events) - 1):
            # Time from end of frame i to end of frame i+1
            latency = self.frame_end_events[i].elapsed_time(self.frame_end_events[i + 1])
            latencies.append(latency)

        return latencies

    def get_prompt_switch_latency(self) -> Optional[float]:
        """Get the measured prompt switch latency in ms."""
        if not self.enabled or not self.use_cuda_events:
            return None

        if self.prompt_switch_start is None or self.prompt_switch_end is None:
            return None

        torch.cuda.synchronize()
        return self.prompt_switch_start.elapsed_time(self.prompt_switch_end)

    def get_summary(self) -> Dict:
        """
        Get a summary of all measurements.

        Returns:
            Dictionary containing all measurement statistics.
        """
        summary = {
            'components': {name: m.to_dict() for name, m in self.measurements.items()},
            'inter_frame_latencies': [],
            'prompt_switch_latency': None,
        }

        # Add inter-frame latencies
        inter_frame = self.get_inter_frame_latencies()
        if inter_frame:
            summary['inter_frame_latencies'] = {
                'mean': float(np.mean(inter_frame)),
                'std': float(np.std(inter_frame)),
                'p50': float(np.percentile(inter_frame, 50)),
                'p95': float(np.percentile(inter_frame, 95)),
                'p99': float(np.percentile(inter_frame, 99)),
                'max': float(np.max(inter_frame)),
                'min': float(np.min(inter_frame)),
                'all_values': inter_frame,
            }

        # Add prompt switch latency
        ps_latency = self.get_prompt_switch_latency()
        if ps_latency is not None:
            summary['prompt_switch_latency'] = ps_latency

        return summary

    def print_report(self, title: str = "Latency Profiling Report"):
        """Print a formatted profiling report."""
        print("\n" + "=" * 70)
        print(f" {title}")
        print("=" * 70)

        # Print component breakdown
        if self.measurements:
            print("\nComponent Breakdown:")
            print("-" * 70)
            print(f"{'Component':<30} {'Mean':>10} {'P99':>10} {'Max':>10} {'Count':>8}")
            print("-" * 70)

            # Sort by mean time descending
            sorted_measurements = sorted(
                self.measurements.items(),
                key=lambda x: x[1].mean,
                reverse=True
            )

            total_time = sum(m.mean for _, m in sorted_measurements)

            for name, m in sorted_measurements:
                pct = (m.mean / total_time * 100) if total_time > 0 else 0
                print(f"{name:<30} {m.mean:>8.2f}ms {m.p99:>8.2f}ms {m.max:>8.2f}ms {len(m.samples):>8}")

            print("-" * 70)
            print(f"{'Total':<30} {total_time:>8.2f}ms")

        # Print inter-frame latencies
        inter_frame = self.get_inter_frame_latencies()
        if inter_frame:
            print("\nInter-Frame Latency (Steady-State):")
            print("-" * 70)
            print(f"  Mean:  {np.mean(inter_frame):.2f} ms")
            print(f"  Std:   {np.std(inter_frame):.2f} ms")
            print(f"  P50:   {np.percentile(inter_frame, 50):.2f} ms")
            print(f"  P95:   {np.percentile(inter_frame, 95):.2f} ms")
            print(f"  P99:   {np.percentile(inter_frame, 99):.2f} ms (WORST CASE TARGET)")
            print(f"  Max:   {np.max(inter_frame):.2f} ms")
            print(f"  Frames: {len(inter_frame) + 1}")

            # Check 40ms target
            if np.max(inter_frame) <= 40:
                print(f"\n  40ms Target: PASS")
            else:
                print(f"\n  40ms Target: FAIL (max {np.max(inter_frame):.2f}ms > 40ms)")

        # Print prompt switch latency
        ps_latency = self.get_prompt_switch_latency()
        if ps_latency is not None:
            print("\nPrompt Switch Latency:")
            print("-" * 70)
            print(f"  Latency: {ps_latency:.2f} ms")

        # Print sync stats
        sync_stats = self.get_sync_stats()
        if sync_stats['sync_count'] > 0:
            print("\nDevice↔Host Sync:")
            print("-" * 70)
            print(f"  Total: {sync_stats['total_sync_time_ms']:.2f} ms")
            print(f"  Count: {sync_stats['sync_count']}")
            print(f"  Avg:   {sync_stats['avg_sync_ms']:.2f} ms/sync")

        print("=" * 70 + "\n")


class ProfilingContext:
    """
    High-level profiling context for entire inference runs.

    Usage:
        with ProfilingContext("balanced_preset") as ctx:
            for i in range(num_frames):
                ctx.profiler.start_frame()
                frame = pipeline.generate_frame()
                ctx.profiler.end_frame()

        ctx.save_results("results/profiling_balanced.json")
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.profiler = LatencyProfiler(enabled=enabled)
        self.start_memory: Optional[int] = None
        self.peak_memory: Optional[int] = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated()

    def save_results(self, path: str):
        """Save profiling results to JSON."""
        import json

        results = self.profiler.get_summary()
        results['name'] = self.name
        results['start_memory_mb'] = self.start_memory / (1024 ** 2) if self.start_memory else None
        results['peak_memory_mb'] = self.peak_memory / (1024 ** 2) if self.peak_memory else None

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
