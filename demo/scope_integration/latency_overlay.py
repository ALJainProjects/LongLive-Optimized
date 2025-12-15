"""
Latency Overlay for Scope Integration

Provides real-time latency visualization overlay for the WebRTC video stream.
Shows current FPS, latency statistics, and optimization status.

Usage:
    from demo.scope_integration import LatencyOverlay, create_latency_overlay

    overlay = create_latency_overlay(position="top-right")

    # In frame generation loop
    frame = pipeline.generate_frame(prompt)
    frame_with_overlay = overlay.apply(frame, pipeline.get_latency_stats())
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque
import time

import torch
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class LatencyStats:
    """Container for latency statistics."""
    current_ms: float = 0.0
    mean_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = 0.0
    fps: float = 0.0
    frame_count: int = 0

    # Optimization status
    cuda_graphs_active: bool = False
    static_kv_active: bool = False
    async_vae_active: bool = False
    prompt_cache_hits: int = 0
    prompt_cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.prompt_cache_hits + self.prompt_cache_misses
        return self.prompt_cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'current_ms': self.current_ms,
            'mean_ms': self.mean_ms,
            'p99_ms': self.p99_ms,
            'max_ms': self.max_ms,
            'min_ms': self.min_ms,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'cuda_graphs_active': self.cuda_graphs_active,
            'static_kv_active': self.static_kv_active,
            'async_vae_active': self.async_vae_active,
            'cache_hit_rate': self.cache_hit_rate,
        }


@dataclass
class OverlayConfig:
    """Configuration for latency overlay."""
    position: str = "top-right"  # top-left, top-right, bottom-left, bottom-right
    opacity: float = 0.8
    font_scale: float = 0.6
    padding: int = 10
    line_height: int = 25
    background_color: Tuple[int, int, int] = (0, 0, 0)  # BGR
    text_color: Tuple[int, int, int] = (255, 255, 255)  # BGR
    warning_color: Tuple[int, int, int] = (0, 165, 255)  # Orange BGR
    error_color: Tuple[int, int, int] = (0, 0, 255)  # Red BGR
    good_color: Tuple[int, int, int] = (0, 255, 0)  # Green BGR

    # Thresholds for color coding
    latency_warning_ms: float = 35.0
    latency_error_ms: float = 40.0
    fps_warning: float = 28.0
    fps_error: float = 25.0

    # What to show
    show_current_latency: bool = True
    show_mean_latency: bool = True
    show_p99_latency: bool = True
    show_max_latency: bool = True
    show_fps: bool = True
    show_frame_count: bool = True
    show_optimization_status: bool = True
    show_latency_graph: bool = True
    graph_width: int = 150
    graph_height: int = 50
    graph_history: int = 60  # Number of frames to show


class LatencyOverlay:
    """
    Applies a latency information overlay to video frames.

    The overlay shows:
    - Current frame latency
    - Mean/P99/Max latency
    - Current FPS
    - Active optimizations
    - Mini latency graph (last N frames)
    """

    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
        self.latency_history: deque = deque(maxlen=self.config.graph_history)
        self._last_time = time.perf_counter()
        self._frame_times: deque = deque(maxlen=30)  # For FPS calculation

        if not HAS_CV2:
            print("Warning: OpenCV not available. Overlay will be text-only.")

    def apply(
        self,
        frame: torch.Tensor,
        stats: LatencyStats,
    ) -> torch.Tensor:
        """
        Apply latency overlay to a frame.

        Args:
            frame: Frame tensor [C, H, W] or [H, W, C]
            stats: Current latency statistics

        Returns:
            Frame with overlay applied
        """
        if not HAS_CV2:
            return frame

        # Convert tensor to numpy for OpenCV
        if isinstance(frame, torch.Tensor):
            # Handle different tensor formats
            if frame.dim() == 3:
                if frame.shape[0] in [1, 3, 4]:  # [C, H, W]
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()
                else:  # [H, W, C]
                    frame_np = frame.cpu().numpy()
            else:
                frame_np = frame.cpu().numpy()

            # Normalize to 0-255 if needed
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            if frame_np.shape[-1] == 3:
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_np = frame

        # Record latency for graph
        self.latency_history.append(stats.current_ms)

        # Create overlay
        frame_with_overlay = self._draw_overlay(frame_np, stats)

        # Convert back to tensor
        if isinstance(frame, torch.Tensor):
            # BGR to RGB
            frame_with_overlay = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
            result = torch.from_numpy(frame_with_overlay).float() / 255.0
            if frame.shape[0] in [1, 3, 4]:  # Original was [C, H, W]
                result = result.permute(2, 0, 1)
            return result.to(frame.device)

        return frame_with_overlay

    def _draw_overlay(
        self,
        frame: np.ndarray,
        stats: LatencyStats,
    ) -> np.ndarray:
        """Draw the overlay on the frame."""
        h, w = frame.shape[:2]
        cfg = self.config

        # Build text lines
        lines = []

        if cfg.show_current_latency:
            color = self._get_latency_color(stats.current_ms)
            lines.append((f"Latency: {stats.current_ms:.1f}ms", color))

        if cfg.show_mean_latency:
            color = self._get_latency_color(stats.mean_ms)
            lines.append((f"Mean: {stats.mean_ms:.1f}ms", color))

        if cfg.show_p99_latency:
            color = self._get_latency_color(stats.p99_ms)
            lines.append((f"P99: {stats.p99_ms:.1f}ms", color))

        if cfg.show_max_latency:
            color = self._get_latency_color(stats.max_ms)
            lines.append((f"Max: {stats.max_ms:.1f}ms", color))

        if cfg.show_fps:
            color = self._get_fps_color(stats.fps)
            lines.append((f"FPS: {stats.fps:.1f}", color))

        if cfg.show_frame_count:
            lines.append((f"Frame: {stats.frame_count}", cfg.text_color))

        if cfg.show_optimization_status:
            lines.append(("", cfg.text_color))  # Separator
            opts = []
            if stats.cuda_graphs_active:
                opts.append("CUDA Graphs")
            if stats.static_kv_active:
                opts.append("Static KV")
            if stats.async_vae_active:
                opts.append("Async VAE")

            if opts:
                lines.append(("Optimizations:", cfg.text_color))
                for opt in opts:
                    lines.append((f"  + {opt}", cfg.good_color))
            else:
                lines.append(("No optimizations", cfg.warning_color))

        # Calculate overlay dimensions
        text_height = len(lines) * cfg.line_height
        overlay_height = text_height + 2 * cfg.padding

        if cfg.show_latency_graph:
            overlay_height += cfg.graph_height + cfg.padding

        overlay_width = 200

        # Determine position
        if "left" in cfg.position:
            x_offset = cfg.padding
        else:
            x_offset = w - overlay_width - cfg.padding

        if "top" in cfg.position:
            y_offset = cfg.padding
        else:
            y_offset = h - overlay_height - cfg.padding

        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_offset, y_offset),
            (x_offset + overlay_width, y_offset + overlay_height),
            cfg.background_color,
            -1
        )

        # Blend with original frame
        frame = cv2.addWeighted(overlay, cfg.opacity, frame, 1 - cfg.opacity, 0)

        # Draw text
        y = y_offset + cfg.padding + 15
        for text, color in lines:
            if text:  # Skip empty lines
                cv2.putText(
                    frame,
                    text,
                    (x_offset + cfg.padding, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.font_scale,
                    color,
                    1,
                    cv2.LINE_AA
                )
            y += cfg.line_height

        # Draw latency graph
        if cfg.show_latency_graph and len(self.latency_history) > 1:
            graph_y = y + cfg.padding // 2
            self._draw_latency_graph(
                frame,
                x_offset + cfg.padding,
                graph_y,
                cfg.graph_width,
                cfg.graph_height,
            )

        return frame

    def _draw_latency_graph(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Draw a mini latency graph."""
        cfg = self.config
        latencies = list(self.latency_history)

        if len(latencies) < 2:
            return

        # Normalize latencies to graph height
        max_latency = max(max(latencies), cfg.latency_error_ms)
        min_latency = min(latencies)

        # Draw graph background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 30), -1)

        # Draw threshold lines
        warning_y = int(y + height - (cfg.latency_warning_ms / max_latency) * height)
        error_y = int(y + height - (cfg.latency_error_ms / max_latency) * height)

        cv2.line(frame, (x, warning_y), (x + width, warning_y), cfg.warning_color, 1)
        cv2.line(frame, (x, error_y), (x + width, error_y), cfg.error_color, 1)

        # Draw latency line
        points = []
        for i, latency in enumerate(latencies):
            px = int(x + (i / (len(latencies) - 1)) * width)
            py = int(y + height - (latency / max_latency) * height)
            py = max(y, min(y + height, py))
            points.append((px, py))

        for i in range(len(points) - 1):
            # Color based on latency value
            latency = latencies[i]
            if latency >= cfg.latency_error_ms:
                color = cfg.error_color
            elif latency >= cfg.latency_warning_ms:
                color = cfg.warning_color
            else:
                color = cfg.good_color

            cv2.line(frame, points[i], points[i + 1], color, 2)

    def _get_latency_color(self, latency_ms: float) -> Tuple[int, int, int]:
        """Get color for latency value."""
        cfg = self.config
        if latency_ms >= cfg.latency_error_ms:
            return cfg.error_color
        elif latency_ms >= cfg.latency_warning_ms:
            return cfg.warning_color
        else:
            return cfg.good_color

    def _get_fps_color(self, fps: float) -> Tuple[int, int, int]:
        """Get color for FPS value."""
        cfg = self.config
        if fps <= cfg.fps_error:
            return cfg.error_color
        elif fps <= cfg.fps_warning:
            return cfg.warning_color
        else:
            return cfg.good_color

    def reset(self):
        """Reset overlay state."""
        self.latency_history.clear()
        self._frame_times.clear()


def create_latency_overlay(
    position: str = "top-right",
    opacity: float = 0.8,
    show_graph: bool = True,
    **kwargs,
) -> LatencyOverlay:
    """
    Factory function to create a latency overlay.

    Args:
        position: Overlay position (top-left, top-right, bottom-left, bottom-right)
        opacity: Background opacity (0.0-1.0)
        show_graph: Whether to show the latency graph
        **kwargs: Additional config options

    Returns:
        Configured LatencyOverlay instance
    """
    config = OverlayConfig(
        position=position,
        opacity=opacity,
        show_latency_graph=show_graph,
        **kwargs,
    )
    return LatencyOverlay(config)


class LatencyTracker:
    """
    Standalone latency tracker that can be used independently of the overlay.

    Tracks latency history and computes statistics.
    """

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.latencies: deque = deque(maxlen=history_size)
        self.prompt_switches: deque = deque(maxlen=100)
        self._frame_count = 0
        self._start_time = time.perf_counter()

        # Optimization tracking
        self.cuda_graphs_active = False
        self.static_kv_active = False
        self.async_vae_active = False
        self.prompt_cache_hits = 0
        self.prompt_cache_misses = 0

    def record_frame(self, latency_ms: float):
        """Record a frame's latency."""
        self.latencies.append(latency_ms)
        self._frame_count += 1

    def record_prompt_switch(self, latency_ms: float):
        """Record a prompt switch latency."""
        self.prompt_switches.append(latency_ms)

    def record_cache_hit(self):
        """Record a prompt cache hit."""
        self.prompt_cache_hits += 1

    def record_cache_miss(self):
        """Record a prompt cache miss."""
        self.prompt_cache_misses += 1

    def get_stats(self) -> LatencyStats:
        """Get current latency statistics."""
        if not self.latencies:
            return LatencyStats()

        latencies = np.array(self.latencies)
        elapsed = time.perf_counter() - self._start_time

        return LatencyStats(
            current_ms=latencies[-1] if len(latencies) > 0 else 0.0,
            mean_ms=float(np.mean(latencies)),
            p99_ms=float(np.percentile(latencies, 99)) if len(latencies) > 10 else float(np.max(latencies)),
            max_ms=float(np.max(latencies)),
            min_ms=float(np.min(latencies)),
            fps=self._frame_count / elapsed if elapsed > 0 else 0.0,
            frame_count=self._frame_count,
            cuda_graphs_active=self.cuda_graphs_active,
            static_kv_active=self.static_kv_active,
            async_vae_active=self.async_vae_active,
            prompt_cache_hits=self.prompt_cache_hits,
            prompt_cache_misses=self.prompt_cache_misses,
        )

    def reset(self):
        """Reset all statistics."""
        self.latencies.clear()
        self.prompt_switches.clear()
        self._frame_count = 0
        self._start_time = time.perf_counter()
        self.prompt_cache_hits = 0
        self.prompt_cache_misses = 0
