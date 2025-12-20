# Scope Integration for LongLive-Optimized

This module provides integration with [daydreamlive/scope](https://github.com/daydreamlive/scope) for real-time WebRTC streaming with latency optimizations.

## Overview

Scope is a WebRTC-based real-time video generation framework. This integration allows you to use LongLive-Optimized as a drop-in replacement for the base LongLive pipeline with all latency optimizations enabled.

## Features

- **Drop-in Replacement**: `OptimizedScopePipeline` maintains API compatibility with scope's `LongLivePipeline`
- **Real-time Latency Overlay**: Visual overlay showing FPS, latency statistics, and optimization status
- **Preset Selection**: Switch between quality/balanced/speed presets at runtime
- **Latency Tracking**: Built-in statistics for monitoring performance

## Installation

1. Clone scope repository:
```bash
git clone https://github.com/daydreamlive/scope.git
cd scope
```

2. Install LongLive-Optimized as a dependency:
```bash
pip install -e /path/to/LongLive-Optimized
```

## Usage

### Basic Integration

```python
from longlive_optimized.demo.scope_integration import (
    OptimizedScopePipeline,
    ScopeOptimizationConfig,
    create_latency_overlay,
)

# Create optimized pipeline
config = ScopeOptimizationConfig.preset_realtime()
pipeline = OptimizedScopePipeline(
    model_path="/path/to/longlive",
    config=config
)

# Warmup (required for CUDA graph capture)
pipeline.warmup(num_frames=50)

# Create latency overlay
overlay = create_latency_overlay(position="top-right")

# Frame generation loop
while streaming:
    frame = pipeline.generate_frame(prompt=current_prompt)
    stats = pipeline.get_latency_stats()
    frame_with_overlay = overlay.apply(frame, stats)
    stream.send(frame_with_overlay)
```

### Modifying scope's Pipeline Manager

In `scope/server/pipeline_manager.py`:

```python
from longlive_optimized.demo.scope_integration import (
    OptimizedScopePipeline,
    ScopeOptimizationConfig,
)

class PipelineManager:
    def load_longlive_pipeline(self, model_path: str, preset: str = "balanced"):
        # Use optimized pipeline instead of base
        config = getattr(ScopeOptimizationConfig, f"preset_{preset}")()
        return OptimizedScopePipeline(model_path, config)
```

### WebRTC Data Channel Commands

The optimized pipeline supports runtime configuration via WebRTC data channel:

```json
{
    "type": "set_optimization_preset",
    "preset": "balanced"
}
```

```json
{
    "type": "get_latency_stats"
}
```

## Configuration Options

### ScopeOptimizationConfig

Extends `OptimizationConfig` with scope-specific settings:

```python
@dataclass
class ScopeOptimizationConfig(OptimizationConfig):
    # Overlay settings
    enable_latency_overlay: bool = True
    overlay_position: str = "top-right"
    overlay_opacity: float = 0.8

    # Streaming settings
    target_fps: int = 25
    max_queue_depth: int = 3
    drop_frames_on_lag: bool = True

    # Interactive settings
    prompt_debounce_ms: int = 100
    smooth_prompt_transition: bool = False
    transition_frames: int = 5
```

### Presets

| Preset | Use Case | Expected Latency | FPS | Quality |
|--------|----------|------------------|-----|---------|
| `realtime` | Minimum latency streaming | ~460ms | 6.5 | Slight loss |
| `quality_stream` | Higher quality streaming | ~575ms | 5.2 | Identical |
| `balanced` | General use (recommended) | ~575ms | 5.2 | Identical |
| `speed` | Same as balanced | ~575ms | 5.2 | Identical |
| `turbo` | Maximum speed with 3 steps | ~458ms | 6.5 | Slight loss |
| `ultra` | Preview/draft mode (2 steps) | ~312ms | 9.6 | Noticeable loss |

**Note**: CUDA Graphs are disabled in all presets due to incompatibility with LongLive's dynamic KV cache indices. torch.compile with max-autotune provides the primary optimization.

## Latency Overlay

The overlay shows real-time latency information:

```
┌─────────────────────┐
│ Latency: 32.5ms     │  <- Current frame latency
│ Mean: 31.2ms        │  <- Average over window
│ P99: 38.1ms         │  <- 99th percentile
│ Max: 42.3ms         │  <- Worst case seen
│ FPS: 31.2           │  <- Current throughput
│ Frame: 1234         │  <- Total frames generated
│                     │
│ Optimizations:      │
│   + torch.compile   │  <- Active optimizations
│   + Static KV       │
│   + Async VAE       │
│                     │
│ ▂▃▅▆█▄▃▅▆▇█▅▃      │  <- Latency graph
└─────────────────────┘
```

### Customizing the Overlay

```python
from longlive_optimized.demo.scope_integration import LatencyOverlay, OverlayConfig

config = OverlayConfig(
    position="bottom-left",
    opacity=0.7,
    show_latency_graph=True,
    graph_history=120,  # Show last 120 frames
    latency_warning_ms=35.0,
    latency_error_ms=40.0,
)

overlay = LatencyOverlay(config)
```

## API Reference

### OptimizedScopePipeline

```python
class OptimizedScopePipeline:
    def __init__(
        self,
        model_path: str,
        config: Optional[ScopeOptimizationConfig] = None,
        device: str = "cuda",
    ):
        """Create optimized pipeline for scope integration."""

    def warmup(self, num_frames: int = 50):
        """Run warmup frames and capture CUDA graphs."""

    def generate_frame(self, prompt: str, **kwargs) -> torch.Tensor:
        """Generate a single frame. Main API method."""

    def switch_prompt(self, new_prompt: str):
        """Switch to a new prompt with KV-recache."""

    def get_latency_stats(self) -> Dict[str, float]:
        """Get current latency statistics."""

    def set_optimization_preset(self, preset: str):
        """Switch optimization preset at runtime."""

    def reset(self):
        """Reset pipeline state."""
```

### LatencyOverlay

```python
class LatencyOverlay:
    def __init__(self, config: Optional[OverlayConfig] = None):
        """Create latency overlay."""

    def apply(
        self,
        frame: torch.Tensor,
        stats: LatencyStats,
    ) -> torch.Tensor:
        """Apply overlay to frame."""

    def reset(self):
        """Reset overlay state."""
```

### LatencyStats

```python
@dataclass
class LatencyStats:
    current_ms: float
    mean_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    fps: float
    frame_count: int
    cuda_graphs_active: bool
    static_kv_active: bool
    async_vae_active: bool
    prompt_cache_hits: int
    prompt_cache_misses: int
```

## Troubleshooting

### Overlay Not Appearing

- Ensure OpenCV is installed: `pip install opencv-python`
- Check that `enable_latency_overlay` is `True` in config

### torch.compile Issues

- Run warmup before generating frames (allows compilation to complete)
- First few frames will be slower during compilation
- Use `compile_mode="default"` if `max-autotune` causes issues
- CUDA Graphs are intentionally disabled (LongLive KV cache incompatibility)

### High Latency Despite Optimizations

- Verify warmup completed successfully
- Check GPU thermal throttling
- Ensure exclusive GPU access
- Try `speed` preset for more aggressive optimization

## Example: Complete Integration

```python
import asyncio
from aiortc import VideoStreamTrack
from longlive_optimized.demo.scope_integration import (
    OptimizedScopePipeline,
    ScopeOptimizationConfig,
    create_latency_overlay,
    LatencyStats,
)

class OptimizedVideoTrack(VideoStreamTrack):
    def __init__(self, model_path: str, initial_prompt: str):
        super().__init__()

        # Setup pipeline
        config = ScopeOptimizationConfig.preset_realtime()
        self.pipeline = OptimizedScopePipeline(model_path, config)
        self.pipeline.warmup()

        # Setup overlay
        self.overlay = create_latency_overlay()

        # State
        self.prompt = initial_prompt
        self.running = True

    async def recv(self):
        # Generate frame
        frame = self.pipeline.generate_frame(self.prompt)

        # Apply overlay
        stats = LatencyStats(**self.pipeline.get_latency_stats())
        frame = self.overlay.apply(frame, stats)

        # Convert to VideoFrame for WebRTC
        return self._tensor_to_video_frame(frame)

    def set_prompt(self, prompt: str):
        self.pipeline.switch_prompt(prompt)
        self.prompt = prompt
```
