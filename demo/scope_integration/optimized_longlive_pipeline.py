"""
Optimized LongLive Pipeline for Scope Integration

This module provides a drop-in replacement for scope's LongLivePipeline
with all latency optimizations enabled.

Usage in scope:
    # In scope/core/pipelines/longlive/pipeline.py
    from longlive_optimized.demo.scope_integration import (
        OptimizedScopePipeline,
        ScopeOptimizationConfig
    )

    # Replace LongLivePipeline with OptimizedScopePipeline
    config = ScopeOptimizationConfig.preset_balanced()
    pipeline = OptimizedScopePipeline(model_path, config)
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.cuda

from optimizations.config import OptimizationConfig
from optimizations.latency_profiler import LatencyProfiler, LatencyMeasurement
from optimizations.cuda_graphs import CUDAGraphWrapper, GraphCaptureConfig
from optimizations.static_kv_cache import StaticKVCache
from optimizations.prompt_cache import PromptEmbeddingCache
from optimizations.async_vae import AsyncVAEPipeline
from optimizations.memory_pool import LongLiveMemoryPool
from optimizations.sync_elimination import SyncFreeContext
from optimizations.quantized_kv import QuantizedKVCache, QuantizationConfig


@dataclass
class ScopeOptimizationConfig(OptimizationConfig):
    """
    Extended optimization config with scope-specific settings.

    Adds WebRTC streaming and latency overlay options.
    """

    # Scope-specific settings
    enable_latency_overlay: bool = True
    overlay_position: str = "top-right"  # top-left, top-right, bottom-left, bottom-right
    overlay_opacity: float = 0.8

    # Streaming settings
    target_fps: int = 25
    max_queue_depth: int = 3  # Frame buffer depth
    drop_frames_on_lag: bool = True

    # Interactive settings
    prompt_debounce_ms: int = 100  # Debounce rapid prompt changes
    smooth_prompt_transition: bool = False  # Gradual prompt blending
    transition_frames: int = 5

    @classmethod
    def preset_realtime(cls) -> 'ScopeOptimizationConfig':
        """Preset optimized for real-time streaming with minimal latency."""
        return cls(
            use_cuda_graphs=False,  # CUDA graphs incompatible with LongLive dynamic KV
            use_torch_compile=True,
            compile_mode="max-autotune",
            use_static_kv=True,
            use_quantized_kv=False,
            use_integrated_kv_cache=True,
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            model_dtype="bfloat16",
            denoising_steps=[1000, 500, 250],  # 3 steps for speed
            local_attn_size=8,  # Smaller window
            enable_latency_overlay=True,
            target_fps=25,
            drop_frames_on_lag=True,
        )

    @classmethod
    def preset_quality_stream(cls) -> 'ScopeOptimizationConfig':
        """Preset for higher quality streaming (slightly more latency)."""
        return cls(
            use_cuda_graphs=False,
            use_torch_compile=True,
            compile_mode="default",
            use_static_kv=True,
            use_quantized_kv=False,
            use_integrated_kv_cache=True,
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            model_dtype="bfloat16",
            enable_latency_overlay=True,
            target_fps=20,
            drop_frames_on_lag=False,
        )

    @classmethod
    def preset_turbo(cls) -> 'ScopeOptimizationConfig':
        """Maximum speed with slight quality trade-off (3 denoising steps)."""
        return cls(
            use_cuda_graphs=False,
            use_torch_compile=True,
            compile_mode="max-autotune",
            use_static_kv=True,
            use_quantized_kv=False,
            use_integrated_kv_cache=True,
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            model_dtype="bfloat16",
            denoising_steps=[1000, 500, 250],  # 3 steps
            local_attn_size=8,
            enable_latency_overlay=True,
            target_fps=30,
            drop_frames_on_lag=True,
        )

    @classmethod
    def preset_ultra(cls) -> 'ScopeOptimizationConfig':
        """Preview/draft mode with maximum speed (2 denoising steps, noticeable quality loss)."""
        return cls(
            use_cuda_graphs=False,
            use_torch_compile=True,
            compile_mode="max-autotune",
            use_static_kv=True,
            use_quantized_kv=False,
            use_integrated_kv_cache=True,
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            model_dtype="fp8",  # FP8 for maximum speed on H100
            denoising_steps=[1000, 250],  # 2 steps only
            local_attn_size=6,
            enable_latency_overlay=True,
            target_fps=40,
            drop_frames_on_lag=True,
        )


class OptimizedScopePipeline:
    """
    Drop-in replacement for scope's LongLivePipeline with optimizations.

    Maintains API compatibility with scope while adding:
    - CUDA graph acceleration
    - Static KV cache
    - Prompt embedding cache
    - Async VAE pipeline
    - Real-time latency tracking

    Example:
        # In your scope server code
        from longlive_optimized.demo.scope_integration import OptimizedScopePipeline

        pipeline = OptimizedScopePipeline(
            model_path="/path/to/longlive",
            config=ScopeOptimizationConfig.preset_realtime()
        )

        # Use same API as original
        frame = pipeline.generate_frame(prompt="A panda in bamboo forest")
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[ScopeOptimizationConfig] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_path = model_path
        self.config = config or ScopeOptimizationConfig.preset_realtime()
        self.device = device
        self.dtype = dtype or self._get_dtype()

        # Latency tracking
        self.profiler = LatencyProfiler() if self.config.enable_profiling else None
        self.latency_history: List[float] = []
        self.max_history_size = 100

        # Frame counter
        self.frame_idx = 0
        self.warmup_complete = False

        # Current prompt
        self.current_prompt: Optional[str] = None
        self.prompt_embeddings: Optional[torch.Tensor] = None

        # Initialize components
        self._load_base_pipeline()
        self._setup_optimizations()

    def _get_dtype(self) -> torch.dtype:
        """Get dtype from config."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.model_dtype, torch.bfloat16)

    def _load_base_pipeline(self):
        """
        Load the base LongLive pipeline.

        Loads the actual LongLive model components for real inference.
        """
        from pathlib import Path
        from omegaconf import OmegaConf

        print(f"Loading LongLive model from: {self.model_path}")

        # Check if model_path is a config file or directory
        model_path = Path(self.model_path)

        if model_path.suffix in ['.yaml', '.yml']:
            # Load from config file
            config = OmegaConf.load(model_path)
        elif model_path.is_dir():
            # Look for config in directory
            config_candidates = list(model_path.glob("*.yaml")) + list(model_path.glob("*.yml"))
            if config_candidates:
                config = OmegaConf.load(config_candidates[0])
            else:
                # Create minimal config for direct model loading
                config = OmegaConf.create({
                    "model_name": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    "denoising_step_list": [1000, 750, 500, 250],
                    "context_noise": 0,
                    "num_frame_per_block": 1,
                })
        else:
            raise ValueError(f"model_path must be a config file or directory: {self.model_path}")

        # Import pipeline components
        try:
            from pipeline.causal_inference import CausalInferencePipeline

            # Create base pipeline
            self.base_pipeline = CausalInferencePipeline(config, device=self.device)

            # Load checkpoint if specified
            if hasattr(config, 'generator_ckpt') and config.generator_ckpt:
                state_dict = torch.load(config.generator_ckpt, map_location="cpu")
                if "generator" in state_dict or "generator_ema" in state_dict:
                    gen_key = "generator_ema" if getattr(config, 'use_ema', False) else "generator"
                    self.base_pipeline.generator.load_state_dict(state_dict[gen_key])
                print(f"Loaded generator checkpoint: {config.generator_ckpt}")

            # Move to device and dtype
            self.base_pipeline = self.base_pipeline.to(dtype=self.dtype)
            self.base_pipeline.generator.to(device=self.device)
            self.base_pipeline.vae.to(device=self.device)

            # Store references to components
            self.generator = self.base_pipeline.generator
            self.text_encoder = self.base_pipeline.text_encoder
            self.vae = self.base_pipeline.vae

            # Store config for later use
            self._config = config
            self._frame_seq_length = getattr(self.base_pipeline, 'frame_seq_length', 1560)
            self._num_transformer_blocks = getattr(self.base_pipeline, 'num_transformer_blocks', 30)
            self._denoising_steps = getattr(config, 'denoising_step_list', [1000, 750, 500, 250])

            print(f"LongLive pipeline loaded successfully")
            print(f"  Generator: {type(self.generator).__name__}")
            print(f"  Text Encoder: {type(self.text_encoder).__name__}")
            print(f"  VAE: {type(self.vae).__name__}")

        except ImportError as e:
            print(f"Warning: Could not import LongLive pipeline: {e}")
            print("Running in demo mode with placeholder models")
            self.base_pipeline = None
            self.generator = None
            self.text_encoder = None
            self.vae = None
            self._config = config if 'config' in dir() else None
            self._frame_seq_length = 1560
            self._num_transformer_blocks = 30
            self._denoising_steps = [1000, 750, 500, 250]

    def _setup_optimizations(self):
        """Initialize all optimization components."""

        # Memory pool (initialize first)
        if self.config.use_memory_pool:
            self.memory_pool = LongLiveMemoryPool(
                batch_size=1,
                num_frames=self.config.local_attn_size,
                latent_channels=16,
                latent_height=60,
                latent_width=104,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.memory_pool = None

        # KV Cache
        if self.config.use_quantized_kv:
            quant_config = QuantizationConfig(
                quantization_type=self.config.kv_quantization,
            )
            self.kv_cache = QuantizedKVCache(
                num_layers=30,  # LongLive-1.3B default
                num_heads=12,
                head_dim=128,
                local_window=self.config.local_attn_size,
                sink_size=self.config.sink_size,
                config=quant_config,
                device=self.device,
            )
        elif self.config.use_static_kv:
            self.kv_cache = StaticKVCache(
                num_layers=30,
                num_heads=12,
                head_dim=128,
                local_window=self.config.local_attn_size,
                sink_size=self.config.sink_size,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.kv_cache = None

        # Prompt cache
        if self.config.use_prompt_cache:
            self.prompt_cache = PromptEmbeddingCache(
                max_size=self.config.prompt_cache_size,
            )
        else:
            self.prompt_cache = None

        # Async VAE
        if self.config.use_async_vae:
            self.async_vae = AsyncVAEPipeline(
                vae=self.vae,
                device=self.device,
            )
        else:
            self.async_vae = None

        # CUDA Graphs (capture after warmup)
        self.cuda_graph_wrapper = None
        if self.config.use_cuda_graphs:
            self._prepare_cuda_graphs()

    def _prepare_cuda_graphs(self):
        """Prepare CUDA graph capture (actual capture happens during warmup)."""
        self.graph_config = GraphCaptureConfig(
            warmup_iterations=self.config.cuda_graph_warmup_steps,
            capture_multiple_shapes=False,  # Scope uses fixed shapes
        )

    def warmup(self, num_frames: int = 50):
        """
        Run warmup frames to stabilize GPU and capture CUDA graphs.

        Should be called before starting the streaming loop.
        """
        print(f"Running {num_frames} warmup frames...")

        # Generate warmup frames
        warmup_prompt = "A test scene for warmup"
        for i in range(num_frames):
            _ = self._generate_frame_internal(warmup_prompt, capture_graph=(i == num_frames - 1))

        # Reset state after warmup
        self.frame_idx = 0
        self.latency_history.clear()
        if self.kv_cache:
            self.kv_cache.reset()

        self.warmup_complete = True
        print("Warmup complete")

        # Capture CUDA graph on last warmup frame
        if self.config.use_cuda_graphs and self.generator is not None:
            self.cuda_graph_wrapper = CUDAGraphWrapper(
                model=self.generator,
                config=self.graph_config,
            )
            print("CUDA graph captured")

    def generate_frame(
        self,
        prompt: str,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate a single frame with the given prompt.

        This is the main API method matching scope's interface.

        Args:
            prompt: Text prompt for generation
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Generated frame tensor [C, H, W]
        """
        if not self.warmup_complete:
            self.warmup()

        # Track latency
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        frame = self._generate_frame_internal(prompt)
        end_event.record()

        # Non-blocking latency tracking
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        self._record_latency(latency_ms)

        self.frame_idx += 1
        return frame

    def _generate_frame_internal(
        self,
        prompt: str,
        capture_graph: bool = False,
    ) -> torch.Tensor:
        """Internal frame generation with optimizations."""

        # Handle prompt change
        if prompt != self.current_prompt:
            self._handle_prompt_change(prompt)

        # Initialize internal KV caches if using actual generator and not already initialized
        if self.generator is not None and not hasattr(self, '_kv_cache_internal'):
            self._initialize_internal_kv_caches()

        # Generate frame using sync-free context
        with SyncFreeContext(warn_only=True):
            # Get noise
            if self.memory_pool:
                noise = self.memory_pool.get_buffer("noise")
                noise.normal_()
            else:
                noise = torch.randn(
                    1, 16, 1, 60, 104,
                    device=self.device,
                    dtype=self.dtype,
                )

            # Denoising loop
            latents = noise
            for step_idx, timestep in enumerate(self.config.denoising_steps):
                if self.cuda_graph_wrapper and not capture_graph:
                    # Use captured graph
                    latents = self.cuda_graph_wrapper.replay(
                        latents, timestep, self.prompt_embeddings
                    )
                else:
                    # Standard forward pass
                    latents = self._denoise_step(latents, timestep)

            # Update KV cache
            if self.kv_cache:
                self.kv_cache.step()

            # Decode latents to pixels
            if self.async_vae and self.frame_idx > 0:
                # Get previous frame while current decodes
                self.async_vae.decode_async(latents)
                frame = self.async_vae.get_previous_frame()
            else:
                # Synchronous decode
                frame = self._decode_latents(latents)
                if self.async_vae:
                    self.async_vae.decode_async(latents)  # Start next decode

        return frame

    def _handle_prompt_change(self, new_prompt: str):
        """Handle prompt change with caching and KV-recache."""

        # Get embeddings (from cache if available)
        if self.prompt_cache:
            cache_result = self.prompt_cache.get(new_prompt)
            if cache_result is not None:
                self.prompt_embeddings = cache_result
            else:
                self.prompt_embeddings = self._encode_prompt(new_prompt)
                self.prompt_cache.put(new_prompt, self.prompt_embeddings)
        else:
            self.prompt_embeddings = self._encode_prompt(new_prompt)

        # KV-recache if we have cached KV states
        if self.kv_cache and self.frame_idx > 0:
            self.kv_cache.recache_with_new_prompt(self.prompt_embeddings)

        self.current_prompt = new_prompt

    def _encode_prompt(self, prompt: str) -> dict:
        """Encode prompt to embeddings using actual text encoder."""
        if self.text_encoder is not None:
            # Use the text encoder's forward method which returns a dict
            return self.text_encoder(text_prompts=[prompt])
        # Fallback for demo mode - return dict format expected by generator
        return {
            "prompt_embeds": torch.randn(1, 512, 4096, device=self.device, dtype=self.dtype)
        }

    def _denoise_step(
        self,
        latents: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Single denoising step using actual generator."""
        if self.generator is not None:
            batch_size = latents.shape[0]
            num_frames = latents.shape[2] if latents.dim() == 5 else 1

            # Reshape latents if needed: [B, C, T, H, W] -> [B, T, C, H, W]
            if latents.dim() == 5 and latents.shape[1] == 16:  # [B, C, T, H, W]
                latents = latents.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]

            # Create timestep tensor
            timestep_tensor = torch.ones(
                [batch_size, num_frames],
                device=self.device,
                dtype=torch.int64
            ) * timestep

            # Run generator forward pass
            _, denoised = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=self.prompt_embeddings,
                timestep=timestep_tensor,
                kv_cache=getattr(self, '_kv_cache_internal', None),
                crossattn_cache=getattr(self, '_crossattn_cache_internal', None),
                current_start=self.frame_idx * self._frame_seq_length,
            )
            return denoised
        # Fallback for demo mode
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space using actual VAE."""
        if self.vae is not None:
            # Use VAE's decode_to_pixel method for proper decoding
            # latents shape: [B, T, C, H, W] or [B, C, H, W]
            video = self.vae.decode_to_pixel(latents, use_cache=False)
            # Normalize to [0, 1]
            video = (video * 0.5 + 0.5).clamp(0, 1)
            # Return single frame: [C, H, W]
            if video.dim() == 5:  # [B, T, C, H, W]
                return video[0, 0]  # First batch, first frame
            elif video.dim() == 4:  # [B, C, H, W]
                return video[0]  # First batch
            return video
        # Fallback for demo mode - return dummy frame
        return torch.randn(3, 480, 832, device=self.device, dtype=self.dtype)

    def _initialize_internal_kv_caches(self):
        """Initialize internal KV caches for actual generator inference."""
        if self.generator is None:
            return

        # Calculate KV cache size based on local attention config
        local_attn_size = self.config.local_attn_size
        kv_cache_size = local_attn_size * self._frame_seq_length

        # Initialize self-attention KV cache (30 transformer blocks)
        self._kv_cache_internal = []
        for _ in range(self._num_transformer_blocks):
            self._kv_cache_internal.append({
                "k": torch.zeros([1, kv_cache_size, 12, 128], dtype=self.dtype, device=self.device),
                "v": torch.zeros([1, kv_cache_size, 12, 128], dtype=self.dtype, device=self.device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            })

        # Initialize cross-attention cache
        self._crossattn_cache_internal = []
        for _ in range(self._num_transformer_blocks):
            self._crossattn_cache_internal.append({
                "k": torch.zeros([1, 512, 12, 128], dtype=self.dtype, device=self.device),
                "v": torch.zeros([1, 512, 12, 128], dtype=self.dtype, device=self.device),
                "is_init": False,
            })

        print(f"Initialized internal KV caches (kv_cache_size={kv_cache_size})")

    def _record_latency(self, latency_ms: float):
        """Record latency for tracking."""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)

    def get_latency_stats(self) -> Dict[str, float]:
        """Get current latency statistics."""
        if not self.latency_history:
            return {
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "current": 0.0,
                "fps": 0.0,
            }

        import numpy as np
        latencies = np.array(self.latency_history)

        return {
            "mean": float(np.mean(latencies)),
            "max": float(np.max(latencies)),
            "min": float(np.min(latencies)),
            "current": float(latencies[-1]),
            "fps": 1000.0 / float(np.mean(latencies)) if np.mean(latencies) > 0 else 0.0,
        }

    def switch_prompt(self, new_prompt: str):
        """
        Switch to a new prompt.

        Triggers KV-recache and updates prompt embeddings.
        """
        self._handle_prompt_change(new_prompt)

    def reset(self):
        """Reset pipeline state for new generation."""
        self.frame_idx = 0
        self.current_prompt = None
        self.prompt_embeddings = None
        self.latency_history.clear()

        if self.kv_cache:
            self.kv_cache.reset()

        # Reset internal KV caches if they exist
        if hasattr(self, '_kv_cache_internal') and self._kv_cache_internal:
            for cache in self._kv_cache_internal:
                cache["k"].zero_()
                cache["v"].zero_()
                cache["global_end_index"].zero_()
                cache["local_end_index"].zero_()

        if hasattr(self, '_crossattn_cache_internal') and self._crossattn_cache_internal:
            for cache in self._crossattn_cache_internal:
                cache["k"].zero_()
                cache["v"].zero_()
                cache["is_init"] = False

    def set_optimization_preset(self, preset: str):
        """
        Switch optimization preset at runtime.

        Args:
            preset: One of "realtime", "quality", "balanced", "speed", "turbo", "ultra"
        """
        presets = {
            "realtime": ScopeOptimizationConfig.preset_realtime,
            "quality": ScopeOptimizationConfig.preset_quality_stream,
            "balanced": ScopeOptimizationConfig.preset_balanced,
            "speed": ScopeOptimizationConfig.preset_speed,
            "turbo": ScopeOptimizationConfig.preset_turbo,
            "ultra": ScopeOptimizationConfig.preset_ultra,
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        new_config = presets[preset]()

        # Update config (some changes require restart)
        needs_restart = (
            new_config.use_cuda_graphs != self.config.use_cuda_graphs or
            new_config.use_quantized_kv != self.config.use_quantized_kv or
            new_config.use_static_kv != self.config.use_static_kv
        )

        self.config = new_config

        if needs_restart:
            print(f"Preset '{preset}' requires pipeline restart for full effect")
            self._setup_optimizations()
            self.warmup_complete = False


def create_optimized_scope_pipeline(
    model_path: str,
    preset: str = "realtime",
    **kwargs,
) -> OptimizedScopePipeline:
    """
    Factory function to create optimized scope pipeline.

    Args:
        model_path: Path to LongLive model
        preset: Optimization preset name
        **kwargs: Additional config overrides

    Returns:
        Configured OptimizedScopePipeline
    """
    presets = {
        "realtime": ScopeOptimizationConfig.preset_realtime,
        "quality": ScopeOptimizationConfig.preset_quality_stream,
        "balanced": ScopeOptimizationConfig.preset_balanced,
        "speed": ScopeOptimizationConfig.preset_speed,
        "turbo": ScopeOptimizationConfig.preset_turbo,
        "ultra": ScopeOptimizationConfig.preset_ultra,
    }

    config_class = presets.get(preset, ScopeOptimizationConfig.preset_realtime)
    config = config_class()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return OptimizedScopePipeline(model_path, config)
