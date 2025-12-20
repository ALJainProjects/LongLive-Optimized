"""
Optimization Configuration for LongLive inference.

This module provides centralized configuration for all inference-time
optimizations. Configurations can be loaded from YAML files or created
programmatically using preset methods.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import torch


@dataclass
class OptimizationConfig:
    """
    Central configuration for all LongLive optimizations.

    Can be loaded from YAML or created using preset methods:
    - preset_quality(): High quality, moderate speed (~10-15% faster)
    - preset_balanced(): Balance of quality and speed (~25-35% faster) [RECOMMENDED]
    - preset_speed(): Same as balanced (INT8 was found to add overhead)
    - preset_turbo(): 3 denoising steps, max-autotune (~40-50% faster)
    - preset_turbo_fp8(): Turbo + FP8 for H100 (~50-60% faster)
    - preset_ultra(): 2 denoising steps, aggressive (~55-65% faster, quality loss)
    - preset_low_memory(): INT8 KV cache for VRAM-constrained scenarios

    Example:
        config = OptimizationConfig.preset_balanced()
        # or for maximum speed on H100:
        config = OptimizationConfig.preset_turbo_fp8()
        # or from YAML:
        config = OptimizationConfig.from_yaml('opt_config.yaml')
    """

    # === Pipeline Toggle ===
    enabled: bool = True  # Master switch for all optimizations

    # === CUDA Graphs ===
    use_cuda_graphs: bool = True
    cuda_graph_warmup_steps: int = 3
    cuda_graph_pool_size: int = 4  # Number of graph variants for different shapes

    # === KV Cache ===
    use_static_kv: bool = True
    use_quantized_kv: bool = False
    use_integrated_kv_cache: bool = False  # Use ring buffer + optional INT8
    kv_quantization: str = "int8"  # "int8" or "fp8"
    local_attn_size: int = 12  # Match LongLive default (frames)
    sink_size: int = 3  # Frame sink size (first N frames as anchors)

    # === VAE ===
    use_async_vae: bool = True
    vae_double_buffer: bool = True  # Use double buffering for VAE

    # === Prompt ===
    use_prompt_cache: bool = True
    prompt_cache_size: int = 100  # Max cached prompts (LRU eviction)

    # === Memory ===
    use_memory_pool: bool = True
    use_pinned_memory: bool = True  # Faster D2H transfers
    preallocate_output: bool = True  # Pre-allocate output tensors

    # === Precision ===
    model_dtype: str = "bfloat16"  # "bfloat16", "float16", "fp8"

    # === Compile (alternative to CUDA graphs) ===
    use_torch_compile: bool = False  # Mutually exclusive with cuda_graphs
    compile_mode: str = "reduce-overhead"  # "reduce-overhead", "max-autotune"
    compile_fullgraph: bool = True

    # === Profiling ===
    enable_profiling: bool = False
    profile_cuda_events: bool = True  # Use CUDA events for timing
    profile_memory: bool = True  # Track memory usage

    # === Logging ===
    verbose: bool = True  # Print optimization setup info (disable for production)

    # === Generation Defaults ===
    num_frames: int = 120
    num_frame_per_block: int = 3
    denoising_steps: List[int] = field(default_factory=lambda: [1000, 750, 500, 250])

    # === Model Architecture (from LongLive defaults) ===
    num_transformer_blocks: int = 30
    frame_seq_length: int = 1560
    num_heads: int = 12
    head_dim: int = 128

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Mutual exclusions
        if self.use_cuda_graphs and self.use_torch_compile:
            raise ValueError(
                "Cannot use both CUDA graphs and torch.compile simultaneously. "
                "Set use_torch_compile=False or use_cuda_graphs=False."
            )

        # Quantized KV takes precedence over static KV
        if self.use_static_kv and self.use_quantized_kv:
            self.use_static_kv = False

        # Validate dtype
        valid_dtypes = ["bfloat16", "float16", "fp8", "float32"]
        if self.model_dtype not in valid_dtypes:
            raise ValueError(f"model_dtype must be one of {valid_dtypes}")

        # Validate kv_quantization
        valid_kv_quant = ["int8", "fp8", "none"]
        if self.kv_quantization not in valid_kv_quant:
            raise ValueError(f"kv_quantization must be one of {valid_kv_quant}")

    @classmethod
    def from_yaml(cls, path: str) -> 'OptimizationConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        data = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'use_cuda_graphs': self.use_cuda_graphs,
            'cuda_graph_warmup_steps': self.cuda_graph_warmup_steps,
            'cuda_graph_pool_size': self.cuda_graph_pool_size,
            'use_static_kv': self.use_static_kv,
            'use_quantized_kv': self.use_quantized_kv,
            'use_integrated_kv_cache': self.use_integrated_kv_cache,
            'kv_quantization': self.kv_quantization,
            'local_attn_size': self.local_attn_size,
            'sink_size': self.sink_size,
            'use_async_vae': self.use_async_vae,
            'vae_double_buffer': self.vae_double_buffer,
            'use_prompt_cache': self.use_prompt_cache,
            'prompt_cache_size': self.prompt_cache_size,
            'use_memory_pool': self.use_memory_pool,
            'use_pinned_memory': self.use_pinned_memory,
            'preallocate_output': self.preallocate_output,
            'model_dtype': self.model_dtype,
            'use_torch_compile': self.use_torch_compile,
            'compile_mode': self.compile_mode,
            'compile_fullgraph': self.compile_fullgraph,
            'enable_profiling': self.enable_profiling,
            'profile_cuda_events': self.profile_cuda_events,
            'profile_memory': self.profile_memory,
            'num_frames': self.num_frames,
            'num_frame_per_block': self.num_frame_per_block,
            'denoising_steps': self.denoising_steps,
            'num_transformer_blocks': self.num_transformer_blocks,
            'frame_seq_length': self.frame_seq_length,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'verbose': self.verbose,
        }

    @classmethod
    def preset_quality(cls) -> 'OptimizationConfig':
        """
        High quality preset with moderate speed optimizations.

        Focus: Visual quality preservation
        Expected: ~10-15% latency reduction
        Quality impact: Minimal

        Optimizations enabled:
        - Prompt cache (near-zero prompt switch latency)
        - Static KV (buffer reuse, no allocation overhead)
        - Memory pool (pre-allocated buffers)
        - Async VAE (overlapped decode)
        """
        return cls(
            enabled=True,
            use_cuda_graphs=False,
            use_static_kv=True,
            use_quantized_kv=False,
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            model_dtype="bfloat16",
            use_torch_compile=False,
        )

    @classmethod
    def preset_balanced(cls) -> 'OptimizationConfig':
        """
        Balanced preset - good quality with significant speed gains.

        Focus: Balance between quality and speed (RECOMMENDED)
        Expected: ~25-35% latency reduction
        Quality impact: Negligible

        Optimizations enabled:
        - All quality preset optimizations
        - torch.compile with default mode (kernel fusion, PEFT-compatible)

        Note: Uses "default" mode instead of "reduce-overhead" because
        reduce-overhead uses CUDA graphs internally which conflicts with
        dynamic KV cache indexing in LongLive's attention mechanism.
        """
        return cls(
            enabled=True,
            use_cuda_graphs=False,  # torch.compile is more robust with KV cache
            use_static_kv=True,  # Buffer reuse
            use_quantized_kv=False,
            use_integrated_kv_cache=True,  # Ring buffer with full pre-allocated buffers
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            use_pinned_memory=True,
            model_dtype="bfloat16",
            use_torch_compile=True,  # Recommended: handles dynamic shapes well
            compile_mode="default",  # Use default instead of reduce-overhead for PEFT/LoRA
        )

    @classmethod
    def preset_speed(cls) -> 'OptimizationConfig':
        """
        Maximum speed preset - identical to balanced for optimal performance.

        Focus: Minimum latency
        Expected: ~25-35% latency reduction (same as balanced)
        Quality impact: Negligible

        Note: INT8 quantization was found to ADD overhead rather than reduce
        latency on modern GPUs like H100 (3.35 TB/s HBM3). The quant/dequant
        overhead exceeds any memory bandwidth savings. Use preset_low_memory()
        if you need to reduce VRAM usage at the cost of some speed.
        """
        return cls(
            enabled=True,
            use_cuda_graphs=False,  # torch.compile is more robust
            use_static_kv=True,  # Buffer reuse
            use_quantized_kv=False,  # INT8 adds overhead on modern GPUs
            use_integrated_kv_cache=True,  # Ring buffer with full pre-allocated buffers
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            use_pinned_memory=True,
            model_dtype="bfloat16",
            use_torch_compile=True,
            compile_mode="default",  # Use default for PEFT/LoRA compatibility
        )

    @classmethod
    def preset_low_memory(cls) -> 'OptimizationConfig':
        """
        Low memory preset - trades some speed for reduced VRAM usage.

        Focus: Minimize memory usage
        Expected: ~10-15% latency reduction (slower than balanced/speed)
        Memory savings: ~30-40% less KV cache memory (INT8 compression)
        Quality impact: Slight (~1-2% from INT8 quantization)

        Use this preset when:
        - Running on GPUs with limited VRAM (< 24GB)
        - Generating very long videos where KV cache grows large
        - Running multiple concurrent generations
        """
        return cls(
            enabled=True,
            use_cuda_graphs=False,
            use_static_kv=False,  # Quantized KV takes precedence
            use_quantized_kv=True,  # INT8 for memory savings
            use_integrated_kv_cache=True,  # Ring buffer + INT8
            kv_quantization="int8",
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            use_pinned_memory=True,
            model_dtype="bfloat16",
            use_torch_compile=True,
            compile_mode="default",
        )

    @classmethod
    def preset_turbo(cls) -> 'OptimizationConfig':
        """
        Turbo preset - maximum speed with reduced denoising steps.

        Focus: Absolute minimum latency
        Expected: ~40-50% latency reduction vs baseline
        Quality impact: Slight (fewer denoising steps)

        Key changes from balanced:
        - 3 denoising steps instead of 4 (~25% fewer model forward passes)
        - max-autotune compile mode (longer warmup, faster steady-state)
        - Smaller local attention window (8 frames vs 12)

        Use this preset when:
        - Speed is critical and slight quality loss is acceptable
        - Running real-time or interactive applications
        - Generating draft/preview content
        """
        return cls(
            enabled=True,
            use_cuda_graphs=False,
            use_static_kv=True,
            use_quantized_kv=False,
            use_integrated_kv_cache=True,
            local_attn_size=8,  # Smaller window = less compute
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            use_pinned_memory=True,
            model_dtype="bfloat16",
            use_torch_compile=True,
            compile_mode="max-autotune",  # More aggressive optimization
            denoising_steps=[1000, 500, 250],  # 3 steps instead of 4
            verbose=False,  # Disable logging for production
        )

    @classmethod
    def preset_turbo_fp8(cls) -> 'OptimizationConfig':
        """
        Turbo FP8 preset - leverages H100 native FP8 tensor cores.

        Focus: Absolute minimum latency on H100/Hopper GPUs
        Expected: ~50-60% latency reduction vs baseline
        Quality impact: Slight (FP8 precision + fewer denoising steps)

        Key changes from turbo:
        - FP8 inference (2x faster on H100 tensor cores)
        - 3 denoising steps

        Requirements:
        - H100 or newer GPU with FP8 tensor core support
        - PyTorch 2.1+ or transformer-engine installed

        Note: Falls back to bfloat16 if FP8 is not available.
        """
        return cls(
            enabled=True,
            use_cuda_graphs=False,
            use_static_kv=True,
            use_quantized_kv=False,
            use_integrated_kv_cache=True,
            local_attn_size=8,
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            use_pinned_memory=True,
            model_dtype="fp8",  # Native FP8 on H100
            use_torch_compile=True,
            compile_mode="max-autotune",
            denoising_steps=[1000, 500, 250],
            verbose=False,
        )

    @classmethod
    def preset_ultra(cls) -> 'OptimizationConfig':
        """
        Ultra preset - aggressive speed with 2 denoising steps.

        Focus: Extreme speed for real-time applications
        Expected: ~55-65% latency reduction vs baseline
        Quality impact: Moderate (only 2 denoising steps)

        WARNING: Quality degradation is noticeable. Use only when:
        - Real-time performance is essential
        - Output will be post-processed
        - Generating previews/thumbnails

        Key changes:
        - 2 denoising steps instead of 4 (50% fewer model passes)
        - Smallest local attention window (6 frames)
        - FP8 inference on H100
        - max-autotune compilation
        """
        return cls(
            enabled=True,
            use_cuda_graphs=False,
            use_static_kv=True,
            use_quantized_kv=False,
            use_integrated_kv_cache=True,
            local_attn_size=6,  # Minimum practical window
            use_async_vae=True,
            use_prompt_cache=True,
            use_memory_pool=True,
            use_pinned_memory=True,
            model_dtype="fp8",
            use_torch_compile=True,
            compile_mode="max-autotune",
            denoising_steps=[1000, 250],  # Only 2 steps - aggressive
            verbose=False,
        )

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp8": torch.bfloat16,  # FP8 uses bfloat16 as base, actual FP8 handled separately
        }
        return dtype_map.get(self.model_dtype, torch.bfloat16)

    def get_fp8_dtype(self) -> Optional[torch.dtype]:
        """Get FP8 dtype if available and configured."""
        if self.model_dtype != "fp8":
            return None
        # Check for FP8 support (PyTorch 2.1+ on H100)
        if hasattr(torch, 'float8_e4m3fn'):
            return torch.float8_e4m3fn
        return None

    def is_fp8_available(self) -> bool:
        """Check if FP8 inference is available on current hardware."""
        if not torch.cuda.is_available():
            return False
        # Check for Hopper architecture (H100)
        device_name = torch.cuda.get_device_name(0).lower()
        is_hopper = "h100" in device_name or "hopper" in device_name
        # Check for PyTorch FP8 support
        has_fp8 = hasattr(torch, 'float8_e4m3fn')
        return is_hopper and has_fp8

    def get_kv_cache_size(self) -> int:
        """Calculate total KV cache size in tokens."""
        return self.local_attn_size * self.frame_seq_length

    def get_sink_cache_size(self) -> int:
        """Calculate sink token cache size."""
        return self.sink_size * self.frame_seq_length

    def __repr__(self) -> str:
        return (
            f"OptimizationConfig(\n"
            f"  enabled={self.enabled},\n"
            f"  cuda_graphs={self.use_cuda_graphs},\n"
            f"  static_kv={self.use_static_kv},\n"
            f"  quantized_kv={self.use_quantized_kv} ({self.kv_quantization}),\n"
            f"  async_vae={self.use_async_vae},\n"
            f"  prompt_cache={self.use_prompt_cache},\n"
            f"  memory_pool={self.use_memory_pool},\n"
            f"  dtype={self.model_dtype},\n"
            f"  profiling={self.enable_profiling}\n"
            f")"
        )
