# LongLive Latency Optimizations
#
# This module provides inference-time optimizations for LongLive to achieve
# <40ms worst-case inter-frame latency for real-time interaction.
#
# Key optimizations:
# 1. CUDA Graphs - Eliminate kernel launch overhead
# 2. Static KV Cache - Pre-allocated buffers, zero runtime allocation
# 3. Prompt Embedding Cache - Near-zero prompt switch latency
# 4. Async VAE Pipeline - Overlap decode with next frame generation
# 5. Quantized KV Cache - INT8/FP8 for memory bandwidth savings
# 6. Memory Pool - Fixed-shape pre-allocated tensors
# 7. Sync Elimination - Remove unnecessary host-device syncs
#
# Usage:
#     from optimizations import OptimizedCausalInferencePipeline, OptimizationConfig
#
#     # Use preset
#     config = OptimizationConfig.preset_balanced()
#
#     # Or load from YAML
#     config = OptimizationConfig.from_yaml('opt_config.yaml')
#
#     # Create optimized pipeline
#     optimized = OptimizedCausalInferencePipeline.from_base(base_pipeline, config)
#
#     # Run inference
#     video = optimized.inference(noise, text_prompts)

from .config import OptimizationConfig
from .latency_profiler import LatencyProfiler, ProfilingContext, LatencyMeasurement
from .cuda_graphs import CUDAGraphWrapper, MultiStepCUDAGraphWrapper, GraphCaptureConfig
from .static_kv_cache import StaticKVCache
from .prompt_cache import PromptEmbeddingCache, AsyncPromptCache
from .async_vae import AsyncVAEPipeline, SyncVAEPipeline, AsyncVAEDecoder
from .memory_pool import FixedShapeMemoryPool, LongLiveMemoryPool, BufferSpec
from .sync_elimination import SyncFreeContext, SyncPointTracker, AsyncValue, AsyncCPUTransfer
from .quantized_kv import QuantizedKVCache, QuantizationConfig
from .optimized_pipeline import OptimizedCausalInferencePipeline, create_optimized_pipeline
from .optimized_interactive_pipeline import (
    OptimizedInteractiveCausalInferencePipeline,
    create_optimized_interactive_pipeline,
)
from .longlive_integration import (
    add_optimization_args,
    maybe_optimize_pipeline,
    load_pipeline,
    wrap_pipeline,
    get_optimization_config,
)

__all__ = [
    # Main classes
    'OptimizationConfig',
    'OptimizedCausalInferencePipeline',
    'OptimizedInteractiveCausalInferencePipeline',
    'create_optimized_pipeline',
    'create_optimized_interactive_pipeline',

    # Profiling
    'LatencyProfiler',
    'ProfilingContext',
    'LatencyMeasurement',

    # CUDA Graphs
    'CUDAGraphWrapper',
    'MultiStepCUDAGraphWrapper',
    'GraphCaptureConfig',

    # KV Cache
    'StaticKVCache',
    'QuantizedKVCache',
    'QuantizationConfig',

    # Prompt Cache
    'PromptEmbeddingCache',
    'AsyncPromptCache',

    # VAE
    'AsyncVAEPipeline',
    'SyncVAEPipeline',
    'AsyncVAEDecoder',

    # Memory
    'FixedShapeMemoryPool',
    'LongLiveMemoryPool',
    'BufferSpec',

    # Sync
    'SyncFreeContext',
    'SyncPointTracker',
    'AsyncValue',
    'AsyncCPUTransfer',

    # Integration
    'add_optimization_args',
    'maybe_optimize_pipeline',
    'load_pipeline',
    'wrap_pipeline',
    'get_optimization_config',
]
