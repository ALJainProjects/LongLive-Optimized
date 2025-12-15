# Scope Demo Integration for LongLive-Optimized
#
# This module provides integration with daydreamlive/scope for real-time
# interactive video generation with latency optimizations.

from .optimized_longlive_pipeline import (
    OptimizedScopePipeline,
    ScopeOptimizationConfig,
    create_optimized_scope_pipeline,
)
from .latency_overlay import (
    LatencyOverlay,
    LatencyStats,
    create_latency_overlay,
)

__all__ = [
    'OptimizedScopePipeline',
    'ScopeOptimizationConfig',
    'create_optimized_scope_pipeline',
    'LatencyOverlay',
    'LatencyStats',
    'create_latency_overlay',
]
