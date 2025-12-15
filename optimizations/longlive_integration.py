"""
Integration with LongLive's inference.py

This module provides functions to:
1. Add optimization args to argparse
2. Wrap the base pipeline with optimizations

Both inference.py (for generation) and benchmark_suite.py (for measurement)
use this same integration pattern.

Usage:
    # In inference.py - add args and wrap pipeline
    from optimizations.longlive_integration import add_optimization_args, maybe_optimize_pipeline

    add_optimization_args(parser)
    pipeline = CausalInferencePipeline.from_config(config)
    pipeline = maybe_optimize_pipeline(pipeline, args)

    # In benchmark_suite.py - load pipelines for comparison
    from optimizations.longlive_integration import load_pipeline

    baseline = load_pipeline(config, optimized=False)
    optimized = load_pipeline(config, optimized=True, preset='balanced')
"""

import argparse
from typing import Any, Optional
import torch


def add_optimization_args(parser: argparse.ArgumentParser) -> None:
    """Add optimization arguments to any argparse parser."""
    opt_group = parser.add_argument_group('Optimizations')

    opt_group.add_argument(
        '--optimized',
        action='store_true',
        help='Enable latency optimizations'
    )

    opt_group.add_argument(
        '--opt-preset',
        type=str,
        default='balanced',
        choices=['quality', 'balanced', 'speed'],
        help='Optimization preset'
    )

    opt_group.add_argument(
        '--opt-config',
        type=str,
        default=None,
        help='Path to optimization config YAML'
    )

    opt_group.add_argument(
        '--profile',
        action='store_true',
        help='Enable latency profiling'
    )


def get_optimization_config(
    preset: str = 'balanced',
    config_path: str = None,
    enable_profiling: bool = False,
    **overrides
):
    """Get optimization config from preset or file."""
    from .config import OptimizationConfig

    if config_path:
        config = OptimizationConfig.from_yaml(config_path)
    else:
        presets = {
            'quality': OptimizationConfig.preset_quality,
            'balanced': OptimizationConfig.preset_balanced,
            'speed': OptimizationConfig.preset_speed,
        }
        config = presets[preset]()

    config.enable_profiling = enable_profiling

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def maybe_optimize_pipeline(pipeline: Any, args: argparse.Namespace) -> Any:
    """
    Wrap pipeline with optimizations if --optimized flag is set.

    Use this in inference.py after creating the base pipeline.
    """
    from .optimized_pipeline import OptimizedCausalInferencePipeline

    if not getattr(args, 'optimized', False):
        print("Running ORIGINAL pipeline")
        return pipeline

    config = get_optimization_config(
        preset=getattr(args, 'opt_preset', 'balanced'),
        config_path=getattr(args, 'opt_config', None),
        enable_profiling=getattr(args, 'profile', False),
    )

    print(f"Running OPTIMIZED pipeline (preset: {args.opt_preset})")
    return OptimizedCausalInferencePipeline.from_base(pipeline, config)


def load_pipeline(
    inference_config_path: str,
    optimized: bool = False,
    preset: str = 'balanced',
    enable_profiling: bool = False,
):
    """
    Load pipeline from LongLive config.

    This is the main function used by benchmark_suite.py to load
    either baseline or optimized pipelines for comparison.

    Args:
        inference_config_path: Path to LongLive inference config YAML
        optimized: Whether to wrap with optimizations
        preset: Optimization preset
        enable_profiling: Enable latency profiling

    Returns:
        Pipeline ready for inference
    """
    from omegaconf import OmegaConf

    # Load LongLive config
    config = OmegaConf.load(inference_config_path)

    # Import and create base pipeline
    # This import path depends on how LongLive is structured
    from pipeline.causal_inference import CausalInferencePipeline

    pipeline = CausalInferencePipeline.from_config(config)

    if not optimized:
        print("Loaded BASELINE pipeline")
        return pipeline

    # Wrap with optimizations
    from .optimized_pipeline import OptimizedCausalInferencePipeline

    opt_config = get_optimization_config(
        preset=preset,
        enable_profiling=enable_profiling,
    )

    print(f"Loaded OPTIMIZED pipeline (preset: {preset})")
    return OptimizedCausalInferencePipeline.from_base(pipeline, opt_config)


def wrap_pipeline(
    pipeline: Any,
    preset: str = 'balanced',
    enable_profiling: bool = False,
    **config_overrides
):
    """
    Programmatically wrap a pipeline with optimizations.

    Convenience function when not using argparse.
    """
    from .optimized_pipeline import OptimizedCausalInferencePipeline

    config = get_optimization_config(
        preset=preset,
        enable_profiling=enable_profiling,
        **config_overrides
    )

    return OptimizedCausalInferencePipeline.from_base(pipeline, config)
