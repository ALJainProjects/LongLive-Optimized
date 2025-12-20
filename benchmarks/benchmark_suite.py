"""
Benchmark Suite for LongLive Latency Optimizations.

Measures project-defined latency metrics:
1. Steady-state inter-frame latency (1000 frames for P99/max)
2. Prompt-switch latency (50 switches across 8 prompts)
3. Throughput (60 seconds continuous)

Usage:
    # Benchmark baseline
    python benchmarks/benchmark_suite.py \
        --config configs/longlive_inference.yaml

    # Benchmark with optimizations
    python benchmarks/benchmark_suite.py \
        --config configs/longlive_inference.yaml \
        --optimized --preset balanced

    # Compare baseline vs optimized
    python benchmarks/benchmark_suite.py \
        --config configs/longlive_inference.yaml \
        --compare --preset balanced
"""

import torch
import torch.cuda
import json
import time
import argparse
import sys
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Sample sizes - reasonable for statistical validity
STEADY_STATE_FRAMES = 1000  # Enough for P99/max
WARMUP_FRAMES = 50
PROMPT_SWITCHES = 50
FRAMES_BETWEEN_SWITCHES = 10
THROUGHPUT_DURATION_SEC = 60
BREAKDOWN_FRAMES = 100

# Quick mode sample sizes
QUICK_STEADY_STATE = 100
QUICK_SWITCHES = 10
QUICK_THROUGHPUT_SEC = 10

# Test prompts
TEST_PROMPTS = [
    "A panda walking through a bamboo forest",
    "Ocean waves crashing on rocks at sunset",
    "A car driving through city streets at night",
    "Fireworks exploding over a cityscape",
    "A person dancing in the rain",
    "Birds flying over a mountain lake",
    "A train passing through snowy mountains",
    "Leaves falling in an autumn park",
]


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    pipeline_type: str = ""
    preset: str = ""
    device: str = ""
    timestamp: str = ""

    # Steady-state (ms)
    ss_mean: float = 0.0
    ss_std: float = 0.0
    ss_p50: float = 0.0
    ss_p95: float = 0.0
    ss_p99: float = 0.0
    ss_max: float = 0.0
    ss_min: float = 0.0
    ss_samples: int = 0

    # Prompt switch (ms)
    ps_mean: float = 0.0
    ps_std: float = 0.0
    ps_p99: float = 0.0
    ps_max: float = 0.0
    ps_samples: int = 0

    # Throughput
    fps: float = 0.0
    total_frames: int = 0

    # Memory
    peak_memory_gb: float = 0.0

    # Component breakdown
    components: Dict[str, float] = field(default_factory=dict)

    # Target
    meets_40ms: bool = False


def measure_steady_state(pipeline, num_frames: int, warmup: int, prompt: str) -> Dict:
    """
    Measure steady-state inter-frame latency using CUDA events.

    Returns dict with latency statistics.
    """
    print(f"  Steady-state: {num_frames} frames (warmup={warmup})...")

    device = next(pipeline.parameters()).device if hasattr(pipeline, 'parameters') else torch.device('cuda')

    # Create sample input - LongLive latent shape: [batch, num_frames, channels, H, W]
    # Using 3 frames per block as per config (num_frame_per_block: 3)
    def get_noise():
        return torch.randn(1, 3, 16, 60, 104, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup):
        noise = get_noise()
        with torch.no_grad():
            _ = pipeline.inference(noise, [prompt])

    torch.cuda.reset_peak_memory_stats()

    # Create events
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_frames)]

    # Measure
    for i in range(num_frames):
        noise = get_noise()
        with torch.no_grad():
            _ = pipeline.inference(noise, [prompt])
        end_events[i].record()

    torch.cuda.synchronize()

    # Compute inter-frame latencies
    latencies = []
    for i in range(num_frames - 1):
        latencies.append(end_events[i].elapsed_time(end_events[i + 1]))

    lat = np.array(latencies)

    return {
        'mean': float(np.mean(lat)),
        'std': float(np.std(lat)),
        'p50': float(np.percentile(lat, 50)),
        'p95': float(np.percentile(lat, 95)),
        'p99': float(np.percentile(lat, 99)),
        'max': float(np.max(lat)),
        'min': float(np.min(lat)),
        'samples': len(lat),
    }


def measure_prompt_switch(pipeline, num_switches: int, frames_between: int) -> Dict:
    """
    Measure prompt-switch latency.

    Returns dict with switch latency statistics.
    """
    print(f"  Prompt switch: {num_switches} switches...")

    device = next(pipeline.parameters()).device if hasattr(pipeline, 'parameters') else torch.device('cuda')

    def get_noise():
        return torch.randn(1, 3, 16, 60, 104, device=device, dtype=torch.bfloat16)

    prompt_idx = 0
    latencies = []

    for _ in range(num_switches):
        # Generate frames with current prompt
        for _ in range(frames_between):
            noise = get_noise()
            with torch.no_grad():
                _ = pipeline.inference(noise, [TEST_PROMPTS[prompt_idx]])

        # Switch prompt
        prompt_idx = (prompt_idx + 1) % len(TEST_PROMPTS)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        # If optimized, call switch_prompt; otherwise just use new prompt
        if hasattr(pipeline, 'switch_prompt'):
            pipeline.switch_prompt([TEST_PROMPTS[prompt_idx]])

        # First frame with new prompt
        noise = get_noise()
        with torch.no_grad():
            _ = pipeline.inference(noise, [TEST_PROMPTS[prompt_idx]])

        end.record()
        torch.cuda.synchronize()

        latencies.append(start.elapsed_time(end))

    lat = np.array(latencies)

    return {
        'mean': float(np.mean(lat)),
        'std': float(np.std(lat)),
        'p99': float(np.percentile(lat, 99)),
        'max': float(np.max(lat)),
        'samples': len(lat),
    }


def measure_throughput(pipeline, duration_sec: int, prompt: str) -> Dict:
    """
    Measure throughput over duration.

    Returns dict with FPS and frame count.
    """
    print(f"  Throughput: {duration_sec} seconds...")

    device = next(pipeline.parameters()).device if hasattr(pipeline, 'parameters') else torch.device('cuda')

    # 3 frames per block
    frames_per_block = 3

    def get_noise():
        return torch.randn(1, frames_per_block, 16, 60, 104, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(WARMUP_FRAMES):
        noise = get_noise()
        with torch.no_grad():
            _ = pipeline.inference(noise, [prompt])

    torch.cuda.synchronize()
    start = time.perf_counter()
    frames = 0

    while time.perf_counter() - start < duration_sec:
        noise = get_noise()
        with torch.no_grad():
            _ = pipeline.inference(noise, [prompt])
        frames += frames_per_block  # Count actual frames generated

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        'fps': frames / elapsed,
        'total_frames': frames,
    }


def run_benchmark(pipeline, pipeline_type: str, preset: str = "", quick: bool = False) -> BenchmarkResults:
    """Run full benchmark suite on pipeline."""
    import datetime

    results = BenchmarkResults()
    results.pipeline_type = pipeline_type
    results.preset = preset
    results.timestamp = datetime.datetime.now().isoformat()
    results.device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    num_frames = QUICK_STEADY_STATE if quick else STEADY_STATE_FRAMES
    num_switches = QUICK_SWITCHES if quick else PROMPT_SWITCHES
    duration = QUICK_THROUGHPUT_SEC if quick else THROUGHPUT_DURATION_SEC

    print(f"\n{'='*60}")
    print(f"Benchmarking: {pipeline_type.upper()}" + (f" ({preset})" if preset else ""))
    print(f"{'='*60}")

    # Steady-state
    ss = measure_steady_state(pipeline, num_frames, WARMUP_FRAMES, TEST_PROMPTS[0])
    results.ss_mean = ss['mean']
    results.ss_std = ss['std']
    results.ss_p50 = ss['p50']
    results.ss_p95 = ss['p95']
    results.ss_p99 = ss['p99']
    results.ss_max = ss['max']
    results.ss_min = ss['min']
    results.ss_samples = ss['samples']

    # Prompt switch
    ps = measure_prompt_switch(pipeline, num_switches, FRAMES_BETWEEN_SWITCHES)
    results.ps_mean = ps['mean']
    results.ps_std = ps['std']
    results.ps_p99 = ps['p99']
    results.ps_max = ps['max']
    results.ps_samples = ps['samples']

    # Throughput
    tp = measure_throughput(pipeline, duration, TEST_PROMPTS[0])
    results.fps = tp['fps']
    results.total_frames = tp['total_frames']

    # Memory
    results.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Target
    results.meets_40ms = results.ss_max <= 40.0

    # Print summary
    print(f"\nResults:")
    print(f"  Steady-state: mean={results.ss_mean:.1f}ms p99={results.ss_p99:.1f}ms max={results.ss_max:.1f}ms")
    print(f"  Prompt switch: mean={results.ps_mean:.1f}ms max={results.ps_max:.1f}ms")
    print(f"  Throughput: {results.fps:.1f} FPS")
    print(f"  Memory: {results.peak_memory_gb:.2f} GB")
    print(f"  40ms target: {'PASS' if results.meets_40ms else 'FAIL'}")

    return results


def load_pipeline(config_path: str, optimized: bool = False, preset: str = "balanced"):
    """
    Load pipeline from LongLive config.

    Args:
        config_path: Path to inference config YAML
        optimized: If True, wrap with optimizations
        preset: Optimization preset name

    Returns:
        Pipeline instance
    """
    from omegaconf import OmegaConf
    import peft

    # Load LongLive config
    config = OmegaConf.load(config_path)
    device = torch.device("cuda")

    # Import LongLive pipeline
    from pipeline.causal_inference import CausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model

    # Create base pipeline
    pipeline = CausalInferencePipeline(config, device=device)

    # Load generator checkpoint
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
        pipeline.generator.load_state_dict(raw_gen_state_dict)

    # Apply LoRA if configured
    if getattr(config, "adapter", None):
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)

    # Move to device
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    if optimized:
        from optimizations import OptimizedCausalInferencePipeline, OptimizationConfig

        # Get preset
        presets = {
            'quality': OptimizationConfig.preset_quality,
            'balanced': OptimizationConfig.preset_balanced,
            'speed': OptimizationConfig.preset_speed,
        }
        opt_config = presets[preset]()

        # Wrap with optimizations
        pipeline = OptimizedCausalInferencePipeline.from_base(pipeline, opt_config)

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='LongLive inference config')
    parser.add_argument('--optimized', action='store_true', help='Use optimized pipeline')
    parser.add_argument('--compare', action='store_true', help='Compare baseline vs optimized')
    parser.add_argument('--preset', type=str, default='balanced', choices=['quality', 'balanced', 'speed'])
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer samples)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.compare:
        # Load both pipelines
        print("Loading baseline pipeline...")
        baseline = load_pipeline(args.config, optimized=False)

        print("Loading optimized pipeline...")
        optimized = load_pipeline(args.config, optimized=True, preset=args.preset)

        # Benchmark both
        base_results = run_benchmark(baseline, "baseline", quick=args.quick)
        opt_results = run_benchmark(optimized, "optimized", preset=args.preset, quick=args.quick)

        # Save
        with open(f"{args.output}/baseline.json", 'w') as f:
            json.dump(asdict(base_results), f, indent=2)
        with open(f"{args.output}/optimized_{args.preset}.json", 'w') as f:
            json.dump(asdict(opt_results), f, indent=2)

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
        print("-" * 60)

        ss_imp = (base_results.ss_max - opt_results.ss_max) / base_results.ss_max * 100
        ps_imp = (base_results.ps_mean - opt_results.ps_mean) / base_results.ps_mean * 100
        fps_imp = (opt_results.fps - base_results.fps) / base_results.fps * 100

        print(f"{'SS Max (ms)':<25} {base_results.ss_max:>12.1f} {opt_results.ss_max:>12.1f} {ss_imp:>+11.1f}%")
        print(f"{'PS Mean (ms)':<25} {base_results.ps_mean:>12.1f} {opt_results.ps_mean:>12.1f} {ps_imp:>+11.1f}%")
        print(f"{'FPS':<25} {base_results.fps:>12.1f} {opt_results.fps:>12.1f} {fps_imp:>+11.1f}%")
        print(f"{'40ms Target':<25} {'PASS' if base_results.meets_40ms else 'FAIL':>12} {'PASS' if opt_results.meets_40ms else 'FAIL':>12}")

    else:
        # Single pipeline benchmark
        pipeline = load_pipeline(args.config, optimized=args.optimized, preset=args.preset)
        results = run_benchmark(
            pipeline,
            "optimized" if args.optimized else "baseline",
            preset=args.preset if args.optimized else "",
            quick=args.quick
        )

        # Save
        filename = f"{args.output}/{'optimized_' + args.preset if args.optimized else 'baseline'}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nSaved: {filename}")


if __name__ == '__main__':
    main()
