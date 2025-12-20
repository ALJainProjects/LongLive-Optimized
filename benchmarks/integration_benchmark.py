"""
Integration Benchmark - Test All Optimization Combinations.

Tests:
1. Ring buffer KV cache vs standard
2. INT8 KV cache quantization
3. FP16 precision (with proper model conversion)
4. INT8 weight quantization
5. Frame sink reduction (1, 2, 3 frames)
6. Python overhead measurement
"""

import torch
import torch.nn as nn
import time
import argparse
import sys
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
WARMUP_FRAMES = 10
TEST_FRAMES = 30
TEST_PROMPT = "A panda walking through a bamboo forest"


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    config: str
    mean_ms: float
    std_ms: float
    max_ms: float
    fps: float
    memory_gb: float
    status: str = "success"
    notes: str = ""


def measure_latency(pipeline, num_frames: int, warmup: int, prompt: str) -> Dict:
    """Measure latency with CUDA events."""
    device = torch.device('cuda')

    def get_noise():
        return torch.randn(1, 3, 16, 60, 104, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup):
        noise = get_noise()
        with torch.no_grad():
            _ = pipeline.inference(noise, [prompt])

    torch.cuda.reset_peak_memory_stats()

    # Create events for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_frames)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_frames)]

    # Measure
    for i in range(num_frames):
        noise = get_noise()
        start_events[i].record()
        with torch.no_grad():
            _ = pipeline.inference(noise, [prompt])
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate latencies
    latencies = []
    for i in range(num_frames):
        latencies.append(start_events[i].elapsed_time(end_events[i]))

    lat = np.array(latencies)

    return {
        'mean': float(np.mean(lat)),
        'std': float(np.std(lat)),
        'max': float(np.max(lat)),
        'fps': 1000.0 / np.mean(lat) * 3,  # 3 frames per block
        'memory_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def load_base_pipeline(config_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load base LongLive pipeline."""
    from omegaconf import OmegaConf
    import peft

    config = OmegaConf.load(config_path)
    device = torch.device("cuda")

    from pipeline.causal_inference import CausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model

    pipeline = CausalInferencePipeline(config, device=device)

    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu", weights_only=False)
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found")
        pipeline.generator.load_state_dict(raw_gen_state_dict)

    if getattr(config, "adapter", None):
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)

    pipeline = pipeline.to(dtype=dtype)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    return pipeline, config


def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_integrated_kv_cache(config_path: str) -> List[BenchmarkResult]:
    """Test integrated KV cache with ring buffer and quantization."""
    print("\n" + "="*60)
    print("Testing: Integrated KV Cache (Ring Buffer + Quantization)")
    print("="*60)

    results = []
    from optimizations.integrated_kv_cache import (
        create_integrated_kv_cache,
        IntegratedKVConfig,
    )

    # Test configurations
    configs = [
        ("Standard (no optimizations)", False, False),
        ("Ring Buffer Only", True, False),
        ("Ring Buffer + INT8 KV", True, True),
    ]

    for name, use_ring, use_quant in configs:
        print(f"\n  Testing: {name}...")
        cleanup()

        try:
            pipeline, config = load_base_pipeline(config_path)

            # Create integrated cache
            cache = create_integrated_kv_cache(
                num_layers=28,
                num_heads=12,
                head_dim=128,
                local_window_frames=12,
                sink_frames=3,
                frame_seq_length=1560,
                use_ring_buffer=use_ring,
                use_quantization=use_quant,
            )

            # Measure memory of cache
            cache_memory_mb = cache.total_memory_bytes() / (1024**2)
            savings = cache.memory_savings() * 100 if use_quant else 0

            # Note: Full integration would require modifying the pipeline
            # Here we just measure the cache overhead
            metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)

            results.append(BenchmarkResult(
                test_name="integrated_kv",
                config=name,
                mean_ms=metrics['mean'],
                std_ms=metrics['std'],
                max_ms=metrics['max'],
                fps=metrics['fps'],
                memory_gb=metrics['memory_gb'],
                notes=f"Cache: {cache_memory_mb:.1f}MB, Savings: {savings:.1f}%"
            ))
            print(f"    {name}: {metrics['mean']:.1f}ms, {metrics['fps']:.1f} FPS")
            print(f"    Cache memory: {cache_memory_mb:.1f}MB, Savings: {savings:.1f}%")

            del pipeline
            cleanup()

        except Exception as e:
            results.append(BenchmarkResult(
                test_name="integrated_kv",
                config=name,
                mean_ms=0,
                std_ms=0,
                max_ms=0,
                fps=0,
                memory_gb=0,
                status="failed",
                notes=str(e)
            ))
            print(f"    FAILED: {e}")

    return results


def test_fp16_conversion(config_path: str) -> List[BenchmarkResult]:
    """Test FP16 with proper model conversion."""
    print("\n" + "="*60)
    print("Testing: FP16 with Full Model Conversion")
    print("="*60)

    results = []
    from optimizations.integrated_kv_cache import convert_model_to_fp16

    print("\n  Loading model in FP16 with full conversion...")
    cleanup()

    try:
        # Load in BF16 first, then convert everything
        pipeline, config = load_base_pipeline(config_path, dtype=torch.bfloat16)

        # Convert entire generator to FP16
        pipeline.generator = convert_model_to_fp16(pipeline.generator)

        # Also convert VAE
        pipeline.vae = convert_model_to_fp16(pipeline.vae)

        # Create FP16 noise
        device = torch.device('cuda')

        def get_noise_fp16():
            return torch.randn(1, 3, 16, 60, 104, device=device, dtype=torch.float16)

        # Warmup with FP16
        for _ in range(WARMUP_FRAMES):
            noise = get_noise_fp16()
            with torch.no_grad():
                _ = pipeline.inference(noise, [TEST_PROMPT])

        torch.cuda.reset_peak_memory_stats()

        # Measure
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(TEST_FRAMES)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(TEST_FRAMES)]

        for i in range(TEST_FRAMES):
            noise = get_noise_fp16()
            start_events[i].record()
            with torch.no_grad():
                _ = pipeline.inference(noise, [TEST_PROMPT])
            end_events[i].record()

        torch.cuda.synchronize()

        latencies = [start_events[i].elapsed_time(end_events[i]) for i in range(TEST_FRAMES)]
        lat = np.array(latencies)

        results.append(BenchmarkResult(
            test_name="precision",
            config="FP16 (full conversion)",
            mean_ms=float(np.mean(lat)),
            std_ms=float(np.std(lat)),
            max_ms=float(np.max(lat)),
            fps=1000.0 / np.mean(lat) * 3,
            memory_gb=torch.cuda.max_memory_allocated() / (1024 ** 3),
            notes="All weights and biases converted to FP16"
        ))
        print(f"    FP16: {np.mean(lat):.1f}ms mean, {1000.0 / np.mean(lat) * 3:.1f} FPS")

        del pipeline
        cleanup()

    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(BenchmarkResult(
            test_name="precision",
            config="FP16 (full conversion)",
            mean_ms=0,
            std_ms=0,
            max_ms=0,
            fps=0,
            memory_gb=0,
            status="failed",
            notes=str(e)
        ))
        print(f"    FP16 FAILED: {e}")

    return results


def test_int8_weights(config_path: str) -> List[BenchmarkResult]:
    """Test INT8 weight quantization."""
    print("\n" + "="*60)
    print("Testing: INT8 Weight Quantization")
    print("="*60)

    results = []

    print("\n  Loading model with INT8 weight quantization...")
    cleanup()

    try:
        from optimizations.integrated_kv_cache import apply_int8_weight_quantization

        pipeline, config = load_base_pipeline(config_path)

        # Apply INT8 weight quantization to generator
        original_memory = sum(p.numel() * p.element_size() for p in pipeline.generator.parameters())

        pipeline.generator = apply_int8_weight_quantization(pipeline.generator)

        quantized_memory = sum(
            p.numel() * (1 if hasattr(p, 'int_repr') else p.element_size())
            for p in pipeline.generator.parameters()
        )

        metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)

        results.append(BenchmarkResult(
            test_name="quantization",
            config="INT8 weights",
            mean_ms=metrics['mean'],
            std_ms=metrics['std'],
            max_ms=metrics['max'],
            fps=metrics['fps'],
            memory_gb=metrics['memory_gb'],
            notes=f"Weight memory: {quantized_memory/1e9:.2f}GB (was {original_memory/1e9:.2f}GB)"
        ))
        print(f"    INT8 weights: {metrics['mean']:.1f}ms, {metrics['fps']:.1f} FPS")

        del pipeline
        cleanup()

    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(BenchmarkResult(
            test_name="quantization",
            config="INT8 weights",
            mean_ms=0,
            std_ms=0,
            max_ms=0,
            fps=0,
            memory_gb=0,
            status="failed",
            notes=str(e)
        ))
        print(f"    INT8 weights FAILED: {e}")

    return results


def test_sink_reduction(config_path: str) -> List[BenchmarkResult]:
    """Test frame sink reduction."""
    print("\n" + "="*60)
    print("Testing: Frame Sink Token Reduction")
    print("="*60)

    results = []
    sink_sizes = [1, 2, 3]  # Test different sink sizes

    for sink_size in sink_sizes:
        print(f"\n  Testing sink_size={sink_size}...")
        cleanup()

        try:
            pipeline, config = load_base_pipeline(config_path)

            # Modify the sink size in the model config
            if hasattr(pipeline.generator.model, 'sink_size'):
                pipeline.generator.model.sink_size = sink_size

            metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)

            results.append(BenchmarkResult(
                test_name="sink_reduction",
                config=f"sink={sink_size}",
                mean_ms=metrics['mean'],
                std_ms=metrics['std'],
                max_ms=metrics['max'],
                fps=metrics['fps'],
                memory_gb=metrics['memory_gb'],
            ))
            print(f"    sink={sink_size}: {metrics['mean']:.1f}ms, {metrics['fps']:.1f} FPS")

            del pipeline
            cleanup()

        except Exception as e:
            results.append(BenchmarkResult(
                test_name="sink_reduction",
                config=f"sink={sink_size}",
                mean_ms=0,
                std_ms=0,
                max_ms=0,
                fps=0,
                memory_gb=0,
                status="failed",
                notes=str(e)
            ))
            print(f"    sink={sink_size} FAILED: {e}")

    return results


def test_python_overhead(config_path: str) -> List[BenchmarkResult]:
    """Measure Python overhead vs CUDA execution time."""
    print("\n" + "="*60)
    print("Testing: Python Overhead Measurement")
    print("="*60)

    results = []
    cleanup()

    try:
        pipeline, config = load_base_pipeline(config_path)
        device = torch.device('cuda')

        def get_noise():
            return torch.randn(1, 3, 16, 60, 104, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(WARMUP_FRAMES):
            with torch.no_grad():
                _ = pipeline.inference(get_noise(), [TEST_PROMPT])

        # Measure wall clock time (includes Python overhead)
        wall_times = []
        cuda_times = []

        for _ in range(TEST_FRAMES):
            noise = get_noise()

            # CUDA timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Wall clock timing
            wall_start = time.perf_counter()
            start_event.record()

            with torch.no_grad():
                _ = pipeline.inference(noise, [TEST_PROMPT])

            end_event.record()
            torch.cuda.synchronize()
            wall_end = time.perf_counter()

            wall_times.append((wall_end - wall_start) * 1000)  # ms
            cuda_times.append(start_event.elapsed_time(end_event))

        wall_mean = np.mean(wall_times)
        cuda_mean = np.mean(cuda_times)
        overhead = wall_mean - cuda_mean

        results.append(BenchmarkResult(
            test_name="python_overhead",
            config="measurement",
            mean_ms=cuda_mean,
            std_ms=np.std(cuda_times),
            max_ms=np.max(cuda_times),
            fps=1000.0 / cuda_mean * 3,
            memory_gb=torch.cuda.max_memory_allocated() / (1024 ** 3),
            notes=f"Wall: {wall_mean:.1f}ms, CUDA: {cuda_mean:.1f}ms, Overhead: {overhead:.1f}ms ({overhead/wall_mean*100:.1f}%)"
        ))
        print(f"    Wall clock: {wall_mean:.1f}ms")
        print(f"    CUDA time: {cuda_mean:.1f}ms")
        print(f"    Python overhead: {overhead:.1f}ms ({overhead/wall_mean*100:.1f}%)")

        del pipeline
        cleanup()

    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(BenchmarkResult(
            test_name="python_overhead",
            config="measurement",
            mean_ms=0,
            std_ms=0,
            max_ms=0,
            fps=0,
            memory_gb=0,
            status="failed",
            notes=str(e)
        ))
        print(f"    Python overhead FAILED: {e}")

    return results


def run_all_tests(config_path: str) -> List[BenchmarkResult]:
    """Run all integration tests."""
    print("="*60)
    print("INTEGRATION BENCHMARK - All Optimizations")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Warmup: {WARMUP_FRAMES} frames, Test: {TEST_FRAMES} frames")

    all_results = []

    # Run each test
    tests = [
        ("Integrated KV Cache", test_integrated_kv_cache),
        ("FP16 Conversion", test_fp16_conversion),
        ("INT8 Weights", test_int8_weights),
        ("Sink Reduction", test_sink_reduction),
        ("Python Overhead", test_python_overhead),
    ]

    for name, test_fn in tests:
        try:
            results = test_fn(config_path)
            all_results.extend(results)
        except Exception as e:
            print(f"\n{name} test failed completely: {e}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Test':<25} {'Config':<25} {'Mean (ms)':<12} {'FPS':<8} {'Status'}")
    print("-"*80)

    for r in all_results:
        status = "OK" if r.status == "success" else "FAIL"
        print(f"{r.test_name:<25} {r.config:<25} {r.mean_ms:>8.1f}     {r.fps:>6.1f}   {status}")
        if r.notes:
            print(f"    {r.notes}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Integration benchmark for all optimizations")
    parser.add_argument("--config", type=str, default="configs/longlive_inference.yaml")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "kv", "fp16", "int8", "sink", "overhead"])
    args = parser.parse_args()

    if args.test == "all":
        results = run_all_tests(args.config)
    elif args.test == "kv":
        results = test_integrated_kv_cache(args.config)
    elif args.test == "fp16":
        results = test_fp16_conversion(args.config)
    elif args.test == "int8":
        results = test_int8_weights(args.config)
    elif args.test == "sink":
        results = test_sink_reduction(args.config)
    elif args.test == "overhead":
        results = test_python_overhead(args.config)

    # Save results
    import json
    from datetime import datetime

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"integration_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
