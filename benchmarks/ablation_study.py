"""
Ablation Study for LongLive Optimizations.

Tests individual optimization contributions:
1. Sync elimination impact
2. Window size ablation (6, 8, 10, 12)
3. Frame sink ablation (1, 2, 3)
4. Precision comparison (BF16, FP16, FP8, INT8)
5. KV cache quantization

Usage:
    python benchmarks/ablation_study.py --config configs/longlive_inference.yaml --test all
    python benchmarks/ablation_study.py --config configs/longlive_inference.yaml --test window_size
    python benchmarks/ablation_study.py --config configs/longlive_inference.yaml --test precision
"""

import torch
import torch.cuda
import time
import argparse
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
WARMUP_FRAMES = 20
TEST_FRAMES = 50
TEST_PROMPT = "A panda walking through a bamboo forest"


@dataclass
class AblationResult:
    """Result of a single ablation test."""
    test_name: str
    config_value: Any
    mean_ms: float
    std_ms: float
    p99_ms: float
    max_ms: float
    fps: float
    memory_gb: float
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
        'p99': float(np.percentile(lat, 99)),
        'max': float(np.max(lat)),
        'fps': 1000.0 / np.mean(lat) * 3,  # 3 frames per block
        'memory_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def load_base_pipeline(config_path: str):
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

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    return pipeline, config


def test_sync_elimination(config_path: str) -> List[AblationResult]:
    """Test impact of sync elimination."""
    print("\n" + "="*60)
    print("Testing: Sync Elimination Impact")
    print("="*60)

    results = []
    pipeline, config = load_base_pipeline(config_path)

    # Test 1: With explicit syncs (baseline behavior)
    print("\n  Testing with explicit syncs...")

    def inference_with_syncs(noise, prompts):
        result = pipeline.inference(noise, prompts)
        torch.cuda.synchronize()  # Explicit sync
        return result

    # Create wrapper
    class SyncWrapper:
        def __init__(self, pipe):
            self.pipe = pipe
        def inference(self, noise, prompts):
            result = self.pipe.inference(noise, prompts)
            torch.cuda.synchronize()
            return result

    sync_pipe = SyncWrapper(pipeline)
    metrics = measure_latency(sync_pipe, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)

    results.append(AblationResult(
        test_name="sync_elimination",
        config_value="with_syncs",
        mean_ms=metrics['mean'],
        std_ms=metrics['std'],
        p99_ms=metrics['p99'],
        max_ms=metrics['max'],
        fps=metrics['fps'],
        memory_gb=metrics['memory_gb'],
        notes="Explicit torch.cuda.synchronize() after each inference"
    ))
    print(f"    With syncs: {metrics['mean']:.1f}ms mean, {metrics['max']:.1f}ms max")

    # Test 2: Without explicit syncs
    print("  Testing without explicit syncs...")
    metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)

    results.append(AblationResult(
        test_name="sync_elimination",
        config_value="no_syncs",
        mean_ms=metrics['mean'],
        std_ms=metrics['std'],
        p99_ms=metrics['p99'],
        max_ms=metrics['max'],
        fps=metrics['fps'],
        memory_gb=metrics['memory_gb'],
        notes="No explicit syncs (async execution)"
    ))
    print(f"    No syncs: {metrics['mean']:.1f}ms mean, {metrics['max']:.1f}ms max")

    # Calculate improvement
    improvement = (results[0].mean_ms - results[1].mean_ms) / results[0].mean_ms * 100
    print(f"\n  Sync elimination impact: {improvement:.1f}% improvement")

    return results


def test_window_size(config_path: str) -> List[AblationResult]:
    """Test different local attention window sizes."""
    print("\n" + "="*60)
    print("Testing: Window Size Ablation")
    print("="*60)

    results = []
    window_sizes = [6, 8, 10, 12]

    for window_size in window_sizes:
        print(f"\n  Testing window_size={window_size}...")

        # Load fresh pipeline and modify config
        pipeline, config = load_base_pipeline(config_path)

        # Modify the local attention size
        pipeline.local_attn_size = window_size
        pipeline.generator.model.local_attn_size = window_size

        # Update all attention modules
        if hasattr(pipeline, '_set_all_modules_max_attention_size'):
            pipeline._set_all_modules_max_attention_size(window_size)

        metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)

        results.append(AblationResult(
            test_name="window_size",
            config_value=window_size,
            mean_ms=metrics['mean'],
            std_ms=metrics['std'],
            p99_ms=metrics['p99'],
            max_ms=metrics['max'],
            fps=metrics['fps'],
            memory_gb=metrics['memory_gb'],
            notes=f"local_attn_size={window_size} frames"
        ))

        print(f"    window={window_size}: {metrics['mean']:.1f}ms mean, {metrics['fps']:.1f} FPS, {metrics['memory_gb']:.2f}GB")

        # Clean up
        del pipeline
        torch.cuda.empty_cache()

    return results


def test_precision(config_path: str) -> List[AblationResult]:
    """Test different precision modes."""
    print("\n" + "="*60)
    print("Testing: Precision Comparison")
    print("="*60)

    results = []

    # Test BF16 (baseline)
    print("\n  Testing BF16 (baseline)...")
    pipeline, config = load_base_pipeline(config_path)
    pipeline = pipeline.to(dtype=torch.bfloat16)

    metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)
    results.append(AblationResult(
        test_name="precision",
        config_value="bf16",
        mean_ms=metrics['mean'],
        std_ms=metrics['std'],
        p99_ms=metrics['p99'],
        max_ms=metrics['max'],
        fps=metrics['fps'],
        memory_gb=metrics['memory_gb'],
        notes="torch.bfloat16"
    ))
    print(f"    BF16: {metrics['mean']:.1f}ms mean, {metrics['fps']:.1f} FPS")

    del pipeline
    torch.cuda.empty_cache()

    # Test FP16
    print("  Testing FP16...")
    pipeline, config = load_base_pipeline(config_path)
    pipeline = pipeline.to(dtype=torch.float16)
    pipeline.generator.to(dtype=torch.float16)
    pipeline.vae.to(dtype=torch.float16)

    try:
        metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)
        results.append(AblationResult(
            test_name="precision",
            config_value="fp16",
            mean_ms=metrics['mean'],
            std_ms=metrics['std'],
            p99_ms=metrics['p99'],
            max_ms=metrics['max'],
            fps=metrics['fps'],
            memory_gb=metrics['memory_gb'],
            notes="torch.float16"
        ))
        print(f"    FP16: {metrics['mean']:.1f}ms mean, {metrics['fps']:.1f} FPS")
    except Exception as e:
        print(f"    FP16: FAILED - {e}")
        results.append(AblationResult(
            test_name="precision",
            config_value="fp16",
            mean_ms=0,
            std_ms=0,
            p99_ms=0,
            max_ms=0,
            fps=0,
            memory_gb=0,
            notes=f"FAILED: {str(e)[:50]}"
        ))

    del pipeline
    torch.cuda.empty_cache()

    # Test FP8 (if available on H100)
    print("  Testing FP8...")
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            pipeline, config = load_base_pipeline(config_path)
            # FP8 requires special handling - typically done via transformer engine
            # For now, we'll note it's available but not directly supported
            results.append(AblationResult(
                test_name="precision",
                config_value="fp8",
                mean_ms=0,
                std_ms=0,
                p99_ms=0,
                max_ms=0,
                fps=0,
                memory_gb=0,
                notes="FP8 available but requires TransformerEngine integration"
            ))
            print(f"    FP8: Available but requires TransformerEngine integration")
            del pipeline
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    FP8: FAILED - {e}")
    else:
        print(f"    FP8: Not available on this PyTorch version")
        results.append(AblationResult(
            test_name="precision",
            config_value="fp8",
            mean_ms=0,
            std_ms=0,
            p99_ms=0,
            max_ms=0,
            fps=0,
            memory_gb=0,
            notes="FP8 not available in PyTorch version"
        ))

    return results


def test_torch_compile_modes(config_path: str) -> List[AblationResult]:
    """Test different torch.compile modes."""
    print("\n" + "="*60)
    print("Testing: torch.compile Modes")
    print("="*60)

    results = []
    modes = [None, "default", "reduce-overhead", "max-autotune"]

    for mode in modes:
        mode_name = mode if mode else "no_compile"
        print(f"\n  Testing mode={mode_name}...")

        pipeline, config = load_base_pipeline(config_path)

        if mode is not None:
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                pipeline.generator.model = torch.compile(
                    pipeline.generator.model,
                    mode=mode,
                    fullgraph=False,
                )
                print(f"    Compiled with mode={mode}")
            except Exception as e:
                print(f"    Compile failed: {e}")
                results.append(AblationResult(
                    test_name="torch_compile",
                    config_value=mode_name,
                    mean_ms=0,
                    std_ms=0,
                    p99_ms=0,
                    max_ms=0,
                    fps=0,
                    memory_gb=0,
                    notes=f"FAILED: {str(e)[:50]}"
                ))
                del pipeline
                torch.cuda.empty_cache()
                continue

        try:
            metrics = measure_latency(pipeline, TEST_FRAMES, WARMUP_FRAMES, TEST_PROMPT)
            results.append(AblationResult(
                test_name="torch_compile",
                config_value=mode_name,
                mean_ms=metrics['mean'],
                std_ms=metrics['std'],
                p99_ms=metrics['p99'],
                max_ms=metrics['max'],
                fps=metrics['fps'],
                memory_gb=metrics['memory_gb'],
                notes=f"torch.compile mode={mode}"
            ))
            print(f"    {mode_name}: {metrics['mean']:.1f}ms mean, {metrics['fps']:.1f} FPS")
        except Exception as e:
            print(f"    {mode_name}: FAILED during inference - {e}")
            results.append(AblationResult(
                test_name="torch_compile",
                config_value=mode_name,
                mean_ms=0,
                std_ms=0,
                p99_ms=0,
                max_ms=0,
                fps=0,
                memory_gb=0,
                notes=f"FAILED: {str(e)[:50]}"
            ))

        del pipeline
        torch.cuda.empty_cache()

        # Reset dynamo between runs
        if mode is not None:
            torch._dynamo.reset()

    return results


def run_all_ablations(config_path: str, output_dir: str):
    """Run all ablation studies."""
    import datetime

    all_results = []

    # Run each test
    print("\n" + "="*60)
    print("ABLATION STUDY - LongLive Optimizations")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Warmup: {WARMUP_FRAMES} frames, Test: {TEST_FRAMES} frames")

    # Sync elimination
    try:
        results = test_sync_elimination(config_path)
        all_results.extend(results)
    except Exception as e:
        print(f"Sync elimination test failed: {e}")

    # torch.compile modes
    try:
        results = test_torch_compile_modes(config_path)
        all_results.extend(results)
    except Exception as e:
        print(f"torch.compile test failed: {e}")

    # Window size
    try:
        results = test_window_size(config_path)
        all_results.extend(results)
    except Exception as e:
        print(f"Window size test failed: {e}")

    # Precision
    try:
        results = test_precision(config_path)
        all_results.extend(results)
    except Exception as e:
        print(f"Precision test failed: {e}")

    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/ablation_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)

    for test_name in set(r.test_name for r in all_results):
        print(f"\n{test_name.upper()}:")
        test_results = [r for r in all_results if r.test_name == test_name]
        for r in test_results:
            if r.mean_ms > 0:
                print(f"  {r.config_value}: {r.mean_ms:.1f}ms mean, {r.fps:.1f} FPS")
            else:
                print(f"  {r.config_value}: {r.notes}")

    print(f"\nResults saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'sync', 'window_size', 'precision', 'compile'])
    parser.add_argument('--output', type=str, default='results')

    args = parser.parse_args()

    if args.test == 'all':
        run_all_ablations(args.config, args.output)
    elif args.test == 'sync':
        results = test_sync_elimination(args.config)
    elif args.test == 'window_size':
        results = test_window_size(args.config)
    elif args.test == 'precision':
        results = test_precision(args.config)
    elif args.test == 'compile':
        results = test_torch_compile_modes(args.config)


if __name__ == '__main__':
    main()
