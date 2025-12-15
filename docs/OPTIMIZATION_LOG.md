# Optimization Log

This document tracks all optimization experiments, their configurations, measured results, and decisions.

## How to Read This Log

Each entry records:
- **Configuration**: What was enabled/disabled
- **Latency**: Steady-state max (worst case), mean, and P99
- **FPS**: Measured throughput
- **Memory**: Peak GPU memory usage
- **Quality**: Any observed impact on visual output
- **Decision**: Keep/Drop and rationale

---

## Baseline Measurement

| Setting | Value |
|---------|-------|
| Model | LongLive-1.3B |
| GPU | NVIDIA H100 80GB |
| Denoising Steps | 4 (1000, 750, 500, 250) |
| Local Attention | 12 frames |
| Frame Sink | 3 frames |
| Dtype | bfloat16 |

**Baseline Results** (to be filled after running):

| Metric | Value |
|--------|-------|
| SS Mean | ___ ms |
| SS P99 | ___ ms |
| SS Max | ___ ms |
| FPS | ___ |
| Memory | ___ GB |

---

## Optimization Experiments

### Experiment 1: CUDA Graphs Only

**Configuration**:
```yaml
use_cuda_graphs: true
use_static_kv: false
use_quantized_kv: false
use_async_vae: false
use_prompt_cache: false
```

**Results**:

| Metric | Baseline | CUDA Graphs | Change |
|--------|----------|-------------|--------|
| SS Mean | ___ ms | ___ ms | ___% |
| SS Max | ___ ms | ___ ms | ___% |
| FPS | ___ | ___ | ___% |
| Memory | ___ GB | ___ GB | ___% |

**Quality**: (Note any visual differences)

**Decision**: [ ] Keep  [ ] Drop

**Notes**:

---

### Experiment 2: Static KV Cache Only

**Configuration**:
```yaml
use_cuda_graphs: false
use_static_kv: true
use_quantized_kv: false
use_async_vae: false
use_prompt_cache: false
```

**Results**:

| Metric | Baseline | Static KV | Change |
|--------|----------|-----------|--------|
| SS Mean | ___ ms | ___ ms | ___% |
| SS Max | ___ ms | ___ ms | ___% |
| FPS | ___ | ___ | ___% |
| Memory | ___ GB | ___ GB | ___% |

**Quality**:

**Decision**: [ ] Keep  [ ] Drop

**Notes**:

---

### Experiment 3: Async VAE Only

**Configuration**:
```yaml
use_cuda_graphs: false
use_static_kv: false
use_quantized_kv: false
use_async_vae: true
use_prompt_cache: false
```

**Results**:

| Metric | Baseline | Async VAE | Change |
|--------|----------|-----------|--------|
| SS Mean | ___ ms | ___ ms | ___% |
| SS Max | ___ ms | ___ ms | ___% |
| FPS | ___ | ___ | ___% |
| Memory | ___ GB | ___ GB | ___% |

**Quality**:

**Decision**: [ ] Keep  [ ] Drop

**Notes**:

---

### Experiment 4: Quality Preset (All Conservative)

**Configuration**:
```yaml
use_cuda_graphs: false
use_static_kv: true
use_quantized_kv: false
use_async_vae: true
use_prompt_cache: true
model_dtype: bfloat16
```

**Results**:

| Metric | Baseline | Quality | Change |
|--------|----------|---------|--------|
| SS Mean | ___ ms | ___ ms | ___% |
| SS Max | ___ ms | ___ ms | ___% |
| PS Mean | ___ ms | ___ ms | ___% |
| FPS | ___ | ___ | ___% |
| Memory | ___ GB | ___ GB | ___% |

**Quality**: (Run quality_eval.py)
- PSNR: ___ dB
- SSIM: ___
- LPIPS: ___
- CLIP Delta: ___

**Decision**: [ ] Keep  [ ] Drop

---

### Experiment 5: Balanced Preset (Recommended)

**Configuration**:
```yaml
use_cuda_graphs: false      # torch.compile handles dynamic shapes better
use_torch_compile: true     # reduce-overhead mode for kernel fusion
use_static_kv: true
use_quantized_kv: false
use_async_vae: true
use_prompt_cache: true
model_dtype: bfloat16
```

**Results**:

| Metric | Baseline | Balanced | Change |
|--------|----------|----------|--------|
| SS Mean | ___ ms | ___ ms | ___% |
| SS Max | ___ ms | ___ ms | ___% |
| PS Mean | ___ ms | ___ ms | ___% |
| FPS | ___ | ___ | ___% |
| Memory | ___ GB | ___ GB | ___% |

**Quality**:
- PSNR: ___ dB
- SSIM: ___
- LPIPS: ___
- CLIP Delta: ___

**Decision**: [ ] Keep  [ ] Drop

---

### Experiment 6: Speed Preset (Aggressive)

**Configuration**:
```yaml
use_cuda_graphs: false      # torch.compile handles dynamic shapes better
use_torch_compile: true     # max-autotune mode for maximum optimization
compile_mode: max-autotune
use_static_kv: false        # Quantized KV takes precedence
use_quantized_kv: true
kv_quantization: int8
use_async_vae: true
use_prompt_cache: true
model_dtype: bfloat16       # Model stays bfloat16, only KV is quantized
```

**Results**:

| Metric | Baseline | Speed | Change |
|--------|----------|-------|--------|
| SS Mean | ___ ms | ___ ms | ___% |
| SS Max | ___ ms | ___ ms | ___% |
| PS Mean | ___ ms | ___ ms | ___% |
| FPS | ___ | ___ | ___% |
| Memory | ___ GB | ___ GB | ___% |

**Quality**:
- PSNR: ___ dB
- SSIM: ___
- LPIPS: ___
- CLIP Delta: ___

**Decision**: [ ] Keep  [ ] Drop

---

## Summary Table

| Preset | SS Max | FPS | Memory | Quality | 40ms Target |
|--------|--------|-----|--------|---------|-------------|
| Baseline | ___ms | ___ | ___GB | Reference | [ ] |
| Quality | ___ms | ___ | ___GB | ___ | [ ] |
| Balanced | ___ms | ___ | ___GB | ___ | [ ] |
| Speed | ___ms | ___ | ___GB | ___ | [ ] |

## Sweet Spot Recommendation

Based on experiments, the recommended configuration is:

**Preset**: ___

**Rationale**:
-
-
-

**When to use other presets**:
- Quality: ___
- Speed: ___

---

## Notes & Observations

(Record any unexpected findings, edge cases, or additional insights)
