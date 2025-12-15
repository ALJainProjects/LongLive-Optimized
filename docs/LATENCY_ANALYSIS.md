# Latency Analysis Report

## Executive Summary

This document provides a comprehensive analysis of latency characteristics in LongLive video generation, defining measurement methodology, identifying bottlenecks, and establishing targets for optimization.

**Target**: 40ms worst-case inter-frame latency for 25 FPS real-time interaction.

---

## 1. Latency Metrics

### 1.1 Steady-State Inter-Frame Latency

The primary metric for real-time performance. Measures the time between consecutive frame completions during continuous generation.

**Formula**:
```
inter_frame_latency[N] = t_complete[N+1] - t_complete[N]

where:
    t_complete[N] = GPU timestamp when frame N's pixels are available
```

**Critical Scenarios**:

| Scenario | Description | Impact |
|----------|-------------|--------|
| Normal frame | Within same batch block | Lowest latency |
| Batch boundary | Frames span different batches | KV rolling overhead |
| Cache full | Local attention window saturated | Eviction + write |

**Why We Measure Max/P99**:
- Mean latency hides worst-case spikes
- Real-time systems fail on worst case, not average
- A single 100ms frame in 1000 frames ruins user experience

**Measurement Implementation**:
```python
# GPU-accurate timing with CUDA events
# Events timestamp when GPU reaches that point in command stream
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_frames)]

for i in range(num_frames):
    # Generate frame (includes all pipeline stages)
    _ = pipeline.generate_frame(noise, prompt, frame_idx=i)
    end_events[i].record()

# Synchronize at end (not per-frame - that would add sync overhead)
torch.cuda.synchronize()

# Compute inter-frame gaps
latencies = []
for i in range(num_frames - 1):
    # elapsed_time gives milliseconds between two events
    latencies.append(end_events[i].elapsed_time(end_events[i + 1]))
```

### 1.2 Prompt-Switch Latency

Measures responsiveness to user input changes. Critical for interactive applications.

**Formula**:
```
prompt_switch_latency = t_first_new_frame - t_prompt_change_triggered

Components:
    1. Prompt encoding time (or cache lookup: ~0ms if cached)
    2. KV-recache time (recompute attention for N recent frames with new prompt)
    3. First frame denoising time
    4. First frame VAE decode time
```

**Why This Matters**:
LongLive's frame-sink mechanism caches KV values computed with the old prompt. On prompt switch, these must be recomputed to avoid visual artifacts where old prompt "bleeds" into new frames.

**Factors Affecting Switch Latency**:
- `local_attn_size`: More frames = more to recache
- Prompt complexity: Longer prompts take longer to encode
- Cache hit: Previously seen prompts are nearly instant

### 1.3 Component-Level Breakdown

Detailed profiling of each pipeline stage to identify optimization targets.

| Stage | Typical % | Description |
|-------|-----------|-------------|
| `denoise_step_0` | 15-18% | First denoising step (t=1000) |
| `denoise_step_1` | 15-18% | Second step (t=750) |
| `denoise_step_2` | 15-18% | Third step (t=500) |
| `denoise_step_3` | 15-18% | Fourth step (t=250) |
| `vae_decode` | 15-20% | Latent → pixel conversion |
| `kv_cache_ops` | 5-10% | Read/write/roll cache |
| `prompt_encode` | 0-5% | Text → embedding (0 if cached) |
| `sync_overhead` | 2-5% | Python/CUDA sync points |

---

## 2. Baseline Profiling

### 2.1 Expected Unoptimized Performance

Based on LongLive paper numbers and architectural analysis:

| Metric | Value | Source |
|--------|-------|--------|
| Throughput | 20.7 FPS | Paper Table 1 |
| Per-frame time | ~48ms | 1000ms / 20.7 |
| With FP8 | 24.8 FPS (~40ms) | Paper Table 1 |

### 2.2 Time Budget Breakdown

For a ~48ms frame at 20.7 FPS baseline:

```
┌─────────────────────────────────────────────────────────┐
│                    Frame Generation (~48ms)              │
├───────────────────────────────────────────────────────┬─┤
│         Denoising Loop (4 steps) ~30-34ms             │ │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │ │
│ │Step 1  │ │Step 2  │ │Step 3  │ │Step 4  │          │ │
│ │~7-8ms  │ │~7-8ms  │ │~7-8ms  │ │~7-8ms  │          │ │
│ └────────┘ └────────┘ └────────┘ └────────┘          │ │
├──────────────────────────────────────────────────────┬┘ │
│            VAE Decode ~7-10ms                        │  │
├──────────────────────────────────────────────────────┴──┤
│ KV Ops ~3-5ms │ Prompt 0-2ms │ Overhead ~2-4ms         │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Per-Denoising-Step Breakdown

Each denoising step runs the full DiT (Diffusion Transformer):

```
┌─────────────────────────────────────────────────────────┐
│              Single Denoising Step (~7-8ms)             │
├─────────────────────────────────────────────────────────┤
│ Self-Attention  │ Cross-Attention │ FFN │ Misc        │
│   ~3-4ms        │    ~1-2ms       │~2ms │ ~0.5ms      │
│                 │                 │     │             │
│ - Q,K,V proj    │ - K,V from text │     │ - timestep  │
│ - Flash attn    │ - Flash attn    │     │   embedding │
│ - O proj        │ - O proj        │     │ - norm      │
│ - KV cache IO   │                 │     │ - residual  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Bottleneck Identification

### 3.1 Primary Bottlenecks

**1. Denoising Loop Sequential Dependency**
- 4 steps must execute in sequence
- Each step depends on previous step's output
- ~70% of total frame time

**2. Kernel Launch Overhead**
- Each PyTorch operation = new CUDA kernel launch
- ~1000+ kernels per frame
- ~0.01-0.05ms per launch, adds up

**3. KV Cache Memory Operations**
- Cache rolling copies data when window slides
- Allocation during rolling in non-optimized code
- Worst case at batch boundaries

**4. Python Interpreter Overhead**
- Control flow decisions
- Tensor shape checks
- Callback/hook invocations

### 3.2 Secondary Bottlenecks

- Memory allocation for intermediate tensors
- Host-device synchronization (`.item()`, `.cpu()`)
- Prompt encoding on cache miss
- VAE cannot overlap with denoising (sequential)

### 3.3 Fundamental Limits

These cannot be optimized without model changes:

| Limit | Description | Workaround |
|-------|-------------|------------|
| Autoregressive dependency | Frame N needs KV from N-1 | None (architectural) |
| 4 denoising steps | Sequential requirement | Distillation (retrain) |
| Attention O(n²) | Context length cost | Window attention (already done) |

---

## 4. Optimization Opportunities

### 4.1 CUDA Graphs (Target: 30-50% reduction)

**Problem**: Kernel launch overhead accumulates across 1000+ ops
**Solution**: Capture operations as graph, replay with single launch

```python
# Without CUDA graphs: 1000+ kernel launches
for step in steps:
    x = model(x)  # Each op is separate launch

# With CUDA graphs: 1 launch replays all ops
graph.replay()  # All 1000+ ops in single launch
```

**Expected Impact**:
- Eliminates ~0.01-0.05ms × 1000 ops = 10-50ms overhead
- Most benefit from denoising loop capture
- Requires static shapes (ring buffer helps)

### 4.2 Static KV Cache (Target: 15-20% reduction)

**Problem**: Dynamic allocation during cache operations
**Solution**: Pre-allocate fixed buffers, use ring buffer

```python
# Without: allocation during rolling
new_cache = torch.cat([old_cache[:, n:], new_kv], dim=1)  # Allocates

# With ring buffer: in-place write, no allocation
cache[:, write_idx:write_idx+n].copy_(new_kv)  # In-place
write_idx = (write_idx + n) % cache_size
```

**Expected Impact**:
- Zero allocation during inference
- CUDA graph compatible (no dynamic shapes)
- Faster cache updates (no concat)

### 4.3 Async VAE Pipeline (Target: 5-10% reduction)

**Problem**: VAE decode blocks next frame generation
**Solution**: Double-buffer, decode on separate stream

```
Without async:
  [Denoise N]----[VAE N]----[Denoise N+1]----[VAE N+1]

With async:
  [Denoise N]----[Denoise N+1]----[Denoise N+2]
        └──[VAE N]────[VAE N+1]────[VAE N+2]
```

**Expected Impact**:
- VAE decode (~8ms) overlaps with denoising
- Net reduction: ~5-8ms per frame

### 4.4 Prompt Cache (Target: Near-zero prompt switch)

**Problem**: Prompt encoding adds latency on change
**Solution**: LRU cache of prompt embeddings

```python
# Without cache: always encode
embeddings = text_encoder(prompt)  # ~1-2ms

# With cache: instant lookup for seen prompts
embeddings = cache.get(prompt)  # <0.01ms
```

**Expected Impact**:
- Eliminates 1-2ms per frame for repeated prompts
- Near-instant prompt switching for cached prompts

---

## 5. Measurement Protocol

Our `benchmark_suite.py` implements exactly this protocol:

### 5.1 Sample Sizes

| Measurement | Samples | Rationale |
|-------------|---------|-----------|
| Steady-state | 1000 frames | P99/max stability |
| Prompt switch | 50 switches | Diverse prompt pairs |
| Throughput | 60 seconds | Sustained performance |
| Breakdown | 100 frames | Component averages |

### 5.2 Warmup

The benchmark runs 50 warmup frames before measurement:
- Stabilizes GPU clock speed
- Allows CUDA graph capture
- Initializes memory pools
- Memory stats reset after warmup

### 5.3 CUDA Events

Uses `torch.cuda.Event` for GPU-accurate timing:
- Events record GPU timestamp when reached in command stream
- Avoids host-side timing inaccuracies
- Single sync at end of measurement (not per-frame)

### 5.4 Reporting

For each latency metric we report:
- Mean ± std
- P50, P95, P99
- Max (worst case - most important for real-time)

---

## 6. Running Benchmarks

```bash
# Full comparison
python benchmarks/benchmark_suite.py \
    --config configs/longlive_inference.yaml \
    --compare --preset balanced

# Quick validation
python benchmarks/benchmark_suite.py \
    --config configs/longlive_inference.yaml \
    --compare --quick

# Single pipeline
python benchmarks/benchmark_suite.py \
    --config configs/longlive_inference.yaml \
    --optimized --preset speed
```

---

## 7. Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Steady-state max | ≤40ms | Primary target |
| Steady-state P99 | ≤38ms | Consistent performance |
| Prompt switch max | ≤60ms | User-perceptible limit |
| Throughput | ≥25 FPS | Real-time threshold |
| Memory | ≤40 GB | H100 80GB headroom |
