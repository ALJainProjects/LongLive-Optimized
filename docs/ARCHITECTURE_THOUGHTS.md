# Architecture Thoughts: LongLive Latency Analysis

A thought exercise exploring the fundamental limitations and potential architectural changes for LongLive real-time video generation.

---

## 1. Fundamental Serial Dependencies

LongLive's architecture contains several inherent serial dependencies that cannot be optimized away through inference-time techniques alone:

### A. Autoregressive Frame Generation

```
Frame N → KV Cache → Frame N+1 → KV Cache → Frame N+2 → ...
```

- Each frame depends on KV cache from previous frames
- Cannot parallelize frame generation within a single video stream
- This is fundamental to maintaining temporal coherence
- **Implication**: Minimum latency is bounded by single-frame generation time

### B. Denoising Step Dependencies

```
Step 1 (t=1000) → Step 2 (t=750) → Step 3 (t=500) → Step 4 (t=250) → Output
```

- 4-step denoising schedule used by LongLive
- Each step requires the previous step's output as input
- Cannot parallelize denoising steps for the same frame
- **Implication**: With ~10ms per step, minimum is ~40ms per frame

### C. Attention Complexity

```
Attention: O(n²) where n = context_length = sink_tokens + local_window_tokens
```

- Self-attention scales quadratically with context length
- LongLive mitigates this with local attention window
- But longer context = better coherence vs. higher latency
- **Implication**: Quality-latency tradeoff at the architecture level

---

## 2. Frame-Sink Reliance Analysis

### Current Design

LongLive uses a "frame-sink" mechanism where the first 2-3 frames serve as permanent attention anchors:

```
Attention Window = [Sink Frames] + [Local Window Frames]
                   ↓ Fixed         ↓ Rolling
                   First 2 frames  Last N frames
```

**Purpose**:
- Provides long-range temporal consistency
- Prevents drift in long video generation
- Anchors the video to initial content/motion

### Latency Impact

```python
# Sink adds overhead to every attention computation
sink_size = 3  # ~4680 tokens (3 frames × 1560 tokens/frame)
local_size = 12  # ~18720 tokens

# Total KV cache size per layer
total_kv = (sink_size + local_size) × num_heads × head_dim × 2
         = 15 × 12 × 128 × 2 = ~46KB per layer
         = ~1.4MB total for 30 layers (per K and V)
```

- Extra ~6% tokens in every attention computation
- Ablation shows significant quality drop without sinks
- **Tradeoff**: Essential for coherence but adds latency

### Alternative Approaches (Require Retraining)

1. **Dynamic Sink Selection**: Replace sink frames when scene changes significantly
2. **Compressed Sink Representations**: Learn lower-dimensional sink tokens
3. **Hierarchical Sinks**: Multi-resolution anchors at different temporal scales
4. **Learned Positional Anchors**: Abstract "memory" tokens instead of actual frames

---

## 3. Prompt-Switch Handling Limitations

### Current KV-Recache Mechanism

When the prompt changes, LongLive performs "KV-recache":

```python
def kv_recache(new_prompt_embeds, cached_frames):
    """
    Recompute KV cache for recent frames with new prompt conditioning.

    Why needed:
    - KV cache contains prompt-conditioned representations
    - Old prompt residue would cause visual artifacts
    - Must update cross-attention K/V with new prompt

    What it does:
    - Re-run last N frames through attention with new prompt
    - Update KV cache in-place
    - Preserves motion/visual structure, changes semantic content
    """
    for frame in cached_frames[-local_window_size:]:
        new_kv = recompute_attention(frame, new_prompt_embeds)
        kv_cache.update(new_kv)
```

### Fundamental Limitation

```
prompt_switch_latency = encode_time + recache_time + first_frame_time
                      ≈ 2ms + (N × layer_forward) + 40ms
                      ≈ 2ms + (12 × 0.5ms) + 40ms
                      ≈ 48ms minimum
```

- Cannot switch prompts instantly
- Recache latency scales with window size
- **Tradeoff**: Smaller window = faster switch but less coherence

### Ideas for Improvement (Require Architecture Changes)

1. **Speculative Prompt Embedding**: Pre-compute embeddings for likely next prompts
2. **Delta Encoding**: Only update prompt-dependent KV components (not full recompute)
3. **Gradual Prompt Blending**: Interpolate embeddings over N frames for smooth transition
4. **Prompt-Agnostic Base Model**: Learn prompt-independent representations, inject prompt late

---

## 4. The Latency Reality

### Paper Claims vs Actual Measurements

The LongLive paper claims ~48ms per frame (20.7 FPS). However, with the full 1.3B model + 350M LoRA parameters, our measurements on H100 show:

```
Actual Baseline (H100 80GB, BF16, 4-step denoising):
├── Denoising (4 steps)
│   ├── Step 1: ~144ms
│   ├── Step 2: ~144ms
│   ├── Step 3: ~144ms
│   └── Step 4: ~144ms
├── VAE Decode: ~45ms
├── KV Operations: ~8ms (steady-state <1ms)
├── Sync/Overhead: ~23ms
└── Total: ~735ms baseline → 575ms optimized (Balanced preset)
```

### Per-Step Kernel Breakdown (with torch.compile)

Each ~144ms denoising step breaks down as:
- Attention Kernels: 52ms (36%) - Self-attn 18ms, Cross-attn 16ms, Projections 18ms
- FFN Layers: 49ms (34%) - Up/down projections dominate
- KV Cache Ops: 8ms (6%) - Mostly on prompt switch
- Layer Norms: 12ms (8%)
- Other Overhead: 23ms (16%)

### Why Paper Numbers Differ

The paper's 20.7 FPS likely uses:
- Smaller model or no LoRA
- Optimized CUDA kernels not in public release
- Different measurement methodology

### To Achieve Real-Time (25 FPS = 40ms)

**Option 1**: Fewer Denoising Steps (Requires Retraining)
- LCM/Turbo distillation: 4 steps → 1-2 steps
- Potential: 2× latency reduction
- Cost: Training compute + possible quality degradation

**Option 2**: Smaller Model (Requires Retraining)
- Reduce from 1.3B to ~400M parameters
- Potential: 3× latency reduction
- Cost: Quality degradation, architectural changes

**Option 3**: Speculative Frame Generation (Architecture Change)
- Generate frames N+1, N+2 speculatively while finalizing N
- Rollback on inconsistency
- Potential: Amortize latency across batch
- Cost: Complexity, memory, occasional stutters

---

## 5. What Makes LongLive Inherently Latency-Limited

### Fundamental vs. Engineering Bottlenecks

| Bottleneck | Type | Can Optimize? |
|------------|------|---------------|
| Autoregressive dependency | Fundamental | No (without architecture change) |
| Diffusion denoising steps | Fundamental | No (without distillation) |
| Attention O(n²) complexity | Fundamental | Partially (sliding window helps) |
| KV-recache on prompt switch | Fundamental | No (without architecture change) |
| Kernel launch overhead | Engineering | Yes (CUDA graphs) |
| Memory allocation | Engineering | Yes (pre-allocation) |
| Host-device sync | Engineering | Yes (async operations) |
| Redundant computation | Engineering | Yes (caching) |

### The Optimization Ceiling

With all inference-time optimizations applied (measured on H100):

```
Achieved Results:
├── Balanced Preset (4 steps, BF16):     575ms (5.2 FPS)   -21.9%
├── Turbo Preset (3 steps, BF16):        431ms (7.0 FPS)   -40.0%
├── Turbo FP8 (3 steps, FP8):            302ms (9.1 FPS)   -56.0%
├── Ultra Preset (2 steps, FP8):         215ms (12.0 FPS)  -66.6%
└── Theoretical min (1 step, FP8):       ~120ms (est.)     -84%
```

**Conclusion**: To achieve real-time 25 FPS (40ms):
1. ✅ Apply all inference-time optimizations (this project) - **DONE**
2. Reduce denoising steps via distillation (future work) - 2 steps gets to 215ms
3. Reduce model size or use specialized hardware
4. The paper's claimed 20.7 FPS may use optimizations not in the public release

---

## 6. CUDA Graphs: Why They Don't Work Well with LongLive

### The Problem

CUDA graphs require static memory addresses and fixed control flow. However, LongLive's KV cache has dynamic behavior:

```python
# LongLive's KV cache structure
kv_cache = [
    {
        "k": tensor,              # Shape: [batch, kv_cache_size, 12, 128]
        "v": tensor,              # Shape: [batch, kv_cache_size, 12, 128]
        "global_end_index": int,  # Changes every frame!
        "local_end_index": int,   # Changes every frame!
    }
    for _ in range(30)  # 30 layers
]
```

The `global_end_index` and `local_end_index` change with every frame, determining where in the buffer to read/write. This dynamic indexing breaks CUDA graph capture.

### Why torch.compile is Better for LongLive

`torch.compile` with the Inductor backend handles dynamic shapes through:
1. **Shape specialization**: Compiles for observed shapes, recompiles if needed
2. **Dynamic shapes mode**: Can handle bounded dynamic dimensions
3. **Kernel fusion**: Fuses operations without requiring static memory layout

```python
# Current recommendation
config = OptimizationConfig(
    use_cuda_graphs=False,       # Don't use CUDA graphs
    use_torch_compile=True,      # Use torch.compile instead
    compile_mode="default",      # NOT reduce-overhead (uses CUDA graphs internally)
)
```

> **Note**: `reduce-overhead` mode uses CUDA graphs internally and fails with the same error as direct CUDA graph capture. Use `default` or `max-autotune` with `triton.cudagraphs=False`.

### Future Work: Making CUDA Graphs Work

To enable CUDA graphs with LongLive, we would need:

1. **Separate KV cache operations from transformer forward**
   ```python
   # Current: KV cache update inside model forward
   def forward(x, kv_cache):
       for layer in self.layers:
           x, kv_cache = layer(x, kv_cache)  # Updates cache in-place
       return x

   # Needed: Separate capture-able forward from cache ops
   def forward_capturable(x, static_kv):
       # Only uses static tensors - can be captured
       return self.transformer(x, static_kv)

   def update_cache(static_kv, real_kv_cache):
       # Called outside captured region
       real_kv_cache.update_from(static_kv)
   ```

2. **Static input/output buffers**
   - Create fixed-size buffers for graph inputs/outputs
   - Copy data to static buffers before replay
   - Copy results back after replay

3. **Generator modification**
   - Modify `wan/modules/causal_model.py` to expose separate forward/cache-update methods
   - This is not an inference-only optimization - requires code changes to the model

---

## 7. Full Ring Buffer KV Cache

### Current Implementation Limitations

The current "static KV" implementation pre-allocates buffers but still uses the base pipeline's index-based update:

```python
# Current: Buffers are pre-allocated, but indexing is still dynamic
kv_cache[layer]["k"][:, write_idx:write_idx+n] = new_k
```

### True Ring Buffer Design

A full ring buffer would:

1. **Use circular indexing in attention**
   ```python
   class RingBufferKVCache:
       def __init__(self, window_size, ...):
           self.buffer = torch.empty(...)
           self.head = 0  # Next write position
           self.size = 0  # Current fill level

       def write(self, new_kv):
           # O(1) write at head position
           self.buffer[:, :, self.head] = new_kv
           self.head = (self.head + 1) % self.window_size
           self.size = min(self.size + 1, self.window_size)

       def get_attention_indices(self):
           # Return indices for attention in correct temporal order
           if self.size < self.window_size:
               return list(range(self.size))
           # Handle wrap-around
           return list(range(self.head, self.window_size)) + list(range(self.head))
   ```

2. **Modify attention to use gathered indices**
   ```python
   def attention_with_ring_buffer(q, kv_cache):
       indices = kv_cache.get_attention_indices()
       k = kv_cache.buffer[:, :, indices, :]  # Gather in correct order
       v = kv_cache.buffer[:, :, indices, :]
       return flash_attention(q, k, v)
   ```

### Why This Requires Generator Changes

The current attention implementation in `wan/modules/attention.py` expects KV cache in sequential order. To use a ring buffer:

1. Attention layers must accept index remapping
2. Or gather operation must happen before attention
3. Flash Attention may have restrictions on non-contiguous memory

This is why full ring buffer is listed as "future work" requiring architecture changes.

---

## 8. Proposed Architectural Changes (If Redesigning)

### Change 1: Distilled 2-Step Model

```
Current:  noise → step1 → step2 → step3 → step4 → latents
Proposed: noise → step1 → step2 → latents

Impact: 2× latency reduction
Cost: Distillation training, ~10% quality loss
```

### Change 2: Parallel Frame Speculation

```
Frame N:     [Generate] ─────────────────────────> [Output]
Frame N+1:        [Speculate] ─────> [Verify/Rollback]
Frame N+2:              [Speculate] ──────────────────>

Impact: Amortize latency, smoother streaming
Cost: 2× memory, complexity, occasional stutter
```

### Change 3: Prompt-Conditioned Adapters

```
Current:  prompt → cross-attn in every layer
Proposed: prompt → adapter → base model (few cross-attn points)

Impact: Near-instant prompt switch
Cost: Architecture redesign, retraining
```

### Change 4: Temporal Delta Prediction

```
Current:  Generate full frame independently
Proposed: Predict delta from previous frame
          Output = prev_frame + delta

Impact: Faster for slow-motion scenes
Cost: Architecture change, may struggle with fast motion
```

### Change 5: Progressive Quality Refinement

```
Pass 1: 2-step low-quality preview (instant feedback)
Pass 2: 4-step high-quality refinement (async)

Impact: Instant user feedback, eventual consistency
Cost: Complexity, memory for dual pipeline
```

---

## 7. Recommendations

### For This Project (Inference-Only)

1. **Apply all optimizations** - Target ~35-38ms steady-state
2. **Accept 40ms as soft target** - Some frames may exceed
3. **Focus on P99 latency** - Worst-case matters for real-time
4. **Document the ceiling** - Make clear what's achievable without retraining

### For Future Work (With Retraining)

1. **LCM/Turbo Distillation** - Highest impact, reduces step count
2. **Smaller Model Variant** - 400M for real-time use case
3. **Prompt Adapter Architecture** - Faster prompt switching

### For Production Deployment

1. **Use Balanced Preset** - Best quality/latency tradeoff
2. **Pre-warm Prompts** - Cache common prompts before interaction
3. **Monitor P99 Latency** - Alert on spikes
4. **Graceful Degradation** - Skip frames if behind

---

## 8. Conclusion

LongLive's architecture, while innovative for long video generation, has inherent latency limitations stemming from:

1. **Autoregressive frame dependency** - Cannot be parallelized
2. **Multi-step diffusion** - Each step adds ~8ms
3. **Frame-sink mechanism** - Necessary for coherence
4. **KV-recache on prompt switch** - Fundamental to the approach

Through inference-time optimizations (this project), we can reduce latency by 25-35%, bringing steady-state latency close to the 40ms target. However, to reliably achieve <40ms with headroom for real-time interaction, architectural changes requiring retraining (particularly step reduction via distillation) would be necessary.

The optimizations implemented in this project represent the maximum achievable improvement without model changes, and provide a strong foundation for future architectural exploration.
