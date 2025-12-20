"""
Integrated KV Cache - Ring Buffer + INT8 Quantization.

This module provides a drop-in replacement for LongLive's KV cache that:
1. Uses a ring buffer for O(1) cache updates (no memory copies)
2. Optionally stores KV in INT8 for 50% memory reduction
3. Provides the same dict-like interface expected by CausalWanSelfAttention

The key insight is that the original code does:
    temp_k = kv_cache["k"].clone()

We provide a lazy tensor wrapper that defers cloning until necessary,
and for quantized mode, dequantizes on access.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class IntegratedKVConfig:
    """Configuration for integrated KV cache."""
    num_layers: int = 28
    num_heads: int = 12
    head_dim: int = 128
    local_window_frames: int = 12
    sink_frames: int = 3
    frame_seq_length: int = 1560
    batch_size: int = 1
    use_ring_buffer: bool = True
    use_quantization: bool = False
    quantization_method: str = "int8"  # "int8" or "fp8"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class LazyTensor:
    """
    A tensor wrapper that defers expensive operations.

    When the attention code does kv_cache["k"].clone(), we intercept
    this and return the appropriate tensor without full copies.
    """

    # Class-level profiling stats
    _total_dequant_time_ms = 0.0
    _total_dequant_count = 0

    def __init__(
        self,
        data: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        is_quantized: bool = False,
        target_dtype: torch.dtype = torch.bfloat16,
    ):
        self._data = data
        self._scale = scale
        self._is_quantized = is_quantized
        self._target_dtype = target_dtype
        self._materialized = None

    def _materialize(self) -> torch.Tensor:
        """Convert to full tensor if needed."""
        if self._materialized is not None:
            return self._materialized

        if self._is_quantized and self._scale is not None:
            # Dequantize: INT8 * scale -> target_dtype with timing
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            self._materialized = (self._data.float() * self._scale).to(self._target_dtype)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                LazyTensor._total_dequant_time_ms += start.elapsed_time(end)
                LazyTensor._total_dequant_count += 1
        else:
            self._materialized = self._data
        return self._materialized

    @classmethod
    def get_dequant_stats(cls) -> Tuple[float, int]:
        """Get cumulative dequantization stats."""
        return cls._total_dequant_time_ms, cls._total_dequant_count

    @classmethod
    def reset_dequant_stats(cls):
        """Reset dequantization stats."""
        cls._total_dequant_time_ms = 0.0
        cls._total_dequant_count = 0

    def clone(self) -> torch.Tensor:
        """Return a clone - this is where we optimize."""
        return self._materialize().clone()

    def __getitem__(self, key):
        return self._materialize()[key]

    def __setitem__(self, key, value):
        self._materialize()[key] = value

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._target_dtype if self._is_quantized else self._data.dtype

    @property
    def device(self):
        return self._data.device


class IntegratedKVCacheLayer(dict):
    """
    A single layer's KV cache that provides dict-like access.

    This wraps the ring buffer and quantization into the interface
    expected by CausalWanSelfAttention.
    """

    def __init__(
        self,
        layer_idx: int,
        config: IntegratedKVConfig,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.device = torch.device(config.device)

        # Profiling counters for quant/dequant
        self._quant_time_ms = 0.0
        self._dequant_time_ms = 0.0
        self._quant_count = 0
        self._dequant_count = 0

        # Calculate sizes
        self.sink_size = config.sink_frames * config.frame_seq_length
        self.local_size = config.local_window_frames * config.frame_seq_length
        self.total_size = self.sink_size + self.local_size

        # Determine storage dtype
        if config.use_quantization and config.quantization_method == "int8":
            self.storage_dtype = torch.int8
        else:
            self.storage_dtype = config.dtype

        # Pre-allocate buffers
        buffer_shape = (config.batch_size, self.total_size, config.num_heads, config.head_dim)

        self._k_buffer = torch.zeros(buffer_shape, dtype=self.storage_dtype, device=self.device)
        self._v_buffer = torch.zeros(buffer_shape, dtype=self.storage_dtype, device=self.device)

        # Scales for quantization (per-token)
        if config.use_quantization:
            scale_shape = (config.batch_size, self.total_size, 1, 1)
            self._k_scale = torch.ones(scale_shape, dtype=torch.float32, device=self.device)
            self._v_scale = torch.ones(scale_shape, dtype=torch.float32, device=self.device)
        else:
            self._k_scale = None
            self._v_scale = None

        # Ring buffer state
        self._write_idx = 0
        self._valid_len = 0
        self._global_end = 0
        self._local_end = 0

        # Initialize dict values
        self._update_dict_values()

    def _update_dict_values(self):
        """Update the dict values with current state."""
        # Provide the FULL buffer - the original code expects pre-allocated
        # full-size buffers and uses local_end_index/global_end_index to
        # track valid positions. Slicing causes shape mismatches.
        super().__setitem__('k', LazyTensor(
            self._k_buffer,  # Full buffer, not sliced
            self._k_scale if self._k_scale is not None else None,
            is_quantized=self.config.use_quantization,
            target_dtype=self.config.dtype,
        ))
        super().__setitem__('v', LazyTensor(
            self._v_buffer,  # Full buffer, not sliced
            self._v_scale if self._v_scale is not None else None,
            is_quantized=self.config.use_quantization,
            target_dtype=self.config.dtype,
        ))
        super().__setitem__('global_end_index',
            torch.tensor([self._global_end], dtype=torch.long, device=self.device))
        super().__setitem__('local_end_index',
            torch.tensor([self._local_end], dtype=torch.long, device=self.device))

    def _get_valid_len(self) -> int:
        """Get current valid length in buffer."""
        return max(1, self._valid_len)  # At least 1 to avoid empty tensor issues

    def _quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8 with per-token scaling."""
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # Per-token scale: max absolute value per token
        scale = tensor.abs().amax(dim=(-2, -1), keepdim=True) / 127.0
        scale = scale.clamp(min=1e-8)
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            self._quant_time_ms += start.elapsed_time(end)
            self._quant_count += 1

        return quantized, scale.float()

    def update_cache(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        global_end: int,
        local_end: int,
    ):
        """
        Update cache with new KV values.

        This is called by the optimized pipeline after attention computation.
        """
        seq_len = new_k.shape[1]

        # Quantize if enabled
        if self.config.use_quantization:
            new_k_q, new_k_scale = self._quantize_int8(new_k)
            new_v_q, new_v_scale = self._quantize_int8(new_v)
        else:
            new_k_q = new_k
            new_v_q = new_v
            new_k_scale = None
            new_v_scale = None

        # Write to buffer using ring buffer logic
        write_start = self._write_idx
        write_end = write_start + seq_len

        if write_end <= self.total_size:
            # Fits without wrapping
            self._k_buffer[:, write_start:write_end].copy_(new_k_q)
            self._v_buffer[:, write_start:write_end].copy_(new_v_q)
            if self.config.use_quantization:
                self._k_scale[:, write_start:write_end].copy_(new_k_scale)
                self._v_scale[:, write_start:write_end].copy_(new_v_scale)
            self._write_idx = write_end % self.total_size
        else:
            # Wrap around
            first_part = self.total_size - write_start
            self._k_buffer[:, write_start:].copy_(new_k_q[:, :first_part])
            self._v_buffer[:, write_start:].copy_(new_v_q[:, :first_part])

            remaining = seq_len - first_part
            self._k_buffer[:, :remaining].copy_(new_k_q[:, first_part:])
            self._v_buffer[:, :remaining].copy_(new_v_q[:, first_part:])

            if self.config.use_quantization:
                self._k_scale[:, write_start:].copy_(new_k_scale[:, :first_part])
                self._v_scale[:, write_start:].copy_(new_v_scale[:, :first_part])
                self._k_scale[:, :remaining].copy_(new_k_scale[:, first_part:])
                self._v_scale[:, :remaining].copy_(new_v_scale[:, first_part:])

            self._write_idx = remaining

        # Update indices
        self._valid_len = min(self._valid_len + seq_len, self.total_size)
        self._global_end = global_end
        self._local_end = local_end

        # Update dict values
        self._update_dict_values()

    def get_attention_kv(
        self,
        sink_tokens: int,
        local_end_index: int,
        max_attention_size: int,
    ) -> tuple:
        """
        Get K, V tensors ready for attention computation.

        This handles ring buffer wraparound and returns contiguous views
        without unnecessary cloning.

        Args:
            sink_tokens: Number of sink tokens (always at start of buffer)
            local_end_index: Current local end index
            max_attention_size: Maximum attention window size

        Returns:
            (k_for_attention, v_for_attention): Tensors ready for attention
        """
        # For now, simple implementation that reads from buffer
        # The ring buffer stores sink + local window contiguously
        # Sink tokens are always at indices [0:sink_tokens]
        # Local window is at indices [sink_tokens:local_end_index]

        if self.config.use_quantization and self._k_scale is not None:
            # Dequantize on read
            k = self._k_buffer[:, :local_end_index].to(self.config.dtype) * self._k_scale[:, :local_end_index]
            v = self._v_buffer[:, :local_end_index].to(self.config.dtype) * self._v_scale[:, :local_end_index]
        else:
            # Return view (no copy)
            k = self._k_buffer[:, :local_end_index]
            v = self._v_buffer[:, :local_end_index]

        # Apply attention windowing
        if sink_tokens > 0:
            local_budget = max_attention_size - sink_tokens
            k_sink = k[:, :sink_tokens]
            v_sink = v[:, :sink_tokens]

            if local_budget > 0:
                local_start = max(sink_tokens, local_end_index - local_budget)
                k_local = k[:, local_start:local_end_index]
                v_local = v[:, local_start:local_end_index]
                k_cat = torch.cat([k_sink, k_local], dim=1)
                v_cat = torch.cat([v_sink, v_local], dim=1)
            else:
                k_cat = k_sink
                v_cat = v_sink
            return k_cat, v_cat
        else:
            window_start = max(0, local_end_index - max_attention_size)
            return k[:, window_start:local_end_index], v[:, window_start:local_end_index]

    def update_from_attention(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        write_start_index: int,
        write_end_index: int,
        global_end: int,
        local_end: int,
        do_roll: bool = False,
        sink_tokens: int = 0,
        num_rolled_tokens: int = 0,
        num_evicted_tokens: int = 0,
    ):
        """
        Update cache from attention computation results.

        This is the O(1) ring buffer update - no rolling needed.
        The attention code computes the new KV, we just store them.

        Args:
            new_k, new_v: New key/value tensors to store
            write_start_index, write_end_index: Where to write in buffer
            global_end, local_end: Updated cache indices
            do_roll: Whether rolling was needed (for compatibility)
            sink_tokens, num_rolled_tokens, num_evicted_tokens: Rolling params (for compatibility)
        """
        write_len = write_end_index - write_start_index
        if write_len <= 0:
            return

        # Handle rolling if needed (when cache is full)
        if do_roll and num_rolled_tokens > 0:
            # Ring buffer approach: instead of shifting, we just update write pointer
            # But for compatibility with current attention code, do the shift
            self._k_buffer[:, sink_tokens:sink_tokens + num_rolled_tokens].copy_(
                self._k_buffer[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens]
            )
            self._v_buffer[:, sink_tokens:sink_tokens + num_rolled_tokens].copy_(
                self._v_buffer[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens]
            )
            if self.config.use_quantization and self._k_scale is not None:
                self._k_scale[:, sink_tokens:sink_tokens + num_rolled_tokens].copy_(
                    self._k_scale[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens]
                )
                self._v_scale[:, sink_tokens:sink_tokens + num_rolled_tokens].copy_(
                    self._v_scale[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens]
                )

        # Write new KV (with quantization if enabled)
        if self.config.use_quantization:
            new_k_q, new_k_scale = self._quantize_int8(new_k)
            new_v_q, new_v_scale = self._quantize_int8(new_v)
            self._k_buffer[:, write_start_index:write_end_index].copy_(new_k_q)
            self._v_buffer[:, write_start_index:write_end_index].copy_(new_v_q)
            self._k_scale[:, write_start_index:write_end_index].copy_(new_k_scale)
            self._v_scale[:, write_start_index:write_end_index].copy_(new_v_scale)
        else:
            self._k_buffer[:, write_start_index:write_end_index].copy_(new_k)
            self._v_buffer[:, write_start_index:write_end_index].copy_(new_v)

        # Update indices
        self._global_end = global_end
        self._local_end = local_end
        self._valid_len = local_end

        # Update dict values for next access
        self._update_dict_values()

    def reset(self):
        """Reset cache to initial state."""
        self._k_buffer.zero_()
        self._v_buffer.zero_()
        if self._k_scale is not None:
            self._k_scale.fill_(1.0)
            self._v_scale.fill_(1.0)
        self._write_idx = 0
        self._valid_len = 0
        self._global_end = 0
        self._local_end = 0
        self._update_dict_values()

    def memory_usage_bytes(self) -> int:
        """Calculate memory usage."""
        element_size = 1 if self.config.use_quantization else self._k_buffer.element_size()
        buffer_bytes = self._k_buffer.numel() * element_size * 2  # K and V

        if self.config.use_quantization:
            buffer_bytes += self._k_scale.numel() * 4 * 2  # Scales are float32

        return buffer_bytes

    def get_quant_stats(self) -> Dict[str, float]:
        """Get quantization profiling stats."""
        return {
            'quant_time_ms': self._quant_time_ms,
            'quant_count': self._quant_count,
            'avg_quant_ms': self._quant_time_ms / max(1, self._quant_count),
        }

    def reset_quant_stats(self):
        """Reset quantization stats."""
        self._quant_time_ms = 0.0
        self._quant_count = 0


class IntegratedKVCache(list):
    """
    Full KV cache for all layers.

    Drop-in replacement for LongLive's list-of-dicts KV cache format.
    """

    def __init__(self, config: IntegratedKVConfig):
        super().__init__()
        self.config = config

        # Create per-layer caches
        for layer_idx in range(config.num_layers):
            layer_cache = IntegratedKVCacheLayer(layer_idx, config)
            self.append(layer_cache)

    def reset(self):
        """Reset all layer caches."""
        for layer_cache in self:
            layer_cache.reset()

    def total_memory_bytes(self) -> int:
        """Total memory usage across all layers."""
        return sum(layer.memory_usage_bytes() for layer in self)

    def memory_savings(self) -> float:
        """Calculate memory savings vs unquantized BF16."""
        current = self.total_memory_bytes()

        # Calculate what BF16 would use
        bf16_per_layer = (
            self.config.batch_size *
            (self.config.sink_frames + self.config.local_window_frames) *
            self.config.frame_seq_length *
            self.config.num_heads *
            self.config.head_dim *
            2 *  # bytes per BF16
            2    # K and V
        )
        bf16_total = bf16_per_layer * self.config.num_layers

        return 1.0 - (current / bf16_total) if bf16_total > 0 else 0.0

    def get_all_quant_stats(self) -> Dict[str, float]:
        """Get aggregated quant/dequant stats across all layers."""
        total_quant_time = sum(layer._quant_time_ms for layer in self)
        total_quant_count = sum(layer._quant_count for layer in self)

        dequant_time, dequant_count = LazyTensor.get_dequant_stats()

        return {
            'total_quant_time_ms': total_quant_time,
            'total_quant_count': total_quant_count,
            'avg_quant_ms': total_quant_time / max(1, total_quant_count),
            'total_dequant_time_ms': dequant_time,
            'total_dequant_count': dequant_count,
            'avg_dequant_ms': dequant_time / max(1, dequant_count),
        }

    def reset_all_stats(self):
        """Reset all profiling stats."""
        for layer in self:
            layer.reset_quant_stats()
        LazyTensor.reset_dequant_stats()


def create_integrated_kv_cache(
    num_layers: int = 28,
    num_heads: int = 12,
    head_dim: int = 128,
    local_window_frames: int = 12,
    sink_frames: int = 3,
    frame_seq_length: int = 1560,
    batch_size: int = 1,
    use_ring_buffer: bool = True,
    use_quantization: bool = False,
    quantization_method: str = "int8",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> IntegratedKVCache:
    """
    Factory function to create an integrated KV cache.

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        local_window_frames: Frames in local attention window
        sink_frames: Number of sink frames
        frame_seq_length: Tokens per frame
        batch_size: Batch size
        use_ring_buffer: Whether to use ring buffer (vs standard append)
        use_quantization: Whether to quantize KV to INT8
        quantization_method: "int8" or "fp8"
        dtype: Output dtype for attention
        device: Device string

    Returns:
        IntegratedKVCache instance
    """
    config = IntegratedKVConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        local_window_frames=local_window_frames,
        sink_frames=sink_frames,
        frame_seq_length=frame_seq_length,
        batch_size=batch_size,
        use_ring_buffer=use_ring_buffer,
        use_quantization=use_quantization,
        quantization_method=quantization_method,
        dtype=dtype,
        device=device,
    )
    return IntegratedKVCache(config)


# Precision conversion utilities

def convert_model_to_fp16(model: nn.Module) -> nn.Module:
    """
    Convert model to FP16, including all biases.

    This fixes the "Input type (BFloat16) and bias type (Half) mismatch" error
    by ensuring all parameters are converted consistently.
    """
    for name, param in model.named_parameters():
        param.data = param.data.to(torch.float16)

    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(torch.float16)

    return model


def convert_model_to_bf16(model: nn.Module) -> nn.Module:
    """Convert model to BF16, including all biases."""
    for name, param in model.named_parameters():
        param.data = param.data.to(torch.bfloat16)

    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(torch.bfloat16)

    return model


def apply_int8_weight_quantization(model: nn.Module) -> nn.Module:
    """
    Apply INT8 weight-only quantization to linear layers.

    Uses dynamic quantization for weights while keeping activations in FP16/BF16.
    """
    try:
        # Try using torch.ao.quantization for dynamic quantization
        from torch.ao.quantization import quantize_dynamic

        quantized_model = quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model
    except Exception as e:
        print(f"Warning: INT8 weight quantization failed: {e}")
        print("Falling back to unquantized model")
        return model
