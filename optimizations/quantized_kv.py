"""
Quantized KV Cache for LongLive.

Implements INT8 and FP8 quantized KV cache to reduce memory bandwidth
and potentially improve throughput. Quantization happens on cache
store, dequantization on cache load.

Key features:
1. INT8 quantization with per-token scaling
2. FP8 quantization (E4M3 format)
3. Ring buffer support (like StaticKVCache)
4. Fused quantize/dequantize kernels
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """Configuration for KV cache quantization."""
    method: str = "int8"  # "int8" or "fp8"
    per_token_scaling: bool = True  # Scale per token (better accuracy)
    symmetric: bool = True  # Symmetric quantization
    clamp_range: float = 6.0  # Clamp outliers for stability


def quantize_int8(
    tensor: torch.Tensor,
    per_token: bool = True,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8 with scaling factors.

    Args:
        tensor: Input tensor to quantize [batch, seq, heads, dim]
        per_token: If True, compute scale per token
        symmetric: If True, use symmetric quantization

    Returns:
        (quantized_tensor, scales) tuple
    """
    if per_token:
        # Compute scale per token (reduce over heads and dim)
        # Shape: [batch, seq, heads, dim] -> [batch, seq, 1, 1]
        if symmetric:
            scales = tensor.abs().amax(dim=(-2, -1), keepdim=True) / 127.0
        else:
            min_val = tensor.amin(dim=(-2, -1), keepdim=True)
            max_val = tensor.amax(dim=(-2, -1), keepdim=True)
            scales = (max_val - min_val) / 255.0
    else:
        # Global scale
        if symmetric:
            scales = tensor.abs().max() / 127.0
        else:
            scales = (tensor.max() - tensor.min()) / 255.0

    # Avoid division by zero
    scales = scales.clamp(min=1e-8)

    # Quantize
    if symmetric:
        quantized = (tensor / scales).round().clamp(-128, 127).to(torch.int8)
    else:
        min_val = tensor.amin(dim=(-2, -1), keepdim=True) if per_token else tensor.min()
        quantized = ((tensor - min_val) / scales).round().clamp(0, 255).to(torch.uint8)

    return quantized, scales


def dequantize_int8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    symmetric: bool = True,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize INT8 tensor back to floating point.

    Args:
        quantized: INT8 quantized tensor
        scales: Scaling factors
        symmetric: If True, symmetric dequantization
        target_dtype: Output dtype

    Returns:
        Dequantized tensor
    """
    return (quantized.to(target_dtype) * scales)


def quantize_fp8(
    tensor: torch.Tensor,
    format: str = "e4m3",  # E4M3 or E5M2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8 format.

    Note: Requires PyTorch 2.1+ with FP8 support.

    Args:
        tensor: Input tensor
        format: FP8 format ("e4m3" for better precision, "e5m2" for larger range)

    Returns:
        (quantized_tensor, scales) tuple
    """
    # Check for FP8 support
    if not hasattr(torch, 'float8_e4m3fn'):
        # Fallback to bfloat16 if FP8 not available
        return tensor.to(torch.bfloat16), torch.ones(1, device=tensor.device)

    # Compute per-tensor scale for FP8
    amax = tensor.abs().max()

    if format == "e4m3":
        fp8_max = 448.0  # Max value for E4M3
        dtype = torch.float8_e4m3fn
    else:  # e5m2
        fp8_max = 57344.0  # Max value for E5M2
        dtype = torch.float8_e5m2

    scale = amax / fp8_max
    scale = scale.clamp(min=1e-12)

    # Scale and convert
    scaled_tensor = tensor / scale
    quantized = scaled_tensor.to(dtype)

    return quantized, scale


def dequantize_fp8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 tensor back to floating point."""
    return quantized.to(target_dtype) * scale


class QuantizedKVCache(nn.Module):
    """
    Quantized KV cache with ring buffer support.

    Stores KV values in INT8 or FP8 format to reduce memory bandwidth.
    Provides same interface as StaticKVCache but with quantization.

    Memory savings:
    - INT8: 4x reduction vs FP32, 2x vs FP16/BF16
    - FP8: 4x reduction vs FP32, 2x vs FP16/BF16

    Usage:
        cache = QuantizedKVCache(num_layers=30, ...)

        # Store (quantizes automatically)
        cache.update(layer_idx, new_k, new_v)

        # Load (dequantizes automatically)
        k, v = cache.get_full_kv(layer_idx)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        local_window_frames: int,
        sink_frames: int,
        frame_seq_length: int,
        batch_size: int = 1,
        quantization: str = "int8",
        output_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        """
        Initialize quantized KV cache.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            local_window_frames: Frames in local attention window
            sink_frames: Number of frame-sink frames
            frame_seq_length: Tokens per frame
            batch_size: Batch size
            quantization: "int8" or "fp8"
            output_dtype: Dtype for dequantized output
            device: Device for cache
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_window_frames = local_window_frames
        self.sink_frames = sink_frames
        self.frame_seq_length = frame_seq_length
        self.batch_size = batch_size
        self.quantization = quantization
        self.output_dtype = output_dtype
        self.device = device or torch.device('cuda')

        # Calculate sizes
        self.sink_size = sink_frames * frame_seq_length
        self.local_size = local_window_frames * frame_seq_length

        # Determine quantized dtype
        if quantization == "int8":
            self.quant_dtype = torch.int8
        elif quantization == "fp8":
            if hasattr(torch, 'float8_e4m3fn'):
                self.quant_dtype = torch.float8_e4m3fn
            else:
                # Fallback
                self.quant_dtype = torch.int8
                self.quantization = "int8"
        else:
            raise ValueError(f"Unknown quantization method: {quantization}")

        # Allocate quantized buffers
        self._allocate_caches()

        # Ring buffer state
        self.register_buffer(
            'write_indices',
            torch.zeros(num_layers, dtype=torch.long, device=self.device)
        )
        self.register_buffer(
            'valid_lengths',
            torch.zeros(num_layers, dtype=torch.long, device=self.device)
        )
        self.register_buffer(
            'sink_filled',
            torch.zeros(num_layers, dtype=torch.bool, device=self.device)
        )
        self.register_buffer(
            'total_frames',
            torch.tensor(0, dtype=torch.long, device=self.device)
        )

    def _allocate_caches(self):
        """Allocate quantized cache buffers."""
        for layer_idx in range(self.num_layers):
            # Sink cache (quantized)
            self.register_buffer(
                f'sink_k_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.sink_size, self.num_heads, self.head_dim,
                    dtype=self.quant_dtype, device=self.device
                )
            )
            self.register_buffer(
                f'sink_v_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.sink_size, self.num_heads, self.head_dim,
                    dtype=self.quant_dtype, device=self.device
                )
            )

            # Sink scales (per-token for INT8)
            self.register_buffer(
                f'sink_k_scale_{layer_idx}',
                torch.ones(
                    self.batch_size, self.sink_size, 1, 1,
                    dtype=torch.float32, device=self.device
                )
            )
            self.register_buffer(
                f'sink_v_scale_{layer_idx}',
                torch.ones(
                    self.batch_size, self.sink_size, 1, 1,
                    dtype=torch.float32, device=self.device
                )
            )

            # Local cache (quantized, ring buffer)
            self.register_buffer(
                f'local_k_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.local_size, self.num_heads, self.head_dim,
                    dtype=self.quant_dtype, device=self.device
                )
            )
            self.register_buffer(
                f'local_v_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.local_size, self.num_heads, self.head_dim,
                    dtype=self.quant_dtype, device=self.device
                )
            )

            # Local scales
            self.register_buffer(
                f'local_k_scale_{layer_idx}',
                torch.ones(
                    self.batch_size, self.local_size, 1, 1,
                    dtype=torch.float32, device=self.device
                )
            )
            self.register_buffer(
                f'local_v_scale_{layer_idx}',
                torch.ones(
                    self.batch_size, self.local_size, 1, 1,
                    dtype=torch.float32, device=self.device
                )
            )

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor using configured method."""
        if self.quantization == "int8":
            return quantize_int8(tensor, per_token=True, symmetric=True)
        else:
            return quantize_fp8(tensor, format="e4m3")

    def _dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize a tensor using configured method."""
        if self.quantization == "int8":
            return dequantize_int8(quantized, scale, symmetric=True, target_dtype=self.output_dtype)
        else:
            return dequantize_fp8(quantized, scale, target_dtype=self.output_dtype)

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        is_sink: bool = False
    ):
        """
        Update cache with new KV values (quantizes automatically).

        Args:
            layer_idx: Transformer layer index
            new_k: New key tensor [batch, seq, heads, dim]
            new_v: New value tensor [batch, seq, heads, dim]
            is_sink: If True, write to sink cache
        """
        # Quantize new values
        new_k_q, new_k_scale = self._quantize(new_k)
        new_v_q, new_v_scale = self._quantize(new_v)

        seq_len = new_k.shape[1]

        if is_sink:
            sink_k = getattr(self, f'sink_k_{layer_idx}')
            sink_v = getattr(self, f'sink_v_{layer_idx}')
            sink_k_scale = getattr(self, f'sink_k_scale_{layer_idx}')
            sink_v_scale = getattr(self, f'sink_v_scale_{layer_idx}')

            if not self.sink_filled[layer_idx]:
                current_len = self.valid_lengths[layer_idx].item()
                end_idx = min(current_len + seq_len, self.sink_size)
                write_len = end_idx - current_len

                sink_k[:, current_len:end_idx].copy_(new_k_q[:, :write_len])
                sink_v[:, current_len:end_idx].copy_(new_v_q[:, :write_len])
                sink_k_scale[:, current_len:end_idx].copy_(new_k_scale[:, :write_len])
                sink_v_scale[:, current_len:end_idx].copy_(new_v_scale[:, :write_len])

                if end_idx >= self.sink_size:
                    self.sink_filled[layer_idx] = True
                    remaining = seq_len - write_len
                    if remaining > 0:
                        self._update_local_ring(
                            layer_idx,
                            new_k_q[:, write_len:],
                            new_v_q[:, write_len:],
                            new_k_scale[:, write_len:],
                            new_v_scale[:, write_len:]
                        )
                else:
                    self.valid_lengths[layer_idx] = end_idx
        else:
            self._update_local_ring(
                layer_idx,
                new_k_q, new_v_q,
                new_k_scale, new_v_scale
            )

    def _update_local_ring(
        self,
        layer_idx: int,
        new_k_q: torch.Tensor,
        new_v_q: torch.Tensor,
        new_k_scale: torch.Tensor,
        new_v_scale: torch.Tensor
    ):
        """Update local cache ring buffer with quantized values."""
        local_k = getattr(self, f'local_k_{layer_idx}')
        local_v = getattr(self, f'local_v_{layer_idx}')
        local_k_scale = getattr(self, f'local_k_scale_{layer_idx}')
        local_v_scale = getattr(self, f'local_v_scale_{layer_idx}')

        seq_len = new_k_q.shape[1]
        write_idx = self.write_indices[layer_idx].item()
        space_before_wrap = self.local_size - write_idx

        if seq_len <= space_before_wrap:
            local_k[:, write_idx:write_idx + seq_len].copy_(new_k_q)
            local_v[:, write_idx:write_idx + seq_len].copy_(new_v_q)
            local_k_scale[:, write_idx:write_idx + seq_len].copy_(new_k_scale)
            local_v_scale[:, write_idx:write_idx + seq_len].copy_(new_v_scale)
            new_write_idx = (write_idx + seq_len) % self.local_size
        else:
            local_k[:, write_idx:].copy_(new_k_q[:, :space_before_wrap])
            local_v[:, write_idx:].copy_(new_v_q[:, :space_before_wrap])
            local_k_scale[:, write_idx:].copy_(new_k_scale[:, :space_before_wrap])
            local_v_scale[:, write_idx:].copy_(new_v_scale[:, :space_before_wrap])

            remaining = seq_len - space_before_wrap
            local_k[:, :remaining].copy_(new_k_q[:, space_before_wrap:])
            local_v[:, :remaining].copy_(new_v_q[:, space_before_wrap:])
            local_k_scale[:, :remaining].copy_(new_k_scale[:, space_before_wrap:])
            local_v_scale[:, :remaining].copy_(new_v_scale[:, space_before_wrap:])

            new_write_idx = remaining

        self.write_indices[layer_idx] = new_write_idx
        current_valid = self.valid_lengths[layer_idx].item()
        new_valid = min(current_valid + seq_len, self.local_size)
        self.valid_lengths[layer_idx] = new_valid

    def get_full_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full dequantized KV cache for attention.

        Returns properly ordered K and V tensors.
        """
        # Get sink cache
        sink_k = getattr(self, f'sink_k_{layer_idx}')
        sink_v = getattr(self, f'sink_v_{layer_idx}')
        sink_k_scale = getattr(self, f'sink_k_scale_{layer_idx}')
        sink_v_scale = getattr(self, f'sink_v_scale_{layer_idx}')

        # Dequantize sink
        sink_k_dq = self._dequantize(sink_k, sink_k_scale)
        sink_v_dq = self._dequantize(sink_v, sink_v_scale)

        # Get local cache
        local_k = getattr(self, f'local_k_{layer_idx}')
        local_v = getattr(self, f'local_v_{layer_idx}')
        local_k_scale = getattr(self, f'local_k_scale_{layer_idx}')
        local_v_scale = getattr(self, f'local_v_scale_{layer_idx}')

        write_idx = self.write_indices[layer_idx].item()
        valid_len = self.valid_lengths[layer_idx].item()

        if valid_len == 0:
            return sink_k_dq, sink_v_dq

        if valid_len < self.local_size:
            local_k_dq = self._dequantize(
                local_k[:, :valid_len],
                local_k_scale[:, :valid_len]
            )
            local_v_dq = self._dequantize(
                local_v[:, :valid_len],
                local_v_scale[:, :valid_len]
            )
            full_k = torch.cat([sink_k_dq, local_k_dq], dim=1)
            full_v = torch.cat([sink_v_dq, local_v_dq], dim=1)
        else:
            # Ring buffer wrapped - reorder
            local_k_reordered = torch.cat([
                local_k[:, write_idx:],
                local_k[:, :write_idx]
            ], dim=1)
            local_v_reordered = torch.cat([
                local_v[:, write_idx:],
                local_v[:, :write_idx]
            ], dim=1)
            local_k_scale_reordered = torch.cat([
                local_k_scale[:, write_idx:],
                local_k_scale[:, :write_idx]
            ], dim=1)
            local_v_scale_reordered = torch.cat([
                local_v_scale[:, write_idx:],
                local_v_scale[:, :write_idx]
            ], dim=1)

            local_k_dq = self._dequantize(local_k_reordered, local_k_scale_reordered)
            local_v_dq = self._dequantize(local_v_reordered, local_v_scale_reordered)

            full_k = torch.cat([sink_k_dq, local_k_dq], dim=1)
            full_v = torch.cat([sink_v_dq, local_v_dq], dim=1)

        return full_k, full_v

    def reset(self):
        """Reset cache to initial state."""
        self.write_indices.zero_()
        self.valid_lengths.zero_()
        self.sink_filled.zero_()
        self.total_frames.zero_()

    def memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        quant_element_size = 1 if self.quantization == "int8" else 1  # FP8 is also 1 byte
        scale_element_size = 4  # float32

        # Per layer: 2 * (sink + local) * (quant + scale)
        per_layer_quant = 2 * (self.sink_size + self.local_size) * self.num_heads * self.head_dim * quant_element_size
        per_layer_scale = 2 * (self.sink_size + self.local_size) * scale_element_size

        total_bytes = self.num_layers * (per_layer_quant + per_layer_scale) * self.batch_size

        # Compare to unquantized (bfloat16)
        unquant_bytes = self.num_layers * 2 * (self.sink_size + self.local_size) * self.num_heads * self.head_dim * 2 * self.batch_size

        return {
            'quantized_mb': total_bytes / (1024 ** 2),
            'unquantized_mb': unquant_bytes / (1024 ** 2),
            'savings_ratio': unquant_bytes / total_bytes if total_bytes > 0 else 0,
        }

    def __repr__(self) -> str:
        usage = self.memory_usage()
        return (
            f"QuantizedKVCache(\n"
            f"  quantization={self.quantization},\n"
            f"  layers={self.num_layers},\n"
            f"  sink_size={self.sink_size},\n"
            f"  local_size={self.local_size},\n"
            f"  memory={usage['quantized_mb']:.2f}MB (vs {usage['unquantized_mb']:.2f}MB unquantized),\n"
            f"  savings={usage['savings_ratio']:.1f}x\n"
            f")"
        )
