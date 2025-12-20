"""
KV Cache Wrapper for Transparent Quantization.

This module provides a wrapper that allows the existing LongLive pipeline
to use quantized KV caches without modifying the generator code.

The wrapper intercepts KV cache reads/writes and automatically handles
quantization/dequantization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .quantized_kv import quantize_int8, dequantize_int8


class QuantizedKVCacheDict(dict):
    """
    A dict-like object that wraps KV cache entries with quantization.

    When values are stored, they are quantized to INT8.
    When values are read, they are dequantized to the target dtype.

    This allows transparent integration with existing code that expects
    the standard KV cache format: {"k": tensor, "v": tensor, ...}
    """

    def __init__(
        self,
        initial_dict: Dict[str, Any] = None,
        quantize_kv: bool = True,
        target_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.quantize_kv = quantize_kv
        self.target_dtype = target_dtype

        # Storage for quantized values and scales
        self._quantized_k = None
        self._quantized_v = None
        self._k_scale = None
        self._v_scale = None

        # Initialize from dict if provided
        if initial_dict:
            for key, value in initial_dict.items():
                self[key] = value

    def __setitem__(self, key: str, value: Any):
        """Store value, quantizing K and V if enabled."""
        if self.quantize_kv and key in ('k', 'K'):
            # Quantize key tensor
            self._quantized_k, self._k_scale = quantize_int8(
                value, per_token=True, symmetric=True
            )
            # Store original shape for reconstruction
            self._k_shape = value.shape
            self._k_dtype = value.dtype
            self._k_device = value.device
            # Store placeholder
            super().__setitem__(key, None)
        elif self.quantize_kv and key in ('v', 'V'):
            # Quantize value tensor
            self._quantized_v, self._v_scale = quantize_int8(
                value, per_token=True, symmetric=True
            )
            self._v_shape = value.shape
            self._v_dtype = value.dtype
            self._v_device = value.device
            super().__setitem__(key, None)
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        """Get value, dequantizing K and V if needed."""
        if self.quantize_kv and key in ('k', 'K') and self._quantized_k is not None:
            # Dequantize on read
            return dequantize_int8(
                self._quantized_k,
                self._k_scale,
                symmetric=True,
                target_dtype=self.target_dtype,
            )
        elif self.quantize_kv and key in ('v', 'V') and self._quantized_v is not None:
            return dequantize_int8(
                self._quantized_v,
                self._v_scale,
                symmetric=True,
                target_dtype=self.target_dtype,
            )
        else:
            return super().__getitem__(key)

    def update_slice(self, key: str, indices: slice, value: torch.Tensor):
        """Update a slice of K or V cache (for in-place updates)."""
        if self.quantize_kv and key in ('k', 'K'):
            # Quantize the new values
            new_q, new_scale = quantize_int8(value, per_token=True, symmetric=True)
            # Update quantized storage
            self._quantized_k[:, indices] = new_q
            self._k_scale[:, indices] = new_scale
        elif self.quantize_kv and key in ('v', 'V'):
            new_q, new_scale = quantize_int8(value, per_token=True, symmetric=True)
            self._quantized_v[:, indices] = new_q
            self._v_scale[:, indices] = new_scale
        else:
            super().__getitem__(key)[:, indices] = value

    def memory_usage_bytes(self) -> int:
        """Get memory usage of quantized cache."""
        total = 0
        if self._quantized_k is not None:
            total += self._quantized_k.numel() * 1  # INT8 = 1 byte
            total += self._k_scale.numel() * 4  # float32 scale
        if self._quantized_v is not None:
            total += self._quantized_v.numel() * 1
            total += self._v_scale.numel() * 4
        return total


class QuantizedKVCacheList(list):
    """
    A list wrapper that wraps KV cache layer entries with quantization.

    Each element in the list is a dict with {"k": ..., "v": ..., ...}.
    This wrapper ensures each dict uses QuantizedKVCacheDict for
    transparent quantization.
    """

    def __init__(
        self,
        initial_list: List[Dict] = None,
        quantize_kv: bool = True,
        target_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.quantize_kv = quantize_kv
        self.target_dtype = target_dtype

        if initial_list:
            for item in initial_list:
                self.append(item)

    def append(self, item: Dict):
        """Append a new layer's KV cache, wrapping with quantization."""
        if isinstance(item, QuantizedKVCacheDict):
            super().append(item)
        else:
            wrapped = QuantizedKVCacheDict(
                item,
                quantize_kv=self.quantize_kv,
                target_dtype=self.target_dtype,
            )
            super().append(wrapped)

    def __setitem__(self, index: int, item: Dict):
        """Set a layer's KV cache, wrapping with quantization."""
        if isinstance(item, QuantizedKVCacheDict):
            super().__setitem__(index, item)
        else:
            wrapped = QuantizedKVCacheDict(
                item,
                quantize_kv=self.quantize_kv,
                target_dtype=self.target_dtype,
            )
            super().__setitem__(index, wrapped)

    def total_memory_bytes(self) -> int:
        """Get total memory usage across all layers."""
        total = 0
        for item in self:
            if isinstance(item, QuantizedKVCacheDict):
                total += item.memory_usage_bytes()
        return total


def wrap_kv_cache_with_quantization(
    kv_cache: List[Dict],
    quantize: bool = True,
    target_dtype: torch.dtype = torch.bfloat16,
) -> QuantizedKVCacheList:
    """
    Wrap an existing KV cache list with quantization.

    Args:
        kv_cache: List of dicts, each with {"k": tensor, "v": tensor, ...}
        quantize: Whether to enable quantization
        target_dtype: Output dtype after dequantization

    Returns:
        QuantizedKVCacheList that behaves like the original but uses INT8 storage
    """
    return QuantizedKVCacheList(
        kv_cache,
        quantize_kv=quantize,
        target_dtype=target_dtype,
    )


def create_quantized_kv_cache(
    num_layers: int,
    batch_size: int,
    kv_cache_size: int,
    num_heads: int = 12,
    head_dim: int = 128,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    quantize: bool = True,
) -> QuantizedKVCacheList:
    """
    Create a new quantized KV cache with the standard LongLive format.

    This creates the same structure as the base pipeline but with
    transparent quantization.

    Args:
        num_layers: Number of transformer layers
        batch_size: Batch size
        kv_cache_size: Total KV cache size in tokens
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: CUDA device
        dtype: Data type for dequantized values
        quantize: Whether to enable quantization

    Returns:
        QuantizedKVCacheList ready for use
    """
    device = device or torch.device('cuda')

    kv_cache = []
    for layer_idx in range(num_layers):
        layer_cache = {
            'k': torch.zeros(
                [batch_size, kv_cache_size, num_heads, head_dim],
                dtype=dtype,
                device=device,
            ),
            'v': torch.zeros(
                [batch_size, kv_cache_size, num_heads, head_dim],
                dtype=dtype,
                device=device,
            ),
            'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
            'local_end_index': torch.tensor([0], dtype=torch.long, device=device),
        }
        kv_cache.append(layer_cache)

    return QuantizedKVCacheList(
        kv_cache,
        quantize_kv=quantize,
        target_dtype=dtype,
    )
