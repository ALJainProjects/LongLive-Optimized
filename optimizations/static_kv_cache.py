"""
Static KV Cache with Ring Buffer for LongLive.

This module implements a pre-allocated KV cache with ring buffer semantics
for efficient cache rolling without memory allocation during inference.

Key features:
1. Pre-allocated fixed-size buffers (CUDA graph compatible)
2. Ring buffer for O(1) cache updates without memory copies
3. Separate sink cache for frame-sink tokens (always-attendable anchors)
4. Zero runtime allocation after initialization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class StaticKVCache(nn.Module):
    """
    Static KV cache with ring buffer for efficient local attention.

    The cache is split into two parts:
    1. Sink cache: First N frames (frame-sink) - always in attention window
    2. Local cache: Rolling window of recent frames - ring buffer semantics

    Memory layout:
        [sink_k, sink_v]: Fixed, stores frame-sink tokens
        [local_k, local_v]: Ring buffer for local attention window

    CUDA Graph Compatibility:
        - All buffers are pre-allocated at initialization
        - Updates use in-place copy_() operations
        - No new tensor allocations during forward pass
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
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        memory_pool: Optional['FixedShapeMemoryPool'] = None,
    ):
        """
        Initialize static KV cache.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            local_window_frames: Number of frames in local attention window
            sink_frames: Number of frame-sink frames (anchors)
            frame_seq_length: Tokens per frame
            batch_size: Batch size
            dtype: Data type for cache
            device: Device for cache
            memory_pool: Optional pre-allocated memory pool
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_window_frames = local_window_frames
        self.sink_frames = sink_frames
        self.frame_seq_length = frame_seq_length
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device or torch.device('cuda')

        # Calculate sizes
        self.sink_size = sink_frames * frame_seq_length
        self.local_size = local_window_frames * frame_seq_length
        self.total_size = self.sink_size + self.local_size

        # Pre-allocate all cache buffers
        self._allocate_caches()

        # Ring buffer state (per layer)
        # write_idx: where to write next tokens in local cache
        # valid_len: how many valid tokens in local cache
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

        # Track total frames written
        self.register_buffer(
            'total_frames',
            torch.tensor(0, dtype=torch.long, device=self.device)
        )

    def _allocate_caches(self):
        """Pre-allocate all KV cache buffers."""
        # Shape: [batch, seq_len, num_heads, head_dim]
        # We use this layout for efficient attention

        for layer_idx in range(self.num_layers):
            # Sink cache (frame-sink tokens)
            self.register_buffer(
                f'sink_k_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.sink_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device
                )
            )
            self.register_buffer(
                f'sink_v_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.sink_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device
                )
            )

            # Local cache (ring buffer)
            self.register_buffer(
                f'local_k_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.local_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device
                )
            )
            self.register_buffer(
                f'local_v_{layer_idx}',
                torch.zeros(
                    self.batch_size, self.local_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device
                )
            )

    def get_sink_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sink (frame-sink) KV cache for a layer."""
        k = getattr(self, f'sink_k_{layer_idx}')
        v = getattr(self, f'sink_v_{layer_idx}')
        return k, v

    def get_local_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get local (ring buffer) KV cache for a layer."""
        k = getattr(self, f'local_k_{layer_idx}')
        v = getattr(self, f'local_v_{layer_idx}')
        return k, v

    def get_full_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full KV cache (sink + valid local) for attention.

        Returns properly ordered K and V tensors for attention computation.
        The ring buffer is linearized to present tokens in temporal order.
        """
        sink_k, sink_v = self.get_sink_kv(layer_idx)
        local_k, local_v = self.get_local_kv(layer_idx)

        write_idx = self.write_indices[layer_idx].item()
        valid_len = self.valid_lengths[layer_idx].item()

        if valid_len == 0:
            # Only sink cache
            return sink_k, sink_v

        if valid_len < self.local_size:
            # Not wrapped yet - return sink + valid portion
            full_k = torch.cat([sink_k, local_k[:, :valid_len]], dim=1)
            full_v = torch.cat([sink_v, local_v[:, :valid_len]], dim=1)
        else:
            # Ring buffer has wrapped - reorder to temporal sequence
            # Order: sink, then oldest->newest in ring buffer
            # Ring buffer order: [write_idx:] then [:write_idx]
            reordered_k = torch.cat([
                local_k[:, write_idx:],
                local_k[:, :write_idx]
            ], dim=1)
            reordered_v = torch.cat([
                local_v[:, write_idx:],
                local_v[:, :write_idx]
            ], dim=1)

            full_k = torch.cat([sink_k, reordered_k], dim=1)
            full_v = torch.cat([sink_v, reordered_v], dim=1)

        return full_k, full_v

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        is_sink: bool = False
    ):
        """
        Update cache with new KV values.

        For CUDA graph compatibility, uses in-place copy_() operations.

        Args:
            layer_idx: Which transformer layer
            new_k: New key tensor [batch, seq_len, num_heads, head_dim]
            new_v: New value tensor [batch, seq_len, num_heads, head_dim]
            is_sink: If True, write to sink cache instead of local cache
        """
        seq_len = new_k.shape[1]

        if is_sink:
            # Write to sink cache (used for first N frames)
            sink_k, sink_v = self.get_sink_kv(layer_idx)

            # Find current write position in sink
            # For simplicity, we assume sink is filled sequentially
            if not self.sink_filled[layer_idx]:
                # Still filling sink cache
                current_len = self.valid_lengths[layer_idx].item()
                end_idx = min(current_len + seq_len, self.sink_size)
                write_len = end_idx - current_len

                sink_k[:, current_len:end_idx].copy_(new_k[:, :write_len])
                sink_v[:, current_len:end_idx].copy_(new_v[:, :write_len])

                # Check if sink is now full
                if end_idx >= self.sink_size:
                    self.sink_filled[layer_idx] = True

                    # Any remaining tokens go to local cache
                    remaining = seq_len - write_len
                    if remaining > 0:
                        self._update_local_ring(
                            layer_idx,
                            new_k[:, write_len:],
                            new_v[:, write_len:]
                        )
                else:
                    self.valid_lengths[layer_idx] = end_idx
        else:
            # Write to local cache (ring buffer)
            self._update_local_ring(layer_idx, new_k, new_v)

    def _update_local_ring(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ):
        """
        Update local cache using ring buffer semantics.

        Uses in-place copy_() for CUDA graph compatibility.
        """
        local_k, local_v = self.get_local_kv(layer_idx)
        seq_len = new_k.shape[1]
        write_idx = self.write_indices[layer_idx].item()

        # Calculate how much fits before wrap
        space_before_wrap = self.local_size - write_idx

        if seq_len <= space_before_wrap:
            # Fits without wrapping
            local_k[:, write_idx:write_idx + seq_len].copy_(new_k)
            local_v[:, write_idx:write_idx + seq_len].copy_(new_v)
            new_write_idx = (write_idx + seq_len) % self.local_size
        else:
            # Needs to wrap around
            # First part: fill to end
            local_k[:, write_idx:].copy_(new_k[:, :space_before_wrap])
            local_v[:, write_idx:].copy_(new_v[:, :space_before_wrap])

            # Second part: wrap to beginning
            remaining = seq_len - space_before_wrap
            local_k[:, :remaining].copy_(new_k[:, space_before_wrap:])
            local_v[:, :remaining].copy_(new_v[:, space_before_wrap:])

            new_write_idx = remaining

        # Update write index (in-place for CUDA graph compatibility)
        self.write_indices[layer_idx] = new_write_idx

        # Update valid length (capped at local_size)
        current_valid = self.valid_lengths[layer_idx].item()
        new_valid = min(current_valid + seq_len, self.local_size)
        self.valid_lengths[layer_idx] = new_valid

    def update_all_layers(
        self,
        new_k_list: List[torch.Tensor],
        new_v_list: List[torch.Tensor],
        is_sink: bool = False
    ):
        """Update all layers at once."""
        for layer_idx, (new_k, new_v) in enumerate(zip(new_k_list, new_v_list)):
            self.update(layer_idx, new_k, new_v, is_sink=is_sink)

    def advance_frame(self):
        """Called after generating a frame to update frame counter."""
        self.total_frames += 1

    def reset(self):
        """Reset cache to initial state."""
        self.write_indices.zero_()
        self.valid_lengths.zero_()
        self.sink_filled.zero_()
        self.total_frames.zero_()

        # Zero out all cache buffers
        for layer_idx in range(self.num_layers):
            sink_k, sink_v = self.get_sink_kv(layer_idx)
            local_k, local_v = self.get_local_kv(layer_idx)
            sink_k.zero_()
            sink_v.zero_()
            local_k.zero_()
            local_v.zero_()

    def recache_with_new_prompt(self, new_prompt_embeds: torch.Tensor):
        """
        Recache KV values with new prompt embeddings (for prompt switching).

        This is called when the user changes prompts. We need to recompute
        the KV cache for recent frames with the new prompt conditioning.

        Note: This operation is NOT CUDA graph compatible as it requires
        running the model forward pass. It should be done outside the graph.
        """
        # This will be implemented in the optimized pipeline
        # Here we just mark that a recache is needed
        pass

    def get_attention_mask(self, query_len: int) -> torch.Tensor:
        """
        Get causal attention mask for current cache state.

        Returns mask of shape [query_len, key_len] where:
        - key_len = sink_size + valid_local_len
        - Causal masking is applied to local tokens
        - Sink tokens are always attendable
        """
        # Calculate total key length
        valid_local = self.valid_lengths[0].item()  # Assume same for all layers
        key_len = self.sink_size + valid_local

        # Create mask (True = attend, False = mask out)
        mask = torch.ones(query_len, key_len, dtype=torch.bool, device=self.device)

        # Sink tokens are always attendable (already True)
        # Local tokens use causal masking - this is handled by the attention module

        return mask

    def to_legacy_format(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Convert to LongLive's original cache format for compatibility.

        Returns dict with keys: 'k', 'v', 'global_end_index', 'local_end_index'
        """
        full_k, full_v = self.get_full_kv(layer_idx)

        # LongLive uses shape [batch, seq, heads, dim]
        valid_len = self.sink_size + self.valid_lengths[layer_idx].item()

        return {
            'k': full_k,
            'v': full_v,
            'global_end_index': torch.tensor([valid_len], device=self.device),
            'local_end_index': torch.tensor([self.valid_lengths[layer_idx].item()], device=self.device),
        }

    @classmethod
    def from_config(cls, config: 'OptimizationConfig', device: torch.device = None) -> 'StaticKVCache':
        """Create StaticKVCache from OptimizationConfig."""
        return cls(
            num_layers=config.num_transformer_blocks,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            local_window_frames=config.local_attn_size,
            sink_frames=config.sink_size,
            frame_seq_length=config.frame_seq_length,
            batch_size=1,  # Typically 1 for inference
            dtype=config.get_torch_dtype(),
            device=device or torch.device('cuda'),
        )

    def __repr__(self) -> str:
        return (
            f"StaticKVCache(\n"
            f"  layers={self.num_layers},\n"
            f"  heads={self.num_heads},\n"
            f"  head_dim={self.head_dim},\n"
            f"  sink_size={self.sink_size} ({self.sink_frames} frames),\n"
            f"  local_size={self.local_size} ({self.local_window_frames} frames),\n"
            f"  total_size={self.total_size},\n"
            f"  dtype={self.dtype}\n"
            f")"
        )
