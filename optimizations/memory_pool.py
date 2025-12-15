"""
Fixed-Shape Memory Pool for LongLive.

Pre-allocates tensors with fixed shapes to eliminate runtime memory allocation.
This is especially important for CUDA graph compatibility where allocations
during graph capture can cause issues.

Key features:
1. Pre-allocated buffers for all known tensor shapes
2. Zero allocation during inference
3. Buffer recycling with reference counting
4. CUDA graph compatible
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class BufferSpec:
    """Specification for a memory buffer."""
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    pinned: bool = False  # For CPU tensors, use pinned memory


class FixedShapeMemoryPool:
    """
    Memory pool with pre-allocated fixed-shape buffers.

    Eliminates runtime allocation by providing pre-allocated tensors.
    All tensors have fixed shapes determined at initialization.

    Usage:
        # Define buffer specifications
        specs = {
            'latents': BufferSpec('latents', (1, 16, 3, 60, 104), torch.bfloat16, device),
            'noise': BufferSpec('noise', (1, 16, 3, 60, 104), torch.bfloat16, device),
            'vae_output': BufferSpec('vae_output', (3, 480, 832), torch.bfloat16, device),
        }

        # Create pool
        pool = FixedShapeMemoryPool(specs)

        # Get pre-allocated buffer
        latents = pool.get_buffer('latents')

        # Return buffer when done
        pool.return_buffer('latents', latents)
    """

    def __init__(
        self,
        buffer_specs: Dict[str, BufferSpec] = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize memory pool.

        Args:
            buffer_specs: Dictionary of buffer name -> BufferSpec
            device: Default device for buffers
            dtype: Default dtype for buffers
        """
        self.device = device or torch.device('cuda')
        self.dtype = dtype
        self.buffer_specs = buffer_specs or {}

        # Allocated buffers: name -> tensor
        self._buffers: Dict[str, torch.Tensor] = {}

        # Buffer availability tracking
        self._available: Dict[str, bool] = {}

        # Pre-allocate all specified buffers
        self._allocate_all()

    def _allocate_all(self):
        """Pre-allocate all buffers from specs."""
        for name, spec in self.buffer_specs.items():
            self._allocate_buffer(spec)

    def _allocate_buffer(self, spec: BufferSpec):
        """Allocate a single buffer."""
        if spec.pinned and spec.device.type == 'cpu':
            # Pinned CPU memory for faster D2H transfers
            buffer = torch.empty(
                spec.shape,
                dtype=spec.dtype,
                device='cpu',
                pin_memory=True
            )
        else:
            buffer = torch.empty(
                spec.shape,
                dtype=spec.dtype,
                device=spec.device
            )

        self._buffers[spec.name] = buffer
        self._available[spec.name] = True

    def get_buffer(self, name: str) -> torch.Tensor:
        """
        Get a pre-allocated buffer by name.

        Args:
            name: Buffer name

        Returns:
            Pre-allocated tensor

        Raises:
            KeyError: If buffer name not found
            RuntimeError: If buffer is currently in use
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found in pool")

        if not self._available[name]:
            raise RuntimeError(f"Buffer '{name}' is currently in use")

        self._available[name] = False
        return self._buffers[name]

    def return_buffer(self, name: str, buffer: torch.Tensor = None):
        """
        Return a buffer to the pool.

        Args:
            name: Buffer name
            buffer: The buffer being returned (for validation)
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found in pool")

        # Optionally validate it's the same buffer
        if buffer is not None and buffer.data_ptr() != self._buffers[name].data_ptr():
            raise ValueError(f"Returned buffer does not match pool buffer '{name}'")

        self._available[name] = True

    def add_buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = None,
        device: torch.device = None,
        pinned: bool = False
    ):
        """
        Add a new buffer specification and allocate it.

        Args:
            name: Buffer name
            shape: Tensor shape
            dtype: Data type (uses pool default if not specified)
            device: Device (uses pool default if not specified)
            pinned: Use pinned memory for CPU tensors
        """
        spec = BufferSpec(
            name=name,
            shape=shape,
            dtype=dtype or self.dtype,
            device=device or self.device,
            pinned=pinned
        )
        self.buffer_specs[name] = spec
        self._allocate_buffer(spec)

    def has_buffer(self, name: str) -> bool:
        """Check if a buffer exists."""
        return name in self._buffers

    def is_available(self, name: str) -> bool:
        """Check if a buffer is available for use."""
        return name in self._available and self._available[name]

    def get_or_allocate(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Get existing buffer or allocate new one if needed.

        This is a convenience method for dynamic buffer management.
        """
        if name in self._buffers:
            # Check if shape matches
            if self._buffers[name].shape == shape:
                return self.get_buffer(name)
            else:
                # Shape mismatch - reallocate
                del self._buffers[name]
                del self._available[name]

        # Allocate new buffer
        self.add_buffer(name, shape, dtype, device)
        return self.get_buffer(name)

    def memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics.

        Returns:
            Dict with 'total_bytes', 'gpu_bytes', 'cpu_bytes'
        """
        total_bytes = 0
        gpu_bytes = 0
        cpu_bytes = 0

        for name, buffer in self._buffers.items():
            size = buffer.numel() * buffer.element_size()
            total_bytes += size

            if buffer.is_cuda:
                gpu_bytes += size
            else:
                cpu_bytes += size

        return {
            'total_bytes': total_bytes,
            'gpu_bytes': gpu_bytes,
            'cpu_bytes': cpu_bytes,
            'total_mb': total_bytes / (1024 ** 2),
            'gpu_mb': gpu_bytes / (1024 ** 2),
            'cpu_mb': cpu_bytes / (1024 ** 2),
        }

    def reset(self):
        """Reset all buffers to available state and zero them."""
        for name in self._buffers:
            self._available[name] = True
            self._buffers[name].zero_()

    def clear(self):
        """Clear all buffers and free memory."""
        self._buffers.clear()
        self._available.clear()

    def __repr__(self) -> str:
        usage = self.memory_usage()
        return (
            f"FixedShapeMemoryPool(\n"
            f"  num_buffers={len(self._buffers)},\n"
            f"  gpu_mb={usage['gpu_mb']:.2f},\n"
            f"  cpu_mb={usage['cpu_mb']:.2f}\n"
            f")"
        )


class LongLiveMemoryPool(FixedShapeMemoryPool):
    """
    Memory pool pre-configured for LongLive inference.

    Automatically creates buffers for:
    - Latent tensors
    - Noise tensors
    - VAE output frames
    - KV cache buffers (if not using StaticKVCache)
    - Intermediate attention tensors
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_frames: int = 3,  # Frames per block
        latent_channels: int = 16,
        latent_height: int = 60,
        latent_width: int = 104,
        output_height: int = 480,
        output_width: int = 832,
        num_layers: int = 30,
        num_heads: int = 12,
        head_dim: int = 128,
        local_window_frames: int = 12,
        frame_seq_length: int = 1560,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        include_kv_cache: bool = False,
        use_pinned_output: bool = True,
    ):
        """
        Initialize LongLive-specific memory pool.

        Args:
            batch_size: Batch size
            num_frames: Frames per generation block
            latent_*: Latent space dimensions
            output_*: Output frame dimensions
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            local_window_frames: Local attention window size
            frame_seq_length: Sequence length per frame
            dtype: Data type
            device: Device
            include_kv_cache: Include KV cache buffers
            use_pinned_output: Use pinned memory for output frames
        """
        device = device or torch.device('cuda')

        # Calculate shapes
        latent_shape = (batch_size, latent_channels, num_frames, latent_height, latent_width)
        output_shape = (3, output_height, output_width)  # Single frame output
        seq_len = num_frames * frame_seq_length

        # Define all buffer specs
        specs = {
            # Latent tensors
            'latents': BufferSpec('latents', latent_shape, dtype, device),
            'noise': BufferSpec('noise', latent_shape, dtype, device),
            'denoised': BufferSpec('denoised', latent_shape, dtype, device),

            # VAE outputs (double buffer for async)
            'vae_output_0': BufferSpec('vae_output_0', output_shape, dtype, device),
            'vae_output_1': BufferSpec('vae_output_1', output_shape, dtype, device),

            # Intermediate tensors
            'hidden_states': BufferSpec(
                'hidden_states',
                (batch_size, seq_len, num_heads * head_dim),
                dtype, device
            ),

            # Timestep embeddings
            'timestep_embed': BufferSpec(
                'timestep_embed',
                (batch_size, num_heads * head_dim),
                dtype, device
            ),
        }

        # Add CPU pinned buffer for output transfer
        if use_pinned_output:
            specs['output_pinned'] = BufferSpec(
                'output_pinned',
                output_shape,
                dtype,
                torch.device('cpu'),
                pinned=True
            )

        # Optionally add KV cache buffers
        if include_kv_cache:
            kv_shape = (
                batch_size,
                local_window_frames * frame_seq_length,
                num_heads,
                head_dim
            )
            for i in range(num_layers):
                specs[f'kv_k_{i}'] = BufferSpec(f'kv_k_{i}', kv_shape, dtype, device)
                specs[f'kv_v_{i}'] = BufferSpec(f'kv_v_{i}', kv_shape, dtype, device)

        super().__init__(specs, device, dtype)

    @classmethod
    def from_config(cls, config: 'OptimizationConfig', device: torch.device = None) -> 'LongLiveMemoryPool':
        """Create memory pool from OptimizationConfig."""
        return cls(
            batch_size=1,
            num_frames=config.num_frame_per_block,
            num_layers=config.num_transformer_blocks,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            local_window_frames=config.local_attn_size,
            frame_seq_length=config.frame_seq_length,
            dtype=config.get_torch_dtype(),
            device=device or torch.device('cuda'),
            include_kv_cache=not config.use_static_kv,  # Only if not using StaticKVCache
            use_pinned_output=config.use_pinned_memory,
        )


class TensorCache:
    """
    Simple tensor cache with LRU eviction.

    For caching computed tensors that may be reused (e.g., attention patterns).
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: Dict[str, torch.Tensor] = {}
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached tensor if available."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, tensor: torch.Tensor):
        """Cache a tensor."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict oldest
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = tensor
        self._access_order.append(key)

    def clear(self):
        """Clear all cached tensors."""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)
