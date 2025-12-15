"""
Async VAE Pipeline for LongLive.

Implements double-buffered VAE decoding to overlap latent-to-pixel
conversion with the next frame's generation.

Key features:
1. Double buffering - decode frame N while generating frame N+1
2. Separate CUDA stream for VAE operations
3. Pre-allocated output buffers
4. CUDA graph compatible (with static shapes)
"""

import torch
import torch.cuda
from typing import Optional, Tuple, List


class AsyncVAEPipeline:
    """
    Double-buffered async VAE decoder.

    Overlaps VAE decode with next frame generation to hide VAE latency.

    Architecture:
        Main Stream:     [Generate N] -----> [Generate N+1] -----> [Generate N+2]
        VAE Stream:           [Decode N-1] -----> [Decode N] -----> [Decode N+1]
        Output:          Frame N-1 ready    Frame N ready     Frame N+1 ready

    Usage:
        async_vae = AsyncVAEPipeline(vae_model)

        # Start decoding first frame
        async_vae.decode_async(latents_0)

        # Generate next frame while decode happens
        latents_1 = generate_frame()

        # Get previous frame (blocks until decode complete)
        frame_0 = async_vae.get_completed_frame()

        # Start next decode
        async_vae.decode_async(latents_1)
    """

    def __init__(
        self,
        vae,
        num_buffers: int = 2,
        height: int = 480,
        width: int = 832,
        num_channels: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        memory_pool: Optional['FixedShapeMemoryPool'] = None,
    ):
        """
        Initialize async VAE pipeline.

        Args:
            vae: VAE model (WanVAEWrapper)
            num_buffers: Number of output buffers (2 for double buffering)
            height: Output frame height
            width: Output frame width
            num_channels: Output channels (3 for RGB)
            dtype: Data type for buffers
            device: Device for buffers
            memory_pool: Optional pre-allocated memory pool
        """
        self.vae = vae
        self.num_buffers = num_buffers
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.dtype = dtype
        self.device = device or torch.device('cuda')
        self.memory_pool = memory_pool

        # Create separate CUDA stream for VAE operations
        self.vae_stream = torch.cuda.Stream(device=self.device)

        # Pre-allocate output buffers
        self._output_buffers: List[torch.Tensor] = []
        self._events: List[torch.cuda.Event] = []
        self._buffer_ready: List[bool] = []

        for _ in range(num_buffers):
            if memory_pool is not None:
                # Use memory pool if available
                buffer = memory_pool.get_buffer('vae_output')
            else:
                # Allocate new buffer
                buffer = torch.empty(
                    num_channels, height, width,
                    dtype=dtype,
                    device=self.device
                )
            self._output_buffers.append(buffer)
            self._events.append(torch.cuda.Event(enable_timing=False))
            self._buffer_ready.append(False)

        # Current write buffer index
        self._write_idx = 0
        # Current read buffer index (one behind write)
        self._read_idx = 0
        # Number of frames in flight
        self._in_flight = 0

    def decode_async(self, latents: torch.Tensor) -> None:
        """
        Start async VAE decode on separate stream.

        Does not block - returns immediately while decode runs.

        Args:
            latents: Latent tensor to decode [batch, channels, frames, H, W]
                     or [channels, frames, H, W]
        """
        # Get current write buffer
        buffer = self._output_buffers[self._write_idx]
        event = self._events[self._write_idx]

        # Launch decode on VAE stream
        with torch.cuda.stream(self.vae_stream):
            # Decode latents
            with torch.no_grad():
                decoded = self.vae.decode_to_pixel(latents, use_cache=False)

                # Normalize to [0, 1]
                decoded = (decoded * 0.5 + 0.5).clamp(0, 1)

                # Handle batch dimension
                if decoded.dim() == 5:  # [B, T, C, H, W]
                    decoded = decoded[0, -1]  # Take last frame of first batch
                elif decoded.dim() == 4:  # [T, C, H, W]
                    decoded = decoded[-1]  # Take last frame

                # Copy to buffer (in-place for CUDA graph compatibility)
                buffer.copy_(decoded)

            # Record completion event
            event.record()

        self._buffer_ready[self._write_idx] = True
        self._write_idx = (self._write_idx + 1) % self.num_buffers
        self._in_flight = min(self._in_flight + 1, self.num_buffers)

    def get_completed_frame(self, block: bool = True) -> Optional[torch.Tensor]:
        """
        Get the most recently completed decoded frame.

        Args:
            block: If True, wait for decode to complete. If False, return
                   None if not ready.

        Returns:
            Decoded frame tensor [C, H, W] or None if not ready
        """
        if self._in_flight == 0:
            return None

        event = self._events[self._read_idx]

        if block:
            # Wait for decode to complete
            event.synchronize()
        elif not event.query():
            # Decode not complete and non-blocking
            return None

        # Get completed frame
        frame = self._output_buffers[self._read_idx].clone()

        self._buffer_ready[self._read_idx] = False
        self._read_idx = (self._read_idx + 1) % self.num_buffers
        self._in_flight -= 1

        return frame

    def get_previous_frame(self) -> torch.Tensor:
        """
        Get the previous completed frame (for use in generation loop).

        Blocks until frame is ready.
        """
        return self.get_completed_frame(block=True)

    def flush(self) -> List[torch.Tensor]:
        """
        Wait for and return all pending decoded frames.

        Returns:
            List of all pending frames in order
        """
        frames = []
        while self._in_flight > 0:
            frame = self.get_completed_frame(block=True)
            if frame is not None:
                frames.append(frame)
        return frames

    def synchronize(self):
        """Wait for all VAE operations to complete."""
        self.vae_stream.synchronize()

    def reset(self):
        """Reset pipeline state for new video generation."""
        self.synchronize()
        self._write_idx = 0
        self._read_idx = 0
        self._in_flight = 0
        for i in range(self.num_buffers):
            self._buffer_ready[i] = False

    @property
    def has_pending(self) -> bool:
        """Check if there are pending decode operations."""
        return self._in_flight > 0


class SyncVAEPipeline:
    """
    Synchronous VAE pipeline (fallback when async not beneficial).

    Simple wrapper that provides same interface as AsyncVAEPipeline
    but runs synchronously on the main stream.
    """

    def __init__(
        self,
        vae,
        height: int = 480,
        width: int = 832,
        num_channels: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        self.vae = vae
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.dtype = dtype
        self.device = device or torch.device('cuda')

        self._last_frame: Optional[torch.Tensor] = None

    def decode_async(self, latents: torch.Tensor) -> None:
        """Decode synchronously (no async)."""
        with torch.no_grad():
            decoded = self.vae.decode_to_pixel(latents, use_cache=False)
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)

            if decoded.dim() == 5:
                decoded = decoded[0, -1]
            elif decoded.dim() == 4:
                decoded = decoded[-1]

            self._last_frame = decoded.clone()

    def get_completed_frame(self, block: bool = True) -> Optional[torch.Tensor]:
        """Return last decoded frame."""
        return self._last_frame

    def get_previous_frame(self) -> torch.Tensor:
        """Return last decoded frame."""
        return self._last_frame

    def flush(self) -> List[torch.Tensor]:
        """Return last frame as list."""
        if self._last_frame is not None:
            return [self._last_frame]
        return []

    def synchronize(self):
        """No-op for sync pipeline."""
        pass

    def reset(self):
        """Reset state."""
        self._last_frame = None

    @property
    def has_pending(self) -> bool:
        return False


class AsyncVAEDecoder:
    """
    Simplified async VAE decoder for full video decode.

    This is a higher-level wrapper that handles the full video decode,
    running on a separate CUDA stream to overlap with other operations.

    Unlike AsyncVAEPipeline (frame-by-frame), this decodes the full
    output latents in one call.

    Usage:
        decoder = AsyncVAEDecoder(vae)
        video = decoder.decode(output_latents)  # Full video decode
    """

    def __init__(
        self,
        vae,
        device: torch.device = None,
        use_async: bool = True,
    ):
        """
        Initialize async VAE decoder.

        Args:
            vae: VAE model (WanVAEWrapper)
            device: Device for output tensors
            use_async: If True, use separate CUDA stream
        """
        self.vae = vae
        self.device = device or torch.device('cuda')
        self.use_async = use_async and torch.cuda.is_available()

        if self.use_async:
            self.vae_stream = torch.cuda.Stream(device=self.device)
        else:
            self.vae_stream = None

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to video.

        Args:
            latents: Latent tensor [batch, frames, channels, H, W]

        Returns:
            Video tensor [batch, frames, channels, H, W] normalized to [0, 1]
        """
        if self.use_async and self.vae_stream is not None:
            with torch.cuda.stream(self.vae_stream):
                video = self._decode_impl(latents)
            # Sync to ensure decode is complete before returning
            self.vae_stream.synchronize()
        else:
            video = self._decode_impl(latents)

        return video

    def _decode_impl(self, latents: torch.Tensor) -> torch.Tensor:
        """Internal decode implementation."""
        with torch.no_grad():
            video = self.vae.decode_to_pixel(latents, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
        return video

    def decode_async_start(self, latents: torch.Tensor) -> torch.cuda.Event:
        """
        Start async decode without blocking.

        Returns an event that can be waited on for completion.

        Args:
            latents: Latent tensor to decode

        Returns:
            CUDA event to synchronize on
        """
        if not self.use_async or self.vae_stream is None:
            # Sync fallback
            self._last_result = self._decode_impl(latents)
            return None

        event = torch.cuda.Event()

        with torch.cuda.stream(self.vae_stream):
            self._last_result = self._decode_impl(latents)
            event.record()

        return event

    def decode_async_wait(self, event: torch.cuda.Event = None) -> torch.Tensor:
        """
        Wait for async decode to complete and return result.

        Args:
            event: Event from decode_async_start (optional)

        Returns:
            Decoded video tensor
        """
        if event is not None:
            event.synchronize()
        elif self.vae_stream is not None:
            self.vae_stream.synchronize()

        return self._last_result
