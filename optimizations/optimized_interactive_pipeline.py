"""
Optimized Interactive Causal Inference Pipeline for LongLive.

This module extends the OptimizedCausalInferencePipeline to support
interactive prompt switching, which is the key feature of InteractiveCausalInferencePipeline.

Usage:
    from optimizations import OptimizedInteractiveCausalInferencePipeline, OptimizationConfig

    # Wrap existing interactive pipeline
    base_pipeline = InteractiveCausalInferencePipeline(config, device)
    opt_config = OptimizationConfig.preset_balanced()
    pipeline = OptimizedInteractiveCausalInferencePipeline(base_pipeline, opt_config)

    # Use exactly like base interactive pipeline
    video = pipeline.inference(
        noise=noise,
        text_prompts_list=[["prompt1"], ["prompt2"]],
        switch_frame_indices=[40],
        return_latents=False,
    )
"""

import torch
import torch.cuda
from typing import Dict, List, Optional, Tuple, Any, Union

from .config import OptimizationConfig
from .optimized_pipeline import OptimizedCausalInferencePipeline
from .latency_profiler import LatencyProfiler
from .sync_elimination import SyncFreeContext


class OptimizedInteractiveCausalInferencePipeline(OptimizedCausalInferencePipeline):
    """
    Optimized wrapper for InteractiveCausalInferencePipeline.

    Extends OptimizedCausalInferencePipeline to support:
    - Multiple prompt segments (text_prompts_list)
    - Dynamic prompt switching at specified frames (switch_frame_indices)
    - Optimized KV-recache after prompt switch

    All optimizations from the base class are preserved:
    - Prompt embedding cache
    - Static/quantized KV cache
    - Async VAE decoding
    - torch.compile
    - Memory pool
    - Sync-free context
    """

    def __init__(
        self,
        base_pipeline: Any,
        config: OptimizationConfig,
        device: torch.device = None,
    ):
        """
        Initialize optimized interactive wrapper.

        Args:
            base_pipeline: InteractiveCausalInferencePipeline to wrap
            config: Optimization configuration
            device: CUDA device
        """
        super().__init__(base_pipeline, config, device)

        # Store interactive-specific attributes
        self.global_sink = getattr(base_pipeline, 'global_sink', False)

        # Track prompt switches for profiling
        self._prompt_switches = 0

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        profile: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Optimized interactive inference with prompt switching.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W)
            text_prompts_list: List of prompt lists for each segment
            switch_frame_indices: Frame indices where prompts change
            return_latents: Whether to return latent tensor
            low_memory: Enable low-memory mode
            profile: Enable profiling

        Returns:
            Generated video tensor, optionally with latents
        """
        # Enable profiling if requested
        if profile:
            self.profiler.enabled = True
            self.profiler.reset()

        # Use sync-free context
        sync_context = SyncFreeContext(
            strict=False,
            warn=self.config.enable_profiling
        ) if self._use_sync_free else None

        if sync_context is not None:
            sync_context.__enter__()

        try:
            return self._interactive_inference_impl(
                noise=noise,
                text_prompts_list=text_prompts_list,
                switch_frame_indices=switch_frame_indices,
                return_latents=return_latents,
                low_memory=low_memory,
                profile=profile,
            )
        finally:
            if sync_context is not None:
                sync_context.__exit__(None, None, None)

    def _interactive_inference_impl(
        self,
        noise: torch.Tensor,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        profile: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Internal implementation of interactive inference with optimizations."""
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        with self.profiler.measure("total_inference"):
            # Step 1: Encode all prompts (uses cached version if enabled)
            with self.profiler.measure("text_encoding_all"):
                cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

            if low_memory:
                from utils.memory import get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, gpu
                gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
                move_model_to_device_with_memory_preservation(
                    self.text_encoder,
                    target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation,
                )

            # Output buffer
            output_device = torch.device('cpu') if low_memory else noise.device
            output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=output_device,
                dtype=noise.dtype
            )

            # Step 2: Initialize KV caches (optimized)
            local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
            if local_attn_cfg != -1:
                kv_cache_size = local_attn_cfg * self.frame_seq_length
            else:
                kv_cache_size = num_output_frames * self.frame_seq_length

            # Use optimized cache initialization
            if self._use_quantized_kv:
                self.base.kv_cache1 = self._initialize_kv_cache_quantized(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device,
                    kv_cache_size=kv_cache_size,
                )
            else:
                self.base.kv_cache1 = self._initialize_kv_cache_optimized(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device,
                    kv_cache_size=kv_cache_size,
                )

            self.base.crossattn_cache = self._initialize_crossattn_cache_optimized(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )

            # Set attention size
            current_start_frame = 0
            self.generator.model.local_attn_size = self.local_attn_size
            self.base._set_all_modules_max_attention_size(self.local_attn_size)

            # Step 3: Temporal denoising with prompt switching
            all_num_frames = [self.num_frame_per_block] * num_blocks
            segment_idx = 0
            next_switch_pos = (
                switch_frame_indices[segment_idx]
                if segment_idx < len(switch_frame_indices)
                else None
            )

            for block_idx, current_num_frames in enumerate(all_num_frames):
                self.profiler.start_frame()

                # Check for prompt switch
                if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                    segment_idx += 1
                    self._prompt_switches += 1

                    with self.profiler.measure("prompt_switch"):
                        self._recache_after_switch_optimized(
                            output, current_start_frame, cond_list[segment_idx]
                        )

                    next_switch_pos = (
                        switch_frame_indices[segment_idx]
                        if segment_idx < len(switch_frame_indices)
                        else None
                    )

                cond_in_use = cond_list[segment_idx]

                with self.profiler.measure(f"block_{block_idx}"):
                    noisy_input = noise[
                        :, current_start_frame:current_start_frame + current_num_frames
                    ]

                    # Spatial denoising loop
                    for step_idx, current_timestep in enumerate(self.denoising_step_list):
                        with self.profiler.measure(f"denoise_step_{step_idx}"):
                            timestep = torch.ones(
                                [batch_size, current_num_frames],
                                device=noise.device,
                                dtype=torch.int64
                            ) * current_timestep

                            if step_idx < len(self.denoising_step_list) - 1:
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=cond_in_use,
                                    timestep=timestep,
                                    kv_cache=self.base.kv_cache1,
                                    crossattn_cache=self.base.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length,
                                )

                                next_timestep = self.denoising_step_list[step_idx + 1]
                                noisy_input = self.scheduler.add_noise(
                                    denoised_pred.flatten(0, 1),
                                    torch.randn_like(denoised_pred.flatten(0, 1)),
                                    next_timestep * torch.ones(
                                        [batch_size * current_num_frames],
                                        device=noise.device,
                                        dtype=torch.long
                                    )
                                ).unflatten(0, denoised_pred.shape[:2])
                            else:
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=cond_in_use,
                                    timestep=timestep,
                                    kv_cache=self.base.kv_cache1,
                                    crossattn_cache=self.base.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length,
                                )

                    # Record output
                    output[:, current_start_frame:current_start_frame + current_num_frames] = \
                        denoised_pred.to(output.device)

                    # Update KV cache with clean context
                    with self.profiler.measure("kv_cache_update"):
                        context_timestep = torch.ones_like(timestep) * self.args.context_noise
                        self.generator(
                            noisy_image_or_video=denoised_pred,
                            conditional_dict=cond_in_use,
                            timestep=context_timestep,
                            kv_cache=self.base.kv_cache1,
                            crossattn_cache=self.base.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

                    current_start_frame += current_num_frames

                self.profiler.end_frame()
                self._frames_generated += 1

            # Step 4: VAE decode
            with self.profiler.measure("vae_decode"):
                if self.async_vae is not None:
                    video = self.async_vae.decode(output.to(noise.device))
                else:
                    video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
                    video = (video * 0.5 + 0.5).clamp(0, 1)

        # Print profiling report
        if profile or self.config.enable_profiling:
            self._print_interactive_stats()
            self.profiler.print_report()

        if return_latents:
            return video, output.to(noise.device)
        return video

    def _recache_after_switch_optimized(
        self,
        output: torch.Tensor,
        current_start_frame: int,
        new_conditional_dict: dict,
    ):
        """
        Optimized KV-recache after prompt switch.

        Uses the same logic as base but with profiling instrumentation.
        """
        with self.profiler.measure("recache_kv_reset"):
            if not self.global_sink:
                # Reset KV cache
                for block_idx in range(self.num_transformer_blocks):
                    cache = self.base.kv_cache1[block_idx]
                    cache["k"].zero_()
                    cache["v"].zero_()

            # Reset cross-attention cache
            for blk in self.base.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

        # Recache previous frames with new prompt
        if current_start_frame == 0:
            return

        with self.profiler.measure("recache_frames"):
            num_recache_frames = (
                current_start_frame if self.local_attn_size == -1
                else min(self.local_attn_size, current_start_frame)
            )
            recache_start_frame = current_start_frame - num_recache_frames

            frames_to_recache = output[:, recache_start_frame:current_start_frame]
            if frames_to_recache.device.type == 'cpu':
                target_device = next(self.generator.parameters()).device
                frames_to_recache = frames_to_recache.to(target_device)

            batch_size = frames_to_recache.shape[0]
            device = frames_to_recache.device

            # Prepare blockwise causal mask
            block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
                device=device,
                num_frames=num_recache_frames,
                frame_seqlen=self.frame_seq_length,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )

            context_timestep = torch.ones(
                [batch_size, num_recache_frames],
                device=device,
                dtype=torch.int64
            ) * self.args.context_noise

            self.generator.model.block_mask = block_mask

            # Recache with new prompt
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=frames_to_recache,
                    conditional_dict=new_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.base.kv_cache1,
                    crossattn_cache=self.base.crossattn_cache,
                    current_start=recache_start_frame * self.frame_seq_length,
                    sink_recache_after_switch=not self.global_sink,
                )

            # Reset cross-attention cache after recache
            for blk in self.base.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def _print_interactive_stats(self):
        """Print interactive-specific statistics."""
        print(f"\n{'='*60}")
        print("Interactive Optimization Statistics")
        print(f"{'='*60}")
        print(f"Frames generated: {self._frames_generated}")
        print(f"Prompt switches: {self._prompt_switches}")

        # Prompt cache stats
        if self.prompt_cache is not None:
            total = self._prompt_cache_hits + self._prompt_cache_misses
            hit_rate = self._prompt_cache_hits / total * 100 if total > 0 else 0
            print(f"Prompt cache: {self._prompt_cache_hits}/{total} hits ({hit_rate:.1f}%)")

        # Active optimizations
        print(f"\nActive Optimizations:")
        print(f"  Prompt Cache:   {'Y' if self.prompt_cache else 'N'}")
        print(f"  Memory Pool:    {'Y' if self.memory_pool else 'N'}")
        print(f"  Static KV:      {'Y' if self._preallocated_kv else 'N'}")
        print(f"  Quantized KV:   {'Y ' + self._kv_quantization if self._use_quantized_kv else 'N'}")
        print(f"  Async VAE:      {'Y' if self.async_vae else 'N'}")
        print(f"  torch.compile:  {'Y' if self._compiled else 'N'}")
        print(f"  Global Sink:    {'Y' if self.global_sink else 'N'}")

        print(f"{'='*60}")

    def reset(self):
        """Reset pipeline state."""
        super().reset()
        self._prompt_switches = 0

    @classmethod
    def from_base(
        cls,
        base_pipeline: Any,
        config: OptimizationConfig = None,
        device: torch.device = None,
    ) -> 'OptimizedInteractiveCausalInferencePipeline':
        """
        Factory method to create optimized interactive pipeline.

        Args:
            base_pipeline: InteractiveCausalInferencePipeline instance
            config: Optimization config (uses balanced preset if None)
            device: CUDA device

        Returns:
            OptimizedInteractiveCausalInferencePipeline wrapping the base
        """
        if config is None:
            config = OptimizationConfig.preset_balanced()

        return cls(base_pipeline, config, device)

    def __repr__(self) -> str:
        return (
            f"OptimizedInteractiveCausalInferencePipeline(\n"
            f"  base={type(self.base).__name__},\n"
            f"  prompt_cache={'enabled' if self.prompt_cache else 'disabled'},\n"
            f"  memory_pool={'enabled' if self.memory_pool else 'disabled'},\n"
            f"  static_kv={'enabled' if self._preallocated_kv else 'disabled'},\n"
            f"  quantized_kv={self._kv_quantization if self._use_quantized_kv else 'disabled'},\n"
            f"  async_vae={'enabled' if self.async_vae else 'disabled'},\n"
            f"  torch_compile={'enabled' if self._compiled else 'disabled'},\n"
            f"  global_sink={'enabled' if self.global_sink else 'disabled'},\n"
            f"  frames_generated={self._frames_generated},\n"
            f"  prompt_switches={self._prompt_switches}\n"
            f")"
        )


# Convenience function
def create_optimized_interactive_pipeline(
    base_pipeline: Any,
    preset: str = "balanced",
    **config_overrides
) -> OptimizedInteractiveCausalInferencePipeline:
    """
    Create optimized interactive pipeline with preset configuration.

    Args:
        base_pipeline: InteractiveCausalInferencePipeline instance
        preset: "quality", "balanced", or "speed"
        **config_overrides: Override specific config values

    Returns:
        OptimizedInteractiveCausalInferencePipeline
    """
    presets = {
        "quality": OptimizationConfig.preset_quality,
        "balanced": OptimizationConfig.preset_balanced,
        "speed": OptimizationConfig.preset_speed,
    }

    config = presets[preset]()

    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return OptimizedInteractiveCausalInferencePipeline.from_base(base_pipeline, config)
