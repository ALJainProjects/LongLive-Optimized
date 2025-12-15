"""
CUDA Graph Wrapper for LongLive.

Captures the denoising loop as a CUDA graph to eliminate kernel launch overhead.
This can provide 30-50% latency reduction by replaying pre-recorded operations.

Key features:
1. Warmup phase to stabilize kernels before capture
2. Graph pool for multiple captured variants (different shapes/timesteps)
3. Static input/output buffers for graph replay
4. Fallback to eager mode when graphs cannot be used
"""

import torch
import torch.cuda
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass


@dataclass
class GraphCaptureConfig:
    """Configuration for CUDA graph capture."""
    warmup_steps: int = 3
    capture_error_mode: str = "thread_local"  # or "global"
    pool_size: int = 4  # Number of graph variants to keep


class CUDAGraphWrapper:
    """
    CUDA graph wrapper for denoising steps.

    Captures and replays the forward pass of the diffusion model to
    eliminate Python overhead and kernel launch latency.

    Usage:
        wrapper = CUDAGraphWrapper(model, config)

        # Warmup and capture during first inference
        wrapper.warmup_and_capture(sample_latents, timesteps, prompt_embeds)

        # Subsequent inferences use graph replay
        output = wrapper.replay(latents, timestep, prompt_embeds)

    Compatibility:
        - Requires static tensor shapes within each graph
        - Supports multiple graph variants via graph pool
        - Falls back to eager mode for unsupported configurations
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: GraphCaptureConfig = None,
        device: torch.device = None,
    ):
        """
        Initialize CUDA graph wrapper.

        Args:
            model: The diffusion model (DiT/transformer)
            config: Graph capture configuration
            device: CUDA device for graph operations
        """
        self.model = model
        self.config = config or GraphCaptureConfig()
        self.device = device or torch.device('cuda')

        # Graph pool: key -> (graph, static_inputs, static_outputs)
        self._graph_pool: Dict[str, Tuple[torch.cuda.CUDAGraph, Dict, torch.Tensor]] = {}

        # Currently active graph key
        self._current_key: Optional[str] = None

        # Capture state
        self._is_capturing = False
        self._warmup_complete = False

        # CUDA stream for graph operations
        self._graph_stream = torch.cuda.Stream(device=self.device)

    def _get_graph_key(
        self,
        latents_shape: Tuple,
        timestep: int,
    ) -> str:
        """Generate a unique key for graph lookup."""
        # Key based on shape and timestep (shapes must match for replay)
        return f"{latents_shape}_{timestep}"

    def _warmup(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: Dict[str, torch.Tensor],
        **kwargs
    ) -> None:
        """
        Run warmup iterations to stabilize kernels.

        This ensures consistent kernel behavior before graph capture.
        """
        for i in range(self.config.warmup_steps):
            with torch.no_grad():
                # Run forward pass (discards output)
                _ = self._eager_forward(latents, timestep, prompt_embeds, **kwargs)

        # Sync to ensure all warmup operations complete
        torch.cuda.synchronize(self.device)

    def _eager_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Run model forward pass in eager mode (no graph)."""
        return self.model(
            latents,
            timestep=timestep,
            context=prompt_embeds.get('context'),
            **kwargs
        )

    def warmup_and_capture(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: Dict[str, torch.Tensor],
        **kwargs
    ) -> None:
        """
        Warmup and capture a CUDA graph for the given input configuration.

        Args:
            latents: Sample latent tensor [batch, channels, frames, H, W]
            timestep: Diffusion timestep
            prompt_embeds: Dict with prompt embeddings
            **kwargs: Additional model arguments
        """
        graph_key = self._get_graph_key(latents.shape, timestep.item() if torch.is_tensor(timestep) else timestep)

        if graph_key in self._graph_pool:
            # Already captured for this configuration
            return

        print(f"Capturing CUDA graph for key: {graph_key}")

        # Warmup phase
        if not self._warmup_complete:
            print(f"  Running {self.config.warmup_steps} warmup iterations...")
            self._warmup(latents, timestep, prompt_embeds, **kwargs)
            self._warmup_complete = True

        # Create static input copies (graph will read from these)
        static_inputs = {
            'latents': latents.clone(),
            'timestep': timestep.clone() if torch.is_tensor(timestep) else torch.tensor([timestep], device=self.device),
        }

        # Clone prompt embeddings
        for key, value in prompt_embeds.items():
            if isinstance(value, torch.Tensor):
                static_inputs[f'prompt_{key}'] = value.clone()

        # Clone any additional kwargs that are tensors
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                static_inputs[key] = value.clone()

        # Create the CUDA graph
        graph = torch.cuda.CUDAGraph()

        # Capture
        with torch.cuda.graph(graph, stream=self._graph_stream):
            with torch.no_grad():
                # Build prompt_embeds dict from static inputs
                static_prompt_embeds = {}
                for key in prompt_embeds.keys():
                    static_key = f'prompt_{key}'
                    if static_key in static_inputs:
                        static_prompt_embeds[key] = static_inputs[static_key]

                # Build kwargs from static inputs
                static_kwargs = {}
                for key in kwargs.keys():
                    if key in static_inputs:
                        static_kwargs[key] = static_inputs[key]

                static_outputs = self._eager_forward(
                    static_inputs['latents'],
                    static_inputs['timestep'],
                    static_prompt_embeds,
                    **static_kwargs
                )

        # Store in pool
        self._graph_pool[graph_key] = (graph, static_inputs, static_outputs)

        # Enforce pool size limit (LRU eviction)
        if len(self._graph_pool) > self.config.pool_size:
            # Remove oldest entry
            oldest_key = next(iter(self._graph_pool))
            del self._graph_pool[oldest_key]

        print(f"  Graph captured successfully. Pool size: {len(self._graph_pool)}")

    def replay(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Replay captured CUDA graph with new inputs.

        Args:
            latents: Input latent tensor
            timestep: Diffusion timestep
            prompt_embeds: Prompt embeddings
            **kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        timestep_val = timestep.item() if torch.is_tensor(timestep) else timestep
        graph_key = self._get_graph_key(latents.shape, timestep_val)

        if graph_key not in self._graph_pool:
            # Graph not captured for this config - capture now or fallback
            if len(self._graph_pool) < self.config.pool_size:
                # Capture new graph
                self.warmup_and_capture(latents, timestep, prompt_embeds, **kwargs)
            else:
                # Pool full, use eager mode
                return self._eager_forward(latents, timestep, prompt_embeds, **kwargs)

        graph, static_inputs, static_outputs = self._graph_pool[graph_key]

        # Copy new inputs to static buffers (in-place for graph compatibility)
        static_inputs['latents'].copy_(latents)
        if torch.is_tensor(timestep):
            static_inputs['timestep'].copy_(timestep)

        # Copy prompt embeddings
        for key, value in prompt_embeds.items():
            static_key = f'prompt_{key}'
            if static_key in static_inputs and isinstance(value, torch.Tensor):
                static_inputs[static_key].copy_(value)

        # Copy additional kwargs
        for key, value in kwargs.items():
            if key in static_inputs and isinstance(value, torch.Tensor):
                static_inputs[key].copy_(value)

        # Replay the graph
        graph.replay()

        # Return a clone of outputs (graph output buffer is reused)
        return static_outputs.clone()

    def has_graph(self, latents_shape: Tuple, timestep: int) -> bool:
        """Check if a graph exists for the given configuration."""
        graph_key = self._get_graph_key(latents_shape, timestep)
        return graph_key in self._graph_pool

    def clear_graphs(self):
        """Clear all captured graphs."""
        self._graph_pool.clear()
        self._warmup_complete = False

    def get_pool_size(self) -> int:
        """Get current graph pool size."""
        return len(self._graph_pool)

    def __repr__(self) -> str:
        return (
            f"CUDAGraphWrapper(\n"
            f"  pool_size={len(self._graph_pool)}/{self.config.pool_size},\n"
            f"  warmup_steps={self.config.warmup_steps},\n"
            f"  warmup_complete={self._warmup_complete}\n"
            f")"
        )


class MultiStepCUDAGraphWrapper:
    """
    CUDA graph wrapper for entire denoising loop (multiple timesteps).

    Captures all denoising steps as a single graph for maximum efficiency.
    This is more restrictive (fixed timestep schedule) but provides
    the best performance.

    Usage:
        wrapper = MultiStepCUDAGraphWrapper(model, timesteps=[1000, 750, 500, 250])
        wrapper.warmup_and_capture(sample_latents, prompt_embeds)

        # Replay full denoising loop
        output = wrapper.replay(noise, prompt_embeds)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        timesteps: List[int],
        config: GraphCaptureConfig = None,
        device: torch.device = None,
    ):
        """
        Initialize multi-step graph wrapper.

        Args:
            model: The diffusion model
            timesteps: List of timesteps for denoising schedule
            config: Graph capture configuration
            device: CUDA device
        """
        self.model = model
        self.timesteps = timesteps
        self.config = config or GraphCaptureConfig()
        self.device = device or torch.device('cuda')

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Dict[str, torch.Tensor] = {}
        self._static_outputs: Optional[torch.Tensor] = None
        self._captured = False

        self._graph_stream = torch.cuda.Stream(device=self.device)

    def warmup_and_capture(
        self,
        latents: torch.Tensor,
        prompt_embeds: Dict[str, torch.Tensor],
        **kwargs
    ) -> None:
        """Warmup and capture the full denoising loop as a graph."""
        if self._captured:
            return

        print(f"Capturing multi-step CUDA graph with {len(self.timesteps)} steps...")

        # Create timestep tensors
        timestep_tensors = [
            torch.tensor([t], device=self.device, dtype=torch.long)
            for t in self.timesteps
        ]

        # Warmup
        print(f"  Running {self.config.warmup_steps} warmup iterations...")
        for _ in range(self.config.warmup_steps):
            x = latents.clone()
            for t in timestep_tensors:
                with torch.no_grad():
                    x = self.model(x, timestep=t, context=prompt_embeds.get('context'), **kwargs)
        torch.cuda.synchronize(self.device)

        # Create static inputs
        self._static_inputs = {
            'latents': latents.clone(),
        }
        for key, value in prompt_embeds.items():
            if isinstance(value, torch.Tensor):
                self._static_inputs[f'prompt_{key}'] = value.clone()
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                self._static_inputs[key] = value.clone()

        # Create static intermediate buffers
        static_x = self._static_inputs['latents']

        # Capture graph
        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._graph, stream=self._graph_stream):
            with torch.no_grad():
                static_prompt = {}
                for key in prompt_embeds.keys():
                    static_key = f'prompt_{key}'
                    if static_key in self._static_inputs:
                        static_prompt[key] = self._static_inputs[static_key]

                static_kwargs = {}
                for key in kwargs.keys():
                    if key in self._static_inputs:
                        static_kwargs[key] = self._static_inputs[key]

                for t in timestep_tensors:
                    static_x = self.model(
                        static_x,
                        timestep=t,
                        context=static_prompt.get('context'),
                        **static_kwargs
                    )

                self._static_outputs = static_x

        self._captured = True
        print("  Multi-step graph captured successfully")

    def replay(
        self,
        latents: torch.Tensor,
        prompt_embeds: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Replay the captured multi-step graph."""
        if not self._captured:
            raise RuntimeError("Graph not captured. Call warmup_and_capture first.")

        # Copy inputs
        self._static_inputs['latents'].copy_(latents)
        for key, value in prompt_embeds.items():
            static_key = f'prompt_{key}'
            if static_key in self._static_inputs and isinstance(value, torch.Tensor):
                self._static_inputs[static_key].copy_(value)
        for key, value in kwargs.items():
            if key in self._static_inputs and isinstance(value, torch.Tensor):
                self._static_inputs[key].copy_(value)

        # Replay
        self._graph.replay()

        return self._static_outputs.clone()

    @property
    def is_captured(self) -> bool:
        return self._captured

    def clear(self):
        """Clear captured graph."""
        self._graph = None
        self._static_inputs.clear()
        self._static_outputs = None
        self._captured = False


def is_cuda_graph_compatible(model: torch.nn.Module) -> bool:
    """
    Check if a model is compatible with CUDA graphs.

    Incompatible operations:
    - Dynamic control flow based on tensor values
    - Host-device synchronization (.item(), .cpu())
    - Dynamic tensor shapes
    - Non-deterministic operations without seeding
    """
    # This is a basic check - full validation requires trying to capture
    if not torch.cuda.is_available():
        return False

    # Check for known incompatible layers
    for module in model.modules():
        module_name = type(module).__name__

        # Dropout with training=True is problematic
        if isinstance(module, torch.nn.Dropout) and module.training:
            return False

    return True


def create_cuda_graph_wrapper(
    model: torch.nn.Module,
    timesteps: Optional[List[int]] = None,
    config: Optional[GraphCaptureConfig] = None,
) -> CUDAGraphWrapper:
    """
    Factory function to create appropriate CUDA graph wrapper.

    Args:
        model: The diffusion model
        timesteps: If provided, creates MultiStepCUDAGraphWrapper
        config: Graph capture configuration

    Returns:
        CUDAGraphWrapper or MultiStepCUDAGraphWrapper
    """
    if timesteps is not None:
        return MultiStepCUDAGraphWrapper(model, timesteps, config)
    return CUDAGraphWrapper(model, config)
