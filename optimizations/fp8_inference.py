"""
FP8 Inference Support for LongLive.

This module provides utilities for FP8 inference on H100/Hopper GPUs.
FP8 (8-bit floating point) provides ~2x speedup over BF16 on H100 tensor cores
with minimal quality degradation.

Two FP8 formats are supported:
- E4M3: 4 exponent bits, 3 mantissa bits - better for weights/activations
- E5M2: 5 exponent bits, 2 mantissa bits - better for gradients

Usage:
    from optimizations.fp8_inference import FP8InferenceWrapper, is_fp8_available

    if is_fp8_available():
        model = FP8InferenceWrapper.wrap_model(model)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings


def is_fp8_available() -> bool:
    """Check if FP8 inference is available on current hardware."""
    if not torch.cuda.is_available():
        return False

    # Check for Hopper architecture (H100, H200)
    try:
        device_name = torch.cuda.get_device_name(0).lower()
        is_hopper = "h100" in device_name or "hopper" in device_name or "h200" in device_name
    except Exception:
        is_hopper = False

    # Check for PyTorch FP8 support (2.1+)
    has_fp8 = hasattr(torch, 'float8_e4m3fn')

    # Check compute capability (>= 9.0 for Hopper)
    try:
        major, minor = torch.cuda.get_device_capability(0)
        has_capability = major >= 9
    except Exception:
        has_capability = False

    return is_hopper and has_fp8 and has_capability


def get_fp8_dtype() -> Optional[torch.dtype]:
    """Get the FP8 dtype if available."""
    if hasattr(torch, 'float8_e4m3fn'):
        return torch.float8_e4m3fn
    return None


class FP8LinearWrapper(nn.Module):
    """
    Wrapper that converts a Linear layer to use FP8 compute.

    This uses torch's native FP8 support where available, or falls back
    to scaled FP8 emulation for compatibility.
    """

    def __init__(self, linear: nn.Linear, scale: float = 1.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None

        # Store original weight dtype for fallback
        self.original_dtype = linear.weight.dtype

        # Compute scale for FP8 quantization
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

        # Convert weight to FP8 if available
        if hasattr(torch, 'float8_e4m3fn'):
            # Quantize weight to FP8
            weight_fp32 = linear.weight.float()
            max_val = weight_fp32.abs().max()
            self.register_buffer('weight_scale', max_val / 448.0)  # E4M3 max is ~448

            # Store quantized weight
            scaled_weight = weight_fp32 / self.weight_scale
            self.register_buffer('weight_fp8', scaled_weight.to(torch.float8_e4m3fn))

            # Keep bias in higher precision
            if self.has_bias:
                self.register_buffer('bias', linear.bias.clone())
            else:
                self.bias = None

            self._use_fp8 = True
        else:
            # Fallback: keep original weight
            self.register_buffer('weight', linear.weight.clone())
            if self.has_bias:
                self.register_buffer('bias', linear.bias.clone())
            else:
                self.bias = None
            self._use_fp8 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fp8 and hasattr(torch, '_scaled_mm'):
            # Use torch's scaled matrix multiplication for FP8
            # This leverages H100's FP8 tensor cores
            try:
                # Quantize input to FP8
                x_fp32 = x.float()
                x_max = x_fp32.abs().max()
                x_scale = x_max / 448.0 if x_max > 0 else torch.tensor(1.0)
                x_fp8 = (x_fp32 / x_scale).to(torch.float8_e4m3fn)

                # Scaled matmul
                output = torch._scaled_mm(
                    x_fp8,
                    self.weight_fp8.t(),
                    scale_a=x_scale,
                    scale_b=self.weight_scale,
                    out_dtype=self.original_dtype
                )

                if self.bias is not None:
                    output = output + self.bias

                return output
            except Exception:
                # Fallback to dequantized computation
                pass

        # Fallback: dequantize and use standard matmul
        if self._use_fp8:
            weight = (self.weight_fp8.float() * self.weight_scale).to(self.original_dtype)
        else:
            weight = self.weight

        output = torch.nn.functional.linear(x, weight, self.bias)
        return output


class FP8InferenceWrapper:
    """
    Utility class to wrap models for FP8 inference.
    """

    @staticmethod
    def wrap_model(model: nn.Module, verbose: bool = True) -> nn.Module:
        """
        Wrap a model's Linear layers for FP8 inference.

        Args:
            model: The model to wrap
            verbose: Whether to print conversion info

        Returns:
            The model with FP8-wrapped Linear layers
        """
        if not is_fp8_available():
            if verbose:
                warnings.warn(
                    "FP8 inference not available on this hardware. "
                    "Falling back to original precision."
                )
            return model

        converted = 0
        total = 0

        def convert_linear(module: nn.Module, name: str = ""):
            nonlocal converted, total

            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                if isinstance(child, nn.Linear):
                    total += 1
                    try:
                        wrapped = FP8LinearWrapper(child)
                        setattr(module, child_name, wrapped)
                        converted += 1
                    except Exception as e:
                        if verbose:
                            warnings.warn(f"Failed to convert {full_name}: {e}")
                else:
                    convert_linear(child, full_name)

        convert_linear(model)

        if verbose:
            print(f"FP8 Inference: Converted {converted}/{total} Linear layers")

        return model

    @staticmethod
    def estimate_speedup() -> Dict[str, Any]:
        """
        Estimate potential FP8 speedup based on hardware.

        Returns:
            Dictionary with speedup estimates and hardware info
        """
        info = {
            'fp8_available': is_fp8_available(),
            'estimated_speedup': '1.0x',
            'hardware': 'unknown',
            'recommendation': 'Use bfloat16'
        }

        if not torch.cuda.is_available():
            return info

        try:
            device_name = torch.cuda.get_device_name(0)
            info['hardware'] = device_name

            major, minor = torch.cuda.get_device_capability(0)

            if major >= 9:  # Hopper
                info['estimated_speedup'] = '1.5-2.0x'
                info['recommendation'] = 'Use FP8 for maximum speed'
            elif major >= 8:  # Ampere
                info['estimated_speedup'] = '1.0x (no FP8 support)'
                info['recommendation'] = 'Use bfloat16 or float16'
            else:
                info['estimated_speedup'] = '1.0x'
                info['recommendation'] = 'Use float16 for best performance'

        except Exception:
            pass

        return info


# Convenience function for quick FP8 check and conversion
def maybe_convert_to_fp8(model: nn.Module, config) -> nn.Module:
    """
    Convert model to FP8 if configured and available.

    Args:
        model: The model to potentially convert
        config: OptimizationConfig with model_dtype setting

    Returns:
        Model (possibly converted to FP8)
    """
    if config.model_dtype == "fp8" and is_fp8_available():
        return FP8InferenceWrapper.wrap_model(model, verbose=config.verbose)
    elif config.model_dtype == "fp8":
        if config.verbose:
            warnings.warn(
                "FP8 requested but not available. Using bfloat16 instead. "
                "FP8 requires H100/Hopper GPU and PyTorch 2.1+."
            )
    return model
