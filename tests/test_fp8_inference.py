"""
Tests for FP8 Inference Support

Tests FP8 availability detection and conversion utilities.
"""

import pytest
import torch
import torch.nn as nn

from optimizations.fp8_inference import (
    is_fp8_available,
    get_fp8_dtype,
    FP8LinearWrapper,
    FP8InferenceWrapper,
    maybe_convert_to_fp8,
)
from optimizations.config import OptimizationConfig


class TestFP8Availability:
    """Tests for FP8 availability detection."""

    def test_is_fp8_available_returns_bool(self):
        """Test that is_fp8_available returns a boolean."""
        result = is_fp8_available()
        assert isinstance(result, bool)

    def test_get_fp8_dtype(self):
        """Test FP8 dtype retrieval."""
        dtype = get_fp8_dtype()
        # Either None (no FP8 support) or a valid dtype
        if dtype is not None:
            assert hasattr(torch, 'float8_e4m3fn')
            assert dtype == torch.float8_e4m3fn


class TestFP8LinearWrapper:
    """Tests for FP8LinearWrapper."""

    def test_wraps_linear_layer(self):
        """Test that wrapper preserves linear layer properties."""
        linear = nn.Linear(64, 32)
        wrapped = FP8LinearWrapper(linear)

        assert wrapped.in_features == 64
        assert wrapped.out_features == 32
        assert wrapped.has_bias is True

    def test_forward_produces_correct_shape(self):
        """Test that forward pass produces correct output shape."""
        linear = nn.Linear(64, 32)
        wrapped = FP8LinearWrapper(linear)

        x = torch.randn(4, 64)
        output = wrapped(x)

        assert output.shape == (4, 32)

    def test_forward_without_bias(self):
        """Test wrapper with no bias."""
        linear = nn.Linear(64, 32, bias=False)
        wrapped = FP8LinearWrapper(linear)

        assert wrapped.has_bias is False

        x = torch.randn(4, 64)
        output = wrapped(x)
        assert output.shape == (4, 32)


class TestFP8InferenceWrapper:
    """Tests for FP8InferenceWrapper."""

    def test_wrap_model_returns_model(self):
        """Test that wrap_model returns a model."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        wrapped = FP8InferenceWrapper.wrap_model(model, verbose=False)
        assert wrapped is not None

    def test_estimate_speedup_returns_dict(self):
        """Test speedup estimation returns dictionary."""
        info = FP8InferenceWrapper.estimate_speedup()

        assert isinstance(info, dict)
        assert 'fp8_available' in info
        assert 'estimated_speedup' in info
        assert 'hardware' in info
        assert 'recommendation' in info


class TestMaybeConvertToFP8:
    """Tests for maybe_convert_to_fp8 function."""

    def test_no_conversion_when_bfloat16(self):
        """Test that no conversion happens with bfloat16 config."""
        model = nn.Linear(64, 32)
        config = OptimizationConfig(model_dtype="bfloat16")

        result = maybe_convert_to_fp8(model, config)
        # Should return same model (no conversion)
        assert result is model

    def test_attempts_conversion_when_fp8(self):
        """Test that conversion is attempted with fp8 config."""
        model = nn.Linear(64, 32)
        config = OptimizationConfig(model_dtype="fp8", verbose=False)

        result = maybe_convert_to_fp8(model, config)
        # Result should be a model (either converted or original)
        assert result is not None


class TestConfigFP8Methods:
    """Tests for FP8-related config methods."""

    def test_get_fp8_dtype_when_not_fp8(self):
        """Test get_fp8_dtype returns None when not using FP8."""
        config = OptimizationConfig(model_dtype="bfloat16")
        assert config.get_fp8_dtype() is None

    def test_get_fp8_dtype_when_fp8(self):
        """Test get_fp8_dtype when using FP8."""
        config = OptimizationConfig(model_dtype="fp8")
        dtype = config.get_fp8_dtype()
        # Either None (no support) or float8_e4m3fn
        if hasattr(torch, 'float8_e4m3fn'):
            assert dtype == torch.float8_e4m3fn
        else:
            assert dtype is None

    def test_is_fp8_available_method(self):
        """Test config's is_fp8_available method."""
        config = OptimizationConfig()
        result = config.is_fp8_available()
        assert isinstance(result, bool)
