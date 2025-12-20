"""
Tests for OptimizationConfig

Tests configuration loading, presets, and validation.
"""

import pytest
import tempfile
import os

from optimizations.config import OptimizationConfig


class TestOptimizationConfig:
    """Tests for OptimizationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.enabled is True
        assert config.use_cuda_graphs is True
        assert config.use_static_kv is True
        assert config.use_quantized_kv is False
        assert config.use_async_vae is True
        assert config.use_prompt_cache is True
        assert config.local_attn_size == 12
        assert config.sink_size == 3

    def test_preset_quality(self):
        """Test quality preset configuration."""
        config = OptimizationConfig.preset_quality()

        assert config.use_cuda_graphs is False
        assert config.use_static_kv is True
        assert config.use_quantized_kv is False
        assert config.model_dtype == "bfloat16"

    def test_preset_balanced(self):
        """Test balanced preset configuration."""
        config = OptimizationConfig.preset_balanced()

        # Balanced preset uses torch.compile instead of CUDA graphs
        # (torch.compile handles dynamic KV cache shapes better)
        assert config.use_cuda_graphs is False
        assert config.use_torch_compile is True
        assert config.use_static_kv is True
        assert config.use_quantized_kv is False
        assert config.use_async_vae is True

    def test_preset_speed(self):
        """Test speed preset configuration."""
        config = OptimizationConfig.preset_speed()

        # Speed preset now matches balanced (INT8 was found to add overhead)
        assert config.use_cuda_graphs is False
        assert config.use_torch_compile is True
        assert config.use_quantized_kv is False  # INT8 disabled
        assert config.use_static_kv is True

    def test_preset_turbo(self):
        """Test turbo preset configuration."""
        config = OptimizationConfig.preset_turbo()

        assert config.use_torch_compile is True
        assert config.compile_mode == "max-autotune"
        assert config.denoising_steps == [1000, 500, 250]  # 3 steps
        assert config.local_attn_size == 8  # Smaller window
        assert config.verbose is False

    def test_preset_turbo_fp8(self):
        """Test turbo FP8 preset configuration."""
        config = OptimizationConfig.preset_turbo_fp8()

        assert config.model_dtype == "fp8"
        assert config.compile_mode == "max-autotune"
        assert config.denoising_steps == [1000, 500, 250]  # 3 steps

    def test_preset_ultra(self):
        """Test ultra preset configuration."""
        config = OptimizationConfig.preset_ultra()

        assert config.model_dtype == "fp8"
        assert config.denoising_steps == [1000, 250]  # 2 steps
        assert config.local_attn_size == 6  # Minimum window

    def test_preset_low_memory(self):
        """Test low memory preset configuration."""
        config = OptimizationConfig.preset_low_memory()

        assert config.use_quantized_kv is True
        assert config.kv_quantization == "int8"
        assert config.use_static_kv is False  # Quantized takes precedence

    def test_mutual_exclusion_kv_cache(self):
        """Test that static KV and quantized KV are mutually exclusive."""
        config = OptimizationConfig(
            use_static_kv=True,
            use_quantized_kv=True,
        )
        # Quantized takes precedence
        assert config.use_quantized_kv is True
        assert config.use_static_kv is False

    def test_mutual_exclusion_cuda_graphs_compile(self):
        """Test that CUDA graphs and torch.compile are mutually exclusive."""
        with pytest.raises(ValueError):
            OptimizationConfig(
                use_cuda_graphs=True,
                use_torch_compile=True,
            )

    def test_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
enabled: true
use_cuda_graphs: false
use_static_kv: true
local_attn_size: 8
sink_size: 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = OptimizationConfig.from_yaml(f.name)
                assert config.use_cuda_graphs is False
                assert config.use_static_kv is True
                assert config.local_attn_size == 8
                assert config.sink_size == 2
            finally:
                os.unlink(f.name)

    def test_to_yaml(self):
        """Test saving config to YAML file."""
        config = OptimizationConfig(
            use_cuda_graphs=False,
            local_attn_size=16,
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                config.to_yaml(f.name)
                loaded = OptimizationConfig.from_yaml(f.name)
                assert loaded.use_cuda_graphs is False
                assert loaded.local_attn_size == 16
            finally:
                os.unlink(f.name)

    def test_denoising_steps_default(self):
        """Test default denoising steps."""
        config = OptimizationConfig()
        assert config.denoising_steps == [1000, 750, 500, 250]

    def test_custom_denoising_steps(self):
        """Test custom denoising steps."""
        config = OptimizationConfig(
            denoising_steps=[1000, 500]
        )
        assert config.denoising_steps == [1000, 500]

    def test_config_copy(self):
        """Test that config can be copied and modified."""
        config1 = OptimizationConfig.preset_balanced()
        config2 = OptimizationConfig(
            **{k: v for k, v in config1.__dict__.items()}
        )
        # Modify a different attribute since use_cuda_graphs is already False
        config2.use_memory_pool = False

        assert config1.use_memory_pool is True
        assert config2.use_memory_pool is False
