"""
Unit tests for JobConfig dataclass.

Tests validation, defaults, and edge cases for job configuration.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from generator.config import JobConfig


class TestJobConfigCreation:
    """Test JobConfig instantiation and validation."""

    def test_minimal_sft_config(self, minimal_sft_config):
        """Test creating SFT config with minimal required fields."""
        config = JobConfig(**minimal_sft_config)

        assert config.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.method == "sft"
        # Uses gres for GPU allocation, not gpus
        assert "gpu" in config.gres

    def test_minimal_dpo_config(self, minimal_dpo_config):
        """Test creating DPO config with minimal required fields."""
        config = JobConfig(**minimal_dpo_config)

        assert config.method == "dpo"
        # DPO-specific params use trainer defaults, not in config
        assert config.model_id is not None

    def test_minimal_grpo_config(self, minimal_grpo_config):
        """Test creating GRPO config with minimal required fields."""
        config = JobConfig(**minimal_grpo_config)

        assert config.method == "grpo"
        assert config.reward_type == "math"
        assert config.num_gen == 4  # num_gen, not num_generations

    def test_invalid_method_is_accepted_by_dataclass(self, minimal_sft_config):
        """Test that invalid method doesn't raise at dataclass level.

        Note: Python dataclasses with Literal type hints don't enforce
        the type at runtime. Validation would need to be in __post_init__.
        """
        minimal_sft_config["method"] = "invalid_method"

        # Currently JobConfig doesn't validate method enum at runtime
        # This is a known limitation - could be enhanced in future
        config = JobConfig(**minimal_sft_config)
        assert config.method == "invalid_method"

    def test_zero_cpus_raises_error(self, minimal_sft_config):
        """Test that zero CPUs raises ValueError."""
        minimal_sft_config["cpus"] = 0

        with pytest.raises(ValueError):
            JobConfig(**minimal_sft_config)

    def test_negative_batch_size_raises_error(self, minimal_sft_config):
        """Test that negative batch_size raises ValueError."""
        minimal_sft_config["batch_size"] = -1

        with pytest.raises(ValueError):
            JobConfig(**minimal_sft_config)

    def test_grpo_without_reward_type_uses_default(self, minimal_grpo_config):
        """Test that GRPO without reward_type uses default 'combined'."""
        # Remove reward_type to use default
        del minimal_grpo_config["reward_type"]

        config = JobConfig(**minimal_grpo_config)
        assert config.reward_type == "combined"


class TestJobConfigDefaults:
    """Test JobConfig default values."""

    def test_default_lora_r(self, minimal_sft_config):
        """Test default LoRA rank."""
        # Remove lora_r if present to test default
        minimal_sft_config.pop("lora_r", None)
        config = JobConfig(**minimal_sft_config)

        assert config.lora_r == 32  # Default is 32, not 16

    def test_default_lora_alpha(self, minimal_sft_config):
        """Test default LoRA alpha."""
        # Remove lora_alpha if present to test default
        minimal_sft_config.pop("lora_alpha", None)
        config = JobConfig(**minimal_sft_config)

        assert config.lora_alpha == 64  # Default is 64, not 16

    def test_default_streaming(self, minimal_sft_config):
        """Test default streaming setting."""
        config = JobConfig(**minimal_sft_config)

        assert config.streaming is False

    def test_default_bf16(self, minimal_sft_config):
        """Test default bf16 setting."""
        config = JobConfig(**minimal_sft_config)

        assert config.bf16 is True

    def test_default_gradient_checkpointing(self, minimal_sft_config):
        """Test default gradient_checkpointing setting."""
        config = JobConfig(**minimal_sft_config)

        assert config.gradient_checkpointing is True


class TestJobConfigHubSettings:
    """Test Hub-related configuration."""

    def test_hub_model_id_required(self, minimal_sft_config):
        """Test that hub_model_id is required."""
        config = JobConfig(**minimal_sft_config)

        # hub_model_id is a required field
        assert config.hub_model_id == "testuser/test-sft-model"

    def test_hub_model_id_custom(self, minimal_sft_config):
        """Test setting custom hub_model_id."""
        minimal_sft_config["hub_model_id"] = "username/my-model"
        config = JobConfig(**minimal_sft_config)

        assert config.hub_model_id == "username/my-model"

    def test_gguf_repo_id_required(self, minimal_sft_config):
        """Test that gguf_repo_id is required."""
        config = JobConfig(**minimal_sft_config)

        assert config.gguf_repo_id == "testuser/test-sft-model-GGUF"


class TestJobConfigEquality:
    """Test JobConfig equality and hashing."""

    def test_equal_configs(self, minimal_sft_config):
        """Test that identical configs are equal."""
        config1 = JobConfig(**minimal_sft_config)
        config2 = JobConfig(**minimal_sft_config)

        assert config1 == config2

    def test_different_configs_not_equal(self, minimal_sft_config, minimal_dpo_config):
        """Test that different configs are not equal."""
        config1 = JobConfig(**minimal_sft_config)
        config2 = JobConfig(**minimal_dpo_config)

        assert config1 != config2


class TestJobConfigProperties:
    """Test computed properties of JobConfig."""

    def test_effective_batch_size(self, minimal_sft_config):
        """Test effective batch size calculation."""
        config = JobConfig(**minimal_sft_config)

        # batch_size=4, grad_accum=4 -> effective=16
        assert config.effective_batch_size == 16

    def test_model_short_name(self, minimal_sft_config):
        """Test model short name extraction."""
        config = JobConfig(**minimal_sft_config)

        assert config.model_short_name == "Qwen2.5-0.5B-Instruct"

    def test_dataset_short_name(self, minimal_sft_config):
        """Test dataset short name extraction."""
        config = JobConfig(**minimal_sft_config)

        assert config.dataset_short_name == "Capybara"

    def test_is_grpo_false_for_sft(self, minimal_sft_config):
        """Test is_grpo returns False for SFT."""
        config = JobConfig(**minimal_sft_config)

        assert config.is_grpo is False

    def test_is_grpo_true_for_grpo(self, minimal_grpo_config):
        """Test is_grpo returns True for GRPO."""
        config = JobConfig(**minimal_grpo_config)

        assert config.is_grpo is True


class TestJobConfigToDict:
    """Test JobConfig serialization."""

    def test_to_dict(self, minimal_sft_config):
        """Test conversion to dictionary."""
        config = JobConfig(**minimal_sft_config)
        d = config.to_dict()

        assert d["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert d["method"] == "sft"
        assert "gres" in d


class TestJobConfigFromDict:
    """Test JobConfig.from_dict class method."""

    def test_from_dict(self, minimal_sft_config):
        """Test creation from dictionary."""
        config = JobConfig.from_dict(minimal_sft_config)

        assert config.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.method == "sft"

    def test_from_dict_with_old_key_names(self, minimal_sft_config):
        """Test that old key names are mapped correctly."""
        minimal_sft_config["learning_rate"] = 1e-5
        minimal_sft_config["gradient_accumulation_steps"] = 8

        config = JobConfig.from_dict(minimal_sft_config)

        assert config.lr == 1e-5
        assert config.grad_accum == 8
