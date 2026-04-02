"""
Unit tests for TrainingBlockBuilder.

Tests training command generation for SFT, DPO, and GRPO methods.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from generator.config import JobConfig
from generator.builders.training_block import TrainingBlockBuilder


class TestTrainingBlockBuilder:
    """Test TrainingBlockBuilder functionality."""

    def test_sft_training_script(self, job_config_sft):
        """Test SFT training uses correct script."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "train_sft" in output

    def test_dpo_training_script(self, job_config_dpo):
        """Test DPO training uses correct script."""
        builder = TrainingBlockBuilder(job_config_dpo)
        output = builder.build()

        assert "train_dpo" in output

    def test_grpo_training_script(self, job_config_grpo):
        """Test GRPO training uses correct script."""
        builder = TrainingBlockBuilder(job_config_grpo)
        output = builder.build()

        assert "train_grpo" in output


class TestTrainingArguments:
    """Test training argument generation."""

    def test_contains_model_path(self, job_config_sft):
        """Test that output contains model path argument."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--model_name_or_path" in output
        # Uses shell variable reference, not actual value
        assert "$MODEL_NAME" in output

    def test_contains_dataset_name(self, job_config_sft):
        """Test that output contains dataset argument."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--dataset_name" in output
        # Uses shell variable reference
        assert "$DATASET_NAME" in output

    def test_contains_batch_size(self, job_config_sft):
        """Test that output contains batch size argument."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--per_device_train_batch_size" in output

    def test_contains_learning_rate(self, job_config_sft):
        """Test that output contains learning rate argument."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--learning_rate" in output

    def test_contains_output_dir(self, job_config_sft):
        """Test that output contains output directory argument."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--output_dir" in output


class TestMethodSpecificArguments:
    """Test method-specific argument generation."""

    def test_sft_contains_max_length(self, job_config_sft):
        """Test SFT includes max_length argument."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--max_length" in output

    def test_dpo_contains_max_length(self, job_config_dpo):
        """Test DPO includes max_length argument."""
        builder = TrainingBlockBuilder(job_config_dpo)
        output = builder.build()

        assert "--max_length" in output

    def test_grpo_contains_num_generations(self, job_config_grpo):
        """Test GRPO includes num_generations argument."""
        builder = TrainingBlockBuilder(job_config_grpo)
        output = builder.build()

        assert "--num_generations" in output

    def test_grpo_contains_reward_type(self, job_config_grpo):
        """Test GRPO includes reward_type argument."""
        builder = TrainingBlockBuilder(job_config_grpo)
        output = builder.build()

        assert "--reward_type" in output

    def test_grpo_contains_max_completion_length(self, job_config_grpo):
        """Test GRPO includes max_completion_length argument."""
        builder = TrainingBlockBuilder(job_config_grpo)
        output = builder.build()

        assert "--max_completion_length" in output


class TestLoRAArguments:
    """Test LoRA-specific arguments."""

    def test_contains_lora_r(self, job_config_sft):
        """Test that output contains LoRA rank."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--lora_r" in output

    def test_contains_lora_alpha(self, job_config_sft):
        """Test that output contains LoRA alpha."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--lora_alpha" in output

    def test_contains_lora_dropout(self, job_config_sft):
        """Test that output contains LoRA dropout."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "--lora_dropout" in output


class TestPrecisionArguments:
    """Test precision-related arguments."""

    def test_bf16_argument(self, minimal_sft_config):
        """Test bf16 precision argument."""
        minimal_sft_config["bf16"] = True
        config = JobConfig(**minimal_sft_config)
        builder = TrainingBlockBuilder(config)
        output = builder.build()

        assert "--bf16" in output

    def test_gradient_checkpointing(self, minimal_sft_config):
        """Test gradient checkpointing argument."""
        minimal_sft_config["gradient_checkpointing"] = True
        config = JobConfig(**minimal_sft_config)
        builder = TrainingBlockBuilder(config)
        output = builder.build()

        assert "--gradient_checkpointing" in output


class TestHubArguments:
    """Test Hub push arguments."""

    def test_push_to_hub_flag(self, minimal_sft_config):
        """Test push_to_hub flag."""
        config = JobConfig(**minimal_sft_config)
        builder = TrainingBlockBuilder(config)
        output = builder.build()

        assert "--push_to_hub" in output

    def test_hub_model_id_argument(self, minimal_sft_config):
        """Test hub_model_id argument."""
        config = JobConfig(**minimal_sft_config)
        builder = TrainingBlockBuilder(config)
        output = builder.build()

        assert "--hub_model_id" in output
        # Uses shell variable reference
        assert "$HUB_MODEL_ID" in output


class TestPhaseHeader:
    """Test phase header generation."""

    def test_contains_phase_header(self, job_config_sft):
        """Test that output contains phase header."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "Phase 1" in output
        assert "Training" in output


class TestErrorHandling:
    """Test error handling code generation."""

    def test_contains_exit_code_check(self, job_config_sft):
        """Test that output contains exit code check."""
        builder = TrainingBlockBuilder(job_config_sft)
        output = builder.build()

        assert "TRAIN_EXIT_CODE" in output
        assert "exit" in output
