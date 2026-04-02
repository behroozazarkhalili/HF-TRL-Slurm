"""
Pytest Configuration and Shared Fixtures.

This module provides common fixtures for testing the slurm-model-trainer skill:
- JobConfig instances with various configurations
- Mock models and tokenizers
- Sample datasets for different training methods
- Temporary directories for outputs
"""

import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import pytest


# Add project directories to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "generator"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# =============================================================================
# JobConfig Fixtures
# =============================================================================

@pytest.fixture
def minimal_sft_config() -> Dict[str, Any]:
    """Minimal configuration for SFT training."""
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_id": "trl-lib/Capybara",
        "method": "sft",
        "job_name": "test-sft-job",
        "hub_model_id": "testuser/test-sft-model",
        "gguf_repo_id": "testuser/test-sft-model-GGUF",
        "time_limit": "02:00:00",
        "partition": "gpubase_bygpu_b5",
        "cpus": 4,
        "memory": "32G",
        "batch_size": 4,
        "grad_accum": 4,
        "lr": 2e-5,
        "num_train_epochs": 1,
        "max_length": 1024,
    }


@pytest.fixture
def minimal_dpo_config() -> Dict[str, Any]:
    """Minimal configuration for DPO training."""
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_id": "trl-lib/ultrafeedback_binarized",
        "method": "dpo",
        "job_name": "test-dpo-job",
        "hub_model_id": "testuser/test-dpo-model",
        "gguf_repo_id": "testuser/test-dpo-model-GGUF",
        "time_limit": "04:00:00",
        "partition": "gpubase_bygpu_b5",
        "cpus": 4,
        "memory": "32G",
        "batch_size": 2,
        "grad_accum": 8,
        "lr": 5e-7,
        "num_train_epochs": 1,
        "max_length": 1024,
    }


@pytest.fixture
def minimal_grpo_config() -> Dict[str, Any]:
    """Minimal configuration for GRPO training."""
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_id": "nvidia/OpenMathInstruct-2",
        "method": "grpo",
        "job_name": "test-grpo-job",
        "hub_model_id": "testuser/test-grpo-model",
        "gguf_repo_id": "testuser/test-grpo-model-GGUF",
        "time_limit": "08:00:00",
        "partition": "gpubase_bygpu_b5",
        "cpus": 4,
        "memory": "64G",
        "batch_size": 2,
        "grad_accum": 8,
        "lr": 1e-6,
        "num_train_epochs": 1,
        "max_length": 512,
        "num_gen": 4,
        "reward_type": "math",
    }


@pytest.fixture
def large_model_config() -> Dict[str, Any]:
    """Configuration for a large model (14B parameters)."""
    return {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "dataset_id": "trl-lib/Capybara",
        "method": "sft",
        "job_name": "test-large-job",
        "hub_model_id": "testuser/test-large-model",
        "gguf_repo_id": "testuser/test-large-model-GGUF",
        "params_b": 14.0,
        "size_category": "large",
        "time_limit": "24:00:00",
        "partition": "gpubase_bygpu_b5",
        "gres": "gpu:h100:4",
        "cpus": 16,
        "memory": "256G",
        "batch_size": 1,
        "grad_accum": 32,
        "lr": 1e-5,
        "num_train_epochs": 1,
        "max_length": 2048,
        "requires_4bit": True,
    }


# =============================================================================
# JobConfig Object Fixtures
# =============================================================================

@pytest.fixture
def job_config_sft(minimal_sft_config):
    """Create a JobConfig instance for SFT."""
    from config import JobConfig
    return JobConfig(**minimal_sft_config)


@pytest.fixture
def job_config_dpo(minimal_dpo_config):
    """Create a JobConfig instance for DPO."""
    from config import JobConfig
    return JobConfig(**minimal_dpo_config)


@pytest.fixture
def job_config_grpo(minimal_grpo_config):
    """Create a JobConfig instance for GRPO."""
    from config import JobConfig
    return JobConfig(**minimal_grpo_config)


# =============================================================================
# Mock Model and Tokenizer Fixtures
# =============================================================================

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.model_max_length = 2048
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
    )
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.pad_token_id = 0
    model.config.vocab_size = 32000
    model.config.hidden_size = 2048
    model.config.num_hidden_layers = 24
    return model


# =============================================================================
# Sample Dataset Fixtures
# =============================================================================

@pytest.fixture
def sample_sft_data():
    """Sample data for SFT training (conversational format)."""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain gravity."},
                {"role": "assistant", "content": "Gravity is the force of attraction between objects."},
            ]
        },
    ]


@pytest.fixture
def sample_dpo_data():
    """Sample data for DPO training."""
    return [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "I'm not sure.",
        },
        {
            "prompt": "Explain machine learning briefly.",
            "chosen": "Machine learning is a subset of AI that enables systems to learn from data.",
            "rejected": "ML is computers doing stuff.",
        },
    ]


@pytest.fixture
def sample_grpo_data():
    """Sample data for GRPO training (math problems)."""
    return [
        {
            "prompt": "Calculate 15 + 27.",
            "expected_answer": "42",
        },
        {
            "prompt": "If x = 5, what is 2x + 3?",
            "expected_answer": "13",
        },
        {
            "prompt": "What is the square root of 144?",
            "expected_answer": "12",
        },
    ]


# =============================================================================
# Reward Function Test Fixtures
# =============================================================================

@pytest.fixture
def sample_math_completions():
    """Sample completions for testing math reward function."""
    return [
        # Correct with boxed answer
        "Let me solve this step by step.\n15 + 27 = 42\nThe answer is \\boxed{42}",
        # Correct without boxed
        "The answer is 42.",
        # Incorrect answer
        "I think the answer is 41.",
        # No answer extracted
        "This is a complex problem that requires careful analysis...",
    ]


@pytest.fixture
def sample_math_prompts():
    """Sample prompts for testing math reward function."""
    return [
        "Calculate 15 + 27.",
        "Calculate 15 + 27.",
        "Calculate 15 + 27.",
        "Calculate 15 + 27.",
    ]


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth for testing math reward function."""
    return {
        hash("Calculate 15 + 27."): "42",
        hash("If x = 5, what is 2x + 3?"): "13",
    }


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory with sample YAML files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create training subdirectory
    training_dir = config_dir / "training"
    training_dir.mkdir()

    # Create sample SFT config
    sft_yaml = training_dir / "sft.yaml"
    sft_yaml.write_text("""
learning_rate: 2.0e-5
num_epochs: 1
batch_size: 4
gradient_accumulation_steps: 4
max_seq_length: 1024
warmup_ratio: 0.1
""")

    # Create sample DPO config
    dpo_yaml = training_dir / "dpo.yaml"
    dpo_yaml.write_text("""
learning_rate: 5.0e-7
num_epochs: 1
batch_size: 2
gradient_accumulation_steps: 8
max_seq_length: 1024
beta: 0.1
loss_type: sigmoid
""")

    return config_dir


# =============================================================================
# CLI Testing Fixtures
# =============================================================================

@pytest.fixture
def cli_runner():
    """Create a CLI test runner (using click if available, else subprocess)."""
    try:
        from click.testing import CliRunner
        return CliRunner()
    except ImportError:
        return None


# =============================================================================
# Integration Test Helpers
# =============================================================================

@pytest.fixture
def valid_sbatch_markers():
    """Expected markers in valid SBATCH script."""
    return [
        "#!/bin/bash",
        "#SBATCH --job-name=",
        "#SBATCH --time=",
        "#SBATCH --gres=gpu",  # SLURM uses --gres for GPU allocation
        "module load",
        "source",
        "python",
    ]


@pytest.fixture
def expected_training_args():
    """Expected arguments in training command."""
    return [
        "--model_name_or_path",
        "--dataset_name",
        "--output_dir",
        "--per_device_train_batch_size",
        "--learning_rate",
    ]
