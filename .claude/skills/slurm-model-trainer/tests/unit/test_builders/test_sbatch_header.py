"""
Unit tests for SBatchHeaderBuilder.

Tests SBATCH directive generation for SLURM job scripts.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from generator.config import JobConfig
from generator.builders.sbatch_header import SBatchHeaderBuilder


class TestSBatchHeaderBuilder:
    """Test SBatchHeaderBuilder functionality."""

    def test_build_contains_shebang(self, job_config_sft):
        """Test that output starts with bash shebang."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        assert output.startswith("#!/bin/bash")

    def test_build_contains_job_name(self, job_config_sft):
        """Test that output contains job name directive."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        assert "#SBATCH --job-name=test-sft-job" in output

    def test_build_contains_time_limit(self, job_config_sft):
        """Test that output contains time limit directive."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        assert "#SBATCH --time=02:00:00" in output

    def test_build_contains_partition(self, job_config_sft):
        """Test that output contains partition directive."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        # Check for partition directive (could be gpubase_bygpu_b5 or similar)
        assert "#SBATCH --partition=" in output or "#SBATCH -p" in output

    def test_build_contains_gpu_count(self, job_config_sft):
        """Test that output contains GPU directive (gres format)."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        # SLURM uses --gres for GPU allocation
        assert "#SBATCH --gres=gpu" in output

    def test_build_contains_cpu_count(self, job_config_sft):
        """Test that output contains CPU directive."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        assert "#SBATCH --cpus-per-task=" in output

    def test_build_contains_memory(self, job_config_sft):
        """Test that output contains memory directive."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        # Uses --mem for total memory
        assert "#SBATCH --mem=" in output

    def test_multi_gpu_config(self, large_model_config):
        """Test configuration with multiple GPUs."""
        config = JobConfig(**large_model_config)
        builder = SBatchHeaderBuilder(config)
        output = builder.build()

        # Multi-GPU uses gres format with count (e.g., gpu:h100:4)
        assert "#SBATCH --gres=gpu" in output
        # Should contain "4" somewhere for 4 GPUs
        assert ":4" in output or "4" in output

    def test_contains_header_comment(self, job_config_sft):
        """Test that output contains descriptive header comment."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        # Should have auto-generated comment
        assert "SLURM Job Script" in output or "Auto-generated" in output or "SFT" in output


class TestSBatchHeaderValidation:
    """Test header validation and edge cases."""

    def test_special_characters_in_job_name(self, minimal_sft_config):
        """Test handling of special characters in job name."""
        minimal_sft_config["job_name"] = "test-job_v1.0"
        config = JobConfig(**minimal_sft_config)
        builder = SBatchHeaderBuilder(config)
        output = builder.build()

        assert "#SBATCH --job-name=test-job_v1.0" in output

    def test_long_time_limit(self, minimal_sft_config):
        """Test long time limit format."""
        minimal_sft_config["time_limit"] = "48:00:00"
        config = JobConfig(**minimal_sft_config)
        builder = SBatchHeaderBuilder(config)
        output = builder.build()

        assert "#SBATCH --time=48:00:00" in output

    def test_account_directive(self, job_config_sft):
        """Test that output contains account directive."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        assert "#SBATCH --account=" in output or "#SBATCH -A" in output

    def test_output_log_paths(self, job_config_sft):
        """Test that output contains log file paths."""
        builder = SBatchHeaderBuilder(job_config_sft)
        output = builder.build()

        # Should have output and error log paths
        assert "#SBATCH --output=" in output or "#SBATCH -o" in output
        assert "#SBATCH --error=" in output or "#SBATCH -e" in output
