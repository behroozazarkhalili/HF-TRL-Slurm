"""
Integration tests for JobGenerator.

Tests end-to-end SLURM job script generation.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import from generator package
from generator.job_generator import JobGenerator
from generator.config import JobConfig


class TestJobGeneratorEndToEnd:
    """Test complete job generation flow."""

    def test_generate_sft_job(self, job_config_sft, valid_sbatch_markers):
        """Test generating a complete SFT job script."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        # Check all required sections are present
        for marker in valid_sbatch_markers:
            assert marker in script, f"Missing marker: {marker}"

        # Check SFT-specific content
        assert "sft" in script.lower()

    def test_generate_dpo_job(self, job_config_dpo, valid_sbatch_markers):
        """Test generating a complete DPO job script."""
        generator = JobGenerator()
        script = generator.generate(job_config_dpo)

        # Check all required sections are present
        for marker in valid_sbatch_markers:
            assert marker in script, f"Missing marker: {marker}"

        # Check DPO-specific content
        assert "dpo" in script.lower()

    def test_generate_grpo_job(self, job_config_grpo, valid_sbatch_markers):
        """Test generating a complete GRPO job script."""
        generator = JobGenerator()
        script = generator.generate(job_config_grpo)

        # Check all required sections are present
        for marker in valid_sbatch_markers:
            assert marker in script, f"Missing marker: {marker}"

        # Check GRPO-specific content
        assert "grpo" in script.lower()


class TestJobGeneratorFromDict:
    """Test backward-compatible dict-based generation."""

    def test_generate_from_dict_sft(self, minimal_sft_config, valid_sbatch_markers):
        """Test generating SFT job from dictionary config."""
        generator = JobGenerator()
        script = generator.generate_from_dict(
            model_id=minimal_sft_config["model_id"],
            dataset_id=minimal_sft_config["dataset_id"],
            method="sft",
            config=minimal_sft_config
        )

        # Check required sections
        for marker in valid_sbatch_markers:
            assert marker in script, f"Missing marker: {marker}"

    def test_generate_from_dict_dpo(self, minimal_dpo_config, valid_sbatch_markers):
        """Test generating DPO job from dictionary config."""
        generator = JobGenerator()
        script = generator.generate_from_dict(
            model_id=minimal_dpo_config["model_id"],
            dataset_id=minimal_dpo_config["dataset_id"],
            method="dpo",
            config=minimal_dpo_config
        )

        # Check required sections
        for marker in valid_sbatch_markers:
            assert marker in script, f"Missing marker: {marker}"


class TestJobScriptValidity:
    """Test that generated scripts are valid SLURM scripts."""

    def test_script_starts_with_shebang(self, job_config_sft):
        """Test script starts with proper shebang."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        assert script.strip().startswith("#!/bin/bash")

    def test_sbatch_directives_format(self, job_config_sft):
        """Test SBATCH directives have correct format."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        # All SBATCH lines should start with #SBATCH
        lines = script.split("\n")
        sbatch_lines = [l for l in lines if l.strip().startswith("#SBATCH")]

        for line in sbatch_lines:
            assert "--" in line, f"Invalid SBATCH directive: {line}"

    def test_no_empty_arguments(self, job_config_sft):
        """Test that there are no empty argument values in SBATCH directives."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        # Check SBATCH lines specifically for empty values
        # (echo "" is valid, but --arg="" is not)
        lines = script.split("\n")
        sbatch_lines = [l for l in lines if l.strip().startswith("#SBATCH")]

        for line in sbatch_lines:
            # Should not have empty value after =
            if "=" in line:
                value = line.split("=", 1)[1].strip()
                assert value != "", f"Empty value in SBATCH directive: {line}"

    def test_training_command_is_complete(
        self, job_config_sft, expected_training_args
    ):
        """Test that training command has all required arguments."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        for arg in expected_training_args:
            assert arg in script, f"Missing training argument: {arg}"


class TestJobScriptSections:
    """Test that all required sections are present and ordered."""

    def test_sections_order(self, job_config_sft):
        """Test that sections appear in correct order."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        # Find positions of key markers
        shebang_pos = script.find("#!/bin/bash")
        sbatch_pos = script.find("#SBATCH")
        module_pos = script.find("module")
        python_pos = script.find("python")

        # Verify order
        assert shebang_pos < sbatch_pos, "Shebang should come before SBATCH"
        assert sbatch_pos < module_pos, "SBATCH should come before module loads"
        assert module_pos < python_pos, "Module loads should come before python"

    def test_contains_environment_setup(self, job_config_sft):
        """Test that environment setup is present."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        # Check for environment components
        assert "module" in script.lower()
        assert "source" in script or "activate" in script.lower()

    def test_contains_training_section(self, job_config_sft):
        """Test that training section is present."""
        generator = JobGenerator()
        script = generator.generate(job_config_sft)

        # Check for training components
        assert "python" in script
        assert "train" in script.lower()


class TestLargeModelConfiguration:
    """Test generation for large model configurations."""

    def test_large_model_multi_gpu(self, large_model_config, valid_sbatch_markers):
        """Test generating job for large model with multiple GPUs."""
        config = JobConfig(**large_model_config)
        generator = JobGenerator()
        script = generator.generate(config)

        # Check all markers present
        for marker in valid_sbatch_markers:
            assert marker in script, f"Missing marker: {marker}"

        # Check multi-GPU configuration
        assert "4" in script  # Should reference 4 GPUs somewhere

    def test_large_model_quantization(self, large_model_config):
        """Test that quantization settings are included."""
        config = JobConfig(**large_model_config)
        generator = JobGenerator()
        script = generator.generate(config)

        # Check quantization flag
        assert "4bit" in script.lower() or "quantiz" in script.lower()
