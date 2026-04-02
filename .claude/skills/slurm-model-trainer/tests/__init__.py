"""
Tests for slurm-model-trainer skill.

Test Structure:
- unit/: Unit tests for individual components
  - test_builders/: Tests for SLURM job script builders
  - test_rewards/: Tests for reward functions
  - test_config.py: Tests for JobConfig
  - test_validators.py: Tests for validators
- integration/: Integration tests
  - test_job_generator.py: End-to-end job generation
  - test_cli.py: CLI command tests
- fixtures/: Test data
  - sample_configs/: Sample YAML configurations
  - expected_outputs/: Expected script outputs
"""
