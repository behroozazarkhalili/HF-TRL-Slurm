"""
Builder classes for SLURM job script generation.

This module provides a set of composable builders following the
Builder pattern combined with Single Responsibility Principle.

Each builder handles one specific section of the job script:
- SBatchHeaderBuilder: SLURM directives
- EnvironmentBuilder: Module loads, virtualenv, paths
- TrainingBlockBuilder: Training script and arguments
- PostProcessingBuilder: Model card, GGUF, summary

Example:
    from generator.builders import (
        SBatchHeaderBuilder,
        EnvironmentBuilder,
        TrainingBlockBuilder,
        PostProcessingBuilder,
    )
    from generator.config import JobConfig

    config = JobConfig(...)

    script_sections = [
        SBatchHeaderBuilder(config).build(),
        EnvironmentBuilder(config).build(),
        TrainingBlockBuilder(config).build(),
        PostProcessingBuilder(config).build(),
    ]
    full_script = "\\n".join(script_sections)
"""

from .base import BaseBuilder
from .sbatch_header import SBatchHeaderBuilder
from .environment import EnvironmentBuilder
from .training_block import TrainingBlockBuilder
from .post_processing import PostProcessingBuilder

__all__ = [
    "BaseBuilder",
    "SBatchHeaderBuilder",
    "EnvironmentBuilder",
    "TrainingBlockBuilder",
    "PostProcessingBuilder",
]
