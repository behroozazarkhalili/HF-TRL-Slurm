"""
Job Generator Facade for SLURM job script generation.

This module provides a high-level interface for generating complete
SLURM job scripts by composing specialized builder classes.

The JobGenerator class follows the Facade pattern, delegating
the actual script generation to four specialized builders:
- SBatchHeaderBuilder: SLURM directives
- EnvironmentBuilder: Module loads, virtualenv, paths
- TrainingBlockBuilder: Training script and arguments
- PostProcessingBuilder: Model card, GGUF, summary
"""

from typing import Union

from .config import JobConfig
from .builders import (
    SBatchHeaderBuilder,
    EnvironmentBuilder,
    TrainingBlockBuilder,
    PostProcessingBuilder,
)


class JobGenerator:
    """Facade for generating complete SLURM job scripts.

    This class composes specialized builder classes to generate
    a complete SLURM job script. It supports both the new JobConfig
    dataclass and the legacy dict-based configuration.

    Example:
        # Using JobConfig (recommended)
        config = JobConfig(
            model_id="Qwen/Qwen2.5-0.5B",
            dataset_id="trl-lib/Capybara",
            method="sft",
            job_name="qwen-sft-capybara",
            hub_model_id="username/Qwen2.5-0.5B-SFT",
            gguf_repo_id="username/Qwen2.5-0.5B-SFT-GGUF",
        )
        generator = JobGenerator()
        script = generator.generate(config)

        # Legacy dict support
        config_dict = {...}
        script = generator.generate_from_dict(
            model_id="Qwen/Qwen2.5-0.5B",
            dataset_id="trl-lib/Capybara",
            method="sft",
            config=config_dict
        )
    """

    def generate(self, config: JobConfig) -> str:
        """Generate a complete SLURM job script from JobConfig.

        Args:
            config: JobConfig object with all job parameters.

        Returns:
            Complete SLURM job script as a string.
        """
        # Initialize all builders with the config
        header_builder = SBatchHeaderBuilder(config)
        env_builder = EnvironmentBuilder(config)
        training_builder = TrainingBlockBuilder(config)
        post_builder = PostProcessingBuilder(config)

        # Build all sections
        sections = [
            header_builder.build(),
            env_builder.build(),
            training_builder.build(),
            post_builder.build(),
        ]

        return "\n".join(sections)

    def generate_from_dict(
        self,
        model_id: str,
        dataset_id: str,
        method: str,
        config: dict,
    ) -> str:
        """Generate a SLURM job script from legacy dict configuration.

        This method provides backward compatibility with the old
        dict-based configuration system.

        Args:
            model_id: HuggingFace model ID.
            dataset_id: HuggingFace dataset ID.
            method: Training method (sft, grpo, dpo).
            config: Legacy configuration dictionary.

        Returns:
            Complete SLURM job script as a string.
        """
        # Merge model/dataset/method into config dict
        full_config = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "method": method,
            **config,
        }

        # Convert to JobConfig
        job_config = JobConfig.from_dict(full_config)

        return self.generate(job_config)

    def generate_section(
        self,
        config: JobConfig,
        section: str,
    ) -> str:
        """Generate a specific section of the job script.

        Useful for debugging or when you only need part of the script.

        Args:
            config: JobConfig object with all job parameters.
            section: Section name ('header', 'environment', 'training', 'post').

        Returns:
            The requested script section as a string.

        Raises:
            ValueError: If section name is not recognized.
        """
        builders = {
            "header": SBatchHeaderBuilder,
            "environment": EnvironmentBuilder,
            "training": TrainingBlockBuilder,
            "post": PostProcessingBuilder,
        }

        if section not in builders:
            raise ValueError(
                f"Unknown section: {section}. "
                f"Valid sections: {', '.join(builders.keys())}"
            )

        builder_class = builders[section]
        return builder_class(config).build()
