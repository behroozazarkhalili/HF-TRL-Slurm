"""
Base class for all script section builders.

Provides the common interface and shared utilities for building
sections of SLURM job scripts.
"""

from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import JobConfig


class BaseBuilder(ABC):
    """Abstract base class for job script section builders.

    All builders inherit from this class and implement the build() method
    to generate their specific section of the job script.

    Attributes:
        config: The JobConfig object containing all job parameters.
        skill_path: Path to the slurm-model-trainer skill directory.
        venv_path: Path to the Python virtual environment.
        env_file_path: Path to the .env file with HF token.
    """

    # Class-level defaults (can be overridden via environment)
    DEFAULT_SKILL_PATH = (
        "/project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer"
    )
    DEFAULT_VENV_PATH = "/scratch/ermia/venvs/hf_env"
    DEFAULT_ENV_FILE_PATH = "/project/6014832/ermia/HF-TRL/.env"

    def __init__(self, config: "JobConfig") -> None:
        """Initialize the builder with configuration.

        Args:
            config: JobConfig object with all job parameters.
        """
        self.config = config
        self.skill_path = os.environ.get("SKILL_PATH", self.DEFAULT_SKILL_PATH)
        self.venv_path = os.environ.get("VENV_PATH", self.DEFAULT_VENV_PATH)
        self.env_file_path = os.environ.get("ENV_FILE_PATH", self.DEFAULT_ENV_FILE_PATH)

    @abstractmethod
    def build(self) -> str:
        """Build this section of the job script.

        Returns:
            The generated script section as a string.
        """
        pass

    def _comment_block(self, title: str, char: str = "=", width: int = 77) -> str:
        """Generate a formatted comment block header.

        Args:
            title: The title text for the block.
            char: The character to use for the border (default: '=').
            width: The total width of the border (default: 77).

        Returns:
            Formatted comment block as a string.
        """
        border = char * width
        return f"# {border}\n# {title}\n# {border}"

    def _echo_banner(self, title: str, char: str = "=", width: int = 42) -> str:
        """Generate an echo statement with a banner.

        Args:
            title: The banner title text.
            char: The character to use for the border (default: '=').
            width: The total width of the border (default: 42).

        Returns:
            Echo statements creating a banner.
        """
        border = char * width
        return f'echo "{border}"\necho "{title}"\necho "{border}"'

    @property
    def is_grpo(self) -> bool:
        """Check if this is a GRPO training job."""
        return self.config.method == "grpo"

    @property
    def is_sft(self) -> bool:
        """Check if this is an SFT training job."""
        return self.config.method == "sft"

    @property
    def is_dpo(self) -> bool:
        """Check if this is a DPO training job."""
        return self.config.method == "dpo"

    @property
    def method_upper(self) -> str:
        """Get the training method in uppercase."""
        return self.config.method.upper()

    @property
    def model_short_name(self) -> str:
        """Get the short model name (without organization prefix)."""
        return self.config.model_id.split("/")[-1]

    @property
    def dataset_short_name(self) -> str:
        """Get the short dataset name (without organization prefix)."""
        return self.config.dataset_id.split("/")[-1]

    @property
    def hub_username(self) -> str:
        """Get the Hub username from hub_model_id."""
        return self.config.hub_model_id.split("/")[0]

    @property
    def hub_model_name(self) -> str:
        """Get the model name from hub_model_id."""
        return self.config.hub_model_id.split("/")[-1]
