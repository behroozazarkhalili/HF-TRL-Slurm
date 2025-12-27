"""
SLURM Job Generator for HF-TRL Training.

A user-driven job generator with smart defaults based on model size.

Usage:
    # From command line
    python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7

    # Programmatically
    from generator import get_defaults, JobGenerator
    config = get_defaults("Qwen/Qwen2.5-7B", 7.0, "grpo")
    generator = JobGenerator()
    script = generator.generate("Qwen/Qwen2.5-7B", "nvidia/OpenMathInstruct-2", "grpo", config)
"""

from .smart_defaults import (
    SIZE_DEFAULTS,
    HARDWARE_PROFILES,
    COMMON_SETTINGS,
    categorize_model_size,
    get_defaults,
    get_job_name,
    get_hub_model_id,
    get_partition,
)
from .clarifier import Clarifier
from .job_generator import JobGenerator

__all__ = [
    # Constants
    "SIZE_DEFAULTS",
    "HARDWARE_PROFILES",
    "COMMON_SETTINGS",
    # Functions
    "categorize_model_size",
    "get_defaults",
    "get_job_name",
    "get_hub_model_id",
    "get_partition",
    # Classes
    "Clarifier",
    "JobGenerator",
]
