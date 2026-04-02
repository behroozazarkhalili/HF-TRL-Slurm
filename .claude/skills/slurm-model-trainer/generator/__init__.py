"""
SLURM Job Generator for HF-TRL Training.

A user-driven job generator with smart defaults based on model size.

Usage:
    # From command line
    python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7

    # Programmatically
    from generator import JobConfig, JobGenerator
    config = JobConfig(
        model_id="Qwen/Qwen2.5-7B",
        dataset_id="nvidia/OpenMathInstruct-2",
        method="grpo",
        job_name="qwen-grpo-openmath",
        hub_model_id="username/Qwen2.5-7B-GRPO-OpenMath",
        gguf_repo_id="username/Qwen2.5-7B-GRPO-OpenMath-GGUF",
    )
    generator = JobGenerator()
    script = generator.generate(config)
"""

# New type-safe configuration objects
from .config import (
    JobConfig,
    HardwareProfile,
    HARDWARE_PROFILES as HARDWARE_PROFILES_NEW,
    TrainingMethod,
    SizeCategory,
    RewardType,
    HubStrategy,
)
from .config_loader import (
    ConfigLoader,
    DatasetRecommendation,
    get_loader,
    load_training_config,
    get_common_settings,
)

# Legacy smart_defaults (for backward compatibility)
from .smart_defaults import (
    SIZE_DEFAULTS,
    HARDWARE_PROFILES,
    COMMON_SETTINGS,
    DEFAULT_SLURM_ACCOUNT,
    DEFAULT_HF_USERNAME,
    # Functions
    categorize_model_size,
    get_defaults,
    get_job_name,
    get_hub_model_id,
    select_partition,
    format_sample_size,
)
# Builders for script generation
from .builders import (
    BaseBuilder,
    SBatchHeaderBuilder,
    EnvironmentBuilder,
    TrainingBlockBuilder,
    PostProcessingBuilder,
)
from .clarifier import Clarifier
from .job_generator import JobGenerator
from .logging_config import (
    logger,
    setup_logging,
    log_validation_warning,
    log_validation_error,
)
from .validators import (
    ValidationResult,
    validate_config,
    validate_model_id,
    validate_dataset_id,
    parse_time_limit,
)

__all__ = [
    # Constants
    "SIZE_DEFAULTS",
    "HARDWARE_PROFILES",
    "COMMON_SETTINGS",
    "DEFAULT_SLURM_ACCOUNT",
    "DEFAULT_HF_USERNAME",
    # Type aliases
    "SizeCategory",
    "TrainingMethod",
    "RewardType",
    "JobConfig",
    # Functions
    "categorize_model_size",
    "get_defaults",
    "get_job_name",
    "get_hub_model_id",
    "select_partition",
    "format_sample_size",
    # Logging
    "logger",
    "setup_logging",
    "log_validation_warning",
    "log_validation_error",
    # Validation
    "ValidationResult",
    "validate_config",
    "validate_model_id",
    "validate_dataset_id",
    "parse_time_limit",
    # Classes
    "Clarifier",
    "JobGenerator",
    # Builders
    "BaseBuilder",
    "SBatchHeaderBuilder",
    "EnvironmentBuilder",
    "TrainingBlockBuilder",
    "PostProcessingBuilder",
]
