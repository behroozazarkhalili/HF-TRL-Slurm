"""
JobConfig dataclass for type-safe, validated SLURM job configuration.

This module provides an immutable configuration object that replaces
the error-prone dict-based configuration used previously.

Example:
    config = JobConfig(
        model_id="Qwen/Qwen2.5-0.5B",
        dataset_id="trl-lib/Capybara",
        method="sft",
        job_name="qwen-sft-capybara",
        hub_model_id="username/Qwen2.5-0.5B-SFT-Capybara",
        # ... other required fields
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# Type aliases for documentation and IDE support
TrainingMethod = Literal["sft", "dpo", "grpo"]
SizeCategory = Literal["small", "medium", "large"]
RewardType = Literal["combined", "math", "accuracy", "format", "length"]
HubStrategy = Literal["end", "every_save", "checkpoint", "all_checkpoints"]


@dataclass(frozen=False)
class JobConfig:
    """
    Validated SLURM job configuration for training pipelines.

    This dataclass captures all parameters needed to generate a complete
    SLURM job script for SFT, DPO, or GRPO training.

    Attributes:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B")
        dataset_id: HuggingFace dataset ID (e.g., "trl-lib/Capybara")
        method: Training method ("sft", "dpo", or "grpo")
        job_name: SLURM job name for sbatch
        hub_model_id: Target Hub repository for trained model
        gguf_repo_id: Target Hub repository for GGUF conversion
        params_b: Model size in billions of parameters
        size_category: Derived size category ("small", "medium", "large")
    """

    # =========================================================================
    # Required Fields - Model & Dataset
    # =========================================================================
    model_id: str
    dataset_id: str
    method: TrainingMethod

    # =========================================================================
    # Required Fields - Job Identification
    # =========================================================================
    job_name: str
    hub_model_id: str
    gguf_repo_id: str

    # =========================================================================
    # Model Sizing
    # =========================================================================
    params_b: float = 0.5
    size_category: SizeCategory = "small"

    # =========================================================================
    # SLURM Resources
    # =========================================================================
    account: str = "def-maxwl_gpu"
    partition: str = "gpubase_bygpu_b3"
    time_limit: str = "2-00:00:00"
    gres: str = "gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
    memory: str = "64G"
    cpus: int = 8

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    num_train_epochs: int = 1
    max_length: int = 2048
    max_prompt_length: int = 512  # GRPO only

    # =========================================================================
    # LoRA Configuration
    # =========================================================================
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # =========================================================================
    # GRPO-Specific
    # =========================================================================
    num_gen: int = 4
    reward_type: RewardType = "combined"

    # =========================================================================
    # Dataset Options
    # =========================================================================
    streaming: bool = False
    max_samples: Optional[int] = None
    sample_size_label: str = ""

    # =========================================================================
    # Precision & Optimization
    # =========================================================================
    requires_4bit: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # =========================================================================
    # Checkpointing & Logging
    # =========================================================================
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    hub_strategy: HubStrategy = "end"
    report_to: str = "trackio"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_required_fields()
        self._validate_method_specific()
        self._validate_resources()

    def _validate_required_fields(self) -> None:
        """Ensure required string fields are not empty."""
        required_strings = [
            ("model_id", self.model_id),
            ("dataset_id", self.dataset_id),
            ("job_name", self.job_name),
            ("hub_model_id", self.hub_model_id),
        ]
        for name, value in required_strings:
            if not value or not value.strip():
                raise ValueError(f"{name} cannot be empty")

    def _validate_method_specific(self) -> None:
        """Validate method-specific requirements."""
        if self.method == "grpo":
            if self.reward_type not in ("combined", "math", "accuracy", "format", "length"):
                raise ValueError(f"Invalid reward_type: {self.reward_type}")
            if self.num_gen < 1:
                raise ValueError("num_gen must be >= 1 for GRPO")

    def _validate_resources(self) -> None:
        """Validate SLURM resource allocations."""
        if self.cpus < 1:
            raise ValueError("cpus must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size (batch_size * grad_accum)."""
        return self.batch_size * self.grad_accum

    @property
    def model_short_name(self) -> str:
        """Extract short model name from model_id."""
        return self.model_id.split("/")[-1]

    @property
    def dataset_short_name(self) -> str:
        """Extract short dataset name from dataset_id."""
        return self.dataset_id.split("/")[-1]

    @property
    def is_grpo(self) -> bool:
        """Check if this is a GRPO training job."""
        return self.method == "grpo"

    @property
    def is_large_model(self) -> bool:
        """Check if this is a large model requiring special handling."""
        return self.size_category == "large" or self.params_b > 4

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "method": self.method,
            "job_name": self.job_name,
            "hub_model_id": self.hub_model_id,
            "gguf_repo_id": self.gguf_repo_id,
            "params_b": self.params_b,
            "size_category": self.size_category,
            "account": self.account,
            "partition": self.partition,
            "time_limit": self.time_limit,
            "gres": self.gres,
            "memory": self.memory,
            "cpus": self.cpus,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "num_train_epochs": self.num_train_epochs,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "num_gen": self.num_gen,
            "reward_type": self.reward_type,
            "streaming": self.streaming,
            "max_samples": self.max_samples,
            "sample_size_label": self.sample_size_label,
            "requires_4bit": self.requires_4bit,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "hub_strategy": self.hub_strategy,
            "report_to": self.report_to,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobConfig":
        """Create JobConfig from dictionary.

        This provides backward compatibility with existing dict-based configs.

        Args:
            data: Configuration dictionary

        Returns:
            JobConfig instance
        """
        # Map old key names to new ones if needed
        key_mapping = {
            "learning_rate": "lr",
            "gradient_accumulation_steps": "grad_accum",
        }

        normalized = {}
        for key, value in data.items():
            mapped_key = key_mapping.get(key, key)
            normalized[mapped_key] = value

        # Extract only fields that JobConfig accepts
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in normalized.items() if k in valid_fields}

        return cls(**filtered)


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware profile for SLURM resource allocation."""

    name: str
    gres: str
    memory: str
    cpus: int
    vram_gb: int

    @property
    def is_mig(self) -> bool:
        """Check if this is a MIG partition."""
        return "mig" in self.name.lower() or "3g" in self.gres or "2g" in self.gres


# Pre-defined hardware profiles for Fir cluster
HARDWARE_PROFILES = {
    "h100_80gb": HardwareProfile(
        name="H100 80GB",
        gres="gpu:h100:1",
        memory="128G",
        cpus=16,
        vram_gb=80,
    ),
    "mig_40gb": HardwareProfile(
        name="H100 MIG 3g.40gb",
        gres="gpu:nvidia_h100_80gb_hbm3_3g.40gb:1",
        memory="64G",
        cpus=8,
        vram_gb=40,
    ),
    "mig_20gb": HardwareProfile(
        name="H100 MIG 2g.20gb",
        gres="gpu:nvidia_h100_80gb_hbm3_2g.20gb:1",
        memory="32G",
        cpus=8,
        vram_gb=20,
    ),
    "mig_10gb": HardwareProfile(
        name="H100 MIG 1g.10gb",
        gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1",
        memory="32G",
        cpus=4,
        vram_gb=10,
    ),
}
