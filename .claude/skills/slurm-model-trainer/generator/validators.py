"""
Input validation for SLURM job configurations.

Validates job configurations before generation to catch errors early
and provide helpful warnings about potential issues.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .logging_config import log_validation_warning, log_validation_error


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        valid: Whether the configuration is valid (no errors).
        errors: List of error messages that must be fixed.
        warnings: List of warning messages about potential issues.
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return True if validation passed (no errors)."""
        return self.valid


def parse_time_limit(time_str: str) -> float:
    """Parse SLURM time limit string to hours.

    Args:
        time_str: Time limit string (e.g., '7-00:00:00', '12:00:00', '3:00:00')

    Returns:
        Time limit in hours.
    """
    if not time_str:
        return 0.0

    # Format: D-HH:MM:SS or HH:MM:SS or MM:SS
    if "-" in time_str:
        days, time = time_str.split("-")
        days = int(days)
    else:
        days = 0
        time = time_str

    parts = time.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        hours, minutes = map(int, parts)
        seconds = 0
    else:
        hours = int(parts[0])
        minutes = seconds = 0

    return days * 24 + hours + minutes / 60 + seconds / 3600


def validate_config(config: dict) -> ValidationResult:
    """Validate job configuration before generation.

    Args:
        config: Configuration dictionary from smart_defaults.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # =========================================================================
    # Required fields
    # =========================================================================
    required_fields = [
        "job_name",
        "account",
        "time_limit",
        "gres",
        "memory",
        "cpus",
        "partition",
        "hub_model_id",
    ]

    for field_name in required_fields:
        if not config.get(field_name):
            errors.append(f"Missing required field: {field_name}")

    # =========================================================================
    # Model size validation
    # =========================================================================
    params_b = config.get("params_b", 0)

    # Large models without 4-bit quantization
    if params_b > 7 and not config.get("requires_4bit"):
        warnings.append(
            f"Model with {params_b}B parameters may OOM without 4-bit quantization"
        )

    # Very large models (>14B)
    if params_b > 14:
        errors.append(
            f"Model with {params_b}B parameters exceeds supported size (max 14B)"
        )

    # =========================================================================
    # Time limit validation
    # =========================================================================
    time_limit = config.get("time_limit", "")
    time_hours = parse_time_limit(time_limit)

    # 7-day maximum on Fir cluster
    if time_hours > 168:
        errors.append(
            f"Time limit {time_limit} ({time_hours:.1f}h) exceeds 7-day maximum (168h)"
        )

    # GRPO with short time limit
    method = config.get("method", "")
    if method == "grpo" and time_hours < 24:
        warnings.append(
            f"GRPO training typically takes >24h; time limit is only {time_hours:.1f}h"
        )

    # =========================================================================
    # Batch size validation
    # =========================================================================
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("grad_accum", 1)
    effective_batch = batch_size * grad_accum

    if effective_batch < 8:
        warnings.append(
            f"Effective batch size {effective_batch} is small; consider increasing"
        )

    if effective_batch > 64:
        warnings.append(
            f"Effective batch size {effective_batch} is large; may affect convergence"
        )

    # =========================================================================
    # GPU/Memory validation
    # =========================================================================
    gres = config.get("gres", "")

    # Check for MIG partition compatibility with model size
    if "1g.10gb" in gres and params_b > 1.5:
        warnings.append(
            f"10GB MIG may be too small for {params_b}B model; consider 40GB MIG"
        )

    if "2g.20gb" in gres and params_b > 4:
        warnings.append(
            f"20GB MIG may be too small for {params_b}B model; consider 40GB MIG"
        )

    # =========================================================================
    # Hub model ID validation
    # =========================================================================
    hub_model_id = config.get("hub_model_id", "")

    if hub_model_id:
        # Check format: username/model-name
        if "/" not in hub_model_id:
            errors.append(
                f"Invalid hub_model_id format: {hub_model_id} (expected: username/model-name)"
            )

        # Check for invalid characters
        if re.search(r"[^a-zA-Z0-9\-_./]", hub_model_id):
            errors.append(
                f"hub_model_id contains invalid characters: {hub_model_id}"
            )

    # =========================================================================
    # Streaming validation
    # =========================================================================
    if config.get("streaming"):
        max_samples = config.get("max_samples", 0)

        if max_samples <= 0:
            errors.append("Streaming enabled but max_samples not set or invalid")

        if max_samples > 1_000_000:
            warnings.append(
                f"max_samples={max_samples:,} is very large; may take a long time"
            )

    # =========================================================================
    # GRPO-specific validation
    # =========================================================================
    if method == "grpo":
        num_gen = config.get("num_gen", 4)

        if num_gen < 2:
            errors.append("GRPO requires at least 2 generations (num_gen >= 2)")

        if num_gen > 8:
            warnings.append(
                f"num_gen={num_gen} is high; may slow training significantly"
            )

        reward_type = config.get("reward_type", "")
        valid_rewards = ["combined", "math", "accuracy", "format", "length"]
        if reward_type and reward_type not in valid_rewards:
            errors.append(
                f"Invalid reward_type: {reward_type} (valid: {', '.join(valid_rewards)})"
            )

    # =========================================================================
    # Learning rate validation
    # =========================================================================
    lr = config.get("lr", 0)

    if lr > 1e-3:
        warnings.append(f"Learning rate {lr} is very high; may cause instability")

    if lr < 1e-7:
        warnings.append(f"Learning rate {lr} is very low; may train too slowly")

    # =========================================================================
    # Log and return result
    # =========================================================================
    for warning in warnings:
        log_validation_warning(warning)

    for error in errors:
        log_validation_error(error)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_model_id(model_id: str) -> ValidationResult:
    """Validate HuggingFace model ID format.

    Args:
        model_id: Model ID to validate.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not model_id:
        errors.append("Model ID is required")
    elif "/" not in model_id:
        warnings.append(
            f"Model ID '{model_id}' has no organization prefix; assuming local model"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_dataset_id(dataset_id: str) -> ValidationResult:
    """Validate HuggingFace dataset ID format.

    Args:
        dataset_id: Dataset ID to validate.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not dataset_id:
        errors.append("Dataset ID is required")
    elif "/" not in dataset_id:
        warnings.append(
            f"Dataset ID '{dataset_id}' has no organization prefix"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
