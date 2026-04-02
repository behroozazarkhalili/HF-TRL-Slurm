"""
Structured logging configuration for the SLURM job generator.

Usage:
    from .logging_config import logger

    logger.info("Generating job script...")
    logger.debug("Config: %s", config)
    logger.warning("Model >14B may OOM without 4-bit quantization")
    logger.error("Failed to generate script: %s", error)
"""

import logging
import os
from typing import Optional

try:
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(
    level: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """Configure structured logging with optional Rich handler.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
               Can be overridden via LOG_LEVEL environment variable.
        use_rich: Whether to use Rich for pretty console output.

    Returns:
        Configured logger instance.
    """
    # Get level from environment or parameter
    log_level = level or os.environ.get("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure handler
    if use_rich and RICH_AVAILABLE:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt))

    # Create logger
    logger = logging.getLogger("slurm-trainer")
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Module-level logger instance
logger = setup_logging()


# Convenience functions for validation logging (used by validators.py)
def log_validation_warning(message: str) -> None:
    """Log a validation warning.

    Args:
        message: Warning message.
    """
    logger.warning("Validation: %s", message)


def log_validation_error(message: str) -> None:
    """Log a validation error.

    Args:
        message: Error message.
    """
    logger.error("Validation failed: %s", message)
