"""
YAML Configuration Loader for training defaults.

This module provides a centralized way to load training configuration
from YAML files, replacing hardcoded defaults in smart_defaults.py.

The loader implements caching to avoid repeated file I/O and provides
fallback values when config files are missing.

Example:
    loader = ConfigLoader()
    sft_config = loader.load_training_config("sft")
    hardware = loader.load_hardware_profile("mig_40gb")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DatasetRecommendation:
    """A recommended dataset for a training method."""

    id: str
    split: str
    description: str
    streaming: bool = False
    max_samples: Optional[int] = None


class ConfigLoader:
    """
    Load configuration from YAML files with caching.

    The loader searches for config files relative to the skill root directory
    and caches loaded configurations to avoid repeated file I/O.

    Attributes:
        config_dir: Path to the configs/ directory
        _cache: Internal cache of loaded configurations
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the config loader.

        Args:
            config_dir: Path to configs/ directory. Defaults to ../configs
                       relative to this file.
        """
        if config_dir is None:
            # Default: skill_root/configs/
            config_dir = Path(__file__).parent.parent / "configs"

        self.config_dir = config_dir
        self._cache: dict[str, Any] = {}

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load and parse a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            Parsed YAML content as dictionary

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file is not valid YAML
        """
        cache_key = str(path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        self._cache[cache_key] = data
        return data

    def load_training_config(self, method: str) -> dict[str, Any]:
        """Load training configuration for a method (SFT, DPO, GRPO).

        Args:
            method: Training method name (lowercase)

        Returns:
            Configuration dictionary with 'common' and method-specific settings

        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load_training_config("sft")
            >>> config['common']['bf16']
            True
        """
        path = self.config_dir / "training" / f"{method.lower()}.yaml"

        try:
            return self._load_yaml(path)
        except FileNotFoundError:
            # Return empty config if file doesn't exist
            return {"common": {}, "recommended_datasets": []}

    def load_hardware_config(self) -> dict[str, Any]:
        """Load hardware configuration for the cluster.

        Returns:
            Hardware profiles and partition information
        """
        path = self.config_dir / "hardware" / "fir_cluster.yaml"

        try:
            return self._load_yaml(path)
        except FileNotFoundError:
            return {"profiles": {}, "partitions": {}}

    def load_accelerate_config(self, config_type: str) -> dict[str, Any]:
        """Load Accelerate configuration for distributed training.

        Args:
            config_type: Type of config (single_gpu, multi_gpu_ddp, etc.)

        Returns:
            Accelerate configuration dictionary
        """
        path = self.config_dir / "accelerate" / f"{config_type}.yaml"

        try:
            return self._load_yaml(path)
        except FileNotFoundError:
            return {}

    def load_eval_tasks(self, task_set: str = "general") -> dict[str, Any]:
        """Load evaluation task configuration.

        Args:
            task_set: Task set name (reasoning, general, coding, comprehensive)

        Returns:
            Evaluation task configuration
        """
        path = self.config_dir / "eval_tasks" / f"{task_set}.yaml"

        try:
            return self._load_yaml(path)
        except FileNotFoundError:
            return {"tasks": []}

    def get_common_settings(self, method: str) -> dict[str, Any]:
        """Extract common settings from a training config.

        Args:
            method: Training method name

        Returns:
            Common settings dictionary
        """
        config = self.load_training_config(method)
        return config.get("common", {})

    def get_recommended_datasets(self, method: str) -> list[DatasetRecommendation]:
        """Get recommended datasets for a training method.

        Args:
            method: Training method name

        Returns:
            List of DatasetRecommendation objects
        """
        config = self.load_training_config(method)
        raw_datasets = config.get("recommended_datasets", [])

        recommendations = []
        for ds in raw_datasets:
            recommendations.append(
                DatasetRecommendation(
                    id=ds.get("id", ""),
                    split=ds.get("split", "train"),
                    description=ds.get("description", ""),
                    streaming=ds.get("streaming", False),
                    max_samples=ds.get("max_samples"),
                )
            )

        return recommendations

    def get_grpo_settings(self) -> dict[str, Any]:
        """Get GRPO-specific settings.

        Returns:
            GRPO configuration including num_generations and reward_types
        """
        config = self.load_training_config("grpo")
        return config.get("grpo", {})

    def clear_cache(self) -> None:
        """Clear the configuration cache.

        Call this if config files have been modified and need to be reloaded.
        """
        self._cache.clear()


# Singleton instance for convenience
_default_loader: Optional[ConfigLoader] = None


def get_loader() -> ConfigLoader:
    """Get the default ConfigLoader instance.

    Returns:
        Singleton ConfigLoader instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


def load_training_config(method: str) -> dict[str, Any]:
    """Convenience function to load training config.

    Args:
        method: Training method name

    Returns:
        Training configuration dictionary
    """
    return get_loader().load_training_config(method)


def get_common_settings(method: str) -> dict[str, Any]:
    """Convenience function to get common settings.

    Args:
        method: Training method name

    Returns:
        Common settings dictionary
    """
    return get_loader().get_common_settings(method)
