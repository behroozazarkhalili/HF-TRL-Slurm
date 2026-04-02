"""
Base Reward Function interface.

Defines the abstract interface that all reward functions must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class RewardFunction(ABC):
    """Abstract base class for reward computation in GRPO training.

    All reward functions must implement the compute() method which takes
    a list of completions and prompts and returns a list of reward scores.

    The reward function is called during GRPO training for each batch
    of generated completions. The returned rewards are used to compute
    the policy gradient.

    Attributes:
        name: A human-readable name for the reward function.
    """

    def __init__(self, name: str = "base"):
        """Initialize the reward function.

        Args:
            name: Human-readable name for logging/display.
        """
        self.name = name

    @abstractmethod
    def compute(
        self,
        completions: List[str],
        prompts: List[str],
        **kwargs
    ) -> List[float]:
        """Compute rewards for a batch of completions.

        Args:
            completions: List of model-generated completions.
            prompts: List of corresponding prompts.
            **kwargs: Additional arguments (for extensibility).

        Returns:
            List of reward scores (typically in range [0, 1]).
        """
        pass

    def __call__(
        self,
        completions: List[str],
        prompts: List[str],
        **kwargs
    ) -> List[float]:
        """Allow using the reward function as a callable.

        This makes the reward function compatible with TRL's expected
        interface for reward_funcs.

        Args:
            completions: List of model-generated completions.
            prompts: List of corresponding prompts.
            **kwargs: Additional arguments.

        Returns:
            List of reward scores.
        """
        return self.compute(completions, prompts, **kwargs)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
