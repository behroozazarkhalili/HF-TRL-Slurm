"""
Base Reward Function interface.

Defines the abstract interface that all reward functions must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union


def _normalize_completions(completions: list) -> List[str]:
    """Normalize completions to plain strings.

    TRL passes completions as List[str] for non-conversational datasets
    or List[List[Dict]] for conversational datasets. This function
    handles both formats transparently.
    """
    if not completions:
        return []
    first = completions[0]
    # Conversational format: [[{"role": "assistant", "content": "..."}]]
    if isinstance(first, list):
        return [
            c[0]["content"] if c and isinstance(c[0], dict) else str(c)
            for c in completions
        ]
    # Already strings
    if isinstance(first, str):
        return completions
    # Fallback: coerce to string
    return [str(c) for c in completions]


def _normalize_prompts(prompts: list) -> List[str]:
    """Normalize prompts to plain strings.

    TRL passes prompts as List[str] for non-conversational datasets
    or List[List[Dict]] for conversational datasets.
    """
    if not prompts:
        return []
    first = prompts[0]
    # Conversational format: [{"role": "user", "content": "..."}]
    if isinstance(first, list):
        return [
            p[-1]["content"] if p and isinstance(p[-1], dict) else str(p)
            for p in prompts
        ]
    if isinstance(first, str):
        return prompts
    return [str(p) for p in prompts]


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
            completions: List of model-generated completions (always strings).
            prompts: List of corresponding prompts (always strings).
            **kwargs: Additional arguments (for extensibility).

        Returns:
            List of reward scores (typically in range [0, 1]).
        """
        pass

    def __call__(
        self,
        completions,
        prompts,
        **kwargs
    ) -> List[float]:
        """Allow using the reward function as a callable.

        This makes the reward function compatible with TRL's expected
        interface for reward_funcs. Handles both conversational
        (List[List[Dict]]) and non-conversational (List[str]) formats.

        Args:
            completions: Model-generated completions (str or chat format).
            prompts: Corresponding prompts (str or chat format).
            **kwargs: Additional arguments.

        Returns:
            List of reward scores.
        """
        return self.compute(
            _normalize_completions(completions),
            _normalize_prompts(prompts),
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
