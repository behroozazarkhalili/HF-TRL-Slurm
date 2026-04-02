"""
Format Reward Function for GRPO training.

Provides reward computation based on response formatting quality.
Rewards proper mathematical notation and step-by-step reasoning.
"""

import re
from typing import List

from .base import RewardFunction


class FormatRewardFunction(RewardFunction):
    """Reward function for response formatting quality.

    Computes rewards based on:
    - Proper boxed answer format (\\boxed{})
    - Step-by-step reasoning indicators
    - Mathematical notation usage

    This is useful for training models to produce well-formatted
    mathematical solutions.
    """

    def __init__(self):
        """Initialize the format reward function."""
        super().__init__(name="format_quality")

    def compute(
        self,
        completions: List[str],
        prompts: List[str],
        **kwargs
    ) -> List[float]:
        """Compute format-based rewards for completions.

        Args:
            completions: List of model-generated completions.
            prompts: List of corresponding prompts (unused).

        Returns:
            List of reward scores in [0, 1].
        """
        rewards = []

        for completion in completions:
            score = 0.0

            # Check for boxed answer (LaTeX format)
            if re.search(r'\\boxed\{', completion):
                score += 0.5

            # Check for step-by-step reasoning
            step_patterns = r'(step\s*\d|first|second|third|then|next|finally)'
            if re.search(step_patterns, completion.lower()):
                score += 0.3

            # Check for mathematical notation
            if re.search(r'[=+\-*/^]', completion):
                score += 0.2

            rewards.append(min(score, 1.0))

        return rewards
