"""
Length Reward Function for GRPO training.

Provides reward computation based on response length.
Encourages detailed, thorough responses while penalizing
extremely short or excessively long outputs.
"""

from typing import List

from .base import RewardFunction


class LengthRewardFunction(RewardFunction):
    """Reward function based on response length.

    Computes rewards that encourage appropriately detailed responses:
    - Very short responses (< 10 words): 0.0
    - Short responses (10-50 words): 0.3-0.5
    - Medium responses (50-200 words): 0.7-1.0
    - Long responses (200-300 words): 1.0
    - Very long responses (> 300 words): 0.8 (slight penalty)

    This helps train models to provide thorough explanations
    without being excessively verbose.
    """

    def __init__(
        self,
        min_words: int = 10,
        optimal_min: int = 100,
        optimal_max: int = 300,
        max_words: int = 500
    ):
        """Initialize the length reward function.

        Args:
            min_words: Minimum words for any reward.
            optimal_min: Start of optimal length range.
            optimal_max: End of optimal length range.
            max_words: Maximum words before penalty applies.
        """
        super().__init__(name="length_quality")
        self.min_words = min_words
        self.optimal_min = optimal_min
        self.optimal_max = optimal_max
        self.max_words = max_words

    def compute(
        self,
        completions: List[str],
        prompts: List[str],
        **kwargs
    ) -> List[float]:
        """Compute length-based rewards for completions.

        Args:
            completions: List of model-generated completions.
            prompts: List of corresponding prompts (unused).

        Returns:
            List of reward scores in [0, 1].
        """
        rewards = []

        for completion in completions:
            word_count = len(completion.split())
            reward = self._compute_length_reward(word_count)
            rewards.append(reward)

        return rewards

    def _compute_length_reward(self, word_count: int) -> float:
        """Compute reward based on word count.

        Args:
            word_count: Number of words in the completion.

        Returns:
            Reward score in [0, 1].
        """
        if word_count < self.min_words:
            return 0.0

        if word_count < 30:
            return 0.3

        if word_count < self.optimal_min:
            # Linear interpolation from 0.3 to 0.7
            progress = (word_count - 30) / (self.optimal_min - 30)
            return 0.3 + 0.4 * progress

        if word_count <= self.optimal_max:
            # Optimal range
            return 1.0

        if word_count <= self.max_words:
            # Slight decline after optimal
            return 0.9

        # Penalty for very long responses
        return 0.8
