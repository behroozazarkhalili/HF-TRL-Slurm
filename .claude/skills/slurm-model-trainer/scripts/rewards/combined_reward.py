"""
Combined Reward Function for GRPO training.

Provides a weighted combination of multiple reward functions,
allowing flexible reward shaping for different training objectives.
"""

import re
import traceback
from typing import Dict, List, Optional

from .base import RewardFunction
from .math_reward import MathRewardFunction, extract_answer, normalize_answer, _prompt_key, _try_parse_number
from .format_reward import FormatRewardFunction
from .length_reward import LengthRewardFunction


class CombinedRewardFunction(RewardFunction):
    """Combined reward function using weighted components.

    This reward function combines multiple reward signals:
    - Math accuracy (50%): Answer correctness
    - Format quality (25%): Proper formatting
    - Length quality (15%): Response detail level
    - Reasoning quality (10%): Logical flow indicators

    The weights can be customized for different training objectives.

    Attributes:
        ground_truth: Dictionary mapping prompt keys to answers.
        weights: Dictionary of component weights.
    """

    DEFAULT_WEIGHTS = {
        "accuracy": 0.50,
        "format": 0.25,
        "length": 0.15,
        "reasoning": 0.10,
    }

    def __init__(
        self,
        ground_truth: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the combined reward function.

        Args:
            ground_truth: Dictionary mapping prompt keys to answers.
            weights: Custom weights for reward components.
                    Keys: 'accuracy', 'format', 'length', 'reasoning'
        """
        super().__init__(name="combined")
        self.ground_truth: Dict[str, str] = ground_truth or {}
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total != 1.0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def add_ground_truth(self, prompt: str, answer: str) -> None:
        """Add a ground truth answer for a prompt.

        Args:
            prompt: The problem prompt.
            answer: The expected answer.
        """
        self.ground_truth[_prompt_key(prompt)] = answer

    def compute(
        self,
        completions: List[str],
        prompts: List[str],
        **kwargs
    ) -> List[float]:
        """Compute combined rewards for completions.

        Args:
            completions: List of model-generated completions.
            prompts: List of corresponding prompts.

        Returns:
            List of combined reward scores in [0, 1].
        """
        rewards = []

        for completion, prompt in zip(completions, prompts):
            try:
                reward = self._compute_single(completion, prompt)
                rewards.append(reward)
            except (ValueError, TypeError, AttributeError) as e:
                print(f"[CombinedReward] Graceful fallback for: {type(e).__name__}: {e}")
                rewards.append(0.0)
            except Exception as e:
                traceback.print_exc()
                print(f"[CombinedReward] UNEXPECTED error — this may indicate a bug: {e}")
                rewards.append(0.0)

        return rewards

    def _compute_single(self, completion: str, prompt: str) -> float:
        """Compute combined reward for a single completion.

        Args:
            completion: The model-generated completion.
            prompt: The corresponding prompt.

        Returns:
            Combined reward score in [0, 1].
        """
        # Compute component scores
        accuracy_score = self._compute_accuracy(completion, prompt)
        format_score = self._compute_format(completion)
        length_score = self._compute_length(completion)
        reasoning_score = self._compute_reasoning(completion)

        # Weighted combination
        combined = (
            accuracy_score * self.weights.get("accuracy", 0.5) +
            format_score * self.weights.get("format", 0.25) +
            length_score * self.weights.get("length", 0.15) +
            reasoning_score * self.weights.get("reasoning", 0.10)
        )

        return combined

    def _compute_accuracy(self, completion: str, prompt: str) -> float:
        """Compute accuracy score."""
        predicted_answer = extract_answer(completion)
        prompt_key = _prompt_key(prompt)
        ground_truth = self.ground_truth.get(prompt_key)

        if predicted_answer and ground_truth:
            pred_norm = normalize_answer(predicted_answer)
            gt_norm = normalize_answer(ground_truth)

            if pred_norm == gt_norm:
                return 1.0

            # Numerical comparison first (prevents "1" in "10" false positive)
            pred_num = _try_parse_number(pred_norm)
            gt_num = _try_parse_number(gt_norm)
            if pred_num is not None and gt_num is not None:
                if abs(pred_num - gt_num) < 1e-6:
                    return 1.0
                return 0.0  # Both numeric but different — no partial credit

            # Substring match only for non-numeric answers
            if pred_norm in gt_norm or gt_norm in pred_norm:
                return 0.7

            return 0.0

        elif predicted_answer:
            return 0.3  # Has answer but no ground truth

        return 0.0

    def _compute_format(self, completion: str) -> float:
        """Compute format score."""
        score = 0.0

        if re.search(r'\\boxed\{', completion):
            score += 0.5

        step_patterns = r'(step\s*\d|first|second|third|then|next|finally)'
        if re.search(step_patterns, completion.lower()):
            score += 0.3

        if re.search(r'[=+\-*/^]', completion):
            score += 0.2

        return min(score, 1.0)

    def _compute_length(self, completion: str) -> float:
        """Compute length score."""
        word_count = len(completion.split())

        if word_count < 10:
            return 0.0
        elif word_count < 30:
            return 0.3
        elif word_count < 100:
            return 0.7
        elif word_count < 300:
            return 1.0
        else:
            return 0.8  # Slight penalty for very long

    def _compute_reasoning(self, completion: str) -> float:
        """Compute reasoning quality score."""
        score = 0.0
        completion_lower = completion.lower()

        # Check for explanation indicators
        explanation_words = ['because', 'therefore', 'since', 'thus', 'hence']
        if any(word in completion_lower for word in explanation_words):
            score += 0.4

        # Check for clear logical flow
        flow_words = ['let', 'given', 'we have', 'we get', 'we can']
        if any(word in completion_lower for word in flow_words):
            score += 0.3

        # Check for conclusion
        conclusion_words = ['answer', 'result', 'solution', 'final']
        if any(word in completion_lower for word in conclusion_words):
            score += 0.3

        return min(score, 1.0)
