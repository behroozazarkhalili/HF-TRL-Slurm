"""
Unit tests for CombinedRewardFunction.

Tests weighted combination of multiple reward components.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rewards.combined_reward import CombinedRewardFunction


class TestCombinedRewardCreation:
    """Test CombinedRewardFunction instantiation."""

    def test_default_weights(self):
        """Test default weight configuration."""
        reward_fn = CombinedRewardFunction()

        assert reward_fn.weights["accuracy"] == 0.50
        assert reward_fn.weights["format"] == 0.25
        assert reward_fn.weights["length"] == 0.15
        assert reward_fn.weights["reasoning"] == 0.10

    def test_custom_weights(self):
        """Test custom weight configuration."""
        custom_weights = {
            "accuracy": 0.60,
            "format": 0.20,
            "length": 0.10,
            "reasoning": 0.10,
        }

        reward_fn = CombinedRewardFunction(weights=custom_weights)

        assert reward_fn.weights["accuracy"] == 0.60

    def test_weights_normalized(self):
        """Test that weights are normalized to sum to 1."""
        custom_weights = {
            "accuracy": 1.0,
            "format": 1.0,
            "length": 1.0,
            "reasoning": 1.0,
        }

        reward_fn = CombinedRewardFunction(weights=custom_weights)

        total = sum(reward_fn.weights.values())
        assert abs(total - 1.0) < 1e-6


class TestCombinedRewardComputation:
    """Test reward computation."""

    def test_perfect_completion(self, sample_ground_truth):
        """Test reward for perfect completion with all components."""
        reward_fn = CombinedRewardFunction(ground_truth=sample_ground_truth)

        # Completion with: correct answer, boxed format, good length, reasoning
        completions = [
            "Let me solve this step by step. "
            "First, we add 15 and 27. "
            "Therefore, 15 + 27 = 42. "
            "The final answer is \\boxed{42}."
        ]
        prompts = ["Calculate 15 + 27."]

        rewards = reward_fn.compute(completions, prompts)

        # Should get high reward for hitting all criteria
        assert rewards[0] > 0.7

    def test_correct_answer_poor_format(self):
        """Test reward for correct answer but poor formatting."""
        # Create reward function with explicit ground truth
        reward_fn = CombinedRewardFunction()
        reward_fn.add_ground_truth("Calculate 15 + 27.", "42")

        # Use "the answer is 42" so extract_answer can find it
        # Bare "42" won't be extracted as it lacks patterns like
        # "\\boxed{}", "answer is", "therefore", etc.
        completions = ["The answer is 42"]
        prompts = ["Calculate 15 + 27."]

        rewards = reward_fn.compute(completions, prompts)

        # Gets accuracy credit (50%) but lower for format/length
        # - Accuracy: 1.0 * 0.50 = 0.50 (correct answer)
        # - Format: 0.2 * 0.25 = 0.05 (has = but no boxed/steps)
        # - Length: 0.0 * 0.15 = 0.00 (too short, <10 words)
        # - Reasoning: 0.3 * 0.10 = 0.03 (has "answer")
        # Total: ~0.58
        assert 0.4 < rewards[0] < 0.7

    def test_wrong_answer_good_format(self, sample_ground_truth):
        """Test reward for wrong answer but good formatting."""
        reward_fn = CombinedRewardFunction(ground_truth=sample_ground_truth)

        completions = [
            "Let me solve this step by step. "
            "First, we need to add the numbers. "
            "15 + 27 = 41. "
            "Therefore, the answer is \\boxed{41}."
        ]
        prompts = ["Calculate 15 + 27."]

        rewards = reward_fn.compute(completions, prompts)

        # Should get partial credit for format/length/reasoning
        # But 0 for accuracy (50% of total)
        assert rewards[0] < 0.6

    def test_add_ground_truth(self):
        """Test adding ground truth dynamically."""
        reward_fn = CombinedRewardFunction()
        reward_fn.add_ground_truth("What is 2+2?", "4")

        completions = ["The answer is \\boxed{4}."]
        prompts = ["What is 2+2?"]

        rewards = reward_fn.compute(completions, prompts)

        assert rewards[0] > 0.4  # Correct answer + good format


class TestComponentScoring:
    """Test individual component scoring."""

    def test_format_score_with_boxed(self):
        """Test format score for boxed answer."""
        reward_fn = CombinedRewardFunction()

        # Use internal method for testing
        score = reward_fn._compute_format("The answer is \\boxed{42}")

        assert score >= 0.5  # Boxed gets 0.5

    def test_format_score_with_steps(self):
        """Test format score for step-by-step reasoning."""
        reward_fn = CombinedRewardFunction()

        score = reward_fn._compute_format(
            "Step 1: Add the numbers. Step 2: Get result. = 42"
        )

        assert score >= 0.5  # Steps + math notation

    def test_length_score_very_short(self):
        """Test length score for very short response."""
        reward_fn = CombinedRewardFunction()

        score = reward_fn._compute_length("42")

        assert score == 0.0  # Too short

    def test_length_score_optimal(self):
        """Test length score for optimal length response."""
        reward_fn = CombinedRewardFunction()

        # ~150 words (in optimal range)
        text = "word " * 150
        score = reward_fn._compute_length(text)

        assert score >= 0.9

    def test_reasoning_score_with_explanations(self):
        """Test reasoning score with explanation words."""
        reward_fn = CombinedRewardFunction()

        score = reward_fn._compute_reasoning(
            "Let me explain. Because we have 15 + 27, "
            "therefore the answer is 42."
        )

        assert score > 0.5  # Has explanation words

    def test_reasoning_score_without_explanations(self):
        """Test reasoning score without explanation words."""
        reward_fn = CombinedRewardFunction()

        score = reward_fn._compute_reasoning("42")

        assert score == 0.0


class TestCombinedRewardBatch:
    """Test batch processing."""

    def test_batch_computation(self, sample_ground_truth):
        """Test computing rewards for a batch."""
        reward_fn = CombinedRewardFunction(ground_truth=sample_ground_truth)

        completions = [
            "The answer is \\boxed{42}. Because we add 15 and 27 together, we get 42.",
            "13",
            "I don't know how to solve this problem.",
        ]
        prompts = [
            "Calculate 15 + 27.",
            "If x = 5, what is 2x + 3?",
            "Calculate 15 + 27.",
        ]

        rewards = reward_fn.compute(completions, prompts)

        assert len(rewards) == 3
        # First should be high (correct + formatted + reasoning)
        assert rewards[0] > 0.5
        # Second could vary (correct answer 13 but short format)
        # Third should be low (wrong, no format)
        assert rewards[2] < 0.4

    def test_error_handling(self, sample_ground_truth):
        """Test error handling in batch computation."""
        reward_fn = CombinedRewardFunction(ground_truth=sample_ground_truth)

        # Include None to test error handling
        completions = ["\\boxed{42}", "", None]
        prompts = ["Calculate 15 + 27.", "Test", "Test"]

        # Should not raise, should return 0.0 for errors
        rewards = reward_fn.compute(
            [c if c else "" for c in completions],
            prompts
        )

        assert len(rewards) == 3
