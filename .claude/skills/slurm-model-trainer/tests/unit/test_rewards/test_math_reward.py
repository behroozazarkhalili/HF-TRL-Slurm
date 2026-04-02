"""
Unit tests for MathRewardFunction.

Tests answer extraction, normalization, and reward computation.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rewards.math_reward import MathRewardFunction, extract_answer, normalize_answer


class TestExtractAnswer:
    """Test answer extraction from completions."""

    def test_extract_boxed_answer(self):
        """Test extracting answer from LaTeX \\boxed{} format."""
        text = "Let me solve this. The answer is \\boxed{42}."
        assert extract_answer(text) == "42"

    def test_extract_boxed_with_spaces(self):
        """Test extracting boxed answer with spaces."""
        text = "Therefore \\boxed{ 123 } is the result."
        assert extract_answer(text) == "123"

    def test_extract_answer_phrase(self):
        """Test extracting from 'the answer is' phrase."""
        text = "After calculation, the answer is 42"
        assert extract_answer(text) == "42"

    def test_extract_final_answer_phrase(self):
        """Test extracting from 'final answer' phrase."""
        text = "Final answer: 100"
        assert extract_answer(text) == "100"

    def test_extract_therefore_phrase(self):
        """Test extracting from 'therefore' phrase."""
        text = "Therefore, the answer is 55"
        assert extract_answer(text) == "55"

    def test_extract_equation_at_end(self):
        """Test extracting from equation at end."""
        text = "2 + 3 = 5"
        assert extract_answer(text) == "5"

    def test_extract_empty_returns_none(self):
        """Test that empty text returns None."""
        assert extract_answer("") is None
        assert extract_answer(None) is None

    def test_extract_no_answer_returns_none(self):
        """Test that text without answer returns None."""
        text = "Let me think about this problem..."
        assert extract_answer(text) is None


class TestNormalizeAnswer:
    """Test answer normalization."""

    def test_normalize_strips_whitespace(self):
        """Test that whitespace is stripped."""
        assert normalize_answer("  42  ") == "42"

    def test_normalize_lowercase(self):
        """Test conversion to lowercase."""
        assert normalize_answer("ANSWER") == "answer"

    def test_normalize_removes_latex_text(self):
        """Test removal of LaTeX \\text{} wrapper."""
        assert normalize_answer("\\text{yes}") == "yes"

    def test_normalize_removes_braces(self):
        """Test removal of curly braces."""
        assert normalize_answer("{42}") == "42"

    def test_normalize_removes_dollar_signs(self):
        """Test removal of dollar signs."""
        assert normalize_answer("$42$") == "42"

    def test_normalize_fractions(self):
        """Test fraction normalization."""
        assert normalize_answer("1 / 2") == "1/2"

    def test_normalize_removes_punctuation(self):
        """Test removal of trailing punctuation."""
        assert normalize_answer("42.") == "42"
        assert normalize_answer("yes,") == "yes"

    def test_normalize_empty(self):
        """Test empty string normalization."""
        assert normalize_answer("") == ""
        assert normalize_answer(None) == ""


class TestMathRewardFunction:
    """Test MathRewardFunction reward computation."""

    def test_exact_match_reward(self, sample_ground_truth):
        """Test reward for exact match."""
        reward_fn = MathRewardFunction(ground_truth=sample_ground_truth)

        completions = ["The answer is \\boxed{42}"]
        prompts = ["Calculate 15 + 27."]

        rewards = reward_fn.compute(completions, prompts)

        assert rewards[0] == 1.0

    def test_incorrect_answer_reward(self, sample_ground_truth):
        """Test reward for incorrect answer."""
        reward_fn = MathRewardFunction(ground_truth=sample_ground_truth)

        completions = ["The answer is \\boxed{41}"]
        prompts = ["Calculate 15 + 27."]

        rewards = reward_fn.compute(completions, prompts)

        assert rewards[0] == 0.0

    def test_partial_match_reward(self, sample_ground_truth):
        """Test reward for partial match (substring)."""
        reward_fn = MathRewardFunction(ground_truth=sample_ground_truth)

        # When predicted contains ground truth
        completions = ["The answer is approximately 42 units"]
        prompts = ["Calculate 15 + 27."]

        rewards = reward_fn.compute(completions, prompts)

        # Should get partial credit
        assert rewards[0] >= 0.5

    def test_no_ground_truth_with_boxed(self):
        """Test reward when no ground truth but has boxed answer."""
        reward_fn = MathRewardFunction()

        completions = ["The answer is \\boxed{42}"]
        prompts = ["What is 2+2?"]  # No ground truth stored

        rewards = reward_fn.compute(completions, prompts)

        assert rewards[0] == 0.5  # Good format, no ground truth

    def test_no_ground_truth_without_boxed(self):
        """Test reward when no ground truth and no boxed answer."""
        reward_fn = MathRewardFunction()

        completions = ["I think the answer is 4"]
        prompts = ["What is 2+2?"]

        rewards = reward_fn.compute(completions, prompts)

        assert rewards[0] == 0.3  # Has answer but poor format

    def test_no_answer_extracted_long_response(self):
        """Test reward when no answer but long reasoning."""
        reward_fn = MathRewardFunction()

        # Long response without extractable answer
        completions = [
            "This is a complex mathematical problem. "
            "Let me analyze it step by step. "
            "First, we need to consider the variables involved. "
            "Then, we apply the appropriate formulas. "
            "The calculation requires careful attention to detail. "
            "We must ensure accuracy in our intermediate steps."
        ]
        prompts = ["Solve this complex equation."]

        rewards = reward_fn.compute(completions, prompts)

        # Partial credit for reasoning - could be 0.0 or 0.1 depending on implementation
        assert rewards[0] <= 0.2

    def test_add_ground_truth(self):
        """Test adding ground truth dynamically."""
        reward_fn = MathRewardFunction()
        reward_fn.add_ground_truth("What is 5+5?", "10")

        completions = ["The answer is \\boxed{10}"]
        prompts = ["What is 5+5?"]

        rewards = reward_fn.compute(completions, prompts)

        assert rewards[0] == 1.0

    def test_callable_interface(self, sample_ground_truth):
        """Test that reward function is callable."""
        reward_fn = MathRewardFunction(ground_truth=sample_ground_truth)

        completions = ["The answer is \\boxed{42}"]
        prompts = ["Calculate 15 + 27."]

        # Should work like a function call
        rewards = reward_fn(completions, prompts)

        assert rewards[0] == 1.0

    def test_numerical_comparison(self, sample_ground_truth):
        """Test numerical comparison for floating point."""
        # Add ground truth with decimal
        reward_fn = MathRewardFunction()
        reward_fn.add_ground_truth("What is 1/3?", "0.333333")

        completions = ["The answer is \\boxed{0.333333}"]
        prompts = ["What is 1/3?"]

        rewards = reward_fn.compute(completions, prompts)

        assert rewards[0] == 1.0

    def test_batch_computation(self, sample_ground_truth):
        """Test computing rewards for a batch."""
        reward_fn = MathRewardFunction(ground_truth=sample_ground_truth)

        completions = [
            "\\boxed{42}",
            "\\boxed{13}",
            "\\boxed{wrong}",
        ]
        prompts = [
            "Calculate 15 + 27.",
            "If x = 5, what is 2x + 3?",
            "Calculate 15 + 27.",
        ]

        rewards = reward_fn.compute(completions, prompts)

        assert len(rewards) == 3
        assert rewards[0] == 1.0
        assert rewards[1] == 1.0
        assert rewards[2] == 0.0
