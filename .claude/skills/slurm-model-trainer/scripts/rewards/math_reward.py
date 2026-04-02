"""
Math Reward Function for GRPO training.

Provides reward computation based on mathematical answer accuracy.
Extracts answers from completions and compares with ground truth.
"""

import re
from typing import Dict, List, Optional

from .base import RewardFunction


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a normalized answer string as a number.

    Handles plain floats, simple fractions (3/4), and avoids mangling
    non-numeric strings.
    """
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    return None


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from the LAST \\boxed{...} handling nested braces correctly.

    Uses a brace-depth counter instead of regex to handle cases like
    \\boxed{\\frac{3}{4}} or \\boxed{x^{2} + 1}.

    Finds the LAST occurrence because math solutions typically present
    the final answer in the last \\boxed{} (earlier ones may be intermediate steps).
    """
    idx = text.rfind('\\boxed{')
    if idx == -1:
        idx = text.rfind('\\boxed {')
    if idx == -1:
        return None

    brace_start = text.find('{', idx)
    if brace_start == -1:
        return None

    depth = 1
    pos = brace_start + 1
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        return text[brace_start + 1:pos - 1].strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from a math solution.

    Looks for common answer formats:
    - \\boxed{answer} (with proper nested brace handling)
    - The answer is X
    - Final answer: X
    - = X (at the end)

    Args:
        text: The completion text to extract answer from.

    Returns:
        The extracted answer string, or None if not found.
    """
    if not text:
        return None

    # Try to find boxed answer first (LaTeX format) — use brace-depth parser
    boxed = _extract_boxed_content(text)
    if boxed:
        return boxed

    # Try common answer phrases
    answer_patterns = [
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\n.]+)',
        r'[Ff]inal\s+[Aa]nswer[:\s]+([^\n.]+)',
        r'[Aa]nswer[:\s]+([^\n.]+)',
        r'[Tt]herefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)',
        r'[Ss]o[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Try to find a number at the very end
    last_line_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?)\s*$', text)
    if last_line_match:
        return last_line_match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison.

    Removes LaTeX formatting, normalizes whitespace, and handles
    common mathematical notation variations.

    Args:
        answer: The answer string to normalize.

    Returns:
        Normalized answer string.
    """
    if not answer:
        return ""

    # Remove whitespace and convert to lowercase
    answer = answer.strip().lower()

    # Convert LaTeX fractions to a/b BEFORE stripping (prevents \frac{5}{9} -> "59")
    # Pattern handles one level of nested braces: \frac{3\sqrt{3}}{2} -> 3\sqrt{3}/2
    answer = re.sub(r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1/\2', answer)

    # Remove common LaTeX formatting that wraps content
    answer = re.sub(r'\\text\s*\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\s*\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\sqrt\s*\{([^}]+)\}', r'sqrt(\1)', answer)

    # Remove remaining LaTeX commands
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = re.sub(r'[{}$]', '', answer)

    # Normalize fractions (plain text a / b -> a/b)
    answer = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', answer)

    # Remove trailing punctuation
    answer = re.sub(r'[.,;:!?]+$', '', answer)

    # Normalize whitespace
    answer = ' '.join(answer.split())

    return answer


def _prompt_key(prompt: str) -> str:
    """Create a stable lookup key from a prompt string.

    Strips whitespace and chat template tokens to ensure ground truth stored
    during dataset loading can be found during reward computation, even if
    TRL applies a chat template wrapper to the prompt.
    """
    key = re.sub(r'<\|[^|]*\|>', '', prompt)
    return key.strip()


class MathRewardFunction(RewardFunction):
    """Reward function for mathematical problem solving.

    Computes rewards based on answer accuracy by:
    1. Extracting the final answer from completions
    2. Comparing with stored ground truth answers
    3. Handling partial matches and numerical comparisons

    This class encapsulates ground truth storage, eliminating
    the need for global state.

    Attributes:
        ground_truth: Dictionary mapping prompt keys to expected answers.
    """

    def __init__(self, ground_truth: Optional[Dict[str, str]] = None):
        """Initialize the math reward function.

        Args:
            ground_truth: Dictionary mapping prompt keys to expected answers.
        """
        super().__init__(name="math_accuracy")
        self.ground_truth: Dict[str, str] = ground_truth or {}

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
        """Compute accuracy-based rewards for completions.

        Args:
            completions: List of model-generated completions.
            prompts: List of corresponding prompts.

        Returns:
            List of reward scores in [0, 1].
        """
        rewards = []

        for completion, prompt in zip(completions, prompts):
            try:
                reward = self._compute_single(completion, prompt)
                rewards.append(reward)
            except Exception as e:
                print(f"Reward computation error: {e}")
                rewards.append(0.0)

        return rewards

    def _compute_single(self, completion: str, prompt: str) -> float:
        """Compute reward for a single completion.

        Args:
            completion: The model-generated completion.
            prompt: The corresponding prompt.

        Returns:
            Reward score in [0, 1].
        """
        # Extract answer from completion
        predicted_answer = extract_answer(completion)

        # Get ground truth if available
        prompt_key = _prompt_key(prompt)
        ground_truth = self.ground_truth.get(prompt_key)

        if predicted_answer and ground_truth:
            # Compare normalized answers
            pred_norm = normalize_answer(predicted_answer)
            gt_norm = normalize_answer(ground_truth)

            if pred_norm == gt_norm:
                return 1.0  # Exact match

            if pred_norm in gt_norm or gt_norm in pred_norm:
                return 0.7  # Partial match

            # Check if both are numbers and approximately equal
            pred_num = _try_parse_number(pred_norm)
            gt_num = _try_parse_number(gt_norm)
            if pred_num is not None and gt_num is not None:
                if abs(pred_num - gt_num) < 1e-6:
                    return 1.0  # Numerically equal

            return 0.0  # No match

        elif predicted_answer:
            # Has an answer but no ground truth
            if re.search(r'\\boxed\{', completion):
                return 0.5  # Good format
            return 0.3  # Has answer but poor format

        else:
            # No answer extracted
            if len(completion.split()) > 50:
                return 0.1  # At least attempted reasoning
            return 0.0
