"""
Reward Functions for GRPO Training.

This module provides a Strategy pattern implementation for reward
computation in Group Relative Policy Optimization (GRPO) training.

The reward function hierarchy allows:
- Pluggable reward computation strategies
- No global state (answers stored in function instance)
- Easy testing and composition

Available reward functions:
- MathRewardFunction: For math problem-solving (accuracy-based)
- FormatRewardFunction: For proper formatting (\boxed{}, step-by-step)
- LengthRewardFunction: For response length (prefer detailed responses)
- CombinedRewardFunction: Weighted combination of multiple rewards

Example:
    from rewards import MathRewardFunction, CombinedRewardFunction

    # Store ground truth during dataset loading
    ground_truth = {"What is 2+2?": "4", ...}

    # Create reward function
    reward_fn = MathRewardFunction(ground_truth)

    # Use in GRPO training
    trainer = GRPOTrainer(
        ...,
        reward_funcs=reward_fn,
    )
"""

from .base import RewardFunction
from .math_reward import MathRewardFunction, extract_answer, normalize_answer
from .format_reward import FormatRewardFunction
from .length_reward import LengthRewardFunction
from .combined_reward import CombinedRewardFunction

__all__ = [
    # Base class
    "RewardFunction",
    # Concrete implementations
    "MathRewardFunction",
    "FormatRewardFunction",
    "LengthRewardFunction",
    "CombinedRewardFunction",
    # Utility functions
    "extract_answer",
    "normalize_answer",
]
