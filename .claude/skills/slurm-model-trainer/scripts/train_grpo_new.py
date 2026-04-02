#!/usr/bin/env python3
# /// script
# dependencies = [
#     "trl>=0.26.2",
#     "peft>=0.18.0",
#     "transformers>=4.57.3",
#     "accelerate>=1.12.0",
#     "datasets>=4.4.2",
#     "bitsandbytes>=0.49.0",
#     "trackio>=0.13.1",
# ]
# ///
"""
TRL GRPO (Group Relative Policy Optimization) Training Script.

Refactored to use BaseTrainerScript and the rewards module.

Features:
- Online RL with prompt-only dataset format
- Pluggable reward functions (math, format, length, combined)
- Streaming mode support for large datasets
- Trackio monitoring
- Hub push for model persistence

Supported Datasets:
- nvidia/OpenMathInstruct-2 (14M samples, streaming)
- open-r1/OpenR1-Math-220k (math reasoning)
- trl-lib/math_shepherd
- Any dataset with 'prompt' or 'problem' column

Usage:
    python train_grpo_new.py \\
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset_name nvidia/OpenMathInstruct-2 \\
        --streaming --max_samples 500000 --seed 42 \\
        --output_dir ./output \\
        --push_to_hub \\
        --hub_model_id username/my-grpo-model
"""

import argparse
import re
from typing import Any, List, Optional, Tuple

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

from base_trainer import BaseTrainerScript
from rewards import (
    MathRewardFunction,
    FormatRewardFunction,
    LengthRewardFunction,
    CombinedRewardFunction,
)


class GRPOTrainerScript(BaseTrainerScript):
    """GRPO training script using the BaseTrainerScript framework."""

    def __init__(self):
        """Initialize with GRPO-specific attributes."""
        super().__init__()
        self.reward_function = None
        self.seed = 42

    def get_method_name(self) -> str:
        """Return the training method name."""
        return "GRPO"

    def get_default_learning_rate(self) -> float:
        """GRPO uses very low learning rates."""
        return 1e-6

    def add_method_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Add GRPO-specific command line arguments."""
        # GRPO-specific arguments
        parser.add_argument(
            "--num_generations", type=int, default=4,
            help="Number of generations per prompt"
        )
        parser.add_argument(
            "--reward_type", type=str, default="combined",
            choices=["accuracy", "math", "length", "format", "combined"],
            help="Type of reward function"
        )
        parser.add_argument(
            "--max_prompt_length", type=int, default=256,
            help="Maximum prompt length"
        )

        # Dataset arguments
        parser.add_argument(
            "--streaming", action="store_true",
            help="Use streaming mode for large datasets"
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for reproducibility (shuffling)"
        )

    def prepare_dataset(self) -> Tuple[Any, Optional[Any]]:
        """Prepare dataset for GRPO training.

        GRPO requires:
        - Dataset with 'prompt' column
        - Ground truth answers stored for reward computation
        """
        print(f"Loading dataset: {self.args.dataset_name}")
        print(f"Streaming mode: {self.args.streaming}")
        print(f"Seed: {self.args.seed}")

        self.seed = self.args.seed

        # Determine reward type (auto-detect if combined)
        reward_type = self._detect_reward_type()

        # Create reward function first (so we can store ground truth)
        self.reward_function = self._create_reward_function(reward_type)

        if self.args.streaming:
            dataset = self._load_streaming_dataset()
        else:
            dataset = self._load_standard_dataset()

        print(f"Final dataset size: {len(dataset)}")
        print(f"Final columns: {dataset.column_names}")

        # GRPO doesn't use eval dataset
        return dataset, None

    def create_trainer(self) -> GRPOTrainer:
        """Create the GRPO Trainer."""
        # GRPO Training arguments
        training_args = GRPOConfig(
            output_dir=self.args.output_dir,

            # GRPO-specific
            num_generations=self.args.num_generations,
            max_completion_length=self.args.max_length,
            max_prompt_length=self.args.max_prompt_length,

            # Training hyperparameters
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,

            # Precision
            bf16=self.args.bf16,
            fp16=self.args.fp16,

            # Gradient checkpointing
            gradient_checkpointing=self.args.gradient_checkpointing,

            # Saving
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,

            # Logging
            logging_steps=self.args.logging_steps,
            report_to=self.get_report_to_list(),
            run_name=self.args.run_name,

            # Hub
            push_to_hub=self.args.push_to_hub,
            hub_model_id=self.args.hub_model_id,
            hub_strategy=self.args.hub_strategy,
        )

        # Create trainer
        return GRPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
            reward_funcs=self.reward_function,
        )

    # =========================================================================
    # GRPO-Specific Methods
    # =========================================================================

    def _detect_reward_type(self) -> str:
        """Auto-detect appropriate reward type based on dataset."""
        if self.args.reward_type != "combined":
            return self.args.reward_type

        dataset_lower = self.args.dataset_name.lower()

        # Check for math-related keywords
        math_keywords = [
            'math', 'gsm', 'numina', 'openr1', 'hendrycks',
            'olympiad', 'aime', 'amc', 'competition', 'reasoning', 'arithmetic'
        ]

        for keyword in math_keywords:
            if keyword in dataset_lower:
                print(f"Auto-detected math dataset from keyword '{keyword}'")
                return "math"

        return "combined"

    def _create_reward_function(self, reward_type: str):
        """Create the appropriate reward function.

        Args:
            reward_type: Type of reward function to create.

        Returns:
            The configured reward function instance.
        """
        print(f"Creating reward function: {reward_type}")

        if reward_type in ["accuracy", "math"]:
            return MathRewardFunction()
        elif reward_type == "format":
            return FormatRewardFunction()
        elif reward_type == "length":
            return LengthRewardFunction()
        else:  # combined
            return CombinedRewardFunction()

    def _load_streaming_dataset(self) -> Dataset:
        """Load dataset in streaming mode for large datasets."""
        dataset = load_dataset(
            self.args.dataset_name,
            self.args.dataset_config,
            split=self.args.dataset_split,
            streaming=True,
        )
        print("Dataset loaded in streaming mode")

        # Shuffle with seed
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)
        print(f"Shuffled with seed={self.seed}")

        # Take max_samples if specified
        if self.args.max_samples is not None:
            dataset = dataset.take(self.args.max_samples)
            print(f"Taking {self.args.max_samples} samples")

        # Convert streaming dataset to regular dataset
        print("Converting streaming dataset to list...")
        samples = []

        for i, example in enumerate(dataset):
            processed = self._process_example(example)
            if processed:
                samples.append(processed)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} samples...")

        print(f"Stored {len(self._get_ground_truth_count())} ground truth answers")

        return Dataset.from_list(samples)

    def _load_standard_dataset(self) -> Dataset:
        """Load dataset in standard (non-streaming) mode."""
        dataset = load_dataset(
            self.args.dataset_name,
            self.args.dataset_config,
            split=self.args.dataset_split,
        )

        print(f"Dataset columns: {dataset.column_names}")

        # Process based on dataset type
        dataset_name = self.args.dataset_name.lower()

        if "nvidia/openmathinstruc" in dataset_name:
            dataset = self._process_openmath_dataset(dataset)
        elif "open-r1/openr1-math" in dataset_name or "openr1" in dataset_name:
            dataset = self._process_openr1_dataset(dataset)
        else:
            dataset = self._process_generic_dataset(dataset)

        # Shuffle
        dataset = dataset.shuffle(seed=self.seed)
        print(f"Shuffled dataset with seed={self.seed}")

        # Limit samples
        if self.args.max_samples is not None:
            dataset = dataset.select(
                range(min(self.args.max_samples, len(dataset)))
            )

        return dataset

    def _process_example(self, example: dict) -> Optional[dict]:
        """Process a single example based on dataset format.

        Args:
            example: Raw example from dataset.

        Returns:
            Processed example with 'prompt' column.
        """
        dataset_name = self.args.dataset_name.lower()

        if "nvidia/openmathinstruc" in dataset_name:
            prompt = example.get("problem", "")
            answer = example.get("expected_answer", "")
            if prompt and answer:
                self._add_ground_truth(prompt, answer)
            return {"prompt": prompt} if prompt else None

        elif "open-r1/openr1-math" in dataset_name or "openr1" in dataset_name:
            prompt = example.get("problem", "")
            answer = example.get("answer", "")
            if prompt and answer:
                self._add_ground_truth(prompt, answer)
            return {"prompt": prompt} if prompt else None

        else:
            # Generic handling
            prompt = (
                example.get("prompt") or
                example.get("problem") or
                example.get("question") or
                ""
            )
            return {"prompt": prompt} if prompt else None

    def _process_openmath_dataset(self, dataset) -> Dataset:
        """Process nvidia/OpenMathInstruct dataset."""
        print("Detected nvidia/OpenMathInstruct dataset format")

        # Store ground truth
        if "expected_answer" in dataset.column_names:
            for example in dataset:
                problem = example.get("problem", "")
                answer = example.get("expected_answer", "")
                if problem and answer:
                    self._add_ground_truth(problem, answer)
            print(f"Stored {self._get_ground_truth_count()} ground truth answers")

        # Rename and clean columns
        if "problem" in dataset.column_names and "prompt" not in dataset.column_names:
            dataset = dataset.rename_column("problem", "prompt")

        columns_to_remove = [
            col for col in dataset.column_names if col != "prompt"
        ]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

        return dataset

    def _process_openr1_dataset(self, dataset) -> Dataset:
        """Process OpenR1-Math dataset."""
        print("Detected OpenR1-Math dataset format")

        # Store ground truth
        if "answer" in dataset.column_names:
            for example in dataset:
                problem = example.get("problem", "")
                answer = example.get("answer", "")
                if problem and answer:
                    self._add_ground_truth(problem, answer)
            print(f"Stored {self._get_ground_truth_count()} ground truth answers")

        # Rename and clean columns
        if "problem" in dataset.column_names and "prompt" not in dataset.column_names:
            dataset = dataset.rename_column("problem", "prompt")

        columns_to_remove = [
            col for col in dataset.column_names if col != "prompt"
        ]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

        return dataset

    def _process_generic_dataset(self, dataset) -> Dataset:
        """Process generic dataset format."""
        if "prompt" not in dataset.column_names:
            # Try common mappings
            prompt_columns = ["problem", "instruction", "question", "input", "query"]
            for col in prompt_columns:
                if col in dataset.column_names:
                    dataset = dataset.rename_column(col, "prompt")
                    print(f"Renamed '{col}' to 'prompt'")
                    break

        # Remove non-prompt columns
        columns_to_remove = [
            col for col in dataset.column_names if col != "prompt"
        ]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

        return dataset

    def _add_ground_truth(self, prompt: str, answer: str) -> None:
        """Add ground truth to the reward function."""
        if hasattr(self.reward_function, 'add_ground_truth'):
            self.reward_function.add_ground_truth(prompt, answer)

    def _get_ground_truth_count(self) -> int:
        """Get count of stored ground truth answers."""
        if hasattr(self.reward_function, 'ground_truth'):
            return len(self.reward_function.ground_truth)
        return 0


if __name__ == "__main__":
    GRPOTrainerScript().run()
