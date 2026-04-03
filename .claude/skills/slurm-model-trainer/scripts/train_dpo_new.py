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
TRL DPO (Direct Preference Optimization) Training Script.

Refactored to use BaseTrainerScript for code reuse and maintainability.

Features:
- Preference learning from chosen/rejected pairs
- Support for multiple DPO loss variants (sigmoid, hinge, ipo, kto_pair)
- Automatic column mapping for various dataset formats
- Train/eval splitting for validation
- Trackio monitoring
- Hub push for model persistence

Supported Datasets:
- trl-lib/ultrafeedback_binarized
- Anthropic/hh-rlhf
- argilla/ultrafeedback-binarized-preferences
- Any dataset with prompt, chosen, rejected columns

Usage:
    python train_dpo_new.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_name trl-lib/ultrafeedback_binarized \
        --output_dir ./output \
        --push_to_hub \
        --hub_model_id username/my-dpo-model
"""

import argparse
import os
import sys
from typing import Any, Optional, Tuple

# Ensure sibling modules (base_trainer) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

from base_trainer import BaseTrainerScript, is_mig_gpu


class DPOTrainerScript(BaseTrainerScript):
    """DPO training script using the BaseTrainerScript framework."""

    def get_method_name(self) -> str:
        """Return the training method name."""
        return "DPO"

    def get_default_learning_rate(self) -> float:
        """DPO uses very low learning rates."""
        return 5e-7

    def add_method_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Add DPO-specific command line arguments."""
        # DPO-specific arguments
        parser.add_argument(
            "--beta", type=float, default=0.1,
            help="DPO beta (KL penalty coefficient)"
        )
        parser.add_argument(
            "--loss_type", type=str, default="sigmoid",
            choices=["sigmoid", "hinge", "ipo", "kto_pair"],
            help="DPO loss type"
        )
        parser.add_argument(
            "--max_prompt_length", type=int, default=512,
            help="Maximum prompt length"
        )

        # Note: --eval_strategy, --eval_steps, --test_size, --per_device_eval_batch_size
        # are already defined in the base class (BaseTrainerScript._parse_args)

    def prepare_dataset(self) -> Tuple[Any, Optional[Any]]:
        """Prepare dataset for DPO training.

        DPO requires:
        - Dataset with 'prompt', 'chosen', 'rejected' columns
        - Automatic mapping from common alternative column names
        """
        print(f"Loading dataset: {self.args.dataset_name}")

        dataset = load_dataset(
            self.args.dataset_name,
            self.args.dataset_config,
            split=self.args.dataset_split,
        )

        print(f"Dataset columns: {dataset.column_names}")

        # Validate and map columns
        dataset = self._validate_and_map_columns(dataset)

        # Limit samples if specified
        if self.args.max_samples is not None:
            dataset = dataset.select(
                range(min(self.args.max_samples, len(dataset)))
            )

        # Create train/eval split
        train_dataset, eval_dataset = self._create_splits(dataset)

        return train_dataset, eval_dataset

    def create_trainer(self) -> DPOTrainer:
        """Create the DPO Trainer."""
        # Determine if eval is enabled
        eval_enabled = self.eval_dataset is not None

        # DPO Training arguments
        training_args = DPOConfig(
            output_dir=self.args.output_dir,

            # DPO-specific
            beta=self.args.beta,
            loss_type=self.args.loss_type,
            max_length=self.args.max_length,
            max_prompt_length=self.args.max_prompt_length,

            # Training hyperparameters
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            lr_scheduler_type="cosine",

            # Precision
            bf16=self.args.bf16,
            fp16=self.args.fp16,

            # Gradient checkpointing
            gradient_checkpointing=self.args.gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False}
                if self.args.gradient_checkpointing else None
            ),

            # Evaluation
            eval_strategy=self.args.eval_strategy if eval_enabled else "no",
            eval_steps=self.args.eval_steps if eval_enabled else None,

            # Saving
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            load_best_model_at_end=(
                eval_enabled
                and self.args.save_strategy == self.args.eval_strategy
            ),

            # Logging
            logging_steps=self.args.logging_steps,
            logging_first_step=True,
            report_to=self.get_report_to_list(),
            run_name=self.args.run_name,

            # Hub
            push_to_hub=self.args.push_to_hub,
            hub_model_id=self.args.hub_model_id,
            hub_strategy=self.args.hub_strategy,

            # Other
            remove_unused_columns=False,
            dataloader_pin_memory=not is_mig_gpu(),
            dataloader_num_workers=4,
        )

        # Create trainer
        return DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
        )

    # =========================================================================
    # DPO-Specific Methods
    # =========================================================================

    def _validate_and_map_columns(self, dataset):
        """Validate and map dataset columns to required format.

        Args:
            dataset: The loaded dataset.

        Returns:
            Dataset with proper column names.
        """
        required_columns = ["prompt", "chosen", "rejected"]
        missing_columns = [
            col for col in required_columns if col not in dataset.column_names
        ]

        if not missing_columns:
            print("Dataset has all required columns")
            return dataset

        print(f"Dataset missing columns: {missing_columns}")
        print(f"Available columns: {dataset.column_names}")
        print("Attempting to map common column names...")

        # Common column name mappings
        column_mappings = {
            "prompt": ["instruction", "question", "input", "query"],
            "chosen": ["chosen_response", "preferred", "chosen_text", "positive"],
            "rejected": ["rejected_response", "dispreferred", "rejected_text", "negative"],
        }

        def map_columns(example):
            result = {}
            for target, sources in column_mappings.items():
                if target in example:
                    result[target] = example[target]
                else:
                    for source in sources:
                        if source in example:
                            result[target] = example[source]
                            break
            return result

        dataset = dataset.map(map_columns, remove_columns=dataset.column_names)

        # Post-validation: check mapped columns have non-empty content
        sample = dataset[0]
        unmapped = [
            col for col in required_columns
            if col not in sample or sample[col] == ""
        ]
        if unmapped:
            raise ValueError(
                f"Could not map required DPO columns: {unmapped}. "
                f"Dataset must have 'prompt', 'chosen', 'rejected' columns "
                f"or compatible alternatives."
            )

        return dataset

    def _create_splits(self, dataset) -> Tuple[Any, Optional[Any]]:
        """Create train/eval splits.

        Args:
            dataset: The full dataset.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        if self.args.test_size > 0 and self.args.eval_strategy != "no":
            dataset_split = dataset.train_test_split(
                test_size=self.args.test_size,
                seed=42
            )
            train_dataset = dataset_split["train"]
            eval_dataset = dataset_split["test"]
            print(f"Train samples: {len(train_dataset)}")
            print(f"Eval samples: {len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = None
            print(f"Train samples: {len(train_dataset)}")
            print("Eval: disabled")

        return train_dataset, eval_dataset


if __name__ == "__main__":
    DPOTrainerScript().run()
