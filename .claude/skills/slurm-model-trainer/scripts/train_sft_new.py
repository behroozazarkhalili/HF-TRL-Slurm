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
TRL SFT (Supervised Fine-Tuning) Training Script.

Refactored to use BaseTrainerScript for reduced code duplication.

Features:
- LoRA/PEFT for efficient fine-tuning
- Automatic dataset format detection (conversational, ShareGPT, etc.)
- Trackio monitoring
- Train/eval splits for validation
- Hub push for model persistence

Usage:
    python train_sft_new.py \\
        --model_name_or_path Qwen/Qwen2.5-0.5B \\
        --dataset_name trl-lib/Capybara \\
        --output_dir ./output \\
        --push_to_hub \\
        --hub_model_id username/my-model
"""

import argparse
import os
import sys
from typing import Any, Optional, Tuple

# Ensure sibling modules (base_trainer) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from base_trainer import BaseTrainerScript


class SFTTrainerScript(BaseTrainerScript):
    """SFT training script using the BaseTrainerScript framework."""

    def get_method_name(self) -> str:
        """Return the training method name."""
        return "SFT"

    def get_default_learning_rate(self) -> float:
        """SFT typically uses higher learning rates than DPO/GRPO."""
        return 2e-5

    def add_method_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Add SFT-specific command line arguments."""
        parser.add_argument(
            "--dataset_num_proc", type=int, default=None,
            help="Number of processes for dataset preprocessing. "
                 "Auto-detects from SLURM_CPUS_PER_TASK if not set."
        )
        parser.add_argument(
            "--resume_from_checkpoint", type=str, default=None,
            help="Path to checkpoint directory to resume training from"
        )

    def prepare_dataset(self) -> Tuple[Any, Optional[Any]]:
        """Prepare dataset with intelligent format detection.

        SFT supports multiple dataset formats:
        - conversational: 'messages' column (chat format)
        - sharegpt: 'conversations' with from/value keys
        - prompt_completion: 'prompt' and 'completion' columns
        - text: 'text' column (pre-formatted)
        - instruction_output: 'instruction' and 'output' columns
        """
        print(f"Loading dataset: {self.args.dataset_name}")

        dataset = load_dataset(
            self.args.dataset_name,
            self.args.dataset_config,
            split=self.args.dataset_split,
        )

        # Get num_proc for parallel processing
        num_proc = self.args.dataset_num_proc or int(
            os.environ.get("SLURM_CPUS_PER_TASK", 8)
        )
        print(f"  Using {num_proc} processes for dataset operations")

        # Detect and handle dataset format
        format_type, columns_to_keep = self._detect_dataset_format(dataset)
        print(f"  Detected format: {format_type}")
        print(f"  Original columns: {dataset.column_names}")

        # Prepare dataset based on format
        dataset = self._prepare_dataset_for_sft(
            dataset, format_type, columns_to_keep, num_proc
        )

        # Limit samples if specified
        if self.args.max_samples is not None:
            dataset = dataset.select(range(min(self.args.max_samples, len(dataset))))

        # Create train/eval split
        return self.create_train_eval_split(dataset)

    def create_trainer(self) -> SFTTrainer:
        """Create the SFT Trainer."""
        # Dataset preprocessing config
        num_proc = self.args.dataset_num_proc or int(
            os.environ.get("SLURM_CPUS_PER_TASK", 8)
        )

        # Training arguments
        training_args = SFTConfig(
            output_dir=self.args.output_dir,

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
            optim=self.args.optim,

            # Sequence length
            max_length=self.args.max_length,

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
            eval_strategy=self.args.eval_strategy
                if self.eval_dataset is not None else "no",
            eval_steps=self.args.eval_steps
                if self.eval_dataset is not None else None,

            # Saving
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            load_best_model_at_end=(
                self.eval_dataset is not None
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
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            dataset_num_proc=num_proc,
        )

        # Create trainer
        return SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
        )

    # =========================================================================
    # SFT-Specific Dataset Handling
    # =========================================================================

    def _detect_dataset_format(self, dataset) -> Tuple[str, list]:
        """Detect dataset format for SFT training.

        Returns:
            Tuple of (format_type, columns_to_keep)
        """
        columns = dataset.column_names

        # Check for conversational format
        if 'messages' in columns:
            return 'conversational', ['messages']

        # Check for ShareGPT format
        if 'conversations' in columns:
            if self._is_sharegpt_format(dataset):
                return 'sharegpt', ['conversations']
            return 'conversational', ['conversations']

        # Check for prompt/completion format
        if 'prompt' in columns and 'completion' in columns:
            return 'prompt_completion', ['prompt', 'completion']

        # Check for text format
        if 'text' in columns:
            return 'text', ['text']

        # Check for instruction/output format
        if 'instruction' in columns and 'output' in columns:
            cols = ['instruction', 'output']
            if 'input' in columns:
                cols.insert(1, 'input')
            return 'instruction_output', cols

        return 'unknown', columns

    def _is_sharegpt_format(self, dataset) -> bool:
        """Detect if dataset uses ShareGPT format (from/value keys)."""
        if 'conversations' not in dataset.column_names:
            return False

        try:
            sample = dataset[0]['conversations']
            if not sample:
                return False
            first_turn = sample[0]
            return 'from' in first_turn and 'value' in first_turn
        except (KeyError, IndexError, TypeError):
            return False

    def _prepare_dataset_for_sft(
        self,
        dataset,
        format_type: str,
        columns_to_keep: list,
        num_proc: int
    ):
        """Prepare dataset for SFT training based on format."""

        if format_type == 'conversational':
            columns_to_remove = [
                c for c in dataset.column_names if c not in columns_to_keep
            ]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
            print(f"  Prepared conversational dataset: {dataset.column_names}")
            return dataset

        elif format_type == 'sharegpt':
            print("  Converting ShareGPT format to Messages format...")
            dataset = dataset.map(
                self._convert_sharegpt_to_messages,
                remove_columns=['conversations'],
                num_proc=num_proc,
                desc="Converting ShareGPT to Messages"
            )
            columns_to_remove = [
                c for c in dataset.column_names if c != 'messages'
            ]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
            print(f"  Conversion complete: {dataset.column_names}")
            return dataset

        elif format_type == 'instruction_output':
            dataset = dataset.map(
                self._convert_instruction_to_prompt,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
                desc="Converting to prompt/completion"
            )
            print("  Converted instruction/output to prompt/completion")
            return dataset

        elif format_type in ['prompt_completion', 'text']:
            columns_to_remove = [
                c for c in dataset.column_names if c not in columns_to_keep
            ]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
            print(f"  Using {format_type} format: {dataset.column_names}")
            return dataset

        else:
            print(f"  Warning: Unknown format. TRL will auto-detect.")
            return dataset

    @staticmethod
    def _convert_sharegpt_to_messages(example):
        """Convert ShareGPT format to TRL Messages format."""
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = []
        for turn in example["conversations"]:
            role = role_map.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})
        return {"messages": messages}

    @staticmethod
    def _convert_instruction_to_prompt(example):
        """Convert instruction/output to prompt/completion format."""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')

        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        return {'prompt': prompt, 'completion': output}


if __name__ == "__main__":
    SFTTrainerScript().run()
