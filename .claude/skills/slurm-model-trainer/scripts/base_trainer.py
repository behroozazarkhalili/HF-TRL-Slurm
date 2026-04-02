#!/usr/bin/env python3
"""
Base Trainer Script for TRL Training Methods.

This module provides an abstract base class that encapsulates all shared
functionality across SFT, DPO, and GRPO training scripts, eliminating
~50% code duplication.

The class follows the Template Method Pattern:
- Base class defines the skeleton of the training algorithm
- Subclasses override specific steps (dataset prep, trainer creation)

Example:
    class SFTTrainerScript(BaseTrainerScript):
        def add_method_specific_args(self, parser):
            parser.add_argument("--packing", action="store_true")

        def create_trainer(self):
            return SFTTrainer(...)

        def prepare_dataset(self):
            # SFT-specific dataset preparation
            ...

    if __name__ == "__main__":
        SFTTrainerScript().run()
"""

import argparse
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Optional trackio import
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False


@dataclass
class TrainingArgs:
    """Shared training arguments across all methods.

    This dataclass captures the common configuration that applies
    to SFT, DPO, and GRPO training.
    """
    # Model
    model_name_or_path: str = ""
    use_4bit: bool = False
    use_8bit: bool = False

    # Dataset
    dataset_name: str = ""
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    test_size: float = 0.1

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    output_dir: str = "./output"
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 2048

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100

    # Saving
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "end"

    # Logging
    logging_steps: int = 10
    report_to: str = "trackio"
    run_name: Optional[str] = None
    project: str = "trl-training"
    trackio_space_id: Optional[str] = None

    # Performance
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"


class BaseTrainerScript(ABC):
    """Abstract base class for TRL training scripts.

    This class implements the Template Method Pattern, defining the
    overall training algorithm while allowing subclasses to override
    specific steps.

    Subclasses must implement:
    - add_method_specific_args(): Add method-specific CLI arguments
    - create_trainer(): Create the appropriate TRL Trainer
    - prepare_dataset(): Prepare the dataset for the training method

    Subclasses may optionally override:
    - get_method_name(): Return the training method name
    - get_default_learning_rate(): Return default LR for the method
    - post_training_hook(): Run additional processing after training
    """

    def __init__(self):
        """Initialize the trainer script."""
        self.args: Optional[argparse.Namespace] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.train_dataset: Optional[Any] = None
        self.eval_dataset: Optional[Any] = None
        self.peft_config: Optional[LoraConfig] = None

    # =========================================================================
    # Template Method - Main Entry Point
    # =========================================================================

    def run(self) -> None:
        """Execute the training pipeline.

        This is the main entry point that orchestrates the entire
        training process using the Template Method Pattern.
        """
        # Parse arguments
        self.args = self._parse_args()

        # Print header
        self._print_header()

        # Initialize tracking
        self._init_tracking()

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()

        # Prepare dataset
        self.train_dataset, self.eval_dataset = self.prepare_dataset()

        # Create LoRA config
        self.peft_config = self._create_peft_config()
        print(f"LoRA config: r={self.args.lora_r}, alpha={self.args.lora_alpha}")

        # Create trainer (delegated to subclass)
        trainer = self.create_trainer()

        # Train
        print(f"\nStarting {self.get_method_name()} training...")
        train_result = trainer.train()

        # Save final model
        print("\nSaving final model...")
        trainer.save_model()

        # Push to Hub
        if self.args.push_to_hub:
            print(f"\nPushing to Hub: {self.args.hub_model_id}")
            trainer.push_to_hub()

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Final evaluation (if eval dataset exists)
        if self.eval_dataset is not None:
            print("\nRunning final evaluation...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        # Post-training hook (optional override)
        self.post_training_hook(trainer)

        # Finish tracking
        self._finish_tracking()

        # Print summary
        self._print_summary()

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def add_method_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Add method-specific command line arguments.

        Args:
            parser: The argument parser to add arguments to.
        """
        pass

    @abstractmethod
    def create_trainer(self) -> Any:
        """Create the method-specific TRL Trainer.

        Returns:
            The configured TRL Trainer (SFTTrainer, DPOTrainer, GRPOTrainer).
        """
        pass

    @abstractmethod
    def prepare_dataset(self) -> Tuple[Any, Optional[Any]]:
        """Prepare the dataset for training.

        Returns:
            Tuple of (train_dataset, eval_dataset).
            eval_dataset may be None if evaluation is disabled.
        """
        pass

    # =========================================================================
    # Hook Methods (optional override)
    # =========================================================================

    def get_method_name(self) -> str:
        """Get the training method name for display.

        Override this in subclasses.

        Returns:
            The method name (e.g., "SFT", "DPO", "GRPO").
        """
        return "TRL"

    def get_default_learning_rate(self) -> float:
        """Get the default learning rate for this method.

        Override this in subclasses. Different methods have different
        optimal learning rates:
        - SFT: 2e-5 to 2e-4
        - DPO: 5e-6 to 5e-5
        - GRPO: 1e-6 to 1e-5

        Returns:
            The default learning rate.
        """
        return 2e-5

    def post_training_hook(self, trainer: Any) -> None:
        """Optional hook for post-training processing.

        Override this in subclasses to add method-specific post-processing
        (e.g., evaluation, model merging).

        Args:
            trainer: The completed trainer instance.
        """
        pass

    # =========================================================================
    # Shared Implementation (not meant to be overridden)
    # =========================================================================

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments.

        This method builds the argument parser with common arguments
        and then calls add_method_specific_args() to allow subclasses
        to add their own arguments.

        Returns:
            Parsed arguments namespace.
        """
        parser = argparse.ArgumentParser(
            description=f"{self.get_method_name()} Training with TRL"
        )

        # ===== Model arguments =====
        parser.add_argument(
            "--model_name_or_path", type=str, required=True,
            help="Path to pretrained model or HuggingFace model ID"
        )
        parser.add_argument(
            "--use_4bit", action="store_true",
            help="Use 4-bit quantization (QLoRA)"
        )
        parser.add_argument(
            "--use_8bit", action="store_true",
            help="Use 8-bit quantization"
        )

        # ===== Dataset arguments =====
        parser.add_argument(
            "--dataset_name", type=str, required=True,
            help="The name of the dataset to use"
        )
        parser.add_argument(
            "--dataset_config", type=str, default=None,
            help="Dataset configuration name"
        )
        parser.add_argument(
            "--dataset_split", type=str, default="train",
            help="Dataset split to use"
        )
        parser.add_argument(
            "--max_samples", type=int, default=None,
            help="Maximum number of samples to use"
        )
        parser.add_argument(
            "--test_size", type=float, default=0.1,
            help="Fraction of data to use for evaluation"
        )

        # ===== LoRA arguments =====
        parser.add_argument(
            "--lora_r", type=int, default=16,
            help="LoRA attention dimension"
        )
        parser.add_argument(
            "--lora_alpha", type=int, default=32,
            help="LoRA alpha parameter"
        )
        parser.add_argument(
            "--lora_dropout", type=float, default=0.05,
            help="LoRA dropout"
        )
        parser.add_argument(
            "--target_modules", type=str, nargs="+",
            default=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
            help="Target modules for LoRA"
        )

        # ===== Training arguments =====
        parser.add_argument(
            "--output_dir", type=str, required=True,
            help="Output directory for model and logs"
        )
        parser.add_argument(
            "--num_train_epochs", type=int, default=3,
            help="Number of training epochs"
        )
        parser.add_argument(
            "--max_steps", type=int, default=-1,
            help="Maximum number of training steps (-1 for full epochs)"
        )
        parser.add_argument(
            "--per_device_train_batch_size", type=int, default=4,
            help="Training batch size per device"
        )
        parser.add_argument(
            "--per_device_eval_batch_size", type=int, default=4,
            help="Evaluation batch size per device"
        )
        parser.add_argument(
            "--gradient_accumulation_steps", type=int, default=4,
            help="Gradient accumulation steps"
        )
        parser.add_argument(
            "--learning_rate", type=float, default=self.get_default_learning_rate(),
            help="Learning rate"
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0.01,
            help="Weight decay"
        )
        parser.add_argument(
            "--warmup_ratio", type=float, default=0.1,
            help="Warmup ratio"
        )
        parser.add_argument(
            "--max_length", type=int, default=2048,
            help="Maximum sequence length"
        )

        # ===== Evaluation arguments =====
        parser.add_argument(
            "--eval_strategy", type=str, default="steps",
            choices=["no", "steps", "epoch"],
            help="Evaluation strategy"
        )
        parser.add_argument(
            "--eval_steps", type=int, default=100,
            help="Evaluation frequency in steps"
        )

        # ===== Saving arguments =====
        parser.add_argument(
            "--save_strategy", type=str, default="steps",
            choices=["no", "steps", "epoch"],
            help="Checkpoint save strategy"
        )
        parser.add_argument(
            "--save_steps", type=int, default=100,
            help="Save frequency in steps"
        )
        parser.add_argument(
            "--save_total_limit", type=int, default=3,
            help="Maximum number of checkpoints to keep"
        )

        # ===== Hub arguments =====
        parser.add_argument(
            "--push_to_hub", action="store_true",
            help="Push model to Hugging Face Hub"
        )
        parser.add_argument(
            "--hub_model_id", type=str, default=None,
            help="Hub model ID (e.g., username/model-name)"
        )
        parser.add_argument(
            "--hub_strategy", type=str, default="end",
            choices=["end", "every_save", "checkpoint", "all_checkpoints"],
            help="Hub push strategy"
        )

        # ===== Logging arguments =====
        parser.add_argument(
            "--logging_steps", type=int, default=10,
            help="Logging frequency in steps"
        )
        parser.add_argument(
            "--report_to", type=str, default="trackio",
            help="Reporting integration (trackio, wandb, tensorboard, none)"
        )
        parser.add_argument(
            "--run_name", type=str, default=None,
            help="Run name for tracking"
        )
        parser.add_argument(
            "--project", type=str, default=f"{self.get_method_name().lower()}-training",
            help="Project name for tracking"
        )
        parser.add_argument(
            "--trackio_space_id", type=str, default=None,
            help="HF Space ID for trackio. If not set, logs locally."
        )

        # ===== Performance arguments =====
        parser.add_argument(
            "--bf16", action="store_true",
            help="Use bfloat16 precision"
        )
        parser.add_argument(
            "--fp16", action="store_true",
            help="Use float16 precision"
        )
        parser.add_argument(
            "--gradient_checkpointing", action="store_true",
            help="Enable gradient checkpointing"
        )
        parser.add_argument(
            "--optim", type=str, default="adamw_torch",
            help="Optimizer to use"
        )

        # Add method-specific arguments
        self.add_method_specific_args(parser)

        return parser.parse_args()

    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer with optional quantization.

        Returns:
            Tuple of (model, tokenizer).
        """
        print(f"Loading model: {self.args.model_name_or_path}")

        # Auto-detect precision: bf16 may not be available on some MIG partitions
        if self.args.bf16 and not torch.cuda.is_bf16_supported():
            print("WARNING: bf16 requested but not supported on this GPU. Falling back to fp16.")
            self.args.bf16 = False
            self.args.fp16 = True

        # Quantization config
        compute_dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
        quantization_config = None
        if self.args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.args.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Determine dtype
        if self.args.bf16:
            dtype = torch.bfloat16
        elif self.args.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        print(f"Model loaded successfully")
        print(f"  Dtype: {model.dtype}")
        print(f"  Device: {model.device}")

        return model, tokenizer

    def _create_peft_config(self) -> LoraConfig:
        """Create LoRA configuration.

        Returns:
            Configured LoraConfig object.
        """
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.args.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    def _init_tracking(self) -> None:
        """Initialize experiment tracking (trackio)."""
        if self.args.report_to != "trackio" or not TRACKIO_AVAILABLE:
            return

        run_name = self.args.run_name or \
            f"{self.get_method_name().lower()}-{self.args.model_name_or_path.split('/')[-1]}"

        trackio_kwargs = {
            "project": self.args.project,
            "name": run_name,
            "config": {
                "model": self.args.model_name_or_path,
                "dataset": self.args.dataset_name,
                "learning_rate": self.args.learning_rate,
                "num_epochs": self.args.num_train_epochs,
                "batch_size": self.args.per_device_train_batch_size,
                "lora_r": self.args.lora_r,
                "method": self.get_method_name(),
            }
        }

        if self.args.trackio_space_id:
            trackio_kwargs["space_id"] = self.args.trackio_space_id
            print(f"Trackio logging to: HF Space {self.args.trackio_space_id}")
        else:
            print("Trackio logging locally (default)")

        trackio.init(**trackio_kwargs)

    def _finish_tracking(self) -> None:
        """Finish experiment tracking."""
        if self.args.report_to == "trackio" and TRACKIO_AVAILABLE:
            trackio.finish()

    def _print_header(self) -> None:
        """Print training header with configuration summary."""
        print("=" * 60)
        print(f"{self.get_method_name()} Training with TRL")
        print("=" * 60)

    def _print_summary(self) -> None:
        """Print training completion summary."""
        print("\n" + "=" * 60)
        print(f"{self.get_method_name()} Training Complete!")
        print("=" * 60)
        print(f"Model saved to: {self.args.output_dir}")
        if self.args.push_to_hub:
            print(f"Model pushed to: https://huggingface.co/{self.args.hub_model_id}")

    def get_report_to_list(self) -> list[str]:
        """Get the report_to list for the Trainer.

        When using trackio, we handle logging manually, so we return
        an empty list to avoid duplicate logging.

        Returns:
            List of reporting integrations.
        """
        if self.args.report_to in ["none", "trackio"]:
            return []
        return [self.args.report_to]

    def load_dataset_simple(self) -> Any:
        """Load dataset without any transformation.

        This is a helper method that subclasses can use as a starting
        point for their prepare_dataset() implementation.

        Returns:
            The raw loaded dataset.
        """
        print(f"Loading dataset: {self.args.dataset_name}")

        dataset = load_dataset(
            self.args.dataset_name,
            self.args.dataset_config,
            split=self.args.dataset_split,
        )

        # Limit samples if specified
        if self.args.max_samples is not None:
            dataset = dataset.select(range(min(self.args.max_samples, len(dataset))))

        return dataset

    def create_train_eval_split(self, dataset: Any) -> Tuple[Any, Optional[Any]]:
        """Create train/eval split from a dataset.

        Args:
            dataset: The dataset to split.

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
            print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = None
            print(f"Train samples: {len(train_dataset)}, Eval: disabled")

        return train_dataset, eval_dataset
