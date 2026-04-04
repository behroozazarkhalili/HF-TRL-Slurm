#!/usr/bin/env python3
# /// script
# dependencies = [
#     "unsloth[colab-new]>=2024.12",
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
Unsloth SFT (Supervised Fine-Tuning) Training Script
=====================================================
2-3x faster training with 80% less VRAM using Unsloth optimizations.

Features:
- FastLanguageModel for optimized training
- Automatic dtype selection (bf16 for H100)
- LoRA with optimized kernels
- Trackio monitoring
- Hub push for model persistence

Usage:
    python train_sft_unsloth.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --dataset_name trl-lib/Capybara \
        --output_dir ./output \
        --push_to_hub \
        --hub_model_id username/my-model
"""

import os
import argparse
from typing import Optional

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("ERROR: Unsloth not installed. Install with: pip install unsloth")
    print("Falling back to standard training...")

# Try to import trackio
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False
    print("Warning: trackio not available, using default logging")


def _is_mig() -> bool:
    """Detect broken MIG node where PyTorch can't see the GPU."""
    import os
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "MIG" in cuda_dev.upper() and not torch.cuda.is_available():
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training with Unsloth")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True,
                        help="Load model in 4-bit (default: True, use --no-load_in_4bit to disable)")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="The name of the dataset to use")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration name")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Fraction of data to use for evaluation")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout (0 for Unsloth optimization)")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model and logs")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate (higher for Unsloth)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--dataset_text_field", type=str, default=None,
                        help="Dataset column containing text (auto-detected if not set)")
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable sequence packing for efficiency (use --packing to enable)")

    # Evaluation arguments
    parser.add_argument("--eval_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation frequency in steps")

    # Saving arguments
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save frequency in steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hub model ID")
    parser.add_argument("--hub_strategy", type=str, default="every_save",
                        help="Hub push strategy")

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency in steps")
    parser.add_argument("--report_to", type=str, default="trackio",
                        help="Reporting integration")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for tracking")
    parser.add_argument("--project", type=str, default="sft-unsloth",
                        help="Project name for tracking")
    parser.add_argument("--trackio_space_id", type=str, default=None,
                        help="HF Space ID for trackio (e.g., 'username/space'). If not set, logs locally.")

    # Performance arguments
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="Use bfloat16 precision (use --no-bf16 to disable)")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable gradient checkpointing (use --no-gradient_checkpointing to disable)")

    return parser.parse_args()


def load_model_and_tokenizer_unsloth(args):
    """Load model and tokenizer with Unsloth optimizations."""
    print(f"Loading model with Unsloth: {args.model_name_or_path}")

    # Detect dtype based on GPU capability (not name matching)
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = None

    # Load with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"Model loaded with Unsloth optimizations")
    print(f"Dtype: {dtype}")

    # Apply Unsloth PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=42,
    )

    print(f"LoRA applied: r={args.lora_r}, alpha={args.lora_alpha}")

    return model, tokenizer


def load_and_prepare_dataset(args):
    """Load dataset and create train/eval splits."""
    print(f"Loading dataset: {args.dataset_name}")

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )

    # Limit samples if specified
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Create train/eval split
    if args.test_size > 0 and args.eval_strategy != "no":
        dataset_split = dataset.train_test_split(test_size=args.test_size, seed=42)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"Train samples: {len(train_dataset)}, Eval: disabled")

    return train_dataset, eval_dataset


def main():
    args = parse_args()

    if not UNSLOTH_AVAILABLE:
        print("ERROR: Unsloth is required for this script.")
        print("Install with: pip install unsloth")
        return

    print("=" * 60)
    print("SFT Training with Unsloth (2-3x Faster)")
    print("=" * 60)

    # Validate push_to_hub requires hub_model_id
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--push_to_hub requires --hub_model_id to be set")

    # Auto-detect precision BEFORE loading model
    if args.bf16:
        if _is_mig():
            print("MIG detected (torch.cuda unavailable): keeping bf16 (H100 supports bf16)")
        elif not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            print("WARNING: bf16 not supported, falling back to fp16")
            args.bf16 = False

    # Initialize tracking (logs locally by default, set space_id for HF Space)
    if args.report_to == "trackio" and TRACKIO_AVAILABLE:
        run_name = args.run_name or f"sft-unsloth-{args.model_name_or_path.split('/')[-1]}"
        trackio_kwargs = {
            "project": args.project,
            "name": run_name,
            "config": {
                "model": args.model_name_or_path,
                "dataset": args.dataset_name,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_train_epochs,
                "batch_size": args.per_device_train_batch_size,
                "lora_r": args.lora_r,
                "framework": "unsloth",
            }
        }
        # Use space_id if explicitly set, otherwise log locally (trackio default)
        if args.trackio_space_id:
            trackio_kwargs["space_id"] = args.trackio_space_id
            print(f"Trackio logging to: HF Space {args.trackio_space_id}")
        else:
            print(f"Trackio logging locally (default)")
        trackio.init(**trackio_kwargs)

    # Load model and tokenizer with Unsloth
    model, tokenizer = load_model_and_tokenizer_unsloth(args)

    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(args)

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Training hyperparameters
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",  # Use 8-bit Adam for Unsloth

        # Sequence length
        max_length=args.max_seq_length,

        # Precision
        bf16=args.bf16,
        fp16=not args.bf16,

        # Gradient checkpointing
        gradient_checkpointing=args.gradient_checkpointing,

        # Evaluation
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,

        # Saving
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(
            eval_dataset is not None
            and args.save_strategy == args.eval_strategy
        ),

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Don't pass trackio to Trainer - we handle it manually with local logging
        report_to=[] if args.report_to in ["none", "trackio"] else [args.report_to],
        run_name=args.run_name,

        # Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,

        # Other
        dataset_text_field=args.dataset_text_field,
        packing=args.packing,
        dataloader_pin_memory=not _is_mig(),
        dataloader_num_workers=4,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Print memory stats before training
    if torch.cuda.is_available():
        print(f"\nGPU Memory before training:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Train
    print("\nStarting Unsloth-optimized training...")
    train_result = trainer.train()

    # Print memory stats after training
    if torch.cuda.is_available():
        print(f"\nGPU Memory after training:")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    # Push to Hub
    if args.push_to_hub:
        print(f"\nPushing to Hub: {args.hub_model_id}")
        trainer.push_to_hub()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Final evaluation
    if eval_dataset is not None:
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    # Finish tracking
    if args.report_to == "trackio" and TRACKIO_AVAILABLE:
        trackio.finish()

    print("\n" + "=" * 60)
    print("Unsloth Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    if args.push_to_hub:
        print(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
