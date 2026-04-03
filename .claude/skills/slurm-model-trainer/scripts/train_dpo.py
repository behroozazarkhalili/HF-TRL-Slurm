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
TRL DPO (Direct Preference Optimization) Training Script
=========================================================
Production-ready DPO training with:
- LoRA/PEFT for efficient fine-tuning
- Trackio monitoring
- Train/eval splits for validation
- Hub push for model persistence

DPO Requirements:
- Base model should be instruction-tuned (e.g., Qwen2.5-Instruct)
- Dataset must have: prompt, chosen, rejected columns
- Lower learning rate than SFT (5e-7 recommended)

Usage:
    python train_dpo.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_name trl-lib/ultrafeedback_binarized \
        --output_dir ./output \
        --push_to_hub \
        --hub_model_id username/my-dpo-model
"""

import os
import argparse
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# Try to import trackio
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False
    print("Warning: trackio not available, using default logging")


def _is_mig() -> bool:
    """Detect MIG GPU — disable pin_memory to avoid 12x slowdown."""
    if not torch.cuda.is_available():
        return False
    try:
        p = torch.cuda.get_device_properties(0)
        if "mig" in p.name.lower():
            return True
        if "h100" in p.name.lower() and p.total_mem / 1024**3 < 60:
            return True
    except Exception:
        pass
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training with TRL")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model (should be instruction-tuned)")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="The name of the dataset (must have prompt, chosen, rejected)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration name")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Fraction of data to use for evaluation")

    # DPO-specific arguments
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL penalty coefficient)")
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "kto_pair"],
                        help="DPO loss type")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model and logs")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs (DPO typically needs fewer)")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate (lower than SFT)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length")

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
    parser.add_argument("--hub_strategy", type=str, default="end",
                        help="Hub push strategy (end=push best model only)")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency in steps")
    parser.add_argument("--report_to", type=str, default="trackio",
                        help="Reporting integration")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for tracking")
    parser.add_argument("--project", type=str, default="dpo-training",
                        help="Project name for tracking")
    parser.add_argument("--trackio_space_id", type=str, default=None,
                        help="HF Space ID for trackio (e.g., 'username/space'). If not set, logs locally.")

    # Performance arguments
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")

    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load model and tokenizer with optional quantization."""
    print(f"Loading model: {args.model_name_or_path}")

    # Auto-detect precision: bf16 may not be available on some MIG partitions
    if args.bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        print("WARNING: bf16 requested but not supported on this GPU. Falling back to fp16.")
        args.bf16 = False
        args.fp16 = True

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    # Quantization config
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # device_map="auto" is incompatible with DeepSpeed / multi-GPU accelerate.
    use_device_map = (
        not os.environ.get("ACCELERATE_USE_DEEPSPEED")
        and int(os.environ.get("WORLD_SIZE", "1")) <= 1
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto" if use_device_map else None,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded successfully")

    return model, tokenizer


def load_and_prepare_dataset(args):
    """Load dataset and validate format."""
    print(f"Loading dataset: {args.dataset_name}")

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )

    # Validate required columns
    required_columns = ["prompt", "chosen", "rejected"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]

    if missing_columns:
        print(f"WARNING: Dataset missing columns: {missing_columns}")
        print(f"Available columns: {dataset.column_names}")
        print("Attempting to map common column names...")

        # Try common mappings
        column_mappings = {
            "prompt": ["instruction", "question", "input"],
            "chosen": ["chosen_response", "preferred", "chosen_text"],
            "rejected": ["rejected_response", "dispreferred", "rejected_text"],
        }

        def map_columns(example):
            result = {}
            mapped_from = {}
            for target, sources in column_mappings.items():
                if target in example:
                    result[target] = example[target]
                    mapped_from[target] = target
                else:
                    for source in sources:
                        if source in example:
                            result[target] = example[source]
                            mapped_from[target] = source
                            break
            return result

        dataset = dataset.map(map_columns, remove_columns=dataset.column_names)

        # Post-validation: check mapped columns have non-empty content
        sample = dataset[0]
        unmapped = [col for col in required_columns if col not in sample or sample[col] == ""]
        if unmapped:
            raise ValueError(
                f"Could not map required DPO columns: {unmapped}. "
                f"Dataset must have 'prompt', 'chosen', 'rejected' columns "
                f"or compatible alternatives."
            )

    # Limit samples
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


def create_peft_config(args) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def main():
    args = parse_args()

    print("=" * 60)
    print("DPO Training with TRL")
    print("=" * 60)
    print(f"Beta (KL penalty): {args.beta}")
    print(f"Loss type: {args.loss_type}")

    # Validate push_to_hub requires hub_model_id
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--push_to_hub requires --hub_model_id to be set")

    # Initialize tracking (logs locally by default, set space_id for HF Space)
    if args.report_to == "trackio" and TRACKIO_AVAILABLE:
        run_name = args.run_name or f"dpo-{args.model_name_or_path.split('/')[-1]}"
        trackio_kwargs = {
            "project": args.project,
            "name": run_name,
            "config": {
                "model": args.model_name_or_path,
                "dataset": args.dataset_name,
                "beta": args.beta,
                "loss_type": args.loss_type,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_train_epochs,
            }
        }
        # Use space_id if explicitly set, otherwise log locally (trackio default)
        if args.trackio_space_id:
            trackio_kwargs["space_id"] = args.trackio_space_id
            print(f"Trackio logging to: HF Space {args.trackio_space_id}")
        else:
            print(f"Trackio logging locally (default)")
        trackio.init(**trackio_kwargs)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(args)

    # Create LoRA config
    peft_config = create_peft_config(args)

    # DPO Training arguments
    training_args = DPOConfig(
        output_dir=args.output_dir,

        # DPO-specific
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

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

        # Precision
        bf16=args.bf16,
        fp16=args.fp16,

        # Gradient checkpointing
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
            if args.gradient_checkpointing else None,

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
        remove_unused_columns=False,
        dataloader_pin_memory=not _is_mig(),
        dataloader_num_workers=4,
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting DPO training...")
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

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
    print("DPO Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    if args.push_to_hub:
        print(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
