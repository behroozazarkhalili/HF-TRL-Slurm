#!/usr/bin/env python3
# /// script
# dependencies = [
#     "trl>=0.26.2",
#     "peft>=0.18.0",
#     "transformers>=4.57.3",
#     "accelerate>=1.12.0",
#     "datasets>=4.4.2",
#     "trackio>=0.13.1",
# ]
# ///
"""
Minimal DPO (Direct Preference Optimization) Example for Fir Cluster.

A simple, self-contained script demonstrating DPO training.
Customize and embed in your SBATCH job script.

Usage:
    python train_dpo_example.py

DPO Requirements:
- Base model should be instruction-tuned (e.g., Qwen2.5-*-Instruct)
- Dataset must have: prompt, chosen, rejected columns
- Use lower learning rate than SFT (5e-7 recommended)
"""

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
import trackio

# ============================================================================
# CUSTOMIZE THESE FOR YOUR USE CASE
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Must be instruction-tuned
DATASET_ID = "trl-lib/ultrafeedback_binarized"
OUTPUT_DIR = "./output/dpo-example"
HUB_MODEL_ID = None  # Set to "username/model-name" to push to Hub

# Training hyperparameters
MAX_SAMPLES = 1000  # Set to None for full dataset
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 5e-7  # Lower than SFT!
NUM_EPOCHS = 1
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

# DPO-specific
BETA = 0.1  # KL penalty coefficient
LOSS_TYPE = "sigmoid"  # Options: sigmoid, hinge, ipo, kto_pair

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================================


def main():
    """Run DPO training."""
    print("=" * 60)
    print("DPO Training Example")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Beta: {BETA}, Loss: {LOSS_TYPE}")
    print("=" * 60)

    # Initialize tracking
    trackio.init(
        project="dpo-example",
        name=f"dpo-{MODEL_ID.split('/')[-1]}",
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_ID, split="train")

    # Validate columns
    required = ["prompt", "chosen", "rejected"]
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    # Split for evaluation
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    # DPO configuration
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        loss_type=LOSS_TYPE,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        push_to_hub=HUB_MODEL_ID is not None,
        hub_model_id=HUB_MODEL_ID,
        report_to=[],
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = DPOTrainer(
        model=MODEL_ID,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting DPO training...")
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()

    if HUB_MODEL_ID:
        print(f"\nPushing to Hub: {HUB_MODEL_ID}")
        trainer.push_to_hub()

    trackio.finish()

    print("\n" + "=" * 60)
    print("DPO Training Complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
