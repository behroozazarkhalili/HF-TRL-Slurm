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
Minimal SFT (Supervised Fine-Tuning) Example for Fir Cluster.

A simple, self-contained script demonstrating SFT training.
Customize and embed in your SBATCH job script.

Usage:
    python train_sft_example.py

This example trains a small model on a subset of data.
Modify the constants below for your use case.
"""

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
import trackio

# ============================================================================
# CUSTOMIZE THESE FOR YOUR USE CASE
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "trl-lib/Capybara"
OUTPUT_DIR = "./output/sft-example"
HUB_MODEL_ID = None  # Set to "username/model-name" to push to Hub

# Training hyperparameters
MAX_SAMPLES = 1000  # Set to None for full dataset
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 1024

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================================


def main():
    """Run SFT training."""
    print("=" * 60)
    print("SFT Training Example")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Initialize tracking
    trackio.init(
        project="sft-example",
        name=f"sft-{MODEL_ID.split('/')[-1]}",
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_ID, split="train")

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    # Load tokenizer for chat template
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

    # Training configuration
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        push_to_hub=HUB_MODEL_ID is not None,
        hub_model_id=HUB_MODEL_ID,
        report_to=[],  # We use trackio manually
    )

    # Create trainer
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()

    if HUB_MODEL_ID:
        print(f"\nPushing to Hub: {HUB_MODEL_ID}")
        trainer.push_to_hub()

    trackio.finish()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
