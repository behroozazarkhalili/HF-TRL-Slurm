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
Minimal GRPO (Group Relative Policy Optimization) Example for Fir Cluster.

A simple, self-contained script demonstrating GRPO training for math reasoning.
Customize and embed in your SBATCH job script.

Usage:
    python train_grpo_example.py

GRPO Requirements:
- Base model should be instruction-tuned
- Dataset needs only 'prompt' (or 'problem') column
- Reward function evaluates generated completions

This example uses a simple math accuracy reward function.
"""

import re
from typing import List, Optional

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import trackio

# ============================================================================
# CUSTOMIZE THESE FOR YOUR USE CASE
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "nvidia/OpenMathInstruct-2"
OUTPUT_DIR = "./output/grpo-example"
HUB_MODEL_ID = None  # Set to "username/model-name" to push to Hub

# Training hyperparameters
MAX_SAMPLES = 5000  # Set to None for full dataset
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-6  # Very low for RL
MAX_STEPS = 500  # GRPO often uses step limit
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256
NUM_GENERATIONS = 4  # Completions per prompt

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================================


# Store ground truth for reward computation
GROUND_TRUTH = {}


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from a math solution."""
    if not text:
        return None

    # Try boxed format first
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Try "answer is" format
    match = re.search(r'answer\s+is[:\s]+([^\n.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    answer = answer.strip().lower()
    answer = re.sub(r'[{}$\\]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def math_reward(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """Reward function for math problem solving."""
    rewards = []

    for completion, prompt in zip(completions, prompts):
        try:
            predicted = extract_answer(completion)
            expected = GROUND_TRUTH.get(hash(prompt))

            if predicted and expected:
                pred_norm = normalize_answer(predicted)
                exp_norm = normalize_answer(expected)

                if pred_norm == exp_norm:
                    rewards.append(1.0)  # Exact match
                elif pred_norm in exp_norm or exp_norm in pred_norm:
                    rewards.append(0.7)  # Partial match
                else:
                    rewards.append(0.0)  # Wrong
            elif predicted:
                # Has answer but no ground truth
                if '\\boxed{' in completion:
                    rewards.append(0.5)  # Good format
                else:
                    rewards.append(0.3)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


def main():
    """Run GRPO training."""
    print("=" * 60)
    print("GRPO Training Example (Math Reasoning)")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Generations per prompt: {NUM_GENERATIONS}")
    print("=" * 60)

    # Initialize tracking
    trackio.init(
        project="grpo-example",
        name=f"grpo-{MODEL_ID.split('/')[-1]}",
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)

    # Process streaming dataset
    samples = []
    for i, example in enumerate(dataset):
        if MAX_SAMPLES and i >= MAX_SAMPLES:
            break

        problem = example.get("problem", "")
        answer = example.get("expected_answer", "")

        if problem:
            samples.append({"prompt": problem})
            if answer:
                GROUND_TRUTH[hash(problem)] = answer

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} samples...")

    from datasets import Dataset
    train_dataset = Dataset.from_list(samples)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Ground truth answers: {len(GROUND_TRUTH)}")

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

    # GRPO configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        push_to_hub=HUB_MODEL_ID is not None,
        hub_model_id=HUB_MODEL_ID,
        report_to=[],
    )

    # Create trainer with reward function
    trainer = GRPOTrainer(
        model=MODEL_ID,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=math_reward,
    )

    # Train
    print("\nStarting GRPO training...")
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()

    if HUB_MODEL_ID:
        print(f"\nPushing to Hub: {HUB_MODEL_ID}")
        trainer.push_to_hub()

    trackio.finish()

    print("\n" + "=" * 60)
    print("GRPO Training Complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
