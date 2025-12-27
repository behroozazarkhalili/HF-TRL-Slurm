#!/usr/bin/env python3
"""
Training Time and Cost Estimator
=================================
Estimate walltime and resource usage for training jobs.

Usage:
    python estimate_time.py \
        --model Qwen/Qwen2.5-0.5B \
        --dataset trl-lib/Capybara \
        --epochs 3 \
        --gpu h100
"""

import argparse
from typing import Dict, Tuple


# Model size estimates (in billions of parameters)
MODEL_SIZES = {
    # Qwen family
    "Qwen/Qwen2.5-0.5B": 0.5,
    "Qwen/Qwen2.5-1.5B": 1.5,
    "Qwen/Qwen2.5-3B": 3,
    "Qwen/Qwen2.5-7B": 7,
    "Qwen/Qwen2.5-14B": 14,
    "Qwen/Qwen2.5-32B": 32,
    "Qwen/Qwen2.5-72B": 72,

    # Llama family
    "meta-llama/Llama-2-7b-hf": 7,
    "meta-llama/Llama-2-13b-hf": 13,
    "meta-llama/Llama-2-70b-hf": 70,
    "meta-llama/Llama-3.1-8B": 8,
    "meta-llama/Llama-3.1-70B": 70,

    # Mistral family
    "mistralai/Mistral-7B-v0.1": 7,
    "mistralai/Mixtral-8x7B-v0.1": 47,  # Active params

    # Phi family
    "microsoft/phi-2": 2.7,
    "microsoft/Phi-3-mini-4k-instruct": 3.8,
}

# GPU specs and relative speeds
GPU_SPECS = {
    "h100": {
        "vram_gb": 80,
        "speed_multiplier": 1.0,  # Baseline
        "description": "NVIDIA H100 80GB",
    },
    "h100_3g": {
        "vram_gb": 40,
        "speed_multiplier": 0.5,
        "description": "H100 MIG 3g.40gb",
    },
    "h100_2g": {
        "vram_gb": 20,
        "speed_multiplier": 0.25,
        "description": "H100 MIG 2g.20gb",
    },
    "h100_1g": {
        "vram_gb": 10,
        "speed_multiplier": 0.125,
        "description": "H100 MIG 1g.10gb",
    },
    "a100": {
        "vram_gb": 80,
        "speed_multiplier": 0.7,
        "description": "NVIDIA A100 80GB",
    },
    "a10g": {
        "vram_gb": 24,
        "speed_multiplier": 0.3,
        "description": "NVIDIA A10G 24GB",
    },
}

# Training method multipliers
METHOD_MULTIPLIERS = {
    "sft": 1.0,
    "dpo": 1.5,  # DPO is slower (processes pairs)
    "grpo": 2.0,  # GRPO generates responses online
}

# Unsloth speedup
UNSLOTH_SPEEDUP = 2.5  # 2-3x faster


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate training time")

    parser.add_argument("--model", type=str, required=True,
                        help="Model name or size in billions (e.g., '7' for 7B)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (for known sizes)")
    parser.add_argument("--dataset_size", type=int, default=10000,
                        help="Number of training examples")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--method", type=str, default="sft",
                        choices=["sft", "dpo", "grpo"],
                        help="Training method")
    parser.add_argument("--gpu", type=str, default="h100",
                        choices=list(GPU_SPECS.keys()),
                        help="GPU type")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--use_unsloth", action="store_true",
                        help="Using Unsloth optimization")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Using LoRA (default: True)")

    return parser.parse_args()


def get_model_size(model_name: str) -> float:
    """Get model size in billions of parameters."""
    if model_name in MODEL_SIZES:
        return MODEL_SIZES[model_name]

    # Try to parse size from name
    try:
        return float(model_name)
    except ValueError:
        pass

    # Try to extract from model name
    import re
    match = re.search(r'(\d+\.?\d*)b', model_name.lower())
    if match:
        return float(match.group(1))

    # Default
    print(f"Warning: Unknown model '{model_name}', assuming 7B")
    return 7.0


def estimate_vram(model_size_b: float, use_lora: bool = True) -> float:
    """Estimate VRAM usage in GB."""
    if use_lora:
        # LoRA: ~4GB per billion parameters
        return model_size_b * 4
    else:
        # Full fine-tuning: ~20GB per billion parameters
        return model_size_b * 20


def estimate_training_time(
    model_size_b: float,
    dataset_size: int,
    epochs: int,
    method: str,
    gpu: str,
    num_gpus: int,
    use_unsloth: bool,
) -> Tuple[float, Dict]:
    """Estimate training time in hours."""

    # Base formula: hours = 0.1 * params_B * (examples/1000) * epochs
    base_hours = 0.1 * model_size_b * (dataset_size / 1000) * epochs

    # Apply method multiplier
    method_mult = METHOD_MULTIPLIERS.get(method, 1.0)
    hours = base_hours * method_mult

    # Apply GPU speed
    gpu_spec = GPU_SPECS[gpu]
    hours = hours / gpu_spec["speed_multiplier"]

    # Apply multi-GPU scaling (assume 80% efficiency)
    if num_gpus > 1:
        hours = hours / (num_gpus * 0.8)

    # Apply Unsloth speedup
    if use_unsloth:
        hours = hours / UNSLOTH_SPEEDUP

    # Calculate recommended walltime (with 30% buffer)
    walltime_hours = hours * 1.3

    details = {
        "base_hours": base_hours,
        "method_multiplier": method_mult,
        "gpu_speed": gpu_spec["speed_multiplier"],
        "num_gpus": num_gpus,
        "unsloth_speedup": UNSLOTH_SPEEDUP if use_unsloth else 1.0,
        "estimated_hours": hours,
        "recommended_walltime_hours": walltime_hours,
    }

    return hours, details


def format_time(hours: float) -> str:
    """Format hours as HH:MM:SS."""
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    args = parse_args()

    print("=" * 60)
    print("Training Time Estimator")
    print("=" * 60)

    # Get model size
    model_size = get_model_size(args.model)
    print(f"\nModel: {args.model}")
    print(f"Size: {model_size:.1f}B parameters")

    # Estimate VRAM
    vram_needed = estimate_vram(model_size, args.use_lora)
    gpu_vram = GPU_SPECS[args.gpu]["vram_gb"]
    print(f"\nVRAM Required: {vram_needed:.1f} GB")
    print(f"GPU VRAM: {gpu_vram} GB ({GPU_SPECS[args.gpu]['description']})")

    if vram_needed > gpu_vram:
        print(f"⚠️  WARNING: Model may not fit! Consider:")
        print(f"   - Using LoRA (--use_lora)")
        print(f"   - Using larger GPU")
        print(f"   - Using multi-GPU")

    # Estimate time
    print(f"\nTraining Configuration:")
    print(f"  Dataset: {args.dataset or 'custom'} ({args.dataset_size:,} examples)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Method: {args.method.upper()}")
    print(f"  GPUs: {args.num_gpus}x {args.gpu}")
    print(f"  Unsloth: {'Yes' if args.use_unsloth else 'No'}")

    hours, details = estimate_training_time(
        model_size,
        args.dataset_size,
        args.epochs,
        args.method,
        args.gpu,
        args.num_gpus,
        args.use_unsloth,
    )

    print(f"\n{'=' * 60}")
    print("Estimates")
    print("=" * 60)
    print(f"Estimated Training Time: {hours:.2f} hours ({format_time(hours)})")
    print(f"Recommended Walltime:    {details['recommended_walltime_hours']:.2f} hours ({format_time(details['recommended_walltime_hours'])})")

    # Suggest partition based on time
    if details['recommended_walltime_hours'] <= 3:
        partition = "gpubase_bygpu_b1 (3hr limit)"
    elif details['recommended_walltime_hours'] <= 24:
        partition = "gpubase_bygpu_b3 (1-day limit)"
    elif details['recommended_walltime_hours'] <= 72:
        partition = "gpubase_bygpu_b4 (3-day limit)"
    else:
        partition = "gpubase_bygpu_b5 (7-day limit)"

    print(f"Suggested Partition:     {partition}")

    # SBATCH format
    walltime_fmt = format_time(details['recommended_walltime_hours'])
    print(f"\nSBATCH time parameter:   --time={walltime_fmt}")

    print("\nNote: These are estimates. Actual time may vary based on:")
    print("  - Sequence length and packing efficiency")
    print("  - Evaluation frequency")
    print("  - Checkpoint saving")
    print("  - Network I/O for Hub push")


if __name__ == "__main__":
    main()
