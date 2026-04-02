#!/usr/bin/env python3
"""Validate VRAM estimation formula against known OOM cases."""

import sys
sys.path.insert(0, "/project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/generator")
from smart_defaults import estimate_vram, select_gpu

print("VRAM Estimation Validation")
print("=" * 80)
print()

# Test cases from our OOM analysis
test_cases = [
    # (name, params_b, method, max_length, batch_size, num_gen, expected_outcome)
    ("LFM2-350M GRPO", 0.35, "grpo", 2048, 1, 2, "40GB - COMPLETED"),
    ("LFM2-700M GRPO", 0.7, "grpo", 2048, 1, 2, "40GB - COMPLETED"),
    ("LFM2-1.2B GRPO", 1.2, "grpo", 2048, 1, 2, "40GB - OOM then COMPLETED"),
    ("LFM2-2.6B GRPO", 2.6, "grpo", 2048, 1, 2, "40GB - OOM (needs 80GB)"),
    ("SmolLM2-360M GRPO", 0.36, "grpo", 2048, 2, 4, "40GB - OOM"),
    ("SmolLM2-1.7B GRPO", 1.7, "grpo", 2048, 1, 4, "40GB - OOM"),
    # SFT cases for comparison
    ("LFM2-700M SFT", 0.7, "sft", 2048, 8, 1, "Should fit in 20GB"),
    ("LFM2-2.6B SFT", 2.6, "sft", 2048, 4, 1, "Should fit in 40GB"),
]

print(f"{'Model':<25} {'Est VRAM':>10} {'GPU Rec':>12} {'Actual Outcome':<30}")
print("-" * 80)

for name, params, method, seq, batch, gens, expected in test_cases:
    vram = estimate_vram(params, method, max_length=seq, batch_size=batch, num_generations=gens)
    gpu = select_gpu(params, method, max_length=seq, batch_size=batch, num_generations=gens)

    print(f"{name:<25} {vram:>8.1f}GB {gpu.name:>12} {expected:<30}")

print()
print("=" * 80)
print("Key: Formula should recommend 80GB H100 for cases that OOM'd on 40GB MIG")
print()

# Detailed breakdown for LFM2-2.6B GRPO (the key OOM case)
print("Detailed breakdown for LFM2-2.6B GRPO (key OOM case):")
print("-" * 60)
params_b = 2.6
method = "grpo"
max_length = 2048
batch_size = 1
num_generations = 2

# Manual calculation for visibility
hidden_dim = 3072  # Estimated for ~3B models
num_layers = 36    # Estimated for ~3B models

model_memory = params_b * 2.0
lora_overhead = params_b * 0.5
seq_factor = max_length / 512
activation_memory = params_b * seq_factor * 0.8

kv_cache_per_gen = (2 * hidden_dim * num_layers * max_length * 2) / 1e9
kv_cache_total = batch_size * num_generations * kv_cache_per_gen

vocab_size = 128000
logits_mem = (batch_size * num_generations * max_length * vocab_size * 2) / 1e9

grpo_activation = activation_memory * num_generations

total = model_memory + lora_overhead + grpo_activation + kv_cache_total + logits_mem * 0.3

print(f"  Model memory (bf16):     {model_memory:.2f} GB")
print(f"  LoRA overhead:           {lora_overhead:.2f} GB")
print(f"  Activations (GRPO):      {grpo_activation:.2f} GB")
print(f"  KV Cache (total):        {kv_cache_total:.2f} GB")
print(f"  Logits (30%):            {logits_mem * 0.3:.2f} GB")
print(f"  ---------------------------------")
print(f"  TOTAL:                   {total:.2f} GB")
print(f"  With 25% headroom:       {total * 1.25:.2f} GB")
print()
