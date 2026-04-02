#!/usr/bin/env python3
"""
Model Card Generator for HuggingFace
=====================================
Generates comprehensive README.md files for model uploads following
the established naming conventions and formatting standards.

Supports all training methods: SFT, DPO, GRPO, RLHF

Naming Convention: {BaseModel}-{TrainingMethod}-{Dataset}[-Quantization]
Example: Qwen2.5-7B-SFT-Capybara, Qwen3-8B-GRPO-OpenMath2

Usage:
    # For SFT models:
    python generate_model_card.py \
        --model_name "Qwen2.5-7B-SFT-Capybara" \
        --base_model "Qwen/Qwen2.5-7B" \
        --dataset "trl-lib/Capybara" \
        --training_method "SFT" \
        --training_log /path/to/training.log

    # For GRPO models:
    python generate_model_card.py \
        --model_name "Qwen3-8B-GRPO-OpenMath2" \
        --base_model "Qwen/Qwen3-8B" \
        --dataset "nvidia/OpenMathInstruct-2" \
        --training_method "GRPO" \
        --num_generations 2 \
        --reward_type combined \
        --training_log /path/to/training.log
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, Optional, Any


def get_package_versions() -> Dict[str, str]:
    """Get clean package versions without cluster-specific suffixes."""
    versions = {}

    try:
        import trl
        versions['trl'] = getattr(trl, '__version__', 'unknown').split('+')[0]
    except ImportError:
        versions['trl'] = 'N/A'

    try:
        import transformers
        versions['transformers'] = getattr(transformers, '__version__', 'unknown').split('+')[0]
    except ImportError:
        versions['transformers'] = 'N/A'

    try:
        import torch
        versions['pytorch'] = getattr(torch, '__version__', 'unknown').split('+')[0]
    except ImportError:
        versions['pytorch'] = 'N/A'

    try:
        import datasets
        versions['datasets'] = getattr(datasets, '__version__', 'unknown').split('+')[0]
    except ImportError:
        versions['datasets'] = 'N/A'

    try:
        import tokenizers
        versions['tokenizers'] = getattr(tokenizers, '__version__', 'unknown').split('+')[0]
    except ImportError:
        versions['tokenizers'] = 'N/A'

    try:
        import peft
        versions['peft'] = getattr(peft, '__version__', 'unknown').split('+')[0]
    except ImportError:
        versions['peft'] = 'N/A'

    try:
        import bitsandbytes
        versions['bitsandbytes'] = getattr(bitsandbytes, '__version__', 'unknown').split('+')[0]
    except ImportError:
        versions['bitsandbytes'] = 'N/A'

    return versions


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comprehensive model card for all RL methods")

    # Model information
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., Qwen2.5-7B-SFT-Capybara)")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model ID (e.g., Qwen/Qwen2.5-7B)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., trl-lib/Capybara)")
    parser.add_argument("--training_method", type=str, default="SFT",
                        choices=["SFT", "DPO", "GRPO", "RLHF", "PPO", "ORPO"],
                        help="Training method used")

    # Author information
    parser.add_argument("--author", type=str, default="ermiaazarkhalili",
                        help="Author HuggingFace username")
    parser.add_argument("--license", type=str, default="cc-by-nc-4.0",
                        help="License (e.g., cc-by-nc-4.0, apache-2.0, mit)")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None,
                        help="Maximum sequence length (alias for max_seq_length)")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=None,
                        help="Maximum prompt length (for GRPO/DPO)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization type (e.g., 4-bit, 8-bit)")

    # GRPO-specific parameters
    parser.add_argument("--num_generations", type=int, default=None,
                        help="Number of generations per prompt (GRPO)")
    parser.add_argument("--reward_type", type=str, default=None,
                        choices=["accuracy", "format", "combined", "math", "code"],
                        help="Reward function type (GRPO)")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient (GRPO/DPO)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")

    # DPO-specific parameters
    parser.add_argument("--dpo_loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo"],
                        help="DPO loss type")

    # Metrics
    parser.add_argument("--train_loss", type=float, default=None)
    parser.add_argument("--eval_loss", type=float, default=None)
    parser.add_argument("--accuracy", type=float, default=None,
                        help="Token accuracy (0-1 scale)")
    parser.add_argument("--reward", type=float, default=None,
                        help="Average reward (GRPO)")
    parser.add_argument("--training_time", type=str, default=None)
    parser.add_argument("--hardware", type=str, default="NVIDIA H100 80GB")

    # Dataset info
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Whether dataset was streamed")

    # Files
    parser.add_argument("--training_log", type=str, default=None,
                        help="Path to training log file for auto-extraction")
    parser.add_argument("--training_args", type=str, default=None,
                        help="Path to training_args.json for auto-extraction")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Output directory for README.md")

    # GGUF info
    parser.add_argument("--is_gguf", action="store_true",
                        help="Generate README for GGUF repository")
    parser.add_argument("--gguf_quantizations", type=str, nargs="+",
                        default=["Q4_K_M", "Q5_K_M", "Q8_0"],
                        help="Available GGUF quantizations")

    return parser.parse_args()


def extract_from_training_log(log_path: str, training_method: str = "SFT") -> Dict[str, Any]:
    """Extract training information from log file."""
    info = {}

    if not os.path.exists(log_path):
        return info

    with open(log_path, 'r') as f:
        content = f.read()

    # Extract learning rate
    lr_match = re.search(r"[Ll]earning[_\s]*[Rr]ate['\"]?\s*[:=]\s*([\d.e-]+)", content)
    if lr_match:
        info['learning_rate'] = float(lr_match.group(1))

    # Extract batch size
    bs_match = re.search(r"per_device_train_batch_size['\"]?\s*[:=]\s*(\d+)", content)
    if bs_match:
        info['batch_size'] = int(bs_match.group(1))

    # Extract gradient accumulation steps
    ga_match = re.search(r"[Gg]radient[_\s]*[Aa]ccumulation[_\s]*[Ss]teps?['\"]?\s*[:=]\s*(\d+)", content)
    if ga_match:
        info['gradient_accumulation_steps'] = int(ga_match.group(1))

    # Extract LoRA r
    lora_r_match = re.search(r"(?:lora_r|LoRA\s*[Rr]ank|r)['\"]?\s*[:=]\s*(\d+)", content, re.IGNORECASE)
    if lora_r_match:
        info['lora_r'] = int(lora_r_match.group(1))

    # Extract LoRA alpha
    lora_alpha_match = re.search(r"(?:lora_alpha|LoRA\s*[Aa]lpha)['\"]?\s*[:=]\s*(\d+)", content, re.IGNORECASE)
    if lora_alpha_match:
        info['lora_alpha'] = int(lora_alpha_match.group(1))

    # Extract epochs
    epochs_match = re.search(r"num_train_epochs['\"]?\s*[:=]\s*(\d+)", content)
    if epochs_match:
        info['epochs'] = int(epochs_match.group(1))

    # Extract max length
    max_len_match = re.search(r"max_(?:seq_)?length['\"]?\s*[:=]\s*(\d+)", content)
    if max_len_match:
        info['max_seq_length'] = int(max_len_match.group(1))

    # Extract final loss
    loss_matches = re.findall(r"'loss':\s*([\d.]+)", content)
    if loss_matches:
        info['train_loss'] = float(loss_matches[-1])

    # Extract eval loss
    eval_loss_matches = re.findall(r"'eval_loss':\s*([\d.]+)", content)
    if eval_loss_matches:
        info['eval_loss'] = float(eval_loss_matches[-1])

    # Extract accuracy (different formats)
    acc_patterns = [
        r"mean_token_accuracy['\"]?\s*[:=]\s*([\d.]+)",
        r"token_accuracy['\"]?\s*[:=]\s*([\d.]+)",
        r"accuracy['\"]?\s*[:=]\s*([\d.]+)"
    ]
    for pattern in acc_patterns:
        acc_matches = re.findall(pattern, content, re.IGNORECASE)
        if acc_matches:
            val = float(acc_matches[-1])
            # Normalize to 0-1 if percentage
            info['accuracy'] = val / 100 if val > 1 else val
            break

    # Extract train samples
    train_samples_patterns = [
        r"Train samples:\s*(\d+)",
        r"num_train_examples['\"]?\s*[:=]\s*(\d+)",
        r"training.*?(\d{4,})\s*samples"
    ]
    for pattern in train_samples_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            info['train_samples'] = int(match.group(1))
            break

    # Extract eval samples
    eval_samples_patterns = [
        r"Eval samples:\s*(\d+)",
        r"num_eval_examples['\"]?\s*[:=]\s*(\d+)",
        r"evaluation.*?(\d{3,})\s*samples"
    ]
    for pattern in eval_samples_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            info['eval_samples'] = int(match.group(1))
            break

    # GRPO-specific extractions
    if training_method == "GRPO":
        # Extract num_generations
        ng_match = re.search(r"num_generations['\"]?\s*[:=]\s*(\d+)", content)
        if ng_match:
            info['num_generations'] = int(ng_match.group(1))

        # Extract reward
        reward_matches = re.findall(r"'reward':\s*([\d.-]+)", content)
        if reward_matches:
            info['reward'] = float(reward_matches[-1])

        # Extract reward type
        rt_match = re.search(r"reward_type['\"]?\s*[:=]\s*['\"]?(\w+)", content)
        if rt_match:
            info['reward_type'] = rt_match.group(1)

    return info


def get_model_size_str(base_model: str) -> str:
    """Extract model size from base model name."""
    size_match = re.search(r'(\d+\.?\d*)[Bb]', base_model)
    if size_match:
        return f"{size_match.group(1)}B parameters"
    return "Unknown"


def get_training_method_description(method: str) -> str:
    """Get a description of the training method."""
    descriptions = {
        "SFT": "Supervised Fine-Tuning (SFT) trains the model to follow instructions by learning from high-quality demonstration data.",
        "DPO": "Direct Preference Optimization (DPO) aligns the model with human preferences without requiring a separate reward model.",
        "GRPO": "Group Relative Policy Optimization (GRPO) is an online RL method that optimizes policy using group-relative rewards without a critic model.",
        "RLHF": "Reinforcement Learning from Human Feedback (RLHF) uses a reward model trained on human preferences to guide policy optimization.",
        "PPO": "Proximal Policy Optimization (PPO) is an actor-critic RL algorithm that constrains policy updates for stable training.",
        "ORPO": "Odds Ratio Preference Optimization (ORPO) combines SFT and preference alignment in a single training stage."
    }
    return descriptions.get(method, f"{method} training method.")


def generate_model_card(args) -> str:
    """Generate the complete model card content."""

    # Extract info from log if provided
    log_info = {}
    if args.training_log:
        log_info = extract_from_training_log(args.training_log, args.training_method)

    # Merge with provided args (args take precedence)
    learning_rate = args.learning_rate or log_info.get('learning_rate', 2e-4)
    batch_size = args.batch_size or log_info.get('batch_size', 2)
    grad_accum = args.gradient_accumulation_steps or log_info.get('gradient_accumulation_steps', 8)
    lora_r = args.lora_r or log_info.get('lora_r', 64)
    lora_alpha = args.lora_alpha or log_info.get('lora_alpha', 128)
    train_loss = args.train_loss or log_info.get('train_loss')
    eval_loss = args.eval_loss or log_info.get('eval_loss')
    accuracy = args.accuracy or log_info.get('accuracy')
    train_samples = args.train_samples or log_info.get('train_samples')
    eval_samples = args.eval_samples or log_info.get('eval_samples')
    max_seq_length = args.max_length or args.max_seq_length or log_info.get('max_seq_length', 2048)
    epochs = args.epochs or log_info.get('epochs', 1)

    # GRPO-specific
    num_generations = args.num_generations or log_info.get('num_generations')
    reward_type = args.reward_type or log_info.get('reward_type')
    reward = args.reward or log_info.get('reward')

    effective_batch_size = batch_size * grad_accum
    model_size = get_model_size_str(args.base_model)
    dataset_name = args.dataset.split('/')[-1] if '/' in args.dataset else args.dataset
    base_model_name = args.base_model.split('/')[-1] if '/' in args.base_model else args.base_model

    # Get package versions automatically
    pkg_versions = get_package_versions()
    framework_versions = "\n".join([
        f"- **TRL**: {pkg_versions.get('trl', 'N/A')}",
        f"- **Transformers**: {pkg_versions.get('transformers', 'N/A')}",
        f"- **PyTorch**: {pkg_versions.get('pytorch', 'N/A')}",
        f"- **Datasets**: {pkg_versions.get('datasets', 'N/A')}",
        f"- **PEFT**: {pkg_versions.get('peft', 'N/A')}",
        f"- **BitsAndBytes**: {pkg_versions.get('bitsandbytes', 'N/A')}",
    ])

    # Determine task and description based on training method
    method_desc = get_training_method_description(args.training_method)

    # Build task-specific tags
    task_tags = []
    if args.training_method in ["SFT"]:
        task_tags = ["text-generation", "conversational", "instruction-following"]
    elif args.training_method in ["GRPO", "DPO", "PPO", "RLHF"]:
        task_tags = ["text-generation", "reasoning", "rl-trained"]
        if "math" in dataset_name.lower() or "math" in args.dataset.lower():
            task_tags.append("math")

    # Generate YAML frontmatter
    yaml_content = f"""---
license: {args.license}
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - {base_model_name.lower().replace('-', '').replace('.', '')}
  - {args.training_method.lower()}
  - fine-tuned
  - trl
  - lora"""

    for tag in task_tags:
        yaml_content += f"\n  - {tag}"

    if args.quantization:
        yaml_content += f"\n  - {args.quantization.lower().replace(' ', '-')}"

    yaml_content += f"""
base_model: {args.base_model}
datasets:
  - {args.dataset}
model-index:
  - name: {args.model_name}
    results: []
---"""

    # Build key features based on training method
    if args.training_method == "SFT":
        key_features = f"""- **High-Quality Fine-Tuning**: Trained on {f'{train_samples:,}' if train_samples else 'N/A'} carefully curated examples
- **Efficient Training**: Uses LoRA (Low-Rank Adaptation) with {args.quantization or '4-bit'} quantization
- **Strong Performance**: Achieves {f'{accuracy*100:.2f}%' if accuracy else 'N/A'} token accuracy on evaluation set
- **Optimized for Inference**: Available in multiple formats including GGUF quantizations"""
    elif args.training_method == "GRPO":
        key_features = f"""- **Reinforcement Learning**: Trained with Group Relative Policy Optimization
- **Mathematical Reasoning**: Optimized on {f'{train_samples:,}' if train_samples else 'N/A'} math problems from {dataset_name}
- **Reward Function**: Uses {reward_type or 'combined'} reward (accuracy + format)
- **Efficient Training**: Uses LoRA with {args.quantization or '4-bit'} quantization
- **Optimized for Inference**: Available in GGUF quantizations for local deployment"""
    elif args.training_method == "DPO":
        key_features = f"""- **Preference Alignment**: Trained with Direct Preference Optimization
- **Human-Aligned**: Optimized on preference pairs from {dataset_name}
- **Efficient Training**: Uses LoRA with {args.quantization or '4-bit'} quantization
- **Strong Performance**: Final loss of {train_loss or 'N/A'}
- **Optimized for Inference**: Available in multiple formats"""
    else:
        key_features = f"""- **{args.training_method} Training**: Fine-tuned using {args.training_method}
- **Efficient Training**: Uses LoRA with {args.quantization or '4-bit'} quantization
- **Trained on**: {f'{train_samples:,}' if train_samples else 'N/A'} examples from {dataset_name}"""

    # Build training configuration table
    training_config_rows = f"""| Learning Rate | {learning_rate} |
| Batch Size | {batch_size} per device |
| Gradient Accumulation Steps | {grad_accum} |
| Effective Batch Size | {effective_batch_size} |
| Number of Epochs | {epochs} |
| Max Sequence Length | {max_seq_length:,} tokens |"""

    if args.max_prompt_length:
        training_config_rows += f"\n| Max Prompt Length | {args.max_prompt_length:,} tokens |"

    training_config_rows += f"""
| LR Scheduler | Linear warmup + Cosine annealing |
| Warmup Ratio | {args.warmup_ratio} |
| Precision | BF16 mixed precision |
| Gradient Checkpointing | Enabled |
| Random Seed | {args.seed} |"""

    # Add method-specific config
    method_specific_config = ""
    if args.training_method == "GRPO" and num_generations:
        method_specific_config = f"""
### GRPO Configuration

| Parameter | Value |
|-----------|-------|
| Num Generations | {num_generations} |
| Reward Type | {reward_type or 'combined'} |
| Beta (KL Penalty) | {args.beta} |
| Temperature | {args.temperature} |
"""
    elif args.training_method == "DPO":
        method_specific_config = f"""
### DPO Configuration

| Parameter | Value |
|-----------|-------|
| Loss Type | {args.dpo_loss_type} |
| Beta | {args.beta} |
"""

    # Build metrics table
    metrics_rows = []
    if train_loss:
        metrics_rows.append(f"| Final Training Loss | {train_loss:.4f} |")
    if eval_loss:
        metrics_rows.append(f"| Final Eval Loss | {eval_loss:.4f} |")
    if accuracy:
        metrics_rows.append(f"| Token Accuracy | {accuracy*100:.2f}% |")
    if reward:
        metrics_rows.append(f"| Average Reward | {reward:.4f} |")
    if args.training_time:
        metrics_rows.append(f"| Training Time | {args.training_time} |")
    metrics_rows.append(f"| Hardware | {args.hardware} |")

    metrics_table = "\n".join(metrics_rows) if metrics_rows else "| N/A | N/A |"

    # Dataset section
    dataset_info = f"""This model was trained on the [{args.dataset}](https://huggingface.co/datasets/{args.dataset}) dataset"""
    if args.streaming:
        dataset_info += " using streaming mode"
    dataset_info += "."

    # Generate main content
    content = f"""{yaml_content}

# {args.model_name}

This model is a fine-tuned version of [{args.base_model}](https://huggingface.co/{args.base_model}) trained on the [{args.dataset}](https://huggingface.co/datasets/{args.dataset}) dataset using **{args.training_method}** with LoRA adapters.

## Overview

**{args.model_name}** is a language model optimized using {args.training_method}. {method_desc}

### Key Features

{key_features}

## Model Details

| Property | Value |
|----------|-------|
| **Developed by** | [{args.author}](https://huggingface.co/{args.author}) |
| **License** | {args.license.upper()} |
| **Language** | English |
| **Base Model** | [{args.base_model}](https://huggingface.co/{args.base_model}) |
| **Model Size** | {model_size} |
| **Tensor Type** | BF16 |
| **Context Length** | {max_seq_length:,} tokens |
| **Training Method** | {args.training_method} with LoRA |

## Training Information

### Training Configuration

| Parameter | Value |
|-----------|-------|
{training_config_rows}

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | {lora_r} |
| LoRA Alpha | {lora_alpha} |
| LoRA Dropout | {args.lora_dropout} |
| Target Modules | {', '.join(args.target_modules)} |
| Quantization | {args.quantization or '4-bit NF4'} |
{method_specific_config}
### Training Metrics

| Metric | Value |
|--------|-------|
{metrics_table}

## Dataset

{dataset_info}

| Split | Samples |
|-------|---------|
| Training | {f'{train_samples:,}' if train_samples else 'N/A'} |
| Evaluation | {f'{eval_samples:,}' if eval_samples else 'N/A'} |

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{args.author}/{args.model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "What is the sum of 2 + 2?"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

### Using Pipeline

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{args.author}/{args.model_name}", device_map="auto")
messages = [{{"role": "user", "content": "Explain the concept of machine learning."}}]
output = generator(messages, max_new_tokens=256, return_full_text=False)
print(output[0]["generated_text"])
```

### 4-bit Quantized Inference

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "{args.author}/{args.model_name}",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## GGUF Versions

For CPU or mixed CPU/GPU inference, GGUF quantized versions are available at:
[{args.author}/{args.model_name}-GGUF](https://huggingface.co/{args.author}/{args.model_name}-GGUF)

### Using with Ollama

```bash
ollama pull hf.co/{args.author}/{args.model_name}-GGUF:Q4_K_M
ollama run hf.co/{args.author}/{args.model_name}-GGUF:Q4_K_M "Hello!"
```

## Limitations

- **Language**: Primarily trained on English data
- **Knowledge Cutoff**: Limited to base model's training data cutoff
- **Hallucinations**: May generate plausible-sounding but incorrect information
- **Context Length**: Fine-tuned with {max_seq_length:,} token limit
- **Safety**: Not extensively safety-tuned; use with appropriate guardrails

## Intended Use

### Recommended Uses
- Research on language model fine-tuning
- Educational purposes
- Personal projects
- Prototyping conversational AI

### Out-of-Scope Uses
- Production systems without additional safety measures
- Medical, legal, or financial advice
- Generating harmful or misleading content

## Training Framework

{framework_versions}

## Citation

```bibtex
@misc{{{args.author}_{args.model_name.lower().replace('-', '_')},
    author = {{{args.author}}},
    title = {{{args.model_name}: Fine-tuned {base_model_name} on {dataset_name}}},
    year = {{{datetime.now().year}}},
    publisher = {{Hugging Face}},
    howpublished = {{\\url{{https://huggingface.co/{args.author}/{args.model_name}}}}}
}}
```

## Acknowledgments

- Base model developers at {args.base_model.split('/')[0] if '/' in args.base_model else 'the model creators'}
- [Hugging Face TRL Team](https://github.com/huggingface/trl) for the training library
- Dataset creators and contributors
- Compute Canada / DRAC for HPC resources

## Contact

For questions or issues, please open an issue on the model repository.
"""

    return content


def generate_gguf_readme(args) -> str:
    """Generate README for GGUF repository."""

    model_name_base = args.model_name.replace('-GGUF', '')

    content = f"""---
license: {args.license}
language:
  - en
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - ollama
  - lm-studio
  - quantized
  - {args.training_method.lower()}
base_model: {args.author}/{model_name_base}
---

# {args.model_name}

GGUF quantized versions of [{model_name_base}](https://huggingface.co/{args.author}/{model_name_base}) for use with llama.cpp, Ollama, LM Studio, and other GGUF-compatible tools.

## Available Quantizations

| Quantization | Description | Use Case |
|--------------|-------------|----------|
| Q4_K_M | 4-bit quantization, medium | Best balance of quality and size |
| Q5_K_M | 5-bit quantization, medium | Higher quality, moderate size |
| Q8_0 | 8-bit quantization | Highest quality, larger size |

## Quick Start with Ollama

```bash
# Pull and run directly from HuggingFace
ollama pull hf.co/{args.author}/{args.model_name}:Q4_K_M
ollama run hf.co/{args.author}/{args.model_name}:Q4_K_M "Hello!"
```

## Download

```bash
# Using huggingface-cli
huggingface-cli download {args.author}/{args.model_name} \\
    {model_name_base.lower()}-q4_k_m.gguf --local-dir ./models

# Using wget
wget https://huggingface.co/{args.author}/{args.model_name}/resolve/main/{model_name_base.lower()}-q4_k_m.gguf
```

## Usage

### llama.cpp

```bash
./llama-cli -m {model_name_base.lower()}-q4_k_m.gguf \\
    -p "What is machine learning?" -n 256
```

### llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(model_path="{model_name_base.lower()}-q4_k_m.gguf", n_ctx=2048)
output = llm("What is machine learning?", max_tokens=256)
print(output['choices'][0]['text'])
```

### Ollama (Local Model)

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./{model_name_base.lower()}-q4_k_m.gguf
EOF

# Create and run
ollama create {model_name_base.lower()} -f Modelfile
ollama run {model_name_base.lower()} "Hello!"
```

### LM Studio

1. Download the desired GGUF file
2. Open LM Studio and navigate to the local models directory
3. Load the model and start chatting

## Original Model

This is a quantized version of [{model_name_base}](https://huggingface.co/{args.author}/{model_name_base}).
See the original model card for training details, usage information, and benchmarks.

## License

{args.license.upper()}

## Acknowledgments

- Original model by [{args.author}](https://huggingface.co/{args.author})
- GGUF conversion using [llama.cpp](https://github.com/ggerganov/llama.cpp)
"""

    return content


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate appropriate README
    if args.is_gguf:
        content = generate_gguf_readme(args)
    else:
        content = generate_model_card(args)

    # Write to file
    output_path = os.path.join(args.output_dir, "README.md")
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Model card generated: {output_path}")

    # Also print summary
    print("\n" + "=" * 60)
    print("Model Card Summary")
    print("=" * 60)
    print(f"Model Name: {args.model_name}")
    print(f"Base Model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Training Method: {args.training_method}")
    print(f"License: {args.license}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
