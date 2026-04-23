"""Generate comprehensive model cards for Unsloth-trained models and push to Hub.

Usage:
    python scripts/generate_unsloth_model_card.py --all          # All 16 models (8 merged + 8 GGUF)
    python scripts/generate_unsloth_model_card.py --repo_id <id> # Single model
"""
from __future__ import annotations

import argparse
import os
from huggingface_hub import HfApi

# ── Model registry ──────────────────────────────────────────────────────────

MODELS = {
    # SFT Distillation (Claude Reasoning Traces)
    "ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Opus-Reasoning-Unsloth": {
        "display_name": "Qwen3.5-0.8B-SFT-Claude-Opus-Reasoning-Unsloth",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "base_model_name": "Qwen3.5-0.8B",
        "params": "0.8B",
        "task": "sft_distillation",
        "dataset": "ermiaazarkhalili/claude-reasoning-distillation",
        "dataset_name": "Claude Reasoning Distillation",
        "dataset_samples": "10,477",
        "license": "apache-2.0",
        "model_family": "qwen3.5",
        "unsloth_class": "FastLanguageModel",
        "chat_template": "built-in Qwen3.5",
    },
    "ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Opus-Reasoning-Unsloth": {
        "display_name": "LFM2.5-1.2B-SFT-Claude-Opus-Reasoning-Unsloth",
        "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "base_model_name": "LFM2.5-1.2B-Instruct",
        "params": "1.2B",
        "task": "sft_distillation",
        "dataset": "ermiaazarkhalili/claude-reasoning-distillation",
        "dataset_name": "Claude Reasoning Distillation",
        "dataset_samples": "10,477",
        "license": "apache-2.0",
        "model_family": "lfm2.5",
        "unsloth_class": "FastLanguageModel",
        "chat_template": "LFM2.5 (chatml)",
    },
    "ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Opus-Reasoning-Unsloth": {
        "display_name": "Gemma4-E2B-SFT-Claude-Opus-Reasoning-Unsloth",
        "base_model": "google/gemma-4-E2B-it",
        "base_model_name": "Gemma4-E2B-it",
        "params": "2B",
        "task": "sft_distillation",
        "dataset": "ermiaazarkhalili/claude-reasoning-distillation",
        "dataset_name": "Claude Reasoning Distillation",
        "dataset_samples": "10,477",
        "license": "gemma",
        "model_family": "gemma4",
        "unsloth_class": "FastModel",
        "chat_template": "gemma-4",
    },
    "ermiaazarkhalili/Gemma4-E4B-SFT-Claude-Opus-Reasoning-Unsloth": {
        "display_name": "Gemma4-E4B-SFT-Claude-Opus-Reasoning-Unsloth",
        "base_model": "google/gemma-4-E4B-it",
        "base_model_name": "Gemma4-E4B-it",
        "params": "4B",
        "task": "sft_distillation",
        "dataset": "ermiaazarkhalili/claude-reasoning-distillation",
        "dataset_name": "Claude Reasoning Distillation",
        "dataset_samples": "10,477",
        "license": "gemma",
        "model_family": "gemma4",
        "unsloth_class": "FastModel",
        "chat_template": "gemma-4",
    },
    # xLAM Function Calling
    "ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth": {
        "display_name": "Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "base_model_name": "Qwen3.5-0.8B",
        "params": "0.8B",
        "task": "xlam_function_calling",
        "dataset": "Salesforce/xlam-function-calling-60k",
        "dataset_name": "xLAM Function Calling 60K",
        "dataset_samples": "60,000",
        "license": "apache-2.0",
        "model_family": "qwen3.5",
        "unsloth_class": "FastLanguageModel",
        "chat_template": "built-in Qwen3.5",
    },
    "ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth": {
        "display_name": "Qwen3.5-2B-Function-Calling-xLAM-Unsloth",
        "base_model": "Qwen/Qwen3.5-2B",
        "base_model_name": "Qwen3.5-2B",
        "params": "2B",
        "task": "xlam_function_calling",
        "dataset": "Salesforce/xlam-function-calling-60k",
        "dataset_name": "xLAM Function Calling 60K",
        "dataset_samples": "60,000",
        "license": "apache-2.0",
        "model_family": "qwen3.5",
        "unsloth_class": "FastLanguageModel",
        "chat_template": "built-in Qwen3.5",
    },
    "ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth": {
        "display_name": "LFM2.5-1.2B-Function-Calling-xLAM-Unsloth",
        "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "base_model_name": "LFM2.5-1.2B-Instruct",
        "params": "1.2B",
        "task": "xlam_function_calling",
        "dataset": "Salesforce/xlam-function-calling-60k",
        "dataset_name": "xLAM Function Calling 60K",
        "dataset_samples": "60,000",
        "license": "apache-2.0",
        "model_family": "lfm2.5",
        "unsloth_class": "FastLanguageModel",
        "chat_template": "LFM2.5 (chatml)",
    },
    "ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth": {
        "display_name": "Gemma4-E2B-Function-Calling-xLAM-Unsloth",
        "base_model": "google/gemma-4-E2B-it",
        "base_model_name": "Gemma4-E2B-it",
        "params": "2B",
        "task": "xlam_function_calling",
        "dataset": "Salesforce/xlam-function-calling-60k",
        "dataset_name": "xLAM Function Calling 60K",
        "dataset_samples": "60,000",
        "license": "gemma",
        "model_family": "gemma4",
        "unsloth_class": "FastModel",
        "chat_template": "gemma-4",
    },
    "ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth": {
        "display_name": "Gemma4-E4B-Function-Calling-xLAM-Unsloth",
        "base_model": "google/gemma-4-E4B-it",
        "base_model_name": "Gemma4-E4B-it",
        "params": "4B",
        "task": "xlam_function_calling",
        "dataset": "Salesforce/xlam-function-calling-60k",
        "dataset_name": "xLAM Function Calling 60K",
        "dataset_samples": "60,000",
        "license": "gemma",
        "model_family": "gemma4",
        "unsloth_class": "FastModel",
        "chat_template": "gemma-4",
    },
}


def generate_card(repo_id: str, m: dict) -> str:
    is_xlam = m["task"] == "xlam_function_calling"
    task_desc = "function calling" if is_xlam else "reasoning distillation (chain-of-thought)"
    task_tag = "function-calling" if is_xlam else "reasoning"

    dataset_desc = (
        "the [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset, "
        "which contains 60,000 function calling examples with queries, tool definitions, and structured answers"
        if is_xlam else
        "the [claude-reasoning-distillation](https://huggingface.co/datasets/ermiaazarkhalili/claude-reasoning-distillation) dataset, "
        "which contains 10,477 samples of Claude's reasoning traces with `<think>` blocks for chain-of-thought learning"
    )

    usage_prompt = (
        'Check if the numbers 8 and 1233 are powers of two.'
        if is_xlam else
        'Solve step by step: What is the sum of the first 10 prime numbers?'
    )

    gguf_repo = f"{repo_id}-GGUF"

    card = f"""---
license: {m['license']}
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - unsloth
  - {m['model_family']}
  - sft
  - fine-tuned
  - trl
  - lora
  - qlora
  - text-generation
  - {task_tag}
  - conversational
base_model: {m['base_model']}
datasets:
  - {m['dataset']}
model-index:
  - name: {m['display_name']}
    results: []
---

# {m['display_name']}

This model is a fine-tuned version of [{m['base_model_name']}]({f"https://huggingface.co/{m['base_model']}"}) optimized for **{task_desc}** using [Unsloth](https://github.com/unslothai/unsloth) for **2x faster training** and **60% less VRAM**.

Trained on {dataset_desc}.

## Overview

| Property | Value |
|----------|-------|
| **Developed by** | [ermiaazarkhalili](https://huggingface.co/ermiaazarkhalili) |
| **License** | {m['license'].upper()} |
| **Language** | English |
| **Base Model** | [{m['base_model_name']}](https://huggingface.co/{m['base_model']}) |
| **Model Size** | {m['params']} parameters |
| **Training Framework** | [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) |
| **Training Method** | SFT with QLoRA (4-bit) |
| **Context Length** | 2,048 tokens |
| **GGUF Available** | [{gguf_repo.split('/')[-1]}](https://huggingface.co/{gguf_repo}) |

## Training Configuration

### SFT + LoRA Settings

| Parameter | Value |
|-----------|-------|
| Unsloth Class | `{m['unsloth_class']}` |
| Chat Template | {m['chat_template']} |
| Learning Rate | 2e-4 |
| Batch Size | 2 per device |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 8 |
| Max Steps | 1,000 |
| Optimizer | AdamW 8-bit |
| LR Scheduler | Linear |
| Warmup Steps | 5 |
| Precision | Auto (BF16/FP16) |
| Gradient Checkpointing | Enabled (Unsloth optimized) |
| Seed | 3407 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0 |
| Quantization | 4-bit QLoRA |
| Target Modules | {"attention + MLP (via FastModel)" if m['model_family'] == 'gemma4' else "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"} |

### Dataset

| Property | Value |
|----------|-------|
| Dataset | [{m['dataset_name']}](https://huggingface.co/datasets/{m['dataset']}) |
| Training Samples | {m['dataset_samples']} |
| Format | {"XML-tagged: `<query>`, `<tools>`, `<answers>`" if is_xlam else "Messages with `thinking` field for chain-of-thought"} |

### Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 80GB HBM3 (MIG 3g.40gb slice) |
| Cluster | DRAC Fir (Compute Canada) |
| Execution | [Papermill](https://github.com/nteract/papermill) on SLURM |

## Usage

### Quick Start (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {{"role": "user", "content": "{usage_prompt}"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

### Using with Unsloth (Fastest)

```python
from unsloth import {m['unsloth_class']}

model, {"processor" if m['model_family'] == 'qwen3.5' else "tokenizer"} = {m['unsloth_class']}.from_pretrained(
    "{repo_id}",
    max_seq_length=2048,
    load_in_4bit=True,
)
{"tokenizer = processor.tokenizer  # Extract text tokenizer from processor" if m['model_family'] == 'qwen3.5' else ""}
```

### 4-bit Quantized Inference

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## GGUF Versions

Quantized GGUF versions for CPU and edge inference are available at:
**[{gguf_repo.split('/')[-1]}](https://huggingface.co/{gguf_repo})**

| Format | Description |
|--------|-------------|
| `Q4_K_M` | Recommended — good balance of quality and size |
| `Q5_K_M` | Higher quality, slightly larger |
| `Q8_0` | Near-lossless, largest GGUF size |

### Using with Ollama

```bash
ollama pull hf.co/{gguf_repo}:Q4_K_M
ollama run hf.co/{gguf_repo}:Q4_K_M "{usage_prompt}"
```

### Using with llama.cpp

```bash
./llama-cli -m {m['display_name']}-Q4_K_M.gguf -p "{usage_prompt}" -n 512
```

## Limitations

- **Language**: Primarily trained on English data
- **Knowledge Cutoff**: Limited to base model's training data cutoff
- **Hallucinations**: May generate plausible-sounding but incorrect information
- **Context Length**: Fine-tuned with 2,048 token context window
- **Safety**: Not extensively safety-tuned; use with appropriate guardrails

## Training Framework Versions

| Package | Version |
|---------|---------|
| Unsloth | 2026.4.4 |
| TRL | 0.24.0 |
| Transformers | 5.5.0 |
| PyTorch | 2.9.0 |
| Datasets | 4.3.0 |
| PEFT | 0.18.1 |
| BitsAndBytes | 0.49.2 |

## Citation

```bibtex
@misc{{ermiaazarkhalili_{m['display_name'].lower().replace('-', '_').replace('.', '')},
    author = {{ermiaazarkhalili}},
    title = {{{m['display_name']}: Fine-tuned {m['base_model_name']} with Unsloth}},
    year = {{2026}},
    publisher = {{Hugging Face}},
    howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for 2x faster fine-tuning
- Base model developers ({m['base_model'].split('/')[0]})
- [Hugging Face TRL Team](https://github.com/huggingface/trl) for the training library
- {"[Salesforce xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) for the function calling dataset" if is_xlam else "[Claude Reasoning Distillation](https://huggingface.co/datasets/ermiaazarkhalili/claude-reasoning-distillation) dataset"}
- [Compute Canada / DRAC](https://alliancecan.ca/) for HPC resources
"""
    return card


def push_card(repo_id: str, card_content: str, token: str | None = None) -> None:
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"[OK] Model card pushed to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Unsloth model cards")
    parser.add_argument("--all", action="store_true", help="Generate cards for all models")
    parser.add_argument("--repo_id", type=str, help="Single repo ID to generate card for")
    parser.add_argument("--dry_run", action="store_true", help="Print card without pushing")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    if args.repo_id:
        targets = {args.repo_id: MODELS[args.repo_id]}
    elif args.all:
        targets = MODELS
    else:
        parser.print_help()
        return

    for repo_id, m in targets.items():
        print(f"\n{'='*60}")
        print(f"Generating card for: {repo_id}")
        card = generate_card(repo_id, m)
        if args.dry_run:
            print(card[:500])
            print("... (dry run, not pushing)")
        else:
            push_card(repo_id, card, token)


if __name__ == "__main__":
    main()
