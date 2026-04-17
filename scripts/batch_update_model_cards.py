#!/usr/bin/env python3
"""
Generate comprehensive model cards for all TRL and Unsloth trained models.
Pushes updated README.md to each Hub repo.

Usage:
    python scripts/batch_update_model_cards.py              # Push to all repos on Hub
    python scripts/batch_update_model_cards.py --dry-run    # Print without pushing
    python scripts/batch_update_model_cards.py --model Qwen3.5  # Filter by name
"""
from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from huggingface_hub import HfApi


@dataclass
class ModelSpec:
    hub_id: str
    base_model: str
    base_model_name: str
    model_size: str
    dataset: str
    dataset_name: str
    task: str
    framework: str
    license: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    grad_accum: int
    lr: str
    max_seq_length: int
    epochs: int
    max_steps: int | None
    target_modules: str
    quantization: str
    description: str
    dataset_samples: str
    gguf_repo: str | None


MODELS: list[ModelSpec] = [
    # ── TRL SFT Distillation ──
    ModelSpec("ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Reasoning", "Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "0.8B", "ermiaazarkhalili/claude-reasoning-distillation", "Claude Reasoning Distillation", "SFT Distillation", "TRL", "apache-2.0", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit NF4 (QLoRA)", "Fine-tuned to reproduce Claude's chain-of-thought reasoning traces with `<think>` blocks for transparent, step-by-step problem solving.", "~10,477", "ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Reasoning-GGUF"),
    ModelSpec("ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Reasoning", "LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5-1.2B-Instruct", "1.2B", "ermiaazarkhalili/claude-reasoning-distillation", "Claude Reasoning Distillation", "SFT Distillation", "TRL", "apache-2.0", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit NF4 (QLoRA)", "Fine-tuned to reproduce Claude's chain-of-thought reasoning traces using LiquidAI's state-space model architecture.", "~10,477", "ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Reasoning-GGUF"),
    ModelSpec("ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Reasoning", "google/gemma-4-E2B-it", "Gemma4-E2B-it", "~2B", "ermiaazarkhalili/claude-reasoning-distillation", "Claude Reasoning Distillation", "SFT Distillation", "TRL", "gemma", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "all-linear (Gemma4ClippableLinear workaround)", "4-bit NF4 (QLoRA)", "Fine-tuned to reproduce Claude's chain-of-thought reasoning traces on Google's Gemma 4 architecture.", "~10,477", "ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Reasoning-GGUF"),
    # ── TRL xLAM Function Calling ──
    ModelSpec("ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM", "Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "0.8B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "TRL", "apache-2.0", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit NF4 (QLoRA)", "Fine-tuned for structured function calling and tool use with XML-tagged query/tools/answers format.", "60,000", "ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-GGUF"),
    ModelSpec("ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM", "Qwen/Qwen3.5-2B", "Qwen3.5-2B", "2B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "TRL", "apache-2.0", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit NF4 (QLoRA)", "Fine-tuned for structured function calling and tool use with XML-tagged query/tools/answers format.", "60,000", "ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-GGUF"),
    ModelSpec("ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM", "LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5-1.2B-Instruct", "1.2B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "TRL", "apache-2.0", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit NF4 (QLoRA)", "Fine-tuned for structured function calling and tool use using LiquidAI's efficient state-space architecture.", "60,000", "ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-GGUF"),
    ModelSpec("ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM", "google/gemma-4-E2B-it", "Gemma4-E2B-it", "~2B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "TRL", "gemma", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "all-linear (Gemma4ClippableLinear workaround)", "4-bit NF4 (QLoRA)", "Fine-tuned for structured function calling and tool use on Google's Gemma 4 architecture.", "60,000", "ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-GGUF"),
    ModelSpec("ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM", "google/gemma-4-E4B-it", "Gemma4-E4B-it", "~4B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "TRL", "gemma", 64, 128, 0.05, 2, 8, "2e-4", 2048, 1, None, "all-linear (Gemma4ClippableLinear workaround)", "4-bit NF4 (QLoRA)", "Fine-tuned for structured function calling and tool use on Google's Gemma 4 architecture (4B variant).", "60,000", "ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-GGUF"),
    # ── Unsloth SFT Distillation ──
    ModelSpec("ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Reasoning-Unsloth", "Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "0.8B", "ermiaazarkhalili/claude-reasoning-distillation", "Claude Reasoning Distillation", "SFT Distillation", "Unsloth", "apache-2.0", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) to reproduce Claude's chain-of-thought reasoning traces.", "~10,477", "ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Reasoning-Unsloth-GGUF"),
    ModelSpec("ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Reasoning-Unsloth", "LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5-1.2B-Instruct", "1.2B", "ermiaazarkhalili/claude-reasoning-distillation", "Claude Reasoning Distillation", "SFT Distillation", "Unsloth", "apache-2.0", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) to reproduce Claude's reasoning traces on LiquidAI's state-space architecture.", "~10,477", "ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Reasoning-Unsloth-GGUF"),
    ModelSpec("ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Reasoning-Unsloth", "google/gemma-4-E2B-it", "Gemma4-E2B-it", "~2B", "ermiaazarkhalili/claude-reasoning-distillation", "Claude Reasoning Distillation", "SFT Distillation", "Unsloth", "gemma", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "attention + MLP (via FastModel)", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) to reproduce Claude's reasoning traces on Gemma 4.", "~10,477", "ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Reasoning-Unsloth-GGUF"),
    # ── Unsloth xLAM Function Calling ──
    ModelSpec("ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth", "Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "0.8B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "Unsloth", "apache-2.0", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) for structured function calling and tool use.", "60,000", "ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth-GGUF"),
    ModelSpec("ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth", "Qwen/Qwen3.5-2B", "Qwen3.5-2B", "2B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "Unsloth", "apache-2.0", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) for structured function calling and tool use.", "60,000", "ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth-GGUF"),
    ModelSpec("ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth", "LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5-1.2B-Instruct", "1.2B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "Unsloth", "apache-2.0", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) for function calling on LiquidAI's state-space architecture.", "60,000", "ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth-GGUF"),
    ModelSpec("ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth", "google/gemma-4-E2B-it", "Gemma4-E2B-it", "~2B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "Unsloth", "gemma", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "attention + MLP (via FastModel)", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) for function calling on Gemma 4.", "60,000", "ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth-GGUF"),
    ModelSpec("ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth", "google/gemma-4-E4B-it", "Gemma4-E4B-it", "~4B", "Salesforce/xlam-function-calling-60k", "xLAM Function Calling 60K", "Function Calling", "Unsloth", "gemma", 16, 16, 0, 2, 4, "2e-4", 2048, 1, 1000, "attention + MLP (via FastModel)", "4-bit QLoRA", "Fine-tuned with Unsloth (2x faster) for function calling on Gemma 4 (4B variant).", "60,000", "ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth-GGUF"),
]


def generate_model_card(m: ModelSpec) -> str:
    model_name = m.hub_id.split("/")[1]
    effective_batch = m.batch_size * m.grad_accum
    steps_info = f"{m.max_steps} steps" if m.max_steps else f"{m.epochs} epoch(s)"

    fw_tags = "trl\n  - sft" if m.framework == "TRL" else "unsloth\n  - sft"
    fw_versions = (
        "- TRL: 1.0.0\n- Transformers: 5.5.3\n- PyTorch: 2.11.0\n- PEFT: 0.18.1\n- BitsAndBytes: 0.49.2"
        if m.framework == "TRL"
        else "- Unsloth: 2026.4.4\n- Transformers: 5.5.0\n- PyTorch: 2.9.0\n- PEFT: 0.18.1\n- BitsAndBytes: 0.49.2"
    )
    unsloth_ack = "- [Unsloth Team](https://github.com/unslothai/unsloth) for 2x faster training\n" if m.framework == "Unsloth" else ""

    gguf_section = ""
    if m.gguf_repo:
        gguf_section = f"""## GGUF Versions

Quantized GGUF versions for CPU/edge inference:
**[{m.gguf_repo}](https://huggingface.co/{m.gguf_repo})**

| Quantization | Bits | Use Case |
|---|---|---|
| Q2_K | 2 | Edge devices, mobile |
| Q3_K_M | 3 | Constrained environments |
| Q4_K_M | 4 | Best quality/size balance (recommended) |
| Q5_K_M | 5 | Higher quality |
| Q6_K | 6 | Near-lossless |
| Q8_0 | 8 | Maximum quality |

### Ollama

```bash
ollama pull hf.co/{m.gguf_repo}:Q4_K_M
ollama run hf.co/{m.gguf_repo}:Q4_K_M "Hello!"
```

### llama.cpp

```bash
llama-cli -m {model_name}-Q4_K_M.gguf -p "Your prompt here" -n 256
```
"""

    return f"""---
license: {m.license}
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - {fw_tags}
  - fine-tuned
  - lora
  - qlora
  - text-generation
  - conversational
  - function-calling
  - reasoning
base_model: {m.base_model}
datasets:
  - {m.dataset}
model-index:
  - name: {model_name}
    results: []
---

# {model_name}

{m.description}

Fine-tuned from [{m.base_model}](https://huggingface.co/{m.base_model}) on [{m.dataset_name}](https://huggingface.co/datasets/{m.dataset}) using **{m.framework}** with LoRA adapters.

## Overview

| Property | Value |
|----------|-------|
| **Developed by** | [Behrooz Azarkhalili](https://huggingface.co/ermiaazarkhalili) |
| **License** | {m.license.upper()} |
| **Language** | English |
| **Base Model** | [{m.base_model}](https://huggingface.co/{m.base_model}) |
| **Model Size** | {m.model_size} parameters |
| **Training Method** | SFT with LoRA ({m.framework}) |
| **Dataset** | [{m.dataset_name}](https://huggingface.co/datasets/{m.dataset}) ({m.dataset_samples} samples) |
| **Context Length** | {m.max_seq_length:,} tokens |
| **Hardware** | NVIDIA H100 80GB HBM3 (MIG 3g.40gb) |
| **Cluster** | DRAC / Fir (Compute Canada) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | {m.framework} |
| Learning Rate | {m.lr} |
| Batch Size | {m.batch_size} per device |
| Gradient Accumulation | {m.grad_accum} |
| Effective Batch Size | {effective_batch} |
| Training | {steps_info} |
| Max Sequence Length | {m.max_seq_length:,} tokens |
| Precision | BF16 mixed precision |
| Gradient Checkpointing | Enabled |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | {m.lora_r} |
| LoRA Alpha | {m.lora_alpha} |
| LoRA Dropout | {m.lora_dropout} |
| Target Modules | {m.target_modules} |
| Quantization | {m.quantization} |

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{m.hub_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Explain step by step how to solve 2x + 5 = 13."}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Pipeline

```python
from transformers import pipeline
generator = pipeline("text-generation", model="{m.hub_id}", device_map="auto")
output = generator([{{"role": "user", "content": "What is 2+2?"}}], max_new_tokens=256, return_full_text=False)
print(output[0]["generated_text"])
```

### 4-bit Quantized

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
model = AutoModelForCausalLM.from_pretrained(
    "{m.hub_id}",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto"
)
```

{gguf_section}

## Limitations

- **Language**: Primarily trained on English data
- **Knowledge Cutoff**: Limited to base model's training data cutoff
- **Hallucinations**: May generate plausible-sounding but incorrect information
- **Context Length**: Fine-tuned with {m.max_seq_length:,} token limit
- **Safety**: Not extensively safety-tuned; use with appropriate guardrails

## Intended Use

- Research on language model fine-tuning and reasoning distillation
- Educational purposes and experimentation
- Prototyping conversational AI and tool-use agents
- **Not recommended** for production without additional safety measures

## Framework Versions

{fw_versions}

## Citation

```bibtex
@misc{{azarkhalili2026{model_name.lower().replace('-', '_').replace('.', '')},
    author = {{Azarkhalili, Behrooz}},
    title = {{{model_name}: Fine-tuned {m.base_model_name} for {m.task}}},
    year = {{2026}},
    publisher = {{Hugging Face}},
    url = {{https://huggingface.co/{m.hub_id}}}
}}
```

> To generate a citable DOI, click **"Cite this model"** on the [model page](https://huggingface.co/{m.hub_id}).

## Acknowledgments

- Base model developers ({m.base_model.split("/")[0]})
- [Hugging Face TRL Team](https://github.com/huggingface/trl)
{unsloth_ack}- [Salesforce xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) for the function calling dataset
- [Compute Canada / DRAC](https://docs.alliancecan.ca/) for HPC resources
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, help="Filter by hub_id substring")
    args = parser.parse_args()

    api = HfApi()
    pushed, skipped, failed = 0, 0, 0

    for m in MODELS:
        if args.model and args.model not in m.hub_id:
            continue
        try:
            api.repo_info(m.hub_id, repo_type="model")
        except Exception:
            print(f"[SKIP] {m.hub_id} — not on Hub yet")
            skipped += 1
            continue

        card = generate_model_card(m)
        if args.dry_run:
            print(f"[DRY] {m.hub_id} ({len(card)} chars)")
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(card)
            tmp = f.name
        try:
            api.upload_file(path_or_fileobj=tmp, path_in_repo="README.md",
                          repo_id=m.hub_id, repo_type="model")
            print(f"[OK] {m.hub_id}")
            pushed += 1
        except Exception as e:
            print(f"[FAIL] {m.hub_id} — {e}")
            failed += 1
        finally:
            os.unlink(tmp)

    print(f"\nDone: {pushed} pushed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
