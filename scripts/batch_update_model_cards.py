#!/usr/bin/env python3
"""
Batch Model Card Updater
========================
Updates model cards for multiple HuggingFace repositories based on
training metrics and the Capybara template format.

Usage:
    python batch_update_model_cards.py --push_to_hub
    python batch_update_model_cards.py --dry_run  # Preview only
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

# Add the skills scripts to path
sys.path.insert(0, "/project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts")

from huggingface_hub import HfApi


@dataclass
class ModelConfig:
    """Configuration for a model to update."""
    hub_model_id: str
    model_name: str
    base_model: str
    dataset: str
    training_method: str
    output_dir: Optional[str] = None  # Path to training outputs with all_results.json
    lora_r: int = 32
    lora_alpha: int = 64
    batch_size: int = 8
    epochs: int = 1
    max_length: int = 2048
    learning_rate: float = 2e-4
    hardware: str = "NVIDIA H100 80GB HBM3"
    # Optional metrics (extracted from all_results.json)
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    accuracy: Optional[float] = None
    train_samples: int = 197471
    eval_samples: int = 10394
    # GRPO-specific
    num_generations: Optional[int] = None
    reward_type: Optional[str] = None


# ============================================================================
# SFT UltraChat Models
# ============================================================================
SFT_ULTRACHAT_MODELS = [
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen2.5-3B-SFT-UltraChat",
        model_name="Qwen2.5-3B-SFT-UltraChat",
        base_model="Qwen/Qwen2.5-3B",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        output_dir="/scratch/ermia/outputs/qwen2.5-3b-sft-16615022",
        lora_r=32,
        lora_alpha=64,
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen2.5-7B-SFT-UltraChat",
        model_name="Qwen2.5-7B-SFT-UltraChat",
        base_model="Qwen/Qwen2.5-7B",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        output_dir="/scratch/ermia/outputs/qwen7b-sft-16453144",
        lora_r=64,
        lora_alpha=128,
        batch_size=2,
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen2.5-14B-SFT-UltraChat",
        model_name="Qwen2.5-14B-SFT-UltraChat",
        base_model="Qwen/Qwen2.5-14B",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        lora_r=64,
        lora_alpha=128,
        batch_size=1,
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen3-0.6B-SFT-UltraChat",
        model_name="Qwen3-0.6B-SFT-UltraChat",
        base_model="Qwen/Qwen3-0.6B-Base",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        output_dir="/scratch/ermia/outputs/qwen3-0.6b-sft-16610693",
        lora_r=32,
        lora_alpha=64,
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen3-1.7B-SFT-UltraChat",
        model_name="Qwen3-1.7B-SFT-UltraChat",
        base_model="Qwen/Qwen3-1.7B-Base",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        output_dir="/scratch/ermia/outputs/qwen3-1.7b-sft-16615024",
        lora_r=32,
        lora_alpha=64,
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen3-4B-SFT-UltraChat",
        model_name="Qwen3-4B-SFT-UltraChat",
        base_model="Qwen/Qwen3-4B-Base",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        output_dir="/scratch/ermia/outputs/qwen3-4b-sft-16600897",
        lora_r=32,
        lora_alpha=64,
        batch_size=4,
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen3-8B-SFT-UltraChat",
        model_name="Qwen3-8B-SFT-UltraChat",
        base_model="Qwen/Qwen3-8B-Base",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        output_dir="/scratch/ermia/outputs/qwen3-8b-sft-16602000",
        lora_r=64,
        lora_alpha=128,
        batch_size=2,
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen3-14B-SFT-UltraChat",
        model_name="Qwen3-14B-SFT-UltraChat",
        base_model="Qwen/Qwen3-14B-Base",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
        lora_r=64,
        lora_alpha=128,
        batch_size=1,
    ),
]

# ============================================================================
# GRPO NuminaMath Models
# ============================================================================
GRPO_NUMINA_MODELS = [
    ModelConfig(
        hub_model_id="ermiaazarkhalili/LFM2-1.2B-GRPO-NuminaMath",
        model_name="LFM2-1.2B-GRPO-NuminaMath",
        base_model="LiquidAI/LFM2-1.2B",
        dataset="AI-MO/NuminaMath-CoT",
        training_method="GRPO",
        output_dir="/scratch/ermia/outputs/lfm2-1.2b-grpo-numina-16737228",
        lora_r=16,
        lora_alpha=32,
        batch_size=1,
        max_length=4096,
        learning_rate=1e-6,
        train_samples=10000,
        eval_samples=0,
        num_generations=2,
        reward_type="combined",
        hardware="NVIDIA H100 20GB MIG",
    ),
]

# ============================================================================
# GGUF Models (use generate_gguf_readme format)
# ============================================================================
GGUF_MODELS = [
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen2.5-0.5B-SFT-UltraChat-GGUF",
        model_name="Qwen2.5-0.5B-SFT-UltraChat-GGUF",
        base_model="ermiaazarkhalili/Qwen2.5-0.5B-SFT-UltraChat",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen2.5-1.5B-SFT-UltraChat-GGUF",
        model_name="Qwen2.5-1.5B-SFT-UltraChat-GGUF",
        base_model="ermiaazarkhalili/Qwen2.5-1.5B-SFT-UltraChat",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen2.5-3B-SFT-UltraChat-GGUF",
        model_name="Qwen2.5-3B-SFT-UltraChat-GGUF",
        base_model="ermiaazarkhalili/Qwen2.5-3B-SFT-UltraChat",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen3-0.6B-SFT-UltraChat-GGUF",
        model_name="Qwen3-0.6B-SFT-UltraChat-GGUF",
        base_model="ermiaazarkhalili/Qwen3-0.6B-SFT-UltraChat",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
    ),
    ModelConfig(
        hub_model_id="ermiaazarkhalili/Qwen3-1.7B-SFT-UltraChat-GGUF",
        model_name="Qwen3-1.7B-SFT-UltraChat-GGUF",
        base_model="ermiaazarkhalili/Qwen3-1.7B-SFT-UltraChat",
        dataset="HuggingFaceH4/ultrachat_200k",
        training_method="SFT",
    ),
]


def load_metrics_from_output(output_dir: str) -> Dict[str, Any]:
    """Load training metrics from output directory."""
    metrics = {}

    if not output_dir or not os.path.exists(output_dir):
        return metrics

    # Try all_results.json first
    all_results_path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            data = json.load(f)
            metrics['train_loss'] = data.get('train_loss')
            metrics['eval_loss'] = data.get('eval_loss')
            metrics['accuracy'] = data.get('eval_mean_token_accuracy')
            if metrics['accuracy'] and metrics['accuracy'] > 1:
                metrics['accuracy'] = metrics['accuracy'] / 100  # Convert percentage to ratio
            metrics['train_runtime'] = data.get('train_runtime')

    # Try trainer_state.json for additional info
    checkpoint_dirs = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if checkpoint_dirs:
        latest_checkpoint = checkpoint_dirs[-1]
        trainer_state_path = latest_checkpoint / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                data = json.load(f)
                if 'log_history' in data and data['log_history']:
                    last_entry = data['log_history'][-1]
                    if 'loss' in last_entry and 'train_loss' not in metrics:
                        metrics['train_loss'] = last_entry['loss']

    return metrics


def format_training_time(seconds: Optional[float]) -> str:
    """Convert seconds to human-readable format."""
    if not seconds:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def get_model_size_from_name(model_name: str) -> str:
    """Extract model size from name."""
    import re
    match = re.search(r'(\d+\.?\d*[Bb])', model_name)
    if match:
        return match.group(1).upper()
    return "Unknown"


def generate_sft_model_card(config: ModelConfig, metrics: Dict[str, Any]) -> str:
    """Generate model card for SFT models."""

    model_size = get_model_size_from_name(config.model_name)
    train_loss = metrics.get('train_loss', config.train_loss) or "N/A"
    eval_loss = metrics.get('eval_loss', config.eval_loss) or "N/A"
    accuracy = metrics.get('accuracy', config.accuracy)
    accuracy_str = f"{accuracy*100:.2f}%" if accuracy else "N/A"
    training_time = format_training_time(metrics.get('train_runtime'))

    # Format losses
    if isinstance(train_loss, float):
        train_loss = f"{train_loss:.4f}"
    if isinstance(eval_loss, float):
        eval_loss = f"{eval_loss:.4f}"

    card = f'''---
license: cc-by-nc-4.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - {config.base_model.split("/")[1].lower().replace("-", "").replace(".", "")}
  - sft
  - fine-tuned
  - trl
  - lora
  - conversational
  - text-generation
  - instruction-following
base_model: {config.base_model}
datasets:
  - {config.dataset}
model-index:
  - name: {config.model_name}
    results: []
---

# {config.model_name}

This model is a fine-tuned version of [{config.base_model}](https://huggingface.co/{config.base_model}) trained on the [{config.dataset}](https://huggingface.co/datasets/{config.dataset}) dataset using Supervised Fine-Tuning (SFT) with LoRA adapters.

## Overview

**{config.model_name}** is an instruction-following language model optimized for conversational tasks. It combines the powerful {model_size} base model with high-quality instruction-following data from UltraChat, resulting in improved response quality and helpfulness.

### Key Features

- **High-Quality Fine-Tuning**: Trained on {config.train_samples:,} instruction-response pairs
- **Efficient Training**: Uses LoRA (Low-Rank Adaptation) for memory efficiency
- **Strong Performance**: Achieves {accuracy_str} token accuracy on held-out evaluation set
- **Optimized for Inference**: Available in multiple formats including GGUF quantizations

## Model Details

| Property | Value |
|----------|-------|
| **Developed by** | [ermiaazarkhalili](https://huggingface.co/ermiaazarkhalili) |
| **License** | CC-BY-NC-4.0 |
| **Language** | English |
| **Base Model** | [{config.base_model}](https://huggingface.co/{config.base_model}) |
| **Model Size** | {model_size} parameters |
| **Tensor Type** | BF16 |
| **Context Length** | {config.max_length:,} tokens |
| **Training Method** | SFT with LoRA |

## Training Information

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | {config.learning_rate} |
| Batch Size | {config.batch_size} per device |
| Effective Batch Size | {config.batch_size * 2} (with gradient accumulation) |
| Gradient Accumulation Steps | 2 |
| Number of Epochs | {config.epochs} |
| Max Sequence Length | {config.max_length:,} tokens |
| LR Scheduler | Linear warmup + Cosine annealing |
| Precision | BF16 mixed precision |
| Gradient Checkpointing | Enabled |
| Optimizer | AdamW |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | {config.lora_r} |
| LoRA Alpha | {config.lora_alpha} |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### Training Metrics

| Metric | Value |
|--------|-------|
| Final Training Loss | {train_loss} |
| Final Eval Loss | {eval_loss} |
| Token Accuracy | {accuracy_str} |
| Training Time | {training_time} |

### Training Hardware

- **GPU**: {config.hardware}
- **CPU**: 8 vCPUs
- **Memory**: 64GB
- **Platform**: Compute Canada (Fir Cluster)

## Dataset

This model was trained on the [{config.dataset}](https://huggingface.co/datasets/{config.dataset}) dataset:

| Split | Samples |
|-------|---------|
| Training | {config.train_samples:,} |
| Evaluation | {config.eval_samples:,} |

The UltraChat dataset contains high-quality multi-turn conversations designed to improve instruction-following capabilities.

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{config.hub_model_id}"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Chat format
messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "What are the key principles of effective communication?"}}
]

# Apply chat template
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

### Using Pipeline

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="{config.hub_model_id}",
    device_map="auto",
    torch_dtype="auto"
)

messages = [{{"role": "user", "content": "Explain quantum computing in simple terms."}}]
output = generator(messages, max_new_tokens=256, return_full_text=False)
print(output[0]["generated_text"])
```

## GGUF Versions

For CPU or mixed CPU/GPU inference, GGUF quantized versions are available at:
[{config.hub_model_id}-GGUF](https://huggingface.co/{config.hub_model_id}-GGUF)

Available quantizations:
- **Q4_K_M**: Best balance of quality and size
- **Q5_K_M**: Higher quality, larger size
- **Q8_0**: Highest quality quantization

### Using with Ollama

```bash
ollama pull hf.co/{config.hub_model_id}-GGUF:Q4_K_M
ollama run hf.co/{config.hub_model_id}-GGUF:Q4_K_M "Hello, how are you?"
```

## Limitations

- **Language**: Primarily trained on English data; performance on other languages may vary
- **Knowledge Cutoff**: Base model knowledge is limited to its training data cutoff
- **Hallucinations**: Like all LLMs, may generate plausible-sounding but incorrect information
- **Context Length**: Limited to {config.max_length:,} tokens during fine-tuning
- **Safety**: Not extensively safety-tuned; use appropriate content filtering in production

## Intended Use

### Recommended Uses
- Conversational AI assistants
- Question answering systems
- Text generation and completion
- Educational applications
- Research and experimentation

### Out-of-Scope Uses
- Medical, legal, or financial advice without expert oversight
- Generation of harmful, deceptive, or illegal content
- High-stakes decision-making without human verification

## Citation

```bibtex
@misc{{ermiaazarkhalili_{config.model_name.lower().replace("-", "_")},
    author = {{Ermia Azarkhalili}},
    title = {{{config.model_name}: Fine-tuned {config.base_model.split("/")[1]} on UltraChat}},
    year = {{2025}},
    publisher = {{Hugging Face}},
    howpublished = {{\\url{{https://huggingface.co/{config.hub_model_id}}}}}
}}
```

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM) for the excellent base model
- [Hugging Face TRL Team](https://github.com/huggingface/trl) for the training framework
- [UltraChat Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) creators
- Compute Canada for providing HPC resources

## Contact

For questions, issues, or collaborations, please open an issue on the model repository or contact via [HuggingFace](https://huggingface.co/ermiaazarkhalili).
'''
    return card


def generate_grpo_model_card(config: ModelConfig, metrics: Dict[str, Any]) -> str:
    """Generate model card for GRPO models."""

    model_size = get_model_size_from_name(config.model_name)
    train_loss = metrics.get('train_loss', config.train_loss) or "N/A"
    training_time = format_training_time(metrics.get('train_runtime'))

    if isinstance(train_loss, float):
        train_loss = f"{train_loss:.4f}"

    card = f'''---
license: cc-by-nc-4.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - lfm2
  - grpo
  - reinforcement-learning
  - math
  - reasoning
  - trl
  - lora
  - text-generation
base_model: {config.base_model}
datasets:
  - {config.dataset}
model-index:
  - name: {config.model_name}
    results: []
---

# {config.model_name}

This model is a fine-tuned version of [{config.base_model}](https://huggingface.co/{config.base_model}) trained on the [{config.dataset}](https://huggingface.co/datasets/{config.dataset}) dataset using **Group Relative Policy Optimization (GRPO)** - an online reinforcement learning method.

## Overview

**{config.model_name}** is optimized for mathematical reasoning tasks. It uses GRPO to learn from reward signals based on answer correctness and format adherence, enabling it to generate more accurate step-by-step solutions.

### Key Features

- **Reinforcement Learning**: Trained with GRPO for improved reasoning capabilities
- **Math Focus**: Optimized on {config.train_samples:,} math problems from NuminaMath-CoT
- **Multi-Sample Learning**: Uses {config.num_generations} generations per prompt for robust training
- **Combined Reward**: Evaluates both answer accuracy and output format

## Model Details

| Property | Value |
|----------|-------|
| **Developed by** | [ermiaazarkhalili](https://huggingface.co/ermiaazarkhalili) |
| **License** | CC-BY-NC-4.0 |
| **Language** | English |
| **Base Model** | [{config.base_model}](https://huggingface.co/{config.base_model}) |
| **Model Size** | {model_size} parameters |
| **Tensor Type** | BF16 |
| **Context Length** | {config.max_length:,} tokens |
| **Training Method** | GRPO with LoRA |

## Training Information

### GRPO Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | {config.learning_rate} |
| Batch Size | {config.batch_size} per device |
| Gradient Accumulation Steps | 16 |
| Num Generations | {config.num_generations} |
| Reward Type | {config.reward_type} |
| Max Prompt Length | 1024 |
| Max Completion Length | {config.max_length} |
| Temperature | 0.7 |
| Beta (KL penalty) | 0.04 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | {config.lora_r} |
| LoRA Alpha | {config.lora_alpha} |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### Training Metrics

| Metric | Value |
|--------|-------|
| Final Policy Loss | {train_loss} |
| Training Time | {training_time} |

### Reward Function

The model uses a **combined reward** function:
1. **Math Accuracy**: Extracts and validates final numerical answers
2. **Format Compliance**: Checks for proper step-by-step reasoning format
3. **Combined Score**: Weighted combination of accuracy and format rewards

### Training Hardware

- **GPU**: {config.hardware}
- **CPU**: 8 vCPUs
- **Memory**: 64GB
- **Platform**: Compute Canada (Fir Cluster)

## Dataset

This model was trained on the [{config.dataset}](https://huggingface.co/datasets/{config.dataset}) dataset:

| Property | Value |
|----------|-------|
| Training Samples | {config.train_samples:,} |
| Format | Chain-of-Thought reasoning |
| Topics | Math (algebra, geometry, calculus, etc.) |

NuminaMath-CoT provides step-by-step mathematical solutions, enabling the model to learn structured reasoning patterns.

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{config.hub_model_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Math problem
prompt = "Solve step by step: If a train travels 120 km in 2 hours, what is its average speed?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Limitations

- **Domain Specific**: Optimized for math; may not generalize to other reasoning tasks
- **Language**: English only
- **Hallucinations**: May produce incorrect calculations despite correct format
- **Verification Needed**: Always verify mathematical results independently

## Intended Use

### Recommended Uses
- Mathematical problem solving
- Step-by-step reasoning demonstrations
- Educational math tutoring applications
- Research on RL-trained language models

### Out-of-Scope Uses
- Critical calculations requiring absolute accuracy
- Non-mathematical reasoning tasks
- Production systems without verification

## Citation

```bibtex
@misc{{ermiaazarkhalili_{config.model_name.lower().replace("-", "_")},
    author = {{Ermia Azarkhalili}},
    title = {{{config.model_name}: GRPO-trained {config.base_model.split("/")[1]} for Math}},
    year = {{2025}},
    publisher = {{Hugging Face}},
    howpublished = {{\\url{{https://huggingface.co/{config.hub_model_id}}}}}
}}
```

## Acknowledgments

- [LiquidAI](https://www.liquid.ai/) for the LFM2 base model
- [Hugging Face TRL Team](https://github.com/huggingface/trl) for the GRPO implementation
- [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset creators
- Compute Canada for providing HPC resources
'''
    return card


def generate_gguf_model_card(config: ModelConfig) -> str:
    """Generate model card for GGUF repositories."""

    # Extract source model name (remove -GGUF suffix)
    source_model = config.base_model
    source_model_name = config.model_name.replace("-GGUF", "")

    card = f'''---
license: cc-by-nc-4.0
tags:
  - gguf
  - llama.cpp
  - ollama
  - lm-studio
  - quantized
base_model: {source_model}
---

# {config.model_name}

GGUF quantized versions of [{source_model_name}](https://huggingface.co/{source_model}) for efficient CPU and mixed CPU/GPU inference.

## Available Quantizations

| Quantization | Size | Description |
|--------------|------|-------------|
| Q4_K_M | ~40% of original | Best balance of quality and size |
| Q5_K_M | ~50% of original | Higher quality, larger size |
| Q8_0 | ~80% of original | Highest quality quantization |

## Quick Start

### Using Ollama

```bash
# Pull and run directly from HuggingFace
ollama pull hf.co/{config.hub_model_id}:Q4_K_M
ollama run hf.co/{config.hub_model_id}:Q4_K_M "Hello!"
```

### Using llama.cpp

```bash
# Download the GGUF file
huggingface-cli download {config.hub_model_id} \\
    {source_model_name.lower().replace("-", "_")}-q4_k_m.gguf --local-dir ./models

# Run inference
./llama-cli -m ./models/{source_model_name.lower().replace("-", "_")}-q4_k_m.gguf \\
    -p "What is the capital of France?" -n 128
```

### Using LM Studio

1. Download the GGUF file from this repository
2. Import into LM Studio
3. Select the model and start chatting

## Source Model

This is a quantized version of [{source_model_name}](https://huggingface.co/{source_model}), which was fine-tuned on the [{config.dataset}](https://huggingface.co/datasets/{config.dataset}) dataset using {config.training_method}.

See the [source model card](https://huggingface.co/{source_model}) for full training details and usage examples.

## License

CC-BY-NC-4.0 (same as source model)

## Acknowledgments

- Quantization performed using [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Source model by [ermiaazarkhalili](https://huggingface.co/ermiaazarkhalili)
'''
    return card


def update_model_card(config: ModelConfig, dry_run: bool = False, push_to_hub: bool = False) -> bool:
    """Generate and optionally push model card for a single model."""

    print(f"\n{'='*60}")
    print(f"Processing: {config.hub_model_id}")
    print(f"{'='*60}")

    # Load metrics if output directory exists
    metrics = load_metrics_from_output(config.output_dir) if config.output_dir else {}

    if metrics:
        print(f"  Loaded metrics from: {config.output_dir}")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
    else:
        print("  No metrics found, using defaults")

    # Generate appropriate model card
    if config.hub_model_id.endswith("-GGUF"):
        card_content = generate_gguf_model_card(config)
    elif config.training_method == "GRPO":
        card_content = generate_grpo_model_card(config, metrics)
    else:
        card_content = generate_sft_model_card(config, metrics)

    # Preview
    print(f"\n  Generated README.md ({len(card_content)} chars)")
    print(f"  First 500 chars preview:")
    print("-" * 40)
    print(card_content[:500])
    print("-" * 40)

    if dry_run:
        print("  [DRY RUN] Would push to hub")
        return True

    if push_to_hub:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=config.hub_model_id,
                repo_type="model",
            )
            print(f"  ✅ Successfully pushed to {config.hub_model_id}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to push: {e}")
            return False

    # Save locally
    output_path = Path(f"/tmp/model_cards/{config.model_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    readme_path = output_path / "README.md"
    readme_path.write_text(card_content)
    print(f"  📁 Saved to: {readme_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch update model cards")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--dry_run", action="store_true", help="Preview only, don't save/push")
    parser.add_argument("--sft_only", action="store_true", help="Only update SFT models")
    parser.add_argument("--grpo_only", action="store_true", help="Only update GRPO models")
    parser.add_argument("--gguf_only", action="store_true", help="Only update GGUF repos")
    args = parser.parse_args()

    models_to_update: List[ModelConfig] = []

    if args.sft_only:
        models_to_update = SFT_ULTRACHAT_MODELS
    elif args.grpo_only:
        models_to_update = GRPO_NUMINA_MODELS
    elif args.gguf_only:
        models_to_update = GGUF_MODELS
    else:
        models_to_update = SFT_ULTRACHAT_MODELS + GRPO_NUMINA_MODELS + GGUF_MODELS

    print(f"Will update {len(models_to_update)} model cards")
    print(f"Push to hub: {args.push_to_hub}")
    print(f"Dry run: {args.dry_run}")

    success_count = 0
    for config in models_to_update:
        if update_model_card(config, dry_run=args.dry_run, push_to_hub=args.push_to_hub):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(models_to_update)} model cards updated")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
