---
license: cc-by-nc-4.0
tags:
- text-generation
- granite-4.0-micro
- grpo
- math-reasoning
- reinforcement-learning
- finetuned-model
- trl
- lora
- AI-MO/NuminaMath-CoT
datasets:
- AI-MO/NuminaMath-CoT
base_model: ibm-granite/granite-4.0-micro
library_name: transformers
language:
- en
pipeline_tag: text-generation
model-index:
- name: granite-4.0-micro-GRPO-NuminaMath-10K
  results:
  - task:
      type: mathematical-reasoning
    dataset:
      name: NuminaMath-CoT
      type: AI-MO/NuminaMath-CoT
    metrics:
    - name: Training Loss
      type: loss
      value: -0.0159
    - name: Mean Reward
      type: reward
      value: 0.48
---

# Granite 4.0 Micro — GRPO Fine-tuned on NuminaMath-CoT (10K)

## Overview

This is a **GRPO (Group Relative Policy Optimization)** fine-tuned version of IBM's [granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro), trained on 10,000 mathematical reasoning problems from the [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset.

The model was trained to improve mathematical reasoning through reinforcement learning, using a combined reward function that scores accuracy (50%), format quality (25%), response length (15%), and reasoning quality (10%).

## Model Details

| Property | Value |
|----------|-------|
| **Developed by** | [ermiaazarkhalili](https://huggingface.co/ermiaazarkhalili) |
| **Base model** | [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro) |
| **Architecture** | Granite MoE Hybrid (SSM + Attention) |
| **Parameters** | ~3.4B (MoE, ~1B active per token) |
| **License** | CC-BY-NC-4.0 |
| **Training method** | GRPO with LoRA |
| **Dataset** | AI-MO/NuminaMath-CoT (10K samples, streaming) |
| **Training time** | 16 hours 21 minutes |
| **Hardware** | NVIDIA H100 MIG 3g.40gb (40GB) |
| **Precision** | BFloat16 |

## Training Configuration

### GRPO Parameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-6 |
| LR Scheduler | Cosine |
| Batch Size | 2 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 16 |
| Num Generations | 4 |
| Max Completion Length | 512 |
| Max Prompt Length | 512 |
| Temperature | 0.7 |
| Epochs | 1 |
| Total Steps | 2,500 |

### LoRA Configuration
| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### Reward Function (Combined)
| Component | Weight | Description |
|-----------|--------|-------------|
| Accuracy | 50% | Exact/numeric match with ground truth |
| Format | 25% | Presence of \boxed{}, step-by-step, math notation |
| Length | 15% | Optimal response length (50-500 tokens) |
| Reasoning | 10% | Chain-of-thought indicators |

## Training Results

| Metric | Start | End |
|--------|-------|-----|
| **Loss** | -0.028 | **-0.016** |
| **Mean Reward** | 0.49 | **0.48** |
| **Clipped Ratio** | 0.06 | **0.31** |
| **Mean Completion Length** | 242 | **335** |
| **Mean Terminated Length** | 224 | **272** |
| **Ground Truth Stored** | 9,885 / 10,000 (98.9%) |
| **Training Runtime** | 16h 21m |
| **Steps/Second** | 0.042 |

### Key Observations
- **Healthy termination**: Only 6-31% of completions hit max length (vs 100% for Qwen3 models)
- **MoE efficiency**: ~24s/step — fastest among all tested models
- **Negative loss**: Model is learning to differentiate good from bad completions
- **Ground truth extraction**: 98.9% success rate from NuminaMath \boxed{} format

## Usage

### With Transformers + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-4.0-micro",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    "ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-10K",
)

tokenizer = AutoTokenizer.from_pretrained(
    "ibm-granite/granite-4.0-micro",
    trust_remote_code=True,
)

# Generate
prompt = "Solve: If 3x + 7 = 22, what is x?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

## Why Granite 4.0 Micro for GRPO?

IBM's Granite 4.0 Micro uses a **MoE Hybrid architecture** (Mixture of Experts + SSM/Attention hybrid), which provides several advantages for GRPO training:

1. **Natural termination**: Unlike Qwen3/Qwen2 models that fill the entire completion buffer (100% clipped), Granite produces concise answers with only 6-31% clipping
2. **Speed**: MoE activates only ~1B parameters per token despite having 3.4B total, making it 2-3x faster than similarly-sized dense models
3. **Memory efficiency**: Fits comfortably on a 40GB MIG partition in bf16 without quantization

## Dataset

[AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) contains 860K competition-level math problems with chain-of-thought solutions. This model was trained on 10K samples using streaming mode with seed=42 for reproducibility.

## Infrastructure

Trained on the **Fir cluster** (Digital Research Alliance of Canada) using a single NVIDIA H100 MIG 3g.40gb partition with the slurm-model-trainer skill for automated SLURM job generation.

### Framework Versions
- **TRL**: 0.24.0
- **Transformers**: 4.57.3
- **PyTorch**: 2.9.0
- **Datasets**: 4.3.0
- **PEFT**: 0.18.1

## Limitations

- Trained on only 10K samples (1.2% of the full NuminaMath dataset)
- LoRA adapter only — not a full fine-tune
- Optimized for competition-level math, may not generalize to all math domains
- Responses may occasionally lack \boxed{} formatting

## Credits

- **IBM Research** for the [Granite 4.0](https://huggingface.co/ibm-granite) model family
- **AI-MO/Numina** for the [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset
- **Hugging Face** for the [TRL](https://github.com/huggingface/trl) library
- **Digital Research Alliance of Canada** for compute resources

## Citation

```bibtex
@misc{azarkhalili2026granite_grpo,
    title = {Granite 4.0 Micro GRPO Fine-tuned on NuminaMath},
    author = {Behrooz Azarkhalili},
    year = {2026},
    publisher = {Hugging Face},
    howpublished = {\url{https://huggingface.co/ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-10K}},
}
```
