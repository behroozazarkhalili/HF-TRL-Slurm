# SFT (Supervised Fine-Tuning) Base Models

Pretrained models suitable for SFT training. These models have NOT been instruction-tuned.

## Overview Table

| Model | Size | Downloads | License | Context |
|-------|------|-----------|---------|---------|
| [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) | 3B | 289K | Other | 128K |
| [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | 1.5B | 486K | Apache 2.0 | 128K |
| [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 0.5B | 1.1M | Apache 2.0 | 128K |
| [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | 4B | 358K | Apache 2.0 | 128K |
| [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | 1.7B | 427K | Apache 2.0 | 128K |
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 0.6B | 174K | Apache 2.0 | 128K |
| [nvidia/Nemotron-Flash-3B](https://huggingface.co/nvidia/Nemotron-Flash-3B) | 3B | 4.3K | Other | - |
| [nvidia/Nemotron-Flash-1B](https://huggingface.co/nvidia/Nemotron-Flash-1B) | 1B | 1.2K | Other | - |

---

## Detailed Model Information

### 1. Qwen/Qwen2.5-3B

| Property | Value |
|----------|-------|
| **Parameters** | 3B |
| **Downloads** | 289K |
| **License** | Other (Qwen) |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen2.5 |

**Features**:
- Dense transformer architecture
- Strong multilingual capabilities
- 128K context window
- Pretrained on diverse web data

**Recommended Config**:
- Max Seq Length: 2,048 - 4,096
- Batch Size: 2-4
- LoRA Rank: 32-64

---

### 2. Qwen/Qwen2.5-1.5B

| Property | Value |
|----------|-------|
| **Parameters** | 1.5B |
| **Downloads** | 486K |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen2.5 |

**Features**:
- Efficient model size
- Apache 2.0 license (commercial friendly)
- Good balance of capability and efficiency
- Pretrained on diverse web data

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 4-8
- LoRA Rank: 32-64

---

### 3. Qwen/Qwen2.5-0.5B

| Property | Value |
|----------|-------|
| **Parameters** | 0.5B |
| **Downloads** | 1.1M (most popular) |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen2.5 |

**Features**:
- Most downloaded Qwen2.5 model
- Extremely efficient for experiments
- Apache 2.0 license
- Great for quick prototyping

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 8-16
- LoRA Rank: 16-32

---

### 4. Qwen/Qwen3-4B-Base

| Property | Value |
|----------|-------|
| **Parameters** | 4B |
| **Downloads** | 358K |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen3 (Dense) |

**Features**:
- Latest Qwen3 architecture
- Improved reasoning capabilities
- Trained on 36T tokens (2x Qwen2.5)
- Supports 119 languages

**Recommended Config**:
- Max Seq Length: 2,048 - 4,096
- Batch Size: 1-2
- LoRA Rank: 32-64

---

### 5. Qwen/Qwen3-1.7B-Base

| Property | Value |
|----------|-------|
| **Parameters** | 1.7B |
| **Downloads** | 427K |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen3 (Dense) |

**Features**:
- Efficient Qwen3 variant
- Matches Qwen2.5-3B performance
- Trained on 36T tokens
- Apache 2.0 license

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 4-8
- LoRA Rank: 32-64

---

### 6. Qwen/Qwen3-0.6B-Base

| Property | Value |
|----------|-------|
| **Parameters** | 0.6B |
| **Downloads** | 174K |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen3 (Dense) |

**Features**:
- Smallest Qwen3 model
- Great for experiments and edge devices
- Trained on 36T tokens
- Apache 2.0 license

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 8-16
- LoRA Rank: 16-32

---

### 7. nvidia/Nemotron-Flash-3B

| Property | Value |
|----------|-------|
| **Parameters** | 3B |
| **Downloads** | 4.3K |
| **License** | Other (NVIDIA) |
| **Architecture** | Nemotron-Flash |

**Features**:
- NVIDIA's efficient architecture
- Optimized for inference
- New 2025 model
- Strong general capabilities

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 2-4
- LoRA Rank: 32-64

---

### 8. nvidia/Nemotron-Flash-1B

| Property | Value |
|----------|-------|
| **Parameters** | 1B |
| **Downloads** | 1.2K |
| **License** | Other (NVIDIA) |
| **Architecture** | Nemotron-Flash |

**Features**:
- Compact Nemotron-Flash variant
- NVIDIA optimizations
- New 2025 model
- Efficient for deployment

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 4-8
- LoRA Rank: 16-32

---

## Training Recommendations by Use Case

| Use Case | Recommended Model | Max Seq Len | Notes |
|----------|-------------------|-------------|-------|
| Quick experiment | Qwen2.5-0.5B | 2,048 | Most downloaded, fast training |
| Production SFT | Qwen2.5-1.5B | 2,048 | Good balance, Apache 2.0 |
| Latest architecture | Qwen3-1.7B-Base | 2,048 | Qwen3 benefits |
| Maximum capability | Qwen3-4B-Base | 4,096 | Best performance |
| NVIDIA ecosystem | Nemotron-Flash-3B | 2,048 | NVIDIA optimized |

---

## Quick Start Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load pretrained model for SFT
model_id = "Qwen/Qwen2.5-1.5B"  # Choose your model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

# Load SFT dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
```

---

## Model Size Comparison

```
Model Size Comparison:
======================

Qwen3-4B-Base        ████████████████████████████████████  4.0B params
Qwen2.5-3B           ████████████████████████████          3.0B params
Nemotron-Flash-3B    ████████████████████████████          3.0B params
Qwen3-1.7B-Base      ██████████████                        1.7B params
Qwen2.5-1.5B         █████████████                         1.5B params
Nemotron-Flash-1B    █████████                             1.0B params
Qwen3-0.6B-Base      █████                                 0.6B params
Qwen2.5-0.5B         ████                                  0.5B params
```

---

## Last Updated
2025-12-25
