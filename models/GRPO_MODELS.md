# GRPO (Group Relative Policy Optimization) Base Models

Instruction-tuned models suitable for GRPO training. These models have already been instruction-tuned.

## Overview Table

| Model | Size | Downloads | License | Context |
|-------|------|-----------|---------|---------|
| [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | 9.0M | Other | 128K |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | 5.0M | Apache 2.0 | 128K |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | 2.4M | Apache 2.0 | 128K |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B | 3.9M | Apache 2.0 | 128K |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B | 4.3M | Apache 2.0 | 262K |
| [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | 4B | 494K | Apache 2.0 | 262K |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | 5.4M | Apache 2.0 | 128K |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | 8.1M | Apache 2.0 | 128K |
| [nvidia/Nemotron-Flash-3B-Instruct](https://huggingface.co/nvidia/Nemotron-Flash-3B-Instruct) | 3B | 2.1K | Other | - |
| [LiquidAI/LFM2-2.6B](https://huggingface.co/LiquidAI/LFM2-2.6B) | 2.6B | 12.9K | Other | - |
| [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) | 1.2B | 501K | Other | - |
| [LiquidAI/LFM2-700M](https://huggingface.co/LiquidAI/LFM2-700M) | 700M | 6.5K | Other | - |
| [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) | 350M | 19.9K | Other | - |

---

## Detailed Model Information

### Qwen2.5 Instruct Models

#### 1. Qwen/Qwen2.5-3B-Instruct

| Property | Value |
|----------|-------|
| **Parameters** | 3B |
| **Downloads** | 9.0M (most downloaded) |
| **License** | Other (Qwen) |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen2.5 |

**Features**:
- Most downloaded Qwen2.5 Instruct model
- Strong instruction following
- 128K context window
- Excellent for chat applications

**Recommended Config**:
- Max Seq Length: 2,048 - 4,096
- Batch Size: 1-2
- Num Generations: 2-4

---

#### 2. Qwen/Qwen2.5-1.5B-Instruct

| Property | Value |
|----------|-------|
| **Parameters** | 1.5B |
| **Downloads** | 5.0M |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen2.5 |

**Features**:
- Efficient model size
- Apache 2.0 license (commercial friendly)
- Good instruction following
- Balanced capability and efficiency

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

#### 3. Qwen/Qwen2.5-0.5B-Instruct

| Property | Value |
|----------|-------|
| **Parameters** | 0.5B |
| **Downloads** | 2.4M |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen2.5 |

**Features**:
- Smallest Qwen2.5 Instruct model
- Apache 2.0 license
- Fast training and inference
- Great for experiments

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

### Qwen3 Models

#### 4. Qwen/Qwen3-4B

| Property | Value |
|----------|-------|
| **Parameters** | 4B |
| **Downloads** | 3.9M |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen3 (Dense) |

**Features**:
- Latest Qwen3 architecture
- Improved reasoning capabilities
- Hybrid thinking (reasoning + non-reasoning)
- Trained on 36T tokens

**Recommended Config**:
- Max Seq Length: 2,048 - 4,096
- Batch Size: 1
- Num Generations: 2-4

---

#### 5. Qwen/Qwen3-4B-Instruct-2507

| Property | Value |
|----------|-------|
| **Parameters** | 4B |
| **Downloads** | 4.3M |
| **License** | Apache 2.0 |
| **Context Length** | 262K tokens |
| **Architecture** | Qwen3 (2507 update) |

**Features**:
- Latest July 2025 update
- 262K context window
- Enhanced instruction following
- Optimized for business logic

**Recommended Config**:
- Max Seq Length: 4,096 - 8,192
- Batch Size: 1
- Num Generations: 2-4

---

#### 6. Qwen/Qwen3-4B-Thinking-2507

| Property | Value |
|----------|-------|
| **Parameters** | 4B |
| **Downloads** | 494K |
| **License** | Apache 2.0 |
| **Context Length** | 262K tokens |
| **Architecture** | Qwen3 (Thinking) |

**Features**:
- Enhanced reasoning capabilities
- Deep thinking mode
- Best for complex problem solving
- Mathematics and coding focus

**Recommended Config**:
- Max Seq Length: 4,096 - 8,192
- Batch Size: 1
- Num Generations: 2-4

---

#### 7. Qwen/Qwen3-1.7B

| Property | Value |
|----------|-------|
| **Parameters** | 1.7B |
| **Downloads** | 5.4M |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen3 (Dense) |

**Features**:
- Efficient Qwen3 variant
- Matches Qwen2.5-3B performance
- Hybrid thinking capabilities
- Apache 2.0 license

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

#### 8. Qwen/Qwen3-0.6B

| Property | Value |
|----------|-------|
| **Parameters** | 0.6B |
| **Downloads** | 8.1M (most downloaded Qwen3) |
| **License** | Apache 2.0 |
| **Context Length** | 128K tokens |
| **Architecture** | Qwen3 (Dense) |

**Features**:
- Smallest Qwen3 model
- Most downloaded Qwen3
- Great for experiments
- Apache 2.0 license

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

### NVIDIA Nemotron-Flash

#### 9. nvidia/Nemotron-Flash-3B-Instruct

| Property | Value |
|----------|-------|
| **Parameters** | 3B |
| **Downloads** | 2.1K |
| **License** | Other (NVIDIA) |
| **Architecture** | Nemotron-Flash |

**Features**:
- NVIDIA optimized architecture
- New 2025 model
- Efficient inference
- Strong instruction following

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

### LiquidAI LFM2 Models

#### 10. LiquidAI/LFM2-2.6B

| Property | Value |
|----------|-------|
| **Parameters** | 2.6B |
| **Downloads** | 12.9K |
| **License** | Other (LiquidAI) |
| **Architecture** | LFM2 (Liquid Foundation Model) |

**Features**:
- Novel liquid architecture
- Efficient inference
- Instruction-tuned only (no base version)
- Competitive with larger models

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

#### 11. LiquidAI/LFM2-1.2B

| Property | Value |
|----------|-------|
| **Parameters** | 1.2B |
| **Downloads** | 501K (most downloaded LFM2) |
| **License** | Other (LiquidAI) |
| **Architecture** | LFM2 |

**Features**:
- Most popular LFM2 model
- Efficient model size
- Novel architecture
- Good for production

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

#### 12. LiquidAI/LFM2-700M

| Property | Value |
|----------|-------|
| **Parameters** | 700M |
| **Downloads** | 6.5K |
| **License** | Other (LiquidAI) |
| **Architecture** | LFM2 |

**Features**:
- Compact LFM2 variant
- Fast inference
- Novel architecture
- Edge deployment

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 1-2
- Num Generations: 2-4

---

#### 13. LiquidAI/LFM2-350M

| Property | Value |
|----------|-------|
| **Parameters** | 350M |
| **Downloads** | 19.9K |
| **License** | Other (LiquidAI) |
| **Architecture** | LFM2 |

**Features**:
- Smallest LFM2 model
- Extremely efficient
- Novel architecture
- Mobile/edge deployment

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 2-4
- Num Generations: 2-4

---

## Training Recommendations by Use Case

| Use Case | Recommended Model | Max Seq Len | Notes |
|----------|-------------------|-------------|-------|
| Quick experiment | Qwen3-0.6B | 2,048 | Most downloaded, fast |
| Math reasoning | Qwen3-4B-Thinking-2507 | 8,192 | Deep reasoning |
| Production | Qwen2.5-1.5B-Instruct | 2,048 | Apache 2.0, reliable |
| Maximum capability | Qwen3-4B-Instruct-2507 | 4,096 | Latest update |
| Novel architecture | LFM2-1.2B | 2,048 | Liquid architecture |
| NVIDIA ecosystem | Nemotron-Flash-3B-Instruct | 2,048 | NVIDIA optimized |

---

## Quick Start Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load instruction-tuned model for GRPO
model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Choose your model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

# Load GRPO dataset (math reasoning)
dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
```

---

## Model Size Comparison

```
Model Size Comparison:
======================

Qwen3-4B-Instruct-2507    ████████████████████████████████████  4.0B params
Qwen3-4B-Thinking-2507    ████████████████████████████████████  4.0B params
Qwen3-4B                  ████████████████████████████████████  4.0B params
Qwen2.5-3B-Instruct       ████████████████████████████          3.0B params
Nemotron-Flash-3B-Instruct████████████████████████████          3.0B params
LFM2-2.6B                 ██████████████████████                2.6B params
Qwen3-1.7B                ██████████████                        1.7B params
Qwen2.5-1.5B-Instruct     █████████████                         1.5B params
LFM2-1.2B                 ██████████                            1.2B params
LFM2-700M                 ██████                                0.7B params
Qwen3-0.6B                █████                                 0.6B params
Qwen2.5-0.5B-Instruct     ████                                  0.5B params
LFM2-350M                 ███                                   0.35B params
```

---

## Downloads Comparison

```
Downloads (Millions):
=====================

Qwen2.5-3B-Instruct       ████████████████████████████████████  9.0M
Qwen3-0.6B                ████████████████████████████████      8.1M
Qwen3-1.7B                █████████████████████                 5.4M
Qwen2.5-1.5B-Instruct     ████████████████████                  5.0M
Qwen3-4B-Instruct-2507    █████████████████                     4.3M
Qwen3-4B                  ███████████████                       3.9M
Qwen2.5-0.5B-Instruct     ██████████                            2.4M
LFM2-1.2B                 ██                                    0.5M
```

---

## Last Updated
2025-12-25
