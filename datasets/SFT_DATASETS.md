# SFT (Supervised Fine-Tuning) Datasets

Datasets optimized for instruction-following and conversational fine-tuning.

## Overview Table

| Dataset | Samples | Size | Avg Tokens | License | Downloads |
|---------|---------|------|------------|---------|-----------|
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 515,311 | 1.62 GB | ~3,000 | MIT | 500K+ |
| [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 1,001,551 | 1.94 GB | ~2,000 | Apache 2.0 | 200K+ |
| [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | 2.94M | 2.87 GB | ~1,000 | MIT | 150K+ |
| [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) | 100,000 | 117 MB | ~1,200 | Apache 2.0 | 50K+ |

---

## Detailed Dataset Information

### 1. HuggingFaceH4/ultrachat_200k

| Property | Value |
|----------|-------|
| **Total Samples** | 515,311 |
| **Train Split** | 207,865 (train_sft) |
| **Test Split** | 23,110 (test_sft) |
| **File Size** | 1.62 GB |
| **Avg Tokens** | ~3,000 per conversation |
| **License** | MIT |

**Splits**:
| Split | Samples |
|-------|---------|
| train_sft | 207,865 |
| test_sft | 23,110 |
| train_gen | 256,032 |
| test_gen | 28,304 |

**Features**:
- Multi-turn conversations
- High-quality curated subset of UltraChat
- Diverse topics and instruction types

**Recommended Config**:
- Max Seq Length: 4,096
- Batch Size: 4-8
- Best for general instruction-following

---

### 2. teknium/OpenHermes-2.5

| Property | Value |
|----------|-------|
| **Total Samples** | 1,001,551 |
| **File Size** | 1.94 GB |
| **Avg Tokens** | ~2,000 per sample |
| **License** | Apache 2.0 |
| **Generator** | GPT-4 |

**Features**:
- GPT-4 generated responses
- Diverse instruction types
- High-quality reasoning chains
- Wide topic coverage

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 4-8
- Great for general-purpose models

---

### 3. Open-Orca/OpenOrca

| Property | Value |
|----------|-------|
| **Total Samples** | ~2.94M |
| **File Size** | 2.87 GB |
| **Avg Tokens** | ~1,000 per sample |
| **License** | MIT |
| **Source** | GPT-3.5/GPT-4 augmented FLAN |

**Features**:
- Large-scale augmentation of FLAN
- Mix of GPT-3.5 and GPT-4 responses
- Strong baseline for instruction tuning
- Diverse task types

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 8-16
- Good for large-scale training

---

### 4. mlabonne/FineTome-100k

| Property | Value |
|----------|-------|
| **Total Samples** | 100,000 |
| **File Size** | 117 MB |
| **Avg Tokens** | ~1,200 per sample |
| **License** | Apache 2.0 |
| **Source** | Curated from arcee-ai/The-Tome |

**Features**:
- Carefully curated subset
- Efficient for quick fine-tuning
- Quality over quantity approach
- Good for resource-limited training

**Recommended Config**:
- Max Seq Length: 2,048
- Batch Size: 8-16
- Best for quick experiments

---

## Training Recommendations by Use Case

| Use Case | Recommended Dataset | Max Seq Len | Notes |
|----------|---------------------|-------------|-------|
| Quick experiment | FineTome-100k | 2,048 | Small, efficient |
| Multi-turn chat | ultrachat_200k | 4,096 | Best for conversations |
| General purpose | OpenHermes-2.5 | 2,048 | GPT-4 quality |
| Large-scale | OpenOrca | 2,048 | Massive coverage |
| Production SFT | ultrachat_200k | 4,096 | Well-tested, HF standard |

---

## Comparison Chart

```
Dataset Size Comparison:
========================

OpenOrca          ████████████████████████████████  2.94M samples
OpenHermes-2.5    ███████████                       1.00M samples
ultrachat_200k    █████                             515K samples
FineTome-100k     █                                 100K samples

File Size Comparison:
=====================

OpenOrca          ████████████████████████████████  2.87 GB
OpenHermes-2.5    █████████████████████             1.94 GB
ultrachat_200k    █████████████████                 1.62 GB
FineTome-100k     █                                 117 MB
```

---

## Quick Start Code

```python
from datasets import load_dataset

# Load ultrachat_200k (recommended for SFT)
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# Load with streaming for large datasets
dataset = load_dataset("Open-Orca/OpenOrca", streaming=True)

# Load efficient subset
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
```
