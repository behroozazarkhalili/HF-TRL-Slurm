# Hugging Face Hub Push Guide

## Overview

Pushing models to HF Hub is **critical** because:
- Slurm job storage is temporary (may be cleaned up)
- $SCRATCH has a 60-day purge policy
- Hub provides permanent, accessible storage

## Your HF Account

**Username:** `ermiaazarkhalili`
**Models Collection:** https://huggingface.co/ermiaazarkhalili/models

## Setup

### 1. Get HF Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `slurm-training`
4. Type: **Write** (required for pushing)
5. Copy the token

### 2. Add Token to Environment

Create/edit `.env` file:
```bash
# /project/6014832/ermia/HF-TRL/.env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Verify Token

```python
from huggingface_hub import HfApi
api = HfApi()
print(api.whoami())  # Should show: ermiaazarkhalili
```

## Training Script Configuration

### Required Settings

```python
from trl import SFTConfig

config = SFTConfig(
    output_dir="./output",

    # Hub push settings (REQUIRED)
    push_to_hub=True,
    hub_model_id="ermiaazarkhalili/my-model-name",

    # Push strategy
    hub_strategy="every_save",  # Push after each checkpoint
    # Options: "end", "every_save", "checkpoint", "all_checkpoints"

    # Optional: Private repo
    hub_private_repo=False,
)
```

### SBATCH Job Configuration

```bash
#!/bin/bash
#SBATCH --export=ALL  # Export all environment variables

# Load token from .env
source /project/6014832/ermia/HF-TRL/.env
export HF_TOKEN

# Run training with hub push
python train.py \
    --push_to_hub \
    --hub_model_id ermiaazarkhalili/my-model
```

## Model Naming Convention

Recommended naming format:
```
{BaseModel}-{TrainingMethod}-{Dataset}[-{SampleSize}][-GGUF]
```

**Sample Size Formatting:**
| Sample Count | Label |
|--------------|-------|
| < 1,000 | Raw number (e.g., "500") |
| 1,000 - 999,999 | K format (e.g., "10K", "100K") |
| ≥ 1,000,000 | M format (e.g., "1M", "5M") |

Examples:
- `ermiaazarkhalili/Qwen2.5-0.5B-SFT-Capybara` - Full dataset
- `ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-10K` - 10K samples
- `ermiaazarkhalili/SmolLM2-1.7B-GRPO-NuminaMath-100K` - 100K samples
- `ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-10K-GGUF` - GGUF version

## Hub Push Strategies

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| `every_save` | Long training | Recover from crashes | More Hub writes |
| `end` | Short training | Simple, one push | Lose progress on failure |
| `checkpoint` | Experiment | Keep all checkpoints | Uses more Hub space |

## Checkpoint Recovery

If job fails, you can resume from Hub checkpoint:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load the last checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "ermiaazarkhalili/my-model",
    revision="checkpoint-500",  # Specific checkpoint
)

# Continue training
trainer.train(resume_from_checkpoint=True)
```

## Offline Push (Post-Job)

If Hub push fails during job, push manually from login node:

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model directory
api.upload_folder(
    folder_path="./outputs/12345",  # Local path
    repo_id="ermiaazarkhalili/my-model",
    repo_type="model",
)
```

Or using CLI:
```bash
huggingface-cli upload ermiaazarkhalili/my-model ./outputs/12345
```

## Model Card

The training scripts automatically create model cards. To customize:

```python
from huggingface_hub import ModelCard

card = ModelCard.load("ermiaazarkhalili/my-model")
card.text = """
---
base_model: Qwen/Qwen2.5-0.5B
datasets:
  - trl-lib/Capybara
tags:
  - sft
  - fine-tuned
---

# My Fine-tuned Model

## Training Details
- Base model: Qwen/Qwen2.5-0.5B
- Dataset: trl-lib/Capybara
- Method: SFT with LoRA
- Epochs: 3

## Usage
...
"""
card.push_to_hub("ermiaazarkhalili/my-model")
```

## Adding Evaluation Results

After running lm-eval-harness:

```python
from huggingface_hub import ModelCard

card = ModelCard.load("ermiaazarkhalili/my-model")

# Add evaluation section
eval_section = """

## Evaluation Results

| Benchmark | Score |
|-----------|-------|
| MMLU | 65.2% |
| HellaSwag | 78.1% |
| GSM8K | 45.3% |

"""

card.text += eval_section
card.push_to_hub("ermiaazarkhalili/my-model")
```

## Quick Reference

### Training with Hub Push

```bash
# Single command
python scripts/train_sft_unsloth.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir ./outputs \
    --push_to_hub \
    --hub_model_id ermiaazarkhalili/qwen-capybara-sft \
    --hub_strategy every_save
```

### Verify Upload

```bash
# Check model exists
huggingface-cli repo info ermiaazarkhalili/my-model

# List files
huggingface-cli repo files ermiaazarkhalili/my-model
```

### Download Your Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ermiaazarkhalili/my-model")
tokenizer = AutoTokenizer.from_pretrained("ermiaazarkhalili/my-model")
```

## Troubleshooting

### "Permission denied" or 403 Error

1. Check token has write permission
2. Verify you own the repo or have write access
3. Re-create token if needed

### "Repository not found" on Push

```python
# Create repo first
from huggingface_hub import create_repo
create_repo("ermiaazarkhalili/my-model", private=False)
```

### Push Timeout

For large models, increase timeout:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./model",
    repo_id="ermiaazarkhalili/my-model",
    timeout=3600,  # 1 hour
)
```

### Network Issues on Compute Nodes

Compute nodes may not have internet. Solutions:
1. Use `hub_strategy="every_save"` - retries on next save
2. Save locally, push from login node after job
3. Check if cluster has internal proxy
