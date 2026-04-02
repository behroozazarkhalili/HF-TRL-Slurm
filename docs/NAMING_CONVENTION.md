# Model Naming Convention Guide

## Overview

This document defines the naming convention for models trained in the HF-TRL pipeline, ensuring consistent and discoverable model names across HuggingFace Hub.

## Naming Format

### Standard Format
```
{BaseModel}-{TrainingMethod}-{Dataset}[-{SampleSize}][-{Variant}]
```

### Components

| Component | Description | Examples |
|-----------|-------------|----------|
| **BaseModel** | Model architecture and size | `LFM2-350M`, `Qwen2.5-7B`, `SmolLM2-1.7B` |
| **TrainingMethod** | Training algorithm used | `SFT`, `GRPO`, `DPO`, `RLHF` |
| **Dataset** | Short name of training dataset | `NuminaMath`, `UltraChat`, `OpenMath2` |
| **SampleSize** | Number of training samples (optional) | `10K`, `50K`, `100K`, `200K` |
| **Variant** | Special variant identifier (optional) | `Instruct`, `Thinking`, `v2` |

### Sample Size Formatting

| Samples | Format |
|---------|--------|
| 1,000 | `1K` |
| 10,000 | `10K` |
| 50,000 | `50K` |
| 100,000 | `100K` |
| 200,000 | `200K` |
| 1,000,000 | `1M` |

## Examples

### GRPO Models (Math Reasoning)
```
ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-10K
ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-50K
ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-100K
ermiaazarkhalili/LFM2-2.6B-GRPO-NuminaMath-50K
ermiaazarkhalili/SmolLM2-1.7B-GRPO-NuminaMath-100K
```

### SFT Models (Instruction Following)
```
ermiaazarkhalili/Qwen2.5-7B-SFT-UltraChat
ermiaazarkhalili/LFM2-350M-SFT-UltraChat-200K
ermiaazarkhalili/SmolLM2-1.7B-SFT-Capybara-15K
```

### GGUF Versions
Append `-GGUF` to the model name:
```
ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-10K-GGUF
ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-50K-GGUF
```

## When to Include Sample Size

### Include Sample Size When:
- Training multiple versions with different sample counts
- Sample count significantly affects model quality
- Running experiments to compare sample efficiency
- Dataset is large and you're using a subset

### Omit Sample Size When:
- Using the full dataset
- Only training one version
- Sample count is standard/expected for that dataset

## Job Script Variables

In SLURM job scripts, use these variables:

```bash
# Configuration
MODEL_NAME="LiquidAI/LFM2-350M"
DATASET_NAME="AI-MO/NuminaMath-CoT"
MAX_SAMPLES=50000
SAMPLE_SIZE_LABEL="50K"  # Human-readable format

# Derived names (include sample size)
HUB_MODEL_ID="ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-${SAMPLE_SIZE_LABEL}"
GGUF_REPO_ID="ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-${SAMPLE_SIZE_LABEL}-GGUF"

# Job name should match
#SBATCH --job-name=lfm2-350m-grpo-numina-50k
```

## Dataset Short Names

| Full Dataset Name | Short Name |
|-------------------|------------|
| `AI-MO/NuminaMath-CoT` | `NuminaMath` |
| `nvidia/OpenMathInstruct-2` | `OpenMath2` |
| `open-r1/OpenR1-Math-220k` | `OpenR1Math` |
| `HuggingFaceH4/ultrachat_200k` | `UltraChat` |
| `trl-lib/Capybara` | `Capybara` |
| `microsoft/orca-math-word-problems-200k` | `OrcaMath` |
| `openai/gsm8k` | `GSM8K` |
| `meta-math/MetaMathQA` | `MetaMath` |

## Migration Guide

### Renaming Existing Models

If you need to rename an existing model to include sample size:

1. **Don't delete the old model** - keep it for backward compatibility
2. **Create the new model** with proper naming
3. **Update the old model's README** to point to the new location
4. **Deprecation notice** in old model card:

```markdown
> **Note**: This model has been superseded by
> [LFM2-350M-GRPO-NuminaMath-10K](https://huggingface.co/ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-10K)
> which uses the updated naming convention.
```

## File Naming in Job Scripts

Job script filenames should follow:
```
{model}-{method}-{dataset}[-{samples}].sh
```

Examples:
```
lfm2-350m-grpo-numina-10k.sh
lfm2-350m-grpo-numina-50k.sh
lfm2-2.6b-grpo-numina-50k.sh
smollm2-1.7b-grpo-numina-100k.sh
```
