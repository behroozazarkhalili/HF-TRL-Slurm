---
name: slurm-model-trainer
description: Trains, evaluates, and deploys language models on Fir cluster (DRAC) using SLURM with TRL/Unsloth. Use when user mentions training, fine-tuning, SFT, DPO, GRPO, sbatch, SLURM jobs, Fir cluster, DRAC, lm-eval evaluation, GGUF conversion, or Ollama deployment. Supports LoRA/PEFT, Trackio monitoring, and automatic HuggingFace Hub push.
---

# Slurm Model Trainer for Fir Cluster (DRAC)

## Overview

Train, evaluate, and deploy language models on the **Fir cluster** (Digital Research Alliance of Canada) using Slurm. This skill provides comprehensive support for fine-tuning with TRL and Unsloth, evaluation with lm-eval-harness, and deployment via GGUF conversion.

**Automatic HuggingFace Integration:**
- Comprehensive README.md generation for model pages
- GGUF conversion with separate repository
- Proper naming convention: `{BaseModel}-{TrainingMethod}-{Dataset}[-{SampleSize}][-GGUF]`

**Training Methods:**
- **SFT** (Supervised Fine-Tuning) - Standard instruction tuning
- **DPO** (Direct Preference Optimization) - Alignment from preference data
- **GRPO** (Group Relative Policy Optimization) - Online RL training

**Frameworks:**
- **TRL** - Full-featured, production-ready
- **Unsloth** - 2-3x faster training, 80% less VRAM

**For detailed TRL documentation:**
```python
hf_doc_search("your query", product="trl")
hf_doc_fetch("https://huggingface.co/docs/trl/sft_trainer")
```

## When to Use This Skill

**Trigger this skill when user:**
- Asks to "train", "fine-tune", or "finetune" a model
- Mentions "SLURM", "sbatch", "Fir cluster", or "DRAC"
- Wants to run SFT, DPO, or GRPO training
- Needs to evaluate models with lm-eval-harness
- Wants to convert models to GGUF for Ollama/llama.cpp
- Asks to create automated training pipelines (train → eval → convert)
- References training on HPC or compute cluster

**Common user phrases:**
- "Train Qwen on math dataset"
- "Fine-tune LLaMA with DPO"
- "Submit a GRPO training job"
- "Evaluate my model on reasoning benchmarks"
- "Convert model to GGUF for local inference"

## Key Directives

### ALWAYS

1. **Generate complete SBATCH scripts** - Never provide partial snippets; always generate fully executable job scripts
2. **Include Trackio monitoring** - Every training job must use Trackio for real-time metrics tracking
3. **Configure Hub push** - Compute nodes are ephemeral; always push models to HuggingFace Hub to preserve work
4. **Estimate walltime with 30% buffer** - Use the walltime formula and add safety margin
5. **Pre-download models/datasets** - Verify caching on login node before job submission (no internet on compute)
6. **Match GPU to model size** - Use MIG partitions appropriately (see Hardware Selection table)
7. **Generate model cards** - Every trained model needs a comprehensive README.md
8. **Create GGUF versions** - Always offer GGUF conversion for local inference with Ollama

### NEVER

1. **Use hardcoded usernames or paths** - Use environment variables (`$USER`, `$SCRATCH`, `$PROJECT`)
2. **Submit jobs without cache verification** - Always check model/dataset are pre-downloaded
3. **Skip error handling** - Include exit code checks after each pipeline phase
4. **Omit time estimates** - Always calculate expected duration before submission
5. **Ignore OOM risks** - Validate GPU memory requirements for model size + training method

## Prerequisites Checklist

Before starting any training job, verify:

### Account & Authentication
- [ ] DRAC account with GPU allocation on Fir cluster
- [ ] Hugging Face account with write token
- [ ] HF_TOKEN in `.env` file: `echo "HF_TOKEN=hf_xxx" > ~/.env`

### Environment Setup
Run once on login node:
```bash
source /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/utils/setup_env.sh
```

### Pre-download Models/Datasets
**CRITICAL:** Compute nodes have no internet. Download on login node first:
```bash
# Set cache directory
export HF_HOME=$SCRATCH/.cache/huggingface

# Download model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')"

# Download dataset
python -c "from datasets import load_dataset; load_dataset('trl-lib/Capybara')"
```

## Fir Cluster Hardware

### GPU Types

| GPU Type | VRAM | GRES Request | Model Size | Use Case |
|----------|------|--------------|------------|----------|
| **H100 (full)** | 80GB | `--gres=gpu:h100:1` | 7B-70B | Large models, no quantization needed |
| `3g.40gb` (MIG) | 40GB | `--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1` | 3B-13B | Medium models |
| `2g.20gb` (MIG) | 20GB | `--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1` | 0.5B-3B | Small models |
| `1g.10gb` (MIG) | 10GB | `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` | <1B | Quick tests, tiny models |

### Partition Selection

#### Single/Multiple GPU (`gpubase_bygpu_*`)

| Partition | Time Limit | Priority | Use Case |
|-----------|------------|----------|----------|
| `gpubase_bygpu_b1` | 3 hours | Highest | Quick tests, debugging |
| `gpubase_bygpu_b2` | 12 hours | High | Short training runs |
| `gpubase_bygpu_b3` | 1 day | Medium | Standard training |
| `gpubase_bygpu_b4` | 3 days | Medium-Low | Extended training |
| `gpubase_bygpu_b5` | 7 days | Lower | Long GRPO runs, large datasets |

#### Full Node - 4x H100 80GB (`gpubase_bynode_*`)

| Partition | Time Limit | Use Case |
|-----------|------------|----------|
| `gpubase_bynode_b1` | 3 hours | Multi-GPU testing |
| `gpubase_bynode_b2` | 12 hours | Short multi-GPU training |
| `gpubase_bynode_b3` | 1 day | Standard multi-GPU |
| `gpubase_bynode_b4` | 3 days | Extended multi-GPU |
| `gpubase_bynode_b5` | 7 days | Long multi-GPU runs |

#### Special Partitions

| Partition | Time Limit | Notes |
|-----------|------------|-------|
| `gpubackfill` | 1 day | May be preempted, cost-effective |
| `gpupreempt` | 122 days | Will be preempted, lowest priority |
| `gpubase_interac` | 3 hours | Interactive sessions (MIG only) |

### Memory Estimation

```python
# Full fine-tuning (not recommended for large models)
vram_gb = model_params_B * 20

# LoRA fine-tuning (recommended)
vram_gb = model_params_B * 4

# Examples:
# - 0.5B model + LoRA: ~2GB VRAM
# - 7B model + LoRA: ~28GB VRAM
# - 13B model + LoRA: ~52GB VRAM
```

## Job Generator (Recommended)

The new **Job Generator** module provides a user-driven CLI for generating SLURM job scripts with smart defaults based on model size.

### Features

- **Smart Defaults**: Automatically selects optimal hyperparameters based on model size (small/medium/large)
- **Interactive Preview**: Always shows configuration preview before generating
- **User-Driven**: User specifies problem type, model, and dataset
- **Typer CLI**: Beautiful terminal output with Rich

### Model Size Categories

| Category | Parameter Range | Examples |
|----------|-----------------|----------|
| Small | ≤ 1.5B | Qwen2.5-0.5B, Qwen3-0.6B, DeepSeek-R1-1.5B |
| Medium | 1.5B - 4B | Qwen3-1.7B, Qwen2.5-3B, Phi-3-4B |
| Large | 4B - 14B | Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B |

### Usage

```bash
# Navigate to skill directory
cd /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer

# Activate environment
source /scratch/ermia/venvs/hf_env/bin/activate

# Generate GRPO job for 7B model
python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7

# Generate SFT job for small model
python -m generator generate Qwen/Qwen2.5-0.5B HuggingFaceH4/ultrachat_200k -m sft -p 0.5

# Generate with streaming (for large datasets)
python -m generator generate Qwen/Qwen3-1.7B nvidia/OpenMathInstruct-2 -m grpo -p 1.7 --streaming

# Save to file and skip confirmation
python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7 -o jobs/my-job.sh -y

# Show default configs by training method
python -m generator show-defaults -m grpo
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-m, --method` | Training method: sft, grpo, dpo |
| `-p, --params` | Model size in billions (e.g., 7 for 7B) |
| `-o, --output` | Output path for job script |
| `--streaming` | Use streaming for large datasets |
| `--max-samples` | Max samples for streaming (default: 500000) |
| `--reward-type` | Reward type for GRPO (default: combined) |
| `-u, --username` | HuggingFace username |
| `-y, --yes` | Skip confirmation prompt |

### Example Output

```
═══ Job Configuration Preview ═══

Model
  Model ID             Qwen/Qwen2.5-7B
  Size                 7B (large)
  4-bit Quantization   True

Dataset
  Dataset ID           nvidia/OpenMathInstruct-2
  Streaming            True
  Max Samples          500000

Training (GRPO)
  Batch Size           1
  Gradient Accum       16
  Effective Batch      16
  Learning Rate        5e-07
  LoRA Rank            16
  LoRA Alpha           32
  Time Limit           6-00:00:00
  Num Generations      4

═══════════════════════════════════

Generate job script with this configuration? [y/N]:
```

### Python API

```python
from generator import get_defaults, JobGenerator, get_job_name, get_hub_model_id

# Get smart defaults
config = get_defaults(
    model_id="Qwen/Qwen2.5-7B",
    params_b=7.0,
    method="grpo",
    streaming=True,
)

# Add job metadata
config['job_name'] = get_job_name("Qwen/Qwen2.5-7B", "nvidia/OpenMathInstruct-2", "grpo")
config['hub_model_id'] = get_hub_model_id("username", "Qwen/Qwen2.5-7B", "nvidia/OpenMathInstruct-2", "grpo")

# Generate script
generator = JobGenerator()
script = generator.generate("Qwen/Qwen2.5-7B", "nvidia/OpenMathInstruct-2", "grpo", config)
```

## Quick Start Examples

### Example 1: SFT with Unsloth (Fastest)

```bash
# Generate and submit job
sbatch /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/templates/sbatch_single_gpu.sh \
    --export=ALL,SCRIPT=train_sft_unsloth.py,MODEL=Qwen/Qwen2.5-0.5B,DATASET=trl-lib/Capybara,HUB_MODEL_ID=username/my-model
```

### Example 2: DPO Training

```bash
sbatch /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/templates/sbatch_single_gpu.sh \
    --export=ALL,SCRIPT=train_dpo.py,MODEL=Qwen/Qwen2.5-0.5B-Instruct,DATASET=trl-lib/ultrafeedback_binarized,HUB_MODEL_ID=username/my-dpo-model
```

### Example 3: Full Pipeline (Train → Eval → Convert)

```bash
# Submit pipeline job
sbatch /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/templates/sbatch_pipeline.sh \
    --export=ALL,MODEL=Qwen/Qwen2.5-0.5B,DATASET=trl-lib/Capybara,HUB_MODEL_ID=username/my-model
```

### Example 4: Multi-GPU Training (4x H100)

```bash
sbatch /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/templates/sbatch_multi_gpu.sh \
    --export=ALL,SCRIPT=train_sft.py,MODEL=meta-llama/Llama-2-7b-hf,DATASET=trl-lib/Capybara,HUB_MODEL_ID=username/llama2-sft
```

## Training Scripts

### SFT (Supervised Fine-Tuning)

**TRL Version:** `scripts/train_sft.py`
- Standard TRL SFTTrainer
- LoRA/PEFT support
- Trackio monitoring
- Hub push

**Unsloth Version:** `scripts/train_sft_unsloth.py`
- 2-3x faster training
- 80% less VRAM
- Same features as TRL version

**Key Parameters:**
```python
SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    push_to_hub=True,
    hub_model_id="username/model-name",
    report_to="trackio",
)
```

### DPO (Direct Preference Optimization)

**Files:** `scripts/train_dpo.py`, `scripts/train_dpo_unsloth.py`

**Requirements:**
- Base model should be instruction-tuned
- Dataset with `prompt`, `chosen`, `rejected` columns
- Lower learning rate (5e-7 vs 2e-5 for SFT)

**Key Parameters:**
```python
DPOConfig(
    beta=0.1,  # KL penalty (higher = stay closer to reference)
    learning_rate=5e-7,
    num_train_epochs=1,
)
```

### GRPO (Group Relative Policy Optimization)

**File:** `scripts/train_grpo.py`

> **See also**: [Reasoning Models Guide](references/reasoning_models_guide.md) for detailed lessons learned

**Requirements:**
- **Must use Instruct models** - e.g., Qwen2.5-0.5B-Instruct, Qwen3-* (already instruct-tuned)
- Prompt-only dataset format
- Reward function (built-in or custom)
- Online RL (model generates responses during training)
- **Lower learning rate** - 1e-6 for small models, 5e-7 for 7B+

**IMPORTANT: Ask User About Reward Type**

When creating GRPO jobs, always ask the user which reward type they want to use:

| Reward Type | Description | Use Case |
|-------------|-------------|----------|
| `combined` | **Recommended.** Uses all criteria weighted: accuracy (50%) + format (25%) + length (15%) + reasoning (10%) | Math reasoning tasks |
| `math` | Focus on answer correctness with ground truth comparison | Pure accuracy focus |
| `accuracy` | Same as math - extracts and compares answers | General Q&A |
| `format` | Rewards proper formatting (\\boxed{}, step-by-step) | Teaching format habits |
| `length` | Rewards detailed responses | Verbose outputs |

**Key Parameters:**
```python
GRPOConfig(
    num_generations=4,           # Responses per prompt
    max_completion_length=2048,  # Match to dataset distribution!
    learning_rate=1e-6,          # Lower than SFT! (5e-7 for 7B+)
    num_train_epochs=1,
)

# Command line with streaming (for large datasets)
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name nvidia/OpenMathInstruct-2 \
    --streaming \
    --max_samples 500000 \
    --seed 42 \
    --reward_type combined \
    --num_generations 4 \
    --learning_rate 1e-6
```

**Streaming Mode for Large Datasets:**
```bash
# For datasets with millions of samples
--streaming               # Enable streaming mode
--seed 42                 # Reproducible shuffling
--max_samples 500000      # Limit samples
```

**Recommended Math Datasets (by efficiency):**

| Dataset | Size | Max Length | Use Case |
|---------|------|------------|----------|
| `nvidia/OpenMathInstruct-2` | 14M | **2048** | Most efficient (99.9% < 1024 tokens) |
| `open-r1/OpenR1-Math-220k` | 220K | **8192** | Long reasoning chains |
| `AI-MO/NuminaMath-CoT` | 860K | **4096** | Olympiad-level (IMO/AIME) |
| `EleutherAI/hendrycks_math` | 12.5K | **1024** | Competition MATH benchmark |

**Memory Requirements:**

| Model Size | GPU | Quantization | Notes |
|------------|-----|--------------|-------|
| 0.5B-1.5B | 10GB MIG | bf16 | Fast training |
| 1.5B-4B | 40GB MIG | bf16 | Standard |
| 7B-14B | 40GB MIG | **4-bit** | Required for fit |

## Intelligent Dataset Format Detection

The training scripts (`train_sft.py`, `train_sft_unsloth.py`) automatically detect and handle various dataset formats used in the HuggingFace ecosystem. This eliminates the need to manually configure format-specific parameters.

### Supported Formats

| Format | Required Columns | Example Datasets |
|--------|------------------|------------------|
| `conversational` | `messages` or `conversations` | HuggingFaceH4/ultrachat_200k, OpenAssistant |
| `prompt_completion` | `prompt`, `completion` | Custom instruction datasets |
| `text` | `text` | Pre-formatted text datasets |
| `instruction_output` | `instruction`, `output` (+ optional `input`) | Alpaca-style datasets |

### How It Works

1. **Auto-Detection**: The script analyzes dataset columns to determine format
2. **Column Cleanup**: Removes ambiguous columns that could confuse TRL
3. **Format Preparation**: Prepares data structure for SFTTrainer compatibility

### Example: Conversational Dataset

For datasets with `messages` column (e.g., UltraChat-200k):

```python
# Original columns: ['prompt', 'prompt_id', 'messages']
# The 'prompt' column could confuse TRL (expects 'completion' pair)

# Detected format: conversational
# Prepared columns: ['messages']  # Only keep messages for chat template
```

### Example: Alpaca-Style Dataset

For datasets with `instruction`/`output` columns:

```python
# Columns: ['instruction', 'input', 'output']
# Detected format: instruction_output
# TRL handles via chat_template or formats to text
```

### Manual Format Override

If auto-detection doesn't work for your dataset, you can still prepare data manually:

```python
# Map custom columns to expected format
dataset = dataset.rename_columns({"my_prompt": "prompt", "my_response": "completion"})
```

### Key Benefits

1. **Zero Configuration**: Works out-of-the-box with most HuggingFace datasets
2. **TRL Compatibility**: Ensures SFTTrainer receives correctly formatted data
3. **Chat Template Support**: Conversational format uses model's native chat template
4. **Error Prevention**: Removes columns that could cause format ambiguity

## Evaluation with lm-eval-harness

### Available Benchmark Suites

| Suite | Tasks | Use Case |
|-------|-------|----------|
| `reasoning` | GSM8K, MATH, ARC-C, ARC-E | Mathematical/logical reasoning |
| `general` | MMLU, HellaSwag, TruthfulQA, Winogrande | General knowledge |
| `coding` | HumanEval, MBPP | Code generation |
| `comprehensive` | All of the above | Full evaluation |

### Running Evaluation

```bash
# Submit evaluation job after training
sbatch /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/templates/sbatch_eval.sh \
    --export=ALL,MODEL=username/my-model,TASKS=comprehensive
```

### Evaluation Script Usage

```python
python scripts/evaluate_model.py \
    --model username/my-model \
    --tasks reasoning general coding \
    --batch_size auto \
    --output_dir ./eval_results \
    --push_to_hub
```

## GGUF Conversion for Ollama

### Convert Trained Model

```bash
sbatch /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/templates/sbatch_convert.sh \
    --export=ALL,MODEL=username/my-model,OUTPUT_REPO=username/my-model-gguf
```

### Quantization Options

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| Q4_K_M | ~4 bits | Good | Recommended balance |
| Q5_K_M | ~5 bits | Better | Higher quality |
| Q8_0 | 8 bits | Best | Maximum quality |

### Using with Ollama

```bash
# After conversion, on local machine:
ollama pull hf.co/username/my-model-gguf:Q4_K_M
ollama run my-model-gguf
```

## Trackio Monitoring

### How It Works on Slurm

1. **During training**: Trackio logs to local SQLite database
2. **After job completes**: Sync logs to HF Space from login node
3. **View dashboard**: `https://huggingface.co/spaces/username/trackio`

### Configuration

```python
# In training script
import trackio

trackio.init(
    project="my-training-project",
    run_name="sft-qwen-0.5b-lr2e-5",
    space_id="username/trackio",  # Optional: auto-creates Space
)
```

### Post-Job Sync

```bash
# After job completes, on login node:
python -c "import trackio; trackio.sync_to_hub('username/trackio')"
```

## Job Chaining

### Automatic Pipeline

The pipeline template chains jobs with dependencies:

```
Train Job → (on success) → Eval Job → (on success) → GGUF Convert
```

### Manual Chaining

```bash
# Submit training job, get job ID
TRAIN_JOB=$(sbatch train.sh | awk '{print $4}')

# Submit eval job with dependency
EVAL_JOB=$(sbatch --dependency=afterok:$TRAIN_JOB eval.sh | awk '{print $4}')

# Submit conversion with dependency
sbatch --dependency=afterok:$EVAL_JOB convert.sh
```

## Walltime Estimation

```python
# Formula
base_hours = 0.1 * model_params_B * (dataset_size_K / 1000) * epochs
h100_multiplier = 0.5  # H100 is ~2x faster than A100
estimated_hours = base_hours * h100_multiplier

# Add 30% buffer for walltime
walltime_hours = estimated_hours * 1.3

# Examples:
# - 0.5B model, 10K examples, 3 epochs: ~0.25 hours
# - 7B model, 10K examples, 3 epochs: ~3.5 hours
# - 13B model, 50K examples, 3 epochs: ~33 hours
```

## Common Failure Modes

### 1. Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Fix (try in order):**
1. Reduce batch size: `per_device_train_batch_size=1`
2. Increase gradient accumulation: `gradient_accumulation_steps=16`
3. Enable gradient checkpointing: `gradient_checkpointing=True`
4. Use LoRA: Add PEFT config
5. Use larger GPU: h100 instead of MIG

### 2. Job Timeout

**Symptoms:** Job killed, incomplete training

**Fix:**
1. Estimate walltime correctly (see formula above)
2. Add 30% buffer to estimated time
3. Enable checkpointing to resume from last save
4. Use `gpubase_bygpu_b5` for 7-day limit

### 3. Model/Dataset Not Found

**Symptoms:** HTTP 404 or connection timeout

**Fix:**
1. Pre-download on login node (compute nodes have no internet)
2. Set `HF_HOME=$SCRATCH/.cache/huggingface`
3. Verify download completed before submitting job

### 4. Hub Push Failed

**Symptoms:** Training completes but model not on Hub

**Fix:**
1. Check `HF_TOKEN` in environment
2. Verify token has write permissions
3. Enable checkpoint push: `hub_strategy="every_save"`
4. Add retry logic for network issues

### 5. Eval Dataset Missing

**Symptoms:** Training hangs or crashes

**Fix:**
1. Either provide `eval_dataset` when `eval_strategy` is set
2. Or set `eval_strategy="no"` if validation not needed

## File Reference

### Generator Module

| File | Description |
|------|-------------|
| `generator/__init__.py` | Module exports |
| `generator/__main__.py` | CLI entry point |
| `generator/smart_defaults.py` | Size-based training defaults |
| `generator/clarifier.py` | Interactive questioning for ambiguous decisions |
| `generator/cli.py` | Typer CLI interface |
| `generator/job_generator.py` | SBATCH script generation |

### Configs

| File | Description |
|------|-------------|
| `configs/training/sft.yaml` | SFT method settings |
| `configs/training/grpo.yaml` | GRPO method settings |
| `configs/training/dpo.yaml` | DPO method settings |
| `configs/hardware/fir_cluster.yaml` | Fir cluster hardware profiles |

### Scripts

| File | Description |
|------|-------------|
| `scripts/train_sft.py` | TRL SFT training |
| `scripts/train_sft_unsloth.py` | Unsloth SFT (2-3x faster) |
| `scripts/train_dpo.py` | TRL DPO training |
| `scripts/train_dpo_unsloth.py` | Unsloth DPO |
| `scripts/train_grpo.py` | TRL GRPO training |
| `scripts/evaluate_model.py` | lm-eval-harness wrapper |
| `scripts/convert_gguf.py` | GGUF conversion |
| `scripts/validate_dataset.py` | Dataset format checker |

### Templates

| File | Description |
|------|-------------|
| `templates/sbatch_single_gpu.sh` | Single GPU job |
| `templates/sbatch_multi_gpu.sh` | Multi-GPU (4x H100) |
| `templates/sbatch_multi_node.sh` | Multi-node distributed |
| `templates/sbatch_eval.sh` | Evaluation job |
| `templates/sbatch_convert.sh` | GGUF conversion |
| `templates/sbatch_pipeline.sh` | Full pipeline |

### Configs

| File | Description |
|------|-------------|
| `configs/accelerate/single_gpu.yaml` | Single GPU config |
| `configs/accelerate/multi_gpu_ddp.yaml` | Multi-GPU DDP |
| `configs/accelerate/multi_gpu_fsdp.yaml` | Multi-GPU FSDP |
| `configs/accelerate/deepspeed_zero3.yaml` | DeepSpeed ZeRO-3 |
| `configs/eval_tasks/reasoning.yaml` | Reasoning benchmarks |
| `configs/eval_tasks/general.yaml` | General benchmarks |
| `configs/eval_tasks/coding.yaml` | Coding benchmarks |
| `configs/eval_tasks/comprehensive.yaml` | All benchmarks |

### Utilities

| File | Description |
|------|-------------|
| `utils/setup_env.sh` | One-time environment setup |
| `utils/submit_job.py` | Job submission helper |
| `utils/chain_jobs.py` | Job dependency manager |
| `utils/estimate_time.py` | Walltime estimator |

## HuggingFace Integration

### Naming Convention

All models pushed to HuggingFace follow this naming convention:

```
{BaseModel}-{TrainingMethod}-{Dataset}[-{SampleSize}][-GGUF]
```

**Examples:**
- `Qwen2.5-7B-SFT-Capybara` - SFT trained on full Capybara dataset
- `LFM2-350M-GRPO-NuminaMath-10K` - GRPO trained on 10K samples
- `SmolLM2-1.7B-GRPO-NuminaMath-100K` - GRPO trained on 100K samples
- `Qwen2.5-7B-SFT-Capybara-GGUF` - GGUF quantized version

**Sample Size Formatting:**
| Sample Count | Label |
|--------------|-------|
| < 1,000 | Raw number (e.g., "500") |
| 1,000 - 999,999 | K format (e.g., "10K", "100K") |
| ≥ 1,000,000 | M format (e.g., "1M", "5M") |

**Rules:**
- Use hyphens (`-`) not underscores (`_`)
- Base model name preserved (e.g., `Qwen2.5-7B`, not `qwen2.5-7b`)
- Training method in uppercase (SFT, DPO, GRPO)
- Dataset name as-is from source (e.g., `Capybara`, `NuminaMath`)
- Sample size suffix when using streaming with limited samples
- GGUF suffix for quantized versions

### Automatic Model Card Generation

The pipeline automatically generates comprehensive README.md files including:

- Model details (base model, license, size)
- Training configuration (LR, batch size, epochs, LoRA params)
- Training metrics (loss, accuracy, time)
- Dataset information (train/eval splits)
- Usage examples (Transformers, vLLM, 4-bit, llama.cpp)
- Framework versions (auto-extracted, without cluster suffixes)
- Citation in BibTeX format

**Generate manually:**
```bash
python scripts/generate_model_card.py \
    --model_name "Qwen2.5-7B-SFT-Capybara" \
    --base_model "Qwen/Qwen2.5-7B" \
    --dataset "trl-lib/Capybara" \
    --training_method SFT \
    --author ermiaazarkhalili \
    --license cc-by-nc-4.0 \
    --output_dir ./model_card
```

### Pipeline Outputs

The full pipeline (`sbatch_pipeline.sh`) produces:

1. **Original Model** - `username/Model-Method-Dataset`
   - Full model weights or LoRA adapters
   - Comprehensive README.md
   - Training metrics and configs

2. **GGUF Repository** - `username/Model-Method-Dataset-GGUF`
   - Q4_K_M, Q5_K_M, Q8_0 quantizations
   - Usage instructions for Ollama, llama.cpp, LM Studio

## Key Takeaways

1. **Pre-download everything** - Compute nodes have no internet
2. **Use Unsloth for speed** - 2-3x faster, 80% less VRAM
3. **Always set walltime buffer** - Add 30% to estimated time
4. **Enable Hub push** - Environment is not permanent
5. **Include Trackio** - For monitoring (sync post-job)
6. **Auto format detection** - Scripts handle conversational, alpaca, and text formats automatically
7. **Chain jobs for automation** - train → model card → eval → GGUF
8. **Use LoRA for large models** - Reduces VRAM significantly
9. **Follow naming convention** - `{BaseModel}-{Method}-{Dataset}[-GGUF]`
10. **Always generate GGUF** - For local inference (Ollama, llama.cpp)
