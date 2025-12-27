# Comprehensive Usage Guide for Finetuning Skills

This guide provides comprehensive examples and usage patterns for the slurm-model-trainer skill. It covers all training methods, evaluation, GGUF conversion, and HuggingFace Hub integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Job Generator (Recommended)](#job-generator-recommended)
3. [Environment Setup](#environment-setup)
4. [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
5. [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
6. [GRPO (Group Relative Policy Optimization)](#grpo-group-relative-policy-optimization)
7. [Model Evaluation](#model-evaluation)
8. [GGUF Conversion](#gguf-conversion)
9. [HuggingFace Hub Integration](#huggingface-hub-integration)
10. [Model Card Generation](#model-card-generation)
11. [Advanced Configurations](#advanced-configurations)
12. [Multi-GPU Training](#multi-gpu-training)
13. [Job Pipelines](#job-pipelines)
14. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Using Job Generator (Recommended)

```bash
# 1. Navigate to the skill directory
cd /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer

# 2. Generate job script with smart defaults
python -m generator generate Qwen/Qwen2.5-1.5B trl-lib/Capybara -m sft -p 1.5 -o jobs/my-job.sh

# 3. Pre-download model and dataset
export HF_HOME=$SCRATCH/.cache/huggingface
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')"
python -c "from datasets import load_dataset; load_dataset('trl-lib/Capybara')"

# 4. Submit training job
sbatch jobs/my-job.sh
```

### Manual SFT Example

```bash
# 1. Setup environment (run once)
source /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/utils/setup_env.sh

# 2. Pre-download model and dataset (required - compute nodes have no internet)
export HF_HOME=$SCRATCH/.cache/huggingface
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')"
python -c "from datasets import load_dataset; load_dataset('trl-lib/Capybara')"

# 3. Submit training job
sbatch job_script.sh
```

---

## Job Generator (Recommended)

The Job Generator automatically creates SLURM job scripts with smart defaults based on your model size and training method. This is the **recommended way** to create training jobs.

### Basic Usage

```bash
# Navigate to the skill directory
cd /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer

# Generate a job script (shows preview, asks confirmation)
python -m generator generate MODEL_ID DATASET_ID -m METHOD -p SIZE

# Examples:
python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7
python -m generator generate Qwen/Qwen2.5-1.5B HuggingFaceH4/ultrachat_200k -m sft -p 1.5
python -m generator generate Qwen/Qwen2.5-7B-Instruct argilla/capybara-dpo-7k-binarized -m dpo -p 7
```

### Smart Defaults by Model Size

The generator automatically applies optimal parameters based on model size:

| Size Category | Parameters | Batch Size | Grad Accum | LoRA Rank | 4-bit |
|---------------|------------|------------|------------|-----------|-------|
| small | ≤1.5B | 4-8 | 2-4 | 32 | No |
| medium | 1.5B-4B | 2-4 | 4-8 | 64 | No |
| large | 4B-14B | 1 | 16 | 64 | Yes |

### Configuration Preview

Before generating, the tool shows a preview of all settings:

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

### Generator Commands

```bash
# Generate job script (interactive - shows preview)
python -m generator generate MODEL DATASET -m METHOD -p SIZE

# Generate without confirmation (for automation)
python -m generator generate MODEL DATASET -m METHOD -p SIZE --yes

# Save to specific file
python -m generator generate MODEL DATASET -m METHOD -p SIZE -o jobs/my-job.sh

# Use streaming for large datasets
python -m generator generate MODEL DATASET -m METHOD -p SIZE --streaming

# Show smart defaults for a model size
python -m generator show-defaults -p 7 -m grpo

# List available templates
python -m generator list-templates
```

### Complete Example: Generate and Submit GRPO Job

```bash
# 1. Generate job script
cd /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer
python -m generator generate \
    Qwen/Qwen2.5-7B-Instruct \
    nvidia/OpenMathInstruct-2 \
    -m grpo \
    -p 7 \
    --streaming \
    -o /project/6014832/ermia/HF-TRL/jobs/qwen7b-grpo.sh

# 2. Pre-download model and dataset (required - compute nodes have no internet)
export HF_HOME=$SCRATCH/.cache/huggingface
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
python -c "from datasets import load_dataset; ds = load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True); next(iter(ds))"

# 3. Submit job
sbatch /project/6014832/ermia/HF-TRL/jobs/qwen7b-grpo.sh

# 4. Monitor
squeue -u $USER
tail -f /project/6014832/ermia/HF-TRL/jobs/logs/qwen7b-grpo-*.out
```

### Training Method Options

| Method | Flag | Description | Typical Use Case |
|--------|------|-------------|------------------|
| SFT | `-m sft` | Supervised Fine-Tuning | General instruction tuning |
| GRPO | `-m grpo` | Group Relative Policy Optimization | Math reasoning, verifiable tasks |
| DPO | `-m dpo` | Direct Preference Optimization | Preference alignment |

### Recommended Datasets by Method

**SFT (Supervised Fine-Tuning)**:
- `HuggingFaceH4/ultrachat_200k` - Large conversational dataset
- `trl-lib/Capybara` - High-quality chat dataset

**GRPO (Reinforcement Learning)**:
- `nvidia/OpenMathInstruct-2` - 14M math problems (use streaming)
- `open-r1/OpenR1-Math-220k` - Math reasoning dataset

**DPO (Preference Optimization)**:
- `argilla/capybara-dpo-7k-binarized` - Chat preferences
- `HuggingFaceH4/ultrafeedback_binarized` - General preferences

---

## Environment Setup

### One-Time Setup

```bash
# Source the setup script
source /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/utils/setup_env.sh

# This creates a virtual environment with:
# - TRL
# - Transformers
# - PEFT
# - BitsAndBytes
# - Unsloth (optional)
# - Trackio for monitoring
```

### HuggingFace Authentication

```bash
# Create .env file with your HF token
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > ~/.env

# The token must have write permissions for Hub push
```

### Pre-Download Models and Datasets

```bash
# Set cache directory
export HF_HOME=$SCRATCH/.cache/huggingface

# Download base model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
"

# Download dataset
python -c "
from datasets import load_dataset
load_dataset('trl-lib/Capybara')
"
```

---

## SFT (Supervised Fine-Tuning)

### Standard TRL SFT

```bash
# Using train_sft.py
python scripts/train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --dataset_name trl-lib/Capybara \
    --output_dir ./outputs/qwen7b-sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_seq_length 2048 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --push_to_hub \
    --hub_model_id ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara \
    --report_to trackio
```

### Unsloth SFT (2-3x Faster)

```bash
# Using train_sft_unsloth.py
python scripts/train_sft_unsloth.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --dataset_name trl-lib/Capybara \
    --output_dir ./outputs/qwen7b-sft-unsloth \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_seq_length 2048 \
    --lora_r 64 \
    --lora_alpha 128 \
    --push_to_hub \
    --hub_model_id ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara-Unsloth
```

### SFT with Evaluation

```bash
python scripts/train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --dataset_name trl-lib/Capybara \
    --output_dir ./outputs/qwen7b-sft \
    --test_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --load_best_model_at_end \
    --push_to_hub \
    --hub_model_id username/model-name
```

### Custom Dataset Format

```python
# Your dataset should have a column that can be used as prompts
# The script will try to find: 'prompt', 'instruction', 'question', 'input'

# Option 1: Dataset with 'messages' column (chat format)
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

# Option 2: Dataset with 'prompt' and 'completion' columns
# prompt: "Write a poem about..."
# completion: "Roses are red..."

# Option 3: Dataset with 'instruction' and 'output' columns
# instruction: "Translate to French"
# output: "Bonjour"
```

---

## DPO (Direct Preference Optimization)

### Standard DPO Training

```bash
python scripts/train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir ./outputs/qwen7b-dpo \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --loss_type sigmoid \
    --max_length 1024 \
    --max_prompt_length 512 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --push_to_hub \
    --hub_model_id username/Qwen2.5-7B-DPO
```

### DPO Loss Types

```bash
# Sigmoid (default, most common)
--loss_type sigmoid

# Hinge loss (alternative)
--loss_type hinge

# IPO (Identity Preference Optimization)
--loss_type ipo

# KTO Pair
--loss_type kto_pair
```

### DPO Dataset Format

```python
# Required columns: prompt, chosen, rejected
# Example:
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "France's capital is Lyon."
}
```

### DPO with Custom Beta

```bash
# Higher beta = stay closer to reference model
python scripts/train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --beta 0.5 \
    --learning_rate 1e-7 \
    --push_to_hub
```

---

## GRPO (Group Relative Policy Optimization)

### Standard GRPO Training

```bash
python scripts/train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/math_shepherd \
    --output_dir ./outputs/qwen7b-grpo \
    --num_generations 4 \
    --reward_type accuracy \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --max_length 512 \
    --max_prompt_length 256 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --push_to_hub \
    --hub_model_id username/Qwen2.5-7B-GRPO
```

### GRPO Reward Types

```bash
# Accuracy-based reward (for math problems)
--reward_type accuracy

# Length-based reward (prefer detailed responses)
--reward_type length

# Custom reward function
--reward_type custom
```

### GRPO Dataset Format

```python
# Requires 'prompt' column only (online RL generates responses)
{
    "prompt": "Solve the equation: 2x + 5 = 15"
}
```

---

## Model Evaluation

### Using lm-eval-harness

```bash
python scripts/evaluate_model.py \
    --model username/my-trained-model \
    --tasks reasoning general coding \
    --batch_size auto \
    --output_dir ./eval_results \
    --push_to_hub
```

### Available Task Suites

```bash
# Reasoning benchmarks
--tasks reasoning  # GSM8K, MATH, ARC-C, ARC-E

# General knowledge
--tasks general  # MMLU, HellaSwag, TruthfulQA, Winogrande

# Code generation
--tasks coding  # HumanEval, MBPP

# All benchmarks
--tasks comprehensive

# Custom task list
--tasks mmlu gsm8k hellaswag
```

### Evaluation with Quantization

```bash
python scripts/evaluate_model.py \
    --model username/my-model \
    --tasks general \
    --load_in_4bit \
    --batch_size 8
```

---

## GGUF Conversion

### Convert Model to GGUF

```bash
python scripts/convert_gguf.py \
    --model_id username/my-trained-model \
    --output_dir ./gguf_output \
    --quantization Q4_K_M Q5_K_M Q8_0 \
    --push_to_hub \
    --hub_repo_id username/my-model-GGUF
```

### Merge LoRA Before Conversion

```bash
python scripts/convert_gguf.py \
    --model_id username/my-lora-model \
    --merge_lora \
    --base_model Qwen/Qwen2.5-7B \
    --output_dir ./gguf_output \
    --quantization Q4_K_M
```

### Available Quantizations

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| Q4_K_M | ~4 bits | Good | Recommended balance |
| Q5_K_M | ~5 bits | Better | Higher quality |
| Q8_0 | 8 bits | Best | Maximum quality |
| Q2_K | ~2 bits | Low | Maximum compression |
| Q3_K_S | ~3 bits | Medium | Space-constrained |
| Q6_K | 6 bits | High | Near-original quality |

---

## HuggingFace Hub Integration

### Push During Training

```bash
python scripts/train_sft.py \
    --push_to_hub \
    --hub_model_id username/model-name \
    --hub_strategy end  # Push only at end (recommended)
```

### Hub Strategies

```bash
# Push only final model (recommended)
--hub_strategy end

# Push every checkpoint
--hub_strategy checkpoint

# Push at every save
--hub_strategy every_save
```

### Manual Push After Training

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model files
api.upload_folder(
    folder_path="./outputs/my-model",
    repo_id="username/my-model",
    repo_type="model"
)

# Upload single file
api.upload_file(
    path_or_fileobj="./README.md",
    path_in_repo="README.md",
    repo_id="username/my-model"
)
```

---

## Model Card Generation

### Automatic Model Card

```bash
python scripts/generate_model_card.py \
    --model_name "Qwen2.5-7B-SFT-Capybara" \
    --base_model "Qwen/Qwen2.5-7B" \
    --dataset "trl-lib/Capybara" \
    --training_method SFT \
    --author ermiaazarkhalili \
    --license cc-by-nc-4.0 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --epochs 1 \
    --max_seq_length 2048 \
    --lora_r 64 \
    --lora_alpha 128 \
    --train_loss 1.0456 \
    --eval_loss 1.0092 \
    --accuracy 0.7369 \
    --train_samples 14225 \
    --eval_samples 1581 \
    --training_time "~2 hours" \
    --hardware "NVIDIA H100 80GB" \
    --output_dir ./model_card
```

### Extract from Training Log

```bash
python scripts/generate_model_card.py \
    --model_name "Qwen2.5-7B-SFT-Capybara" \
    --base_model "Qwen/Qwen2.5-7B" \
    --dataset "trl-lib/Capybara" \
    --training_log /scratch/ermia/outputs/qwen7b-sft/training.log \
    --output_dir ./model_card
```

### GGUF Model Card

```bash
python scripts/generate_model_card.py \
    --model_name "Qwen2.5-7B-SFT-Capybara-GGUF" \
    --base_model "Qwen/Qwen2.5-7B" \
    --dataset "trl-lib/Capybara" \
    --is_gguf \
    --gguf_quantizations Q4_K_M Q5_K_M Q8_0 \
    --output_dir ./gguf_model_card
```

### Model Naming Convention

```
{BaseModel}-{TrainingMethod}-{Dataset}[-Quantization]

Examples:
- Qwen2.5-7B-SFT-Capybara
- Qwen2.5-7B-DPO-UltraFeedback
- Qwen2.5-7B-SFT-Capybara-GGUF
- Llama-3.1-8B-GRPO-MathShepherd
```

---

## Advanced Configurations

### LoRA Parameters

```bash
# Standard LoRA (recommended for most cases)
--lora_r 64 \
--lora_alpha 128 \
--lora_dropout 0.0 \
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

# Smaller LoRA (for memory constraints)
--lora_r 16 \
--lora_alpha 32

# Larger LoRA (for more capacity)
--lora_r 128 \
--lora_alpha 256
```

### Quantization Options

```bash
# 4-bit NF4 (recommended)
--use_4bit

# 8-bit
--use_8bit

# No quantization (requires more VRAM)
# (omit both flags)
```

### Learning Rate Scheduling

```bash
# Linear warmup + Cosine decay (default)
--warmup_ratio 0.1

# Custom warmup steps
--warmup_steps 100

# No warmup
--warmup_ratio 0.0
```

### Gradient Checkpointing

```bash
# Enable gradient checkpointing (saves VRAM, slightly slower)
--gradient_checkpointing

# Disable (faster but uses more VRAM)
# (omit flag)
```

---

## Multi-GPU Training

### 4x H100 (Full Node)

```bash
# Submit to multi-GPU partition
sbatch templates/sbatch_multi_gpu.sh
```

### Accelerate Config

```yaml
# configs/accelerate/multi_gpu_ddp.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: bf16
```

### DeepSpeed ZeRO-3

```bash
# For very large models (70B+)
accelerate launch --config_file configs/accelerate/deepspeed_zero3.yaml \
    scripts/train_sft.py \
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --output_dir ./outputs/llama70b-sft
```

---

## Job Pipelines

### Train → Eval → Convert Pipeline

```bash
# Submit pipeline job
sbatch templates/sbatch_pipeline.sh

# Pipeline structure:
# 1. Training job runs
# 2. On success: Evaluation job starts
# 3. On success: GGUF conversion runs
```

### Manual Job Chaining

```bash
# Submit training job
TRAIN_JOB=$(sbatch train.sh | awk '{print $4}')
echo "Training job: $TRAIN_JOB"

# Submit eval job with dependency
EVAL_JOB=$(sbatch --dependency=afterok:$TRAIN_JOB eval.sh | awk '{print $4}')
echo "Eval job: $EVAL_JOB (depends on $TRAIN_JOB)"

# Submit conversion with dependency
CONVERT_JOB=$(sbatch --dependency=afterok:$EVAL_JOB convert.sh | awk '{print $4}')
echo "Convert job: $CONVERT_JOB (depends on $EVAL_JOB)"
```

### Job Dependencies

```bash
# Start after another job succeeds
--dependency=afterok:12345

# Start after another job fails
--dependency=afternotok:12345

# Start after another job completes (success or fail)
--dependency=afterany:12345

# Start after multiple jobs succeed
--dependency=afterok:12345:12346:12347
```

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Solutions (try in order):
1. Reduce batch size: --per_device_train_batch_size 1
2. Increase gradient accumulation: --gradient_accumulation_steps 16
3. Enable gradient checkpointing: --gradient_checkpointing
4. Reduce sequence length: --max_seq_length 1024
5. Use smaller LoRA: --lora_r 16 --lora_alpha 32
6. Use Unsloth for 80% less VRAM
7. Use larger GPU partition
```

### Model Not Found

```bash
# Pre-download on login node
export HF_HOME=$SCRATCH/.cache/huggingface
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('model-id')"

# Verify download
ls $SCRATCH/.cache/huggingface/hub/models--*/
```

### Hub Push Failed

```bash
# Check token
echo $HF_TOKEN

# Verify token has write permissions
# Re-create token at https://huggingface.co/settings/tokens

# Enable checkpoint push for retry
--hub_strategy every_save
```

### Job Timeout

```bash
# Estimate time correctly:
# base_hours = 0.1 * model_params_B * (dataset_size_K / 1000) * epochs
# Add 30% buffer

# Use longer partition
#SBATCH --partition=gpubase_bygpu_b5  # 7 day limit
```

### Training Loss Not Decreasing

```bash
# Check learning rate (DPO needs lower LR)
--learning_rate 5e-7  # DPO
--learning_rate 2e-4  # SFT

# Check dataset format
python scripts/validate_dataset.py --dataset trl-lib/Capybara

# Enable gradient clipping
--max_grad_norm 1.0
```

---

## Complete Example: End-to-End Workflow

```bash
# 1. Setup (once)
source /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/utils/setup_env.sh

# 2. Pre-download
export HF_HOME=$SCRATCH/.cache/huggingface
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B')"
python -c "from datasets import load_dataset; load_dataset('trl-lib/Capybara')"

# 3. Create job script
cat > train_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=qwen7b-sft
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source ~/.env
export HF_HOME=$SCRATCH/.cache/huggingface

source /scratch/ermia/venvs/hf_env/bin/activate

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --dataset_name trl-lib/Capybara \
    --output_dir $SCRATCH/outputs/qwen7b-sft-$SLURM_JOB_ID \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_seq_length 2048 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --test_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --push_to_hub \
    --hub_model_id ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara \
    --report_to trackio \
    --project sft-training \
    --run_name qwen7b-sft-$SLURM_JOB_ID
EOF

# 4. Submit job
sbatch train_job.sh

# 5. Monitor
squeue -u $USER
tail -f slurm-*.out

# 6. After training completes - generate model card
python scripts/generate_model_card.py \
    --model_name "Qwen2.5-7B-SFT-Capybara" \
    --base_model "Qwen/Qwen2.5-7B" \
    --dataset "trl-lib/Capybara" \
    --training_log $SCRATCH/outputs/qwen7b-sft-*/training.log \
    --output_dir ./model_card

# 7. Push model card to Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='./model_card/README.md',
    path_in_repo='README.md',
    repo_id='ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara'
)
"

# 8. Convert to GGUF
python scripts/convert_gguf.py \
    --model_id ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara \
    --output_dir ./gguf_output \
    --quantization Q4_K_M Q5_K_M Q8_0 \
    --push_to_hub \
    --hub_repo_id ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara-GGUF
```

---

## Quick Reference

### Training Methods

| Method | Use Case | Learning Rate | Dataset Format |
|--------|----------|---------------|----------------|
| SFT | Instruction tuning | 2e-4 | messages/prompt-completion |
| DPO | Preference alignment | 5e-7 | prompt/chosen/rejected |
| GRPO | Online RL | 1e-6 | prompt only |

### Hardware Selection

| Model Size | GPU | Partition | Time (1 epoch) |
|------------|-----|-----------|----------------|
| 0.5B-3B | 2g.20gb MIG | gpubase_bygpu_b1 | 15-30 min |
| 3B-7B | 3g.40gb MIG | gpubase_bygpu_b3 | 1-3 hours |
| 7B-13B | h100 full | gpubase_bygpu_b3 | 2-6 hours |
| 13B-70B | 4x h100 | gpubase_bynode_b3 | 6-24 hours |

### Key Flags

```bash
--use_4bit              # 4-bit quantization (saves VRAM)
--bf16                  # BFloat16 precision
--gradient_checkpointing # Memory optimization
--push_to_hub           # Push to HuggingFace
--report_to trackio     # Enable Trackio monitoring
```
