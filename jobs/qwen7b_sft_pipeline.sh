#!/bin/bash
#SBATCH --job-name=qwen7b-sft-pipeline
#SBATCH --account=def-maxwl_gpu
#SBATCH --output=logs/qwen7b-%j.out
#SBATCH --error=logs/qwen7b-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ermiaazarkhalili@gmail.com

# ============================================================================
# Qwen2.5-7B-Instruct Training Pipeline
# - SFT with 4-bit quantization + LoRA (r=64)
# - Evaluation on MMLU and GSM8K
# - GGUF conversion (Q4_K_M)
# - Upload to HuggingFace Hub
# ============================================================================

set -e  # Exit on error

# Configuration
MODEL_NAME="Qwen/Qwen2.5-7B"
DATASET_NAME="trl-lib/Capybara"
HUB_MODEL_ID="ermiaazarkhalili/qwen2.5-7b-sft"
OUTPUT_DIR="/scratch/ermia/outputs/qwen7b-sft-${SLURM_JOB_ID}"
LORA_R=64
LORA_ALPHA=128

# Environment paths
PROJECT_DIR="/project/6014832/ermia/HF-TRL"
SKILL_DIR="${PROJECT_DIR}/.claude/skills/slurm-model-trainer"
cd "$PROJECT_DIR"

# Print job information
echo "=============================================================="
echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=============================================================="

# Load required modules (MUST be before venv activation)
module load python/3.11 StdEnv/2023 cudacore/.12.2.2 scipy-stack/2024a gcc arrow/17.0.0

# Activate virtual environment
source /scratch/ermia/venvs/hf_env/bin/activate

# Memory optimization for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load HuggingFace token from .env file
if [ -f "${PROJECT_DIR}/.env" ]; then
    source "${PROJECT_DIR}/.env"
    export HF_TOKEN="${HF_TOKEN}"
    export HF_HOME="/scratch/ermia/hf_cache"
    mkdir -p "$HF_HOME"
    echo "HuggingFace token loaded from .env"
fi

# Create directories
mkdir -p logs $OUTPUT_DIR

# Verify GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU info available"
echo ""

echo "============================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Hub Model: $HUB_MODEL_ID"
echo "LoRA r=$LORA_R, alpha=$LORA_ALPHA"
echo "4-bit quantization: Enabled"
echo "============================================"

# ============================================================================
# PHASE 1: Training
# ============================================================================
echo ""
echo "========== PHASE 1: SFT Training =========="
echo ""

python ${SKILL_DIR}/scripts/train_sft.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --use_4bit \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --report_to "none" \
    --push_to_hub \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy "end"

echo "Training completed!"

# ============================================================================
# PHASE 2: Evaluation
# ============================================================================
echo ""
echo "========== PHASE 2: Evaluation (MMLU, GSM8K) =========="
echo ""

# Run lm-eval-harness
lm_eval --model hf \
    --model_args pretrained=$HUB_MODEL_ID,trust_remote_code=True \
    --tasks mmlu,gsm8k \
    --batch_size auto \
    --output_path ${OUTPUT_DIR}/eval_results

echo "Evaluation completed!"

# ============================================================================
# PHASE 3: GGUF Conversion
# ============================================================================
echo ""
echo "========== PHASE 3: GGUF Conversion (Q4_K_M) =========="
echo ""

python ${SKILL_DIR}/scripts/convert_gguf.py \
    --model_path $HUB_MODEL_ID \
    --output_dir ${OUTPUT_DIR}/gguf \
    --quantization Q4_K_M \
    --push_to_hub \
    --hub_model_id ${HUB_MODEL_ID}-GGUF

echo "GGUF conversion completed!"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
echo "Trained model: https://huggingface.co/$HUB_MODEL_ID"
echo "GGUF model: https://huggingface.co/${HUB_MODEL_ID}-GGUF"
echo "Local outputs: $OUTPUT_DIR"
echo "Evaluation results: ${OUTPUT_DIR}/eval_results"
echo "============================================"
echo "Job finished on $(date)"
echo "============================================"
