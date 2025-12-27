#!/bin/bash
# =============================================================================
# Single GPU Training Job Template
# Fir Cluster (Digital Research Alliance of Canada)
# =============================================================================
# Usage:
#   sbatch sbatch_single_gpu.sh
#   sbatch --export=ALL,SCRIPT=train_sft.py,MODEL=Qwen/Qwen2.5-0.5B sbatch_single_gpu.sh
# =============================================================================

#SBATCH --job-name=hf-trl-train
#SBATCH --account=def-maxwl_gpu    # Replace with your account
#SBATCH --time=24:00:00               # Adjust based on training time
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1             # Options: h100, nvidia_h100_80gb_hbm3_3g.40gb, etc.
#SBATCH --partition=gpubase_bygpu_b3  # 1-day limit partition
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com  # Replace with your email

# =============================================================================
# Configuration (override via --export)
# =============================================================================
SCRIPT=${SCRIPT:-train_sft_unsloth.py}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
DATASET=${DATASET:-trl-lib/Capybara}
HUB_MODEL_ID=${HUB_MODEL_ID:-}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_STEPS=${MAX_STEPS:--1}
EVAL_STEPS=${EVAL_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-100}

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory if not exists
mkdir -p logs

# Get project directory
PROJECT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
SKILL_DIR=$(dirname $(dirname $(realpath $0)))

# Load modules
module load python/3.11.5 cuda/12.2 arrow/17.0.0

# Activate virtual environment
source /scratch/ermia/venvs/hf_env/bin/activate

# Disable tokenizer internal parallelism to avoid contention with dataset_num_proc
export TOKENIZERS_PARALLELISM=false

# Set cache directories
export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/.cache/torch

# Load HF token from .env
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' $PROJECT_DIR/.env | xargs)
fi

# Verify HF_TOKEN is set
if [[ -z "$HF_TOKEN" ]]; then
    echo "WARNING: HF_TOKEN not set. Hub push will fail."
    echo "Please add HF_TOKEN to $PROJECT_DIR/.env"
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Script: $SCRIPT"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Hub Model ID: ${HUB_MODEL_ID:-<not set>}"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# =============================================================================
# Run Training
# =============================================================================
cd $PROJECT_DIR

# Build command
CMD="python $SKILL_DIR/scripts/$SCRIPT \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir ./outputs/$SLURM_JOB_ID \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --logging_steps 10 \
    --bf16 \
    --gradient_checkpointing \
    --report_to trackio"

# Add optional arguments
if [[ -n "$HUB_MODEL_ID" ]]; then
    CMD="$CMD --push_to_hub --hub_model_id $HUB_MODEL_ID"
fi

if [[ "$MAX_STEPS" != "-1" ]]; then
    CMD="$CMD --max_steps $MAX_STEPS"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Execute training
$CMD

# =============================================================================
# Post-Training
# =============================================================================
echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="

# Print final output location
echo "Model saved to: ./outputs/$SLURM_JOB_ID"
if [[ -n "$HUB_MODEL_ID" ]]; then
    echo "Model pushed to: https://huggingface.co/$HUB_MODEL_ID"
fi
