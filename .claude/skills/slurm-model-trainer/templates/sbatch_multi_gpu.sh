#!/bin/bash
# =============================================================================
# Multi-GPU Training Job Template (4x H100 on Single Node)
# Fir Cluster (Digital Research Alliance of Canada)
# =============================================================================
# Usage:
#   sbatch sbatch_multi_gpu.sh
#   sbatch --export=ALL,SCRIPT=train_sft.py,MODEL=meta-llama/Llama-2-7b-hf sbatch_multi_gpu.sh
# =============================================================================

#SBATCH --job-name=hf-trl-multigpu
#SBATCH --account=def-maxwl_gpu    # Replace with your account
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:4             # 4x H100 GPUs
#SBATCH --partition=gpubase_bynode_b3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# =============================================================================
# Configuration (override via --export)
# =============================================================================
SCRIPT=${SCRIPT:-train_sft.py}
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
DATASET=${DATASET:-trl-lib/Capybara}
HUB_MODEL_ID=${HUB_MODEL_ID:-}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-multi_gpu_ddp.yaml}

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 4x H100"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

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

# Load HF token
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' $PROJECT_DIR/.env | xargs)
fi

# NCCL settings for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Print configuration
echo ""
echo "Configuration:"
echo "  Script: $SCRIPT"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Accelerate Config: $ACCELERATE_CONFIG"
echo "  Effective Batch Size: $((BATCH_SIZE * 4 * GRADIENT_ACCUMULATION))"
echo ""

# Print GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# =============================================================================
# Run Training with Accelerate
# =============================================================================
cd $PROJECT_DIR

accelerate launch \
    --config_file $SKILL_DIR/configs/accelerate/$ACCELERATE_CONFIG \
    --num_processes 4 \
    $SKILL_DIR/scripts/$SCRIPT \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir ./outputs/$SLURM_JOB_ID \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --logging_steps 10 \
    --bf16 \
    --gradient_checkpointing \
    --report_to trackio \
    ${HUB_MODEL_ID:+--push_to_hub --hub_model_id $HUB_MODEL_ID}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
