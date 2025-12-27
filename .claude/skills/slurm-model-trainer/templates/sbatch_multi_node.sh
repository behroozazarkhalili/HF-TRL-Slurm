#!/bin/bash
# =============================================================================
# Multi-Node Distributed Training Job Template
# Fir Cluster (Digital Research Alliance of Canada)
# =============================================================================
# Usage:
#   sbatch sbatch_multi_node.sh
#   sbatch --export=ALL,NODES=2,MODEL=meta-llama/Llama-2-13b-hf sbatch_multi_node.sh
# =============================================================================

#SBATCH --job-name=hf-trl-multinode
#SBATCH --account=def-maxwl_gpu    # Replace with your account
#SBATCH --time=48:00:00
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:4             # 4x H100 per node
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# =============================================================================
# Configuration (override via --export)
# =============================================================================
SCRIPT=${SCRIPT:-train_sft.py}
MODEL=${MODEL:-meta-llama/Llama-2-13b-hf}
DATASET=${DATASET:-trl-lib/Capybara}
HUB_MODEL_ID=${HUB_MODEL_ID:-}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-16}
LEARNING_RATE=${LEARNING_RATE:-2e-5}

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Nodes: $SLURM_NNODES"
echo "Node List: $SLURM_NODELIST"
echo "This Node: $SLURMD_NODENAME (Rank: $SLURM_PROCID)"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

PROJECT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
SKILL_DIR=$(dirname $(dirname $(realpath $0)))

# Load modules
module load python/3.11.5 cuda/12.2 arrow/17.0.0

# Activate virtual environment
source /scratch/ermia/venvs/hf_env/bin/activate

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

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand interface

# Get master node info
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo ""
echo "Distributed Configuration:"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  World Size: $((SLURM_NNODES * 4))"
echo "  This Node Rank: $SLURM_PROCID"
echo ""

# Calculate total processes
NUM_PROCESSES=$((SLURM_NNODES * 4))

# Print GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# =============================================================================
# Run Training with Accelerate (Multi-Node)
# =============================================================================
cd $PROJECT_DIR

# Use bash -c to properly expand Slurm variables at runtime
bash -c "accelerate launch \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $NUM_PROCESSES \
    --mixed_precision bf16 \
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
    ${HUB_MODEL_ID:+--push_to_hub --hub_model_id $HUB_MODEL_ID}"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
