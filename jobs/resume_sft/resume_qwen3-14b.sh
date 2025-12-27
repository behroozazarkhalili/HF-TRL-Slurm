#!/bin/bash
# =============================================================================
# Resume Qwen3-14B SFT Training from Checkpoint
# Reduced LoRA rank (32) and grad_accum (8) to prevent OOM
# =============================================================================

#SBATCH --job-name=qwen3-14b-sft-resume
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=/home/ermia/projects/def-maxwl/ermia/HF-TRL/jobs/logs/qwen3-14b-sft-resume-%j.out
#SBATCH --error=/home/ermia/projects/def-maxwl/ermia/HF-TRL/jobs/logs/qwen3-14b-sft-resume-%j.err

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME="Qwen/Qwen3-14B-Base"
DATASET_NAME="HuggingFaceH4/ultrachat_200k"
CHECKPOINT_DIR="/scratch/ermia/outputs/qwen3-14b-sft-16615027/checkpoint-6200"
OUTPUT_DIR="/scratch/ermia/outputs/qwen3-14b-sft-resume-$SLURM_JOB_ID"
HUB_MODEL_ID="ermiaazarkhalili/Qwen3-14B-SFT-UltraChat"

# Reduced settings to prevent OOM
BATCH_SIZE=1
GRAD_ACCUM=4      # Reduced from 16
LORA_R=32         # Reduced from 64
LORA_ALPHA=64     # 2x LORA_R
MAX_SEQ_LENGTH=2048
LEARNING_RATE=2e-4
NUM_EPOCHS=1

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p /home/ermia/projects/def-maxwl/ermia/HF-TRL/jobs/logs

PROJECT_DIR=/home/ermia/projects/def-maxwl/ermia/HF-TRL
SKILL_DIR=$PROJECT_DIR/.claude/skills/slurm-model-trainer

module load python/3.11.5 cuda/12.2 arrow/17.0.0
source /scratch/ermia/venvs/hf_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ermia/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=/scratch/ermia/.cache/torch

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' $PROJECT_DIR/.env | xargs)
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Resume from: $CHECKPOINT_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Hub Model ID: $HUB_MODEL_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM (reduced from 16)"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LoRA Rank: $LORA_R (reduced from 64)"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# =============================================================================
# Run Training (Resume)
# =============================================================================
python $SKILL_DIR/scripts/train_sft.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_split "train_sft" \
    --output_dir $OUTPUT_DIR \
    --resume_from_checkpoint $CHECKPOINT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_SEQ_LENGTH \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --test_size 0.05 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --push_to_hub \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy end \
    --report_to trackio \
    --trackio_dir $OUTPUT_DIR/trackio \
    --project "qwen-sft-ultrachat" \
    --run_name "qwen3-14b-sft-resume-$SLURM_JOB_ID"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
