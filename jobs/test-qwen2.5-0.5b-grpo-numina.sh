#!/bin/bash
# =============================================================================
# TEST JOB: Qwen2.5-0.5B-Instruct GRPO on NuminaMath-CoT
# Quick validation: 100 samples, ~30 min expected runtime
# Validates: ground truth extraction, 4-bit quant OOM fix, training loop
# =============================================================================

#SBATCH --job-name=test-qwen2.5-0.5b-grpo-numina
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Configuration (reduced for testing)
# =============================================================================
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME="AI-MO/NuminaMath-CoT"

MAX_SAMPLES=100
SAMPLE_SIZE_LABEL="test-100"

# No hub push for test runs
HUB_MODEL_ID=""

# GRPO Training parameters (tuned for 40GB MIG slice)
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-6
NUM_EPOCHS=1
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=512
NUM_GENERATIONS=2
REWARD_TYPE="combined"
LORA_R=16
LORA_ALPHA=32
SEED=42
USE_4BIT=true

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "TEST JOB: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

# Load modules
module load gcc arrow python/3.11.5

# Activate virtual environment
source /scratch/ermia/venvs/hf_env/bin/activate

# Set environment variables
export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR=$SCRATCH/outputs/test-qwen2.5-0.5b-grpo-numina-$SLURM_JOB_ID

# Load HF token
if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME (${SAMPLE_SIZE_LABEL} samples, streaming)"
echo "  Reward Type: $REWARD_TYPE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Num Generations: $NUM_GENERATIONS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Samples: $MAX_SAMPLES"
echo "  4-bit Quantization: $USE_4BIT"
echo "  Seed: $SEED"
echo "  LoRA Rank: $LORA_R"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# =============================================================================
# GRPO Training (test run - no hub push, no model card, no GGUF)
# =============================================================================
echo "=========================================="
echo "GRPO Training (test run)"
echo "=========================================="

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_split "train" \
    --streaming \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --num_generations $NUM_GENERATIONS \
    --reward_type $REWARD_TYPE \
    --bf16 \
    --gradient_checkpointing \
    --use_4bit \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --report_to none \
    --run_name "test-qwen2.5-0.5b-grpo-numina-$SLURM_JOB_ID"

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "TEST RESULTS"
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS: Training completed without errors"
    echo "Check logs for:"
    echo "  - 'Stored N ground truth answers' (should be > 0)"
    echo "  - Training loss values"
    echo "  - No OOM errors"
else
    echo "FAILED: Training exited with code $TRAIN_EXIT_CODE"
    echo "Check error log: logs/test-qwen2.5-0.5b-grpo-numina-$SLURM_JOB_ID.err"
fi

echo ""
echo "Output dir: $OUTPUT_DIR"
