#!/bin/bash
# =============================================================================
# SMOKE TEST: Qwen3.5-2B GRPO on Claude Reasoning Distillation
# 100 samples, <1h, validates: GRPO training with claude-reasoning-distillation grpo config
# =============================================================================

#SBATCH --job-name=qwen3.5-2b-grpo-claude
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME='Qwen/Qwen3.5-2B'
DATASET_NAME='ermiaazarkhalili/claude-reasoning-distillation'
DATASET_CONFIG='grpo'

MAX_SAMPLES=100
SAMPLE_SIZE_LABEL="smoke-100"

HUB_MODEL_ID='ermiaazarkhalili/Qwen3.5-2B-GRPO-Claude-Reasoning-smoke-100'

BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=1e-06
NUM_EPOCHS=1
LORA_R=16
LORA_ALPHA=32
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=512
NUM_GENERATIONS=4
REWARD_TYPE='combined'

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "SMOKE: Qwen3.5-2B GRPO Claude Reasoning"
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

mkdir -p logs

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_trl/bin/activate


export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/qwen3.5-2b-grpo-claude-$SLURM_JOB_ID"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "/project/6014832/ermia/HF-TRL/.env"
fi

mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME ($DATASET_CONFIG config)"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Batch Size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Num Generations: $NUM_GENERATIONS"
echo "  Reward Type: $REWARD_TYPE"
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "  Output: $OUTPUT_DIR"
echo ""

# =============================================================================
# Phase 0: GPU Profiler (background)
# =============================================================================
VRAM_LOG="$OUTPUT_DIR/vram_usage.log"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "$VRAM_LOG" 2>/dev/null &
NVIDIA_PID=$!

# =============================================================================
# Phase 1: GRPO Training
# =============================================================================
echo "=========================================="
echo "Phase 1: GRPO Training"
echo "=========================================="

TRAIN_START=$(date +%s)

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --bf16 \
    --use_4bit \
    --gradient_checkpointing \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --save_strategy no \
    --logging_steps 5 \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy end \
    --report_to trackio \
    --project "qwen3.5-2b-grpo-claude-reasoning" \
    --run_name "qwen3.5-2b-grpo-claude-smoke-$SLURM_JOB_ID" \
    --streaming \
    --max_samples $MAX_SAMPLES \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --num_generations $NUM_GENERATIONS \
    --reward_type $REWARD_TYPE

TRAIN_EXIT=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Training failed (exit $TRAIN_EXIT)"
    kill $NVIDIA_PID 2>/dev/null
    exit $TRAIN_EXIT
fi

echo "Training completed in ${TRAIN_DURATION}s"

# =============================================================================
# Phase 2: Metrics Report
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 2: Metrics Report"
echo "=========================================="

kill $NVIDIA_PID 2>/dev/null

echo "Training duration: ${TRAIN_DURATION}s"
echo "Samples: $MAX_SAMPLES"

if [[ -f "$VRAM_LOG" ]]; then
    PEAK_VRAM=$(tail -n +2 "$VRAM_LOG" | awk -F', ' '{gsub(/ MiB/,"",$2); print $2}' | sort -n | tail -1)
    echo "Peak GPU vRAM: ${PEAK_VRAM} MiB"
fi

echo ""
echo "=========================================="
echo "Smoke Test Complete! $(date)"
echo "=========================================="
