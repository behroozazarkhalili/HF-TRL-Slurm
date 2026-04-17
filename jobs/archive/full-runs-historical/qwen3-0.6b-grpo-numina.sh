#!/bin/bash
# =============================================================================
# Qwen3-0.6B GRPO Training on AI-MO/NuminaMath-CoT
# Full Pipeline: Train → Model Card → GGUF Conversion → HF Upload
# Dataset: AI-MO/NuminaMath-CoT (10K samples, streaming mode)
# =============================================================================

#SBATCH --job-name=qwen3-0.6b-grpo-numina
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=7-00:00:00
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
# Configuration
# =============================================================================
MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET_NAME="AI-MO/NuminaMath-CoT"

# Sample size configuration
MAX_SAMPLES=50000
SAMPLE_SIZE_LABEL="50K"  # Human-readable format for model naming

# Model naming with sample size
HUB_MODEL_ID="ermiaazarkhalili/Qwen3-0.6B-GRPO-NuminaMath-${SAMPLE_SIZE_LABEL}"
GGUF_REPO_ID="ermiaazarkhalili/Qwen3-0.6B-GRPO-NuminaMath-${SAMPLE_SIZE_LABEL}-GGUF"

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
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
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
export OUTPUT_DIR=$SCRATCH/outputs/qwen3-0.6b-grpo-numina-$SLURM_JOB_ID

# Load HF token
if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME (${SAMPLE_SIZE_LABEL} samples, streaming)"
echo "  Hub Model ID: $HUB_MODEL_ID"
echo "  Reward Type: $REWARD_TYPE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Num Generations: $NUM_GENERATIONS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Seed: $SEED"
echo "  LoRA Rank: $LORA_R"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# =============================================================================
# Phase 1: GRPO Training
# =============================================================================
echo "=========================================="
echo "Phase 1: GRPO Training"
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
    --use_liger_kernel \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --push_to_hub \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy end \
    --report_to trackio \
    --project "grpo-numina" \
    --run_name "qwen3-0.6b-grpo-numina-$SLURM_JOB_ID"

TRAIN_EXIT_CODE=$?

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully!"

# =============================================================================
# Phase 2: Generate Model Card
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 2: Generating Model Card"
echo "=========================================="

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/generate_model_card.py \
    --model_name "Qwen3-0.6B-GRPO-NuminaMath-${SAMPLE_SIZE_LABEL}" \
    --base_model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --training_method GRPO \
    --author ermiaazarkhalili \
    --license apache-2.0 \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --hardware "NVIDIA H100 20GB MIG" \
    --output_dir $OUTPUT_DIR/model_card

# Push model card to Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='$OUTPUT_DIR/model_card/README.md',
    path_in_repo='README.md',
    repo_id='$HUB_MODEL_ID'
)
print('Model card uploaded to $HUB_MODEL_ID')
"

# =============================================================================
# Phase 3: GGUF Conversion
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 3: GGUF Conversion (Q4_K_M)"
echo "=========================================="

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/convert_gguf.py \
    --model $HUB_MODEL_ID \
    --base_model $MODEL_NAME \
    --output_repo $GGUF_REPO_ID \
    --quantizations "Q4_K_M,Q5_K_M,Q8_0" \
    --output_dir $OUTPUT_DIR/gguf

GGUF_EXIT_CODE=$?

if [[ $GGUF_EXIT_CODE -ne 0 ]]; then
    echo "WARNING: GGUF conversion had issues (exit code $GGUF_EXIT_CODE)"
else
    echo "GGUF conversion completed successfully!"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Training Output: $OUTPUT_DIR"
echo "  Model on Hub: https://huggingface.co/$HUB_MODEL_ID"
echo "  GGUF on Hub: https://huggingface.co/$GGUF_REPO_ID"
echo ""
echo "To use with Ollama:"
echo "  ollama pull hf.co/$GGUF_REPO_ID:Q4_K_M"
echo ""
