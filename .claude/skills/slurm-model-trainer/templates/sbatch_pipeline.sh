#!/bin/bash
# =============================================================================
# Full Pipeline Job Template (Train → Eval → Convert)
# Fir Cluster (Digital Research Alliance of Canada)
# =============================================================================
# This script submits a chain of dependent jobs:
#   1. Training job
#   2. Evaluation job (runs after training completes)
#   3. GGUF conversion job (runs after evaluation completes)
#
# Usage:
#   sbatch sbatch_pipeline.sh
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-0.5B,DATASET=trl-lib/Capybara sbatch_pipeline.sh
# =============================================================================

#SBATCH --job-name=hf-trl-pipeline
#SBATCH --account=def-maxwl_gpu    # Replace with your account
#SBATCH --time=00:10:00               # Just for job submission
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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
EVAL_TASKS=${EVAL_TASKS:-comprehensive}
CONVERT_GGUF=${CONVERT_GGUF:-true}
ACCOUNT=${ACCOUNT:-def-maxwl_gpu}

# =============================================================================
# Setup
# =============================================================================
echo "=========================================="
echo "HF-TRL Pipeline Submission"
echo "Pipeline ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

SKILL_DIR=$(dirname $(realpath $0))

# Auto-generate HUB_MODEL_ID if not provided
# Naming Convention: {BaseModel}-{TrainingMethod}-{Dataset}[-Quantization]
# Example: Qwen2.5-7B-SFT-Capybara
if [[ -z "$HUB_MODEL_ID" ]]; then
    # Extract username from HF token or use default
    HF_USERNAME=${HF_USERNAME:-$(whoami)}
    # Extract model name (e.g., Qwen2.5-7B from Qwen/Qwen2.5-7B)
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///')
    # Extract dataset name (e.g., Capybara from trl-lib/Capybara)
    DATASET_SHORT=$(echo $DATASET | sed 's/.*\///')
    # Determine training method from script name
    if [[ "$SCRIPT" == *"dpo"* ]]; then
        TRAINING_METHOD="DPO"
    elif [[ "$SCRIPT" == *"grpo"* ]]; then
        TRAINING_METHOD="GRPO"
    else
        TRAINING_METHOD="SFT"
    fi
    # Format: {BaseModel}-{TrainingMethod}-{Dataset} (use "-" not "_")
    HUB_MODEL_ID="$HF_USERNAME/${MODEL_SHORT}-${TRAINING_METHOD}-${DATASET_SHORT}"
    echo "Auto-generated HUB_MODEL_ID: $HUB_MODEL_ID"
    echo "Naming Convention: {BaseModel}-{TrainingMethod}-{Dataset}"
fi

echo ""
echo "Pipeline Configuration:"
echo "  Training Script: $SCRIPT"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Hub Model ID: $HUB_MODEL_ID"
echo "  Eval Tasks: $EVAL_TASKS"
echo "  Convert GGUF: $CONVERT_GGUF"
echo ""

# =============================================================================
# Step 1: Submit Training Job
# =============================================================================
echo "Submitting training job..."

TRAIN_JOB=$(sbatch \
    --account=$ACCOUNT \
    --parsable \
    --export=ALL,SCRIPT=$SCRIPT,MODEL=$MODEL,DATASET=$DATASET,HUB_MODEL_ID=$HUB_MODEL_ID,NUM_EPOCHS=$NUM_EPOCHS,BATCH_SIZE=$BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE \
    $SKILL_DIR/sbatch_single_gpu.sh)

if [[ -z "$TRAIN_JOB" ]]; then
    echo "ERROR: Failed to submit training job"
    exit 1
fi

echo "Training job submitted: $TRAIN_JOB"

# =============================================================================
# Step 2: Generate and Push Model Card (depends on training)
# =============================================================================
echo ""
echo "Submitting model card generation job (depends on $TRAIN_JOB)..."

# Determine training method for model card
if [[ "$SCRIPT" == *"dpo"* ]]; then
    CARD_TRAINING_METHOD="DPO"
elif [[ "$SCRIPT" == *"grpo"* ]]; then
    CARD_TRAINING_METHOD="GRPO"
else
    CARD_TRAINING_METHOD="SFT"
fi

MODEL_CARD_JOB=$(sbatch \
    --account=$ACCOUNT \
    --parsable \
    --dependency=afterok:$TRAIN_JOB \
    --job-name=hf-trl-model-card \
    --time=00:15:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --partition=cpubase_bycore_b1 \
    --wrap="source /scratch/ermia/venvs/hf_env/bin/activate && \
            python $(dirname $SKILL_DIR)/scripts/generate_model_card.py \
            --model_name ${HUB_MODEL_ID##*/} \
            --base_model $MODEL \
            --dataset $DATASET \
            --training_method $CARD_TRAINING_METHOD \
            --author ${HUB_MODEL_ID%%/*} \
            --license cc-by-nc-4.0 \
            --output_dir /tmp/model_card_\$SLURM_JOB_ID && \
            python -c \"
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='/tmp/model_card_\$SLURM_JOB_ID/README.md',
    path_in_repo='README.md',
    repo_id='$HUB_MODEL_ID'
)
print('Model card uploaded to $HUB_MODEL_ID')
\"")

if [[ -z "$MODEL_CARD_JOB" ]]; then
    echo "WARNING: Failed to submit model card job"
else
    echo "Model card job submitted: $MODEL_CARD_JOB"
fi

# =============================================================================
# Step 3: Submit Evaluation Job (depends on training)
# =============================================================================
echo ""
echo "Submitting evaluation job (depends on $TRAIN_JOB)..."

EVAL_JOB=$(sbatch \
    --account=$ACCOUNT \
    --parsable \
    --dependency=afterok:$TRAIN_JOB \
    --export=ALL,MODEL=$HUB_MODEL_ID,TASKS=$EVAL_TASKS,PUSH_TO_HUB=true \
    $SKILL_DIR/sbatch_eval.sh)

if [[ -z "$EVAL_JOB" ]]; then
    echo "WARNING: Failed to submit evaluation job"
else
    echo "Evaluation job submitted: $EVAL_JOB"
fi

# =============================================================================
# Step 4: Submit GGUF Conversion Job (depends on evaluation)
# Naming Convention: {BaseModel}-{TrainingMethod}-{Dataset}-GGUF
# =============================================================================
if [[ "$CONVERT_GGUF" == "true" ]]; then
    echo ""
    echo "Submitting GGUF conversion job (depends on $EVAL_JOB)..."

    # GGUF repo follows naming convention: add -GGUF suffix
    GGUF_REPO="${HUB_MODEL_ID}-GGUF"

    CONVERT_JOB=$(sbatch \
        --account=$ACCOUNT \
        --parsable \
        --dependency=afterok:$EVAL_JOB \
        --export=ALL,MODEL=$HUB_MODEL_ID,BASE_MODEL=$MODEL,OUTPUT_REPO=$GGUF_REPO,QUANTIZATIONS=Q4_K_M,Q5_K_M,Q8_0 \
        $SKILL_DIR/sbatch_convert.sh)

    if [[ -z "$CONVERT_JOB" ]]; then
        echo "WARNING: Failed to submit conversion job"
    else
        echo "Conversion job submitted: $CONVERT_JOB"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Pipeline Submitted Successfully!"
echo "=========================================="
echo ""
echo "Job Chain:"
echo "  1. Training:    $TRAIN_JOB"
echo "  2. Model Card:  ${MODEL_CARD_JOB:-<not submitted>}"
echo "  3. Evaluation:  ${EVAL_JOB:-<not submitted>}"
echo "  4. GGUF Conv:   ${CONVERT_JOB:-<not submitted>}"
echo ""
echo "Naming Convention: {BaseModel}-{TrainingMethod}-{Dataset}[-GGUF]"
echo ""
echo "Monitor progress:"
echo "  squeue -u $USER"
echo "  scontrol show job $TRAIN_JOB"
echo ""
echo "View logs:"
echo "  tail -f logs/hf-trl-train-$TRAIN_JOB.out"
echo ""
echo "Expected outputs:"
echo "  Model:      https://huggingface.co/$HUB_MODEL_ID"
echo "  README.md:  Comprehensive model card (auto-generated)"
if [[ "$CONVERT_GGUF" == "true" ]]; then
    echo "  GGUF:       https://huggingface.co/$GGUF_REPO"
    echo "  GGUF Files: Q4_K_M, Q5_K_M, Q8_0"
fi
