#!/bin/bash
# =============================================================================
# Smoke test: 3-stage chain with 5 samples each
# Validates the quoted-heredoc + sed chain generator fix
# =============================================================================

set -euo pipefail

CHAIN_DIR="/scratch/$USER/outputs/granite-chain-5sample"
rm -rf "$CHAIN_DIR"
mkdir -p "$CHAIN_DIR" logs

MODEL_NAME='ibm-granite/granite-4.0-micro'
DATASET_NAME='AI-MO/NuminaMath-CoT'
NUM_STAGES=3
BASE_SEED=42

BATCH_SIZE=2
GRAD_ACCUM=1
LEARNING_RATE=1e-06
LORA_R=16
LORA_ALPHA=32
MAX_COMPLETION_LENGTH=512
MAX_PROMPT_LENGTH=256
NUM_GENERATIONS=2
REWARD_TYPE='combined'

echo "=== CHAIN SMOKE TEST: 3 stages x 5 samples ==="
echo ""

PREV_JOB_ID=""

for ((STAGE=1; STAGE<=NUM_STAGES; STAGE++)); do
    SEED=$((BASE_SEED + STAGE))
    STAGE_SAMPLES=5
    IS_FINAL=$([[ $STAGE -eq $NUM_STAGES ]] && echo "true" || echo "false")

    STAGE_SCRIPT="$CHAIN_DIR/stage-${STAGE}.sh"

    # Quoted heredoc — no escaping needed inside
    cat > "$STAGE_SCRIPT" << 'STAGE_EOF'
#!/bin/bash
#SBATCH --job-name=chain-5s-stage-__STAGE__
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=== CHAIN STAGE __STAGE__ / __NUM_STAGES__ (Job $SLURM_JOB_ID) ==="
echo "Node: $SLURMD_NODENAME | $(date)"

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/granite-chain-5sample/stage__STAGE__-$SLURM_JOB_ID"

# Load HF token — no escaping issues with quoted heredoc
if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "/project/6014832/ermia/HF-TRL/.env"
fi

CHAIN_DIR="__CHAIN_DIR__"
mkdir -p "$OUTPUT_DIR" logs

# Resolve adapter from previous stage
ADAPTER_ARG=""
STAGE_NUM=__STAGE__
if [[ $STAGE_NUM -gt 1 ]]; then
    PREV_STAGE=$((STAGE_NUM - 1))
    ADAPTER_FILE="$CHAIN_DIR/stage-${PREV_STAGE}.adapter"
    if [[ ! -f "$ADAPTER_FILE" ]]; then
        echo "ERROR: Adapter not found: $ADAPTER_FILE"
        exit 1
    fi
    ADAPTER_DIR=$(cat "$ADAPTER_FILE")
    if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
        echo "ERROR: No adapter_model.safetensors at $ADAPTER_DIR"
        exit 1
    fi
    echo "Loading adapter from stage $PREV_STAGE: $ADAPTER_DIR"
    ADAPTER_ARG="--continue_from_adapter $ADAPTER_DIR"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"
echo ""

PYTHONUNBUFFERED=1 python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path __MODEL_NAME__ \
    --dataset_name __DATASET_NAME__ \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size __BATCH_SIZE__ \
    --gradient_accumulation_steps __GRAD_ACCUM__ \
    --learning_rate __LEARNING_RATE__ \
    --bf16 \
    --gradient_checkpointing \
    --lora_r __LORA_R__ \
    --lora_alpha __LORA_ALPHA__ \
    --lora_dropout 0.05 \
    --save_strategy no \
    --logging_steps 1 \
    --streaming \
    --max_samples __STAGE_SAMPLES__ \
    --seed __SEED__ \
    --max_completion_length __MAX_COMPLETION_LENGTH__ \
    --max_prompt_length __MAX_PROMPT_LENGTH__ \
    --num_generations __NUM_GENERATIONS__ \
    --reward_type __REWARD_TYPE__ \
    $ADAPTER_ARG

TRAIN_EXIT_CODE=$?
if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Stage __STAGE__ failed (exit $TRAIN_EXIT_CODE)"
    exit $TRAIN_EXIT_CODE
fi

# Save adapter path for next stage
echo "$OUTPUT_DIR" > "$CHAIN_DIR/stage-__STAGE__.adapter"
echo "Stage __STAGE__ complete! Adapter: $OUTPUT_DIR"
STAGE_EOF

    # Inject build-time constants
    sed -i \
        -e "s|__STAGE__|${STAGE}|g" \
        -e "s|__NUM_STAGES__|${NUM_STAGES}|g" \
        -e "s|__CHAIN_DIR__|${CHAIN_DIR}|g" \
        -e "s|__SEED__|${SEED}|g" \
        -e "s|__STAGE_SAMPLES__|${STAGE_SAMPLES}|g" \
        -e "s|__MODEL_NAME__|${MODEL_NAME}|g" \
        -e "s|__DATASET_NAME__|${DATASET_NAME}|g" \
        -e "s|__BATCH_SIZE__|${BATCH_SIZE}|g" \
        -e "s|__GRAD_ACCUM__|${GRAD_ACCUM}|g" \
        -e "s|__LEARNING_RATE__|${LEARNING_RATE}|g" \
        -e "s|__LORA_R__|${LORA_R}|g" \
        -e "s|__LORA_ALPHA__|${LORA_ALPHA}|g" \
        -e "s|__MAX_COMPLETION_LENGTH__|${MAX_COMPLETION_LENGTH}|g" \
        -e "s|__MAX_PROMPT_LENGTH__|${MAX_PROMPT_LENGTH}|g" \
        -e "s|__NUM_GENERATIONS__|${NUM_GENERATIONS}|g" \
        -e "s|__REWARD_TYPE__|${REWARD_TYPE}|g" \
        "$STAGE_SCRIPT"

    # Verify syntax before submitting
    bash -n "$STAGE_SCRIPT" || { echo "SYNTAX ERROR in stage $STAGE"; exit 1; }

    if [[ -n "$PREV_JOB_ID" ]]; then
        JOB_ID=$(sbatch --parsable --dependency=afterok:$PREV_JOB_ID "$STAGE_SCRIPT" 2>&1 | grep -oE '[0-9]+$')
    else
        JOB_ID=$(sbatch --parsable "$STAGE_SCRIPT" 2>&1 | grep -oE '[0-9]+$')
    fi

    echo "Stage $STAGE: job $JOB_ID" $([[ -n "$PREV_JOB_ID" ]] && echo "(after $PREV_JOB_ID)" || echo "(no dep)")
    PREV_JOB_ID="$JOB_ID"
done

echo ""
echo "Chain submitted: 3 stages x 5 samples"
echo "Monitor: squeue -u \$USER | grep chain-5s"
