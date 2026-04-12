#!/bin/bash
# =============================================================================
# Qwen2.5-3B-SFT-UltraChat GRPO Continual Training Chain on NuminaMath-CoT
# 5 stages × 20K samples = 100K total
# =============================================================================

set -euo pipefail

WAIT_FOR_JOB="${1:-}"
CHAIN_DIR="/scratch/$USER/outputs/qwen2.5-3b-chain"
BASE_SEED=42

MODEL_NAME='ermiaazarkhalili/Qwen2.5-3B-SFT-UltraChat'
DATASET_NAME='AI-MO/NuminaMath-CoT'
HUB_MODEL_ID='ermiaazarkhalili/Qwen2.5-3B-GRPO-NuminaMath-100K'
GGUF_REPO_ID='ermiaazarkhalili/Qwen2.5-3B-GRPO-NuminaMath-100K-GGUF'

BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=1e-06
LORA_R=16
LORA_ALPHA=32
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=512
NUM_GENERATIONS=4
REWARD_TYPE='combined'
SAMPLES_PER_STAGE=20000
NUM_STAGES=5

mkdir -p "$CHAIN_DIR" logs

echo "=========================================="
echo "Qwen2.5-3B GRPO Continual Training Chain"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Stages: 1 to $NUM_STAGES"
echo "Samples per stage: $SAMPLES_PER_STAGE"
echo "Total: $((SAMPLES_PER_STAGE * NUM_STAGES / 1000))K"
echo "Seeds: $(seq -s', ' $((BASE_SEED + 1)) $((BASE_SEED + NUM_STAGES)))"
echo "Wait for job: ${WAIT_FOR_JOB:-'(none)'}"
echo "Chain state: $CHAIN_DIR"
echo ""

PREV_JOB_ID="${WAIT_FOR_JOB:-}"

for ((STAGE=1; STAGE<=NUM_STAGES; STAGE++)); do
    SEED=$((BASE_SEED + STAGE))
    CUMULATIVE=$((STAGE * SAMPLES_PER_STAGE / 1000))K
    IS_FINAL=$([[ $STAGE -eq $NUM_STAGES ]] && echo "true" || echo "false")

    echo "--- Stage $STAGE ---"
    echo "  Samples: $SAMPLES_PER_STAGE (cumulative: $CUMULATIVE)"
    echo "  Seed: $SEED"
    echo "  Push to Hub: $IS_FINAL"

    STAGE_SCRIPT="$CHAIN_DIR/stage-${STAGE}.sh"

    cat > "$STAGE_SCRIPT" << 'STAGE_EOF'
#!/bin/bash
#SBATCH --job-name=qwen25-3b-chain-stage-__STAGE__
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

echo "=========================================="
echo "QWEN2.5-3B CHAIN STAGE __STAGE__ / __NUM_STAGES__ (Job $SLURM_JOB_ID)"
echo "=========================================="
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/qwen2.5-3b-chain-stage__STAGE__-$SLURM_JOB_ID"

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
        echo "ERROR: Previous stage adapter not found: $ADAPTER_FILE"
        exit 1
    fi

    ADAPTER_DIR=$(cat "$ADAPTER_FILE")

    if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
        echo "ERROR: Adapter not found at $ADAPTER_DIR/adapter_model.safetensors"
        exit 1
    fi

    echo "Loading adapter from stage $PREV_STAGE: $ADAPTER_DIR"
    ADAPTER_ARG="--continue_from_adapter $ADAPTER_DIR"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"
echo "Seed: __SEED__"
echo "Samples: __SAMPLES_PER_STAGE__"
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
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to trackio \
    --project "qwen2.5-3b-grpo-numinamath" \
    --run_name "qwen2.5-3b-chain-stage__STAGE__-$SLURM_JOB_ID" \
    --streaming \
    --max_samples __SAMPLES_PER_STAGE__ \
    --seed __SEED__ \
    --max_completion_length __MAX_COMPLETION_LENGTH__ \
    --max_prompt_length __MAX_PROMPT_LENGTH__ \
    --num_generations __NUM_GENERATIONS__ \
    --reward_type __REWARD_TYPE__ \
    $ADAPTER_ARG

TRAIN_EXIT_CODE=$?

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Stage __STAGE__ failed (exit code $TRAIN_EXIT_CODE)"
    exit $TRAIN_EXIT_CODE
fi

echo "$OUTPUT_DIR" > "$CHAIN_DIR/stage-__STAGE__.adapter"
echo "Adapter saved to $CHAIN_DIR/stage-__STAGE__.adapter"

IS_FINAL=__IS_FINAL__
if [[ "$IS_FINAL" == "true" ]]; then
    HUB_MODEL_ID="__HUB_MODEL_ID__"
    GGUF_REPO_ID="__GGUF_REPO_ID__"

    echo ""
    echo "=== FINAL STAGE: Pushing to Hub ==="

    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$OUTPUT_DIR',
    repo_id='$HUB_MODEL_ID',
    repo_type='model',
    ignore_patterns=['checkpoint-*', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'],
)
print(f'Model uploaded to $HUB_MODEL_ID')
"

    python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/generate_model_card.py \
        --model_name "Qwen2.5-3B-GRPO-NuminaMath-100K" \
        --base_model "__MODEL_NAME__" \
        --dataset "__DATASET_NAME__" \
        --training_method GRPO \
        --author ermiaazarkhalili \
        --license apache-2.0 \
        --learning_rate __LEARNING_RATE__ \
        --batch_size __BATCH_SIZE__ \
        --epochs 1 \
        --max_length __MAX_COMPLETION_LENGTH__ \
        --lora_r __LORA_R__ \
        --lora_alpha __LORA_ALPHA__ \
        --hardware "NVIDIA H100 MIG" \
        --output_dir $OUTPUT_DIR/model_card 2>/dev/null || true

    python -c "
from huggingface_hub import HfApi
import os
api = HfApi()
card = os.path.join('$OUTPUT_DIR', 'model_card', 'README.md')
if os.path.exists(card):
    api.upload_file(path_or_fileobj=card, path_in_repo='README.md', repo_id='$HUB_MODEL_ID')
    print('Model card uploaded')
" 2>/dev/null || true

    echo ""
    echo "=== Evaluation ==="
    pip install -q lm-eval 2>/dev/null
    python -c "
import subprocess
cmd = ['lm_eval', '--model', 'hf', '--model_args', 'pretrained=$HUB_MODEL_ID,trust_remote_code=True',
       '--tasks', 'gsm8k,minerva_math', '--batch_size', 'auto',
       '--output_path', '$OUTPUT_DIR/eval_results', '--log_samples']
print(f'Running: {\" \".join(cmd)}')
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0: print(f'Eval issues: {result.stderr[:500]}')
" 2>/dev/null || true

    echo ""
    echo "=== GGUF Conversion ==="
    python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/convert_gguf.py \
        --model $HUB_MODEL_ID \
        --base_model __MODEL_NAME__ \
        --output_repo $GGUF_REPO_ID \
        --quantizations "Q4_K_M,Q5_K_M,Q8_0" \
        --output_dir $OUTPUT_DIR/gguf 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "Stage __STAGE__ Complete! $(date)"
echo "=========================================="
STAGE_EOF

    # Inject build-time constants into the generated script
    sed -i \
        -e "s|__STAGE__|${STAGE}|g" \
        -e "s|__NUM_STAGES__|${NUM_STAGES}|g" \
        -e "s|__CHAIN_DIR__|${CHAIN_DIR}|g" \
        -e "s|__SEED__|${SEED}|g" \
        -e "s|__SAMPLES_PER_STAGE__|${SAMPLES_PER_STAGE}|g" \
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
        -e "s|__IS_FINAL__|${IS_FINAL}|g" \
        -e "s|__HUB_MODEL_ID__|${HUB_MODEL_ID}|g" \
        -e "s|__GGUF_REPO_ID__|${GGUF_REPO_ID}|g" \
        "$STAGE_SCRIPT"

    chmod +x "$STAGE_SCRIPT"

    if [[ -n "$PREV_JOB_ID" ]]; then
        JOB_ID=$(sbatch --parsable --dependency=afterok:$PREV_JOB_ID "$STAGE_SCRIPT" 2>&1 | grep -oE '[0-9]+$')
    else
        JOB_ID=$(sbatch --parsable "$STAGE_SCRIPT" 2>&1 | grep -oE '[0-9]+$')
    fi

    echo "  Submitted: job $JOB_ID" \
         $([[ -n "$PREV_JOB_ID" ]] && echo "(after $PREV_JOB_ID)" || echo "(no dependency)")

    PREV_JOB_ID="$JOB_ID"
done

echo ""
echo "=========================================="
echo "Chain Summary"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Total: $((SAMPLES_PER_STAGE * NUM_STAGES / 1000))K samples"
echo "Final model: $HUB_MODEL_ID"
echo "Chain state: $CHAIN_DIR/"
echo ""
echo "Monitor: squeue -u \$USER | grep qwen25"
