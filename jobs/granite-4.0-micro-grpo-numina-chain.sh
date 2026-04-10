#!/bin/bash
# =============================================================================
# granite-4.0-micro GRPO Continual Training Chain
#
# Submits a chain of dependent jobs, each training on NEW data and loading
# the adapter from the previous stage via --continue_from_adapter.
#
# Usage:
#   ./jobs/granite-4.0-micro-grpo-numina-chain.sh [ADAPTER_DIR] [START_STAGE] [NUM_STAGES] [WAIT_FOR_JOB]
#
# Args:
#   ADAPTER_DIR    - Path to adapter from prior training (or "" for fresh start)
#   START_STAGE    - First stage number (default: 2)
#   NUM_STAGES     - Last stage number (default: 4)
#   WAIT_FOR_JOB   - SLURM job ID that must finish before stage 1 starts
#
# Examples:
#   # Chain from RUNNING 10K job (stages 2-4, 20K each, total 70K)
#   ./jobs/granite-4.0-micro-grpo-numina-chain.sh /scratch/.../33301494 2 4 33301494
#
#   # Chain from COMPLETED adapter (no wait)
#   ./jobs/granite-4.0-micro-grpo-numina-chain.sh /scratch/.../completed-run 2 4
#
#   # Fresh start, 5 stages of 10K each (total: 50K)
#   ./jobs/granite-4.0-micro-grpo-numina-chain.sh "" 1 5
#
# Each stage writes its adapter path to:
#   /scratch/$USER/outputs/granite-chain/stage-{N}.adapter
# so the next stage knows where to load from.
# =============================================================================

set -euo pipefail

# --- Configuration ---
INITIAL_ADAPTER="${1:-}"
START_STAGE="${2:-2}"
NUM_STAGES="${3:-4}"
WAIT_FOR_JOB="${4:-}"  # SLURM job ID to wait for before starting first stage
SAMPLES_PER_STAGE=20000
SAMPLES_FIRST_STAGE=10000  # Stage 1 trains fewer samples (initial exploration)
BASE_SEED=42               # Stage N uses seed = BASE_SEED + N

MODEL_NAME='ibm-granite/granite-4.0-micro'
DATASET_NAME='AI-MO/NuminaMath-CoT'
CHAIN_DIR="/scratch/$USER/outputs/granite-chain"

# Training params
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=1e-06
LORA_R=16
LORA_ALPHA=32
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=512
NUM_GENERATIONS=4
REWARD_TYPE='combined'

# --- Setup ---
mkdir -p "$CHAIN_DIR" logs

echo "=========================================="
echo "Granite 4.0-micro GRPO Continual Training Chain"
echo "=========================================="
echo "Initial adapter: ${INITIAL_ADAPTER:-'(none - training from scratch)'}"
echo "Stages: $START_STAGE to $NUM_STAGES"
echo "Samples per stage: $SAMPLES_PER_STAGE"
echo "Seeds: $(seq -s', ' $((BASE_SEED + START_STAGE)) $((BASE_SEED + NUM_STAGES)))"
echo "Wait for job: ${WAIT_FOR_JOB:-'(none)'}"
echo "Chain state dir: $CHAIN_DIR"
echo ""

# Record initial adapter for stage 1
if [[ -n "$INITIAL_ADAPTER" ]]; then
    PREV_STAGE=$((START_STAGE - 1))
    echo "$INITIAL_ADAPTER" > "$CHAIN_DIR/stage-${PREV_STAGE}.adapter"
    echo "Registered initial adapter as stage $PREV_STAGE: $INITIAL_ADAPTER"
fi

# If waiting for a prior job, use it as the initial dependency
PREV_JOB_ID="${WAIT_FOR_JOB:-}"
CUMULATIVE_SAMPLES=0

# Calculate cumulative samples from prior stages
if [[ $START_STAGE -gt 1 ]]; then
    CUMULATIVE_SAMPLES=$SAMPLES_FIRST_STAGE
    for ((s=2; s<START_STAGE; s++)); do
        CUMULATIVE_SAMPLES=$((CUMULATIVE_SAMPLES + SAMPLES_PER_STAGE))
    done
fi

for ((STAGE=START_STAGE; STAGE<=NUM_STAGES; STAGE++)); do
    SEED=$((BASE_SEED + STAGE))

    if [[ $STAGE -eq 1 ]]; then
        STAGE_SAMPLES=$SAMPLES_FIRST_STAGE
    else
        STAGE_SAMPLES=$SAMPLES_PER_STAGE
    fi
    CUMULATIVE_SAMPLES=$((CUMULATIVE_SAMPLES + STAGE_SAMPLES))

    # Format cumulative for model naming
    if [[ $CUMULATIVE_SAMPLES -ge 1000000 ]]; then
        CUMULATIVE_LABEL="$((CUMULATIVE_SAMPLES / 1000000))M"
    else
        CUMULATIVE_LABEL="$((CUMULATIVE_SAMPLES / 1000))K"
    fi

    # Only the final stage pushes to Hub
    IS_FINAL=$([[ $STAGE -eq $NUM_STAGES ]] && echo "true" || echo "false")

    echo ""
    echo "--- Stage $STAGE ---"
    echo "  Samples: $STAGE_SAMPLES (cumulative: $CUMULATIVE_LABEL)"
    echo "  Seed: $SEED"
    echo "  Push to Hub: $IS_FINAL"

    # Create the stage job script
    STAGE_SCRIPT="$CHAIN_DIR/stage-${STAGE}.sh"

    cat > "$STAGE_SCRIPT" << STAGE_EOF
#!/bin/bash
#SBATCH --job-name=granite-chain-stage-${STAGE}
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
echo "CHAIN STAGE $STAGE / $NUM_STAGES (Job \$SLURM_JOB_ID)"
echo "=========================================="
echo "Node: \$SLURMD_NODENAME"
echo "Start time: \$(date)"

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=\${SCRATCH:-/scratch/\$USER}
export HF_HOME=\$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=\$HF_HOME/hub
export OUTPUT_DIR="\$SCRATCH/outputs/granite-chain-stage${STAGE}-\$SLURM_JOB_ID"

# Load HF token
if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "\$key" =~ ^#.*$ || -z "\$key" ]] && continue
        value="\${value%\\\"}"; value="\${value#\\\"}"
        value="\${value%\\\'}"; value="\${value#\\\'}"
        export "\$key=\$value"
    done < "/project/6014832/ermia/HF-TRL/.env"
fi

CHAIN_DIR="$CHAIN_DIR"
mkdir -p "\$OUTPUT_DIR" logs

# Resolve adapter from previous stage
ADAPTER_ARG=""
STAGE_NUM=$STAGE
if [[ \$STAGE_NUM -gt 1 ]]; then
    PREV_STAGE=\$((STAGE_NUM - 1))
    ADAPTER_FILE="\$CHAIN_DIR/stage-\${PREV_STAGE}.adapter"

    if [[ ! -f "\$ADAPTER_FILE" ]]; then
        echo "ERROR: Previous stage adapter file not found: \$ADAPTER_FILE"
        exit 1
    fi

    ADAPTER_DIR=\$(cat "\$ADAPTER_FILE")

    if [[ ! -f "\$ADAPTER_DIR/adapter_model.safetensors" ]]; then
        echo "ERROR: Adapter not found at \$ADAPTER_DIR/adapter_model.safetensors"
        exit 1
    fi

    echo "Loading adapter from stage \$PREV_STAGE: \$ADAPTER_DIR"
    ADAPTER_ARG="--continue_from_adapter \$ADAPTER_DIR"
fi

echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
echo "Output: \$OUTPUT_DIR"
echo "Seed: $SEED"
echo "Samples: $STAGE_SAMPLES"
echo ""

# --- Training ---
PYTHONUNBUFFERED=1 python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \\
    --model_name_or_path $MODEL_NAME \\
    --dataset_name $DATASET_NAME \\
    --output_dir \$OUTPUT_DIR \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size $BATCH_SIZE \\
    --gradient_accumulation_steps $GRAD_ACCUM \\
    --learning_rate $LEARNING_RATE \\
    --bf16 \\
    --gradient_checkpointing \\
    --lora_r $LORA_R \\
    --lora_alpha $LORA_ALPHA \\
    --lora_dropout 0.05 \\
    --save_strategy steps \\
    --save_steps 500 \\
    --save_total_limit 3 \\
    --logging_steps 10 \\
    --report_to trackio \\
    --project "granite-4.0-micro-grpo-numinamath" \\
    --run_name "granite-chain-stage${STAGE}-\$SLURM_JOB_ID" \\
    --streaming \\
    --max_samples $STAGE_SAMPLES \\
    --seed $SEED \\
    --max_completion_length $MAX_COMPLETION_LENGTH \\
    --max_prompt_length $MAX_PROMPT_LENGTH \\
    --num_generations $NUM_GENERATIONS \\
    --reward_type $REWARD_TYPE \\
    \$ADAPTER_ARG

TRAIN_EXIT_CODE=\$?

if [[ \$TRAIN_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Stage $STAGE training failed (exit code \$TRAIN_EXIT_CODE)"
    exit \$TRAIN_EXIT_CODE
fi

# Record adapter path for next stage
echo "\$OUTPUT_DIR" > "\$CHAIN_DIR/stage-${STAGE}.adapter"
echo "Adapter path saved to \$CHAIN_DIR/stage-${STAGE}.adapter"

# --- Final stage: push to Hub + eval + GGUF ---
IS_FINAL=$IS_FINAL
if [[ "\$IS_FINAL" == "true" ]]; then
    HUB_MODEL_ID="ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-${CUMULATIVE_LABEL}"
    GGUF_REPO_ID="ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-${CUMULATIVE_LABEL}-GGUF"

    echo ""
    echo "=== FINAL STAGE: Pushing to Hub ==="
    echo "Hub Model: \$HUB_MODEL_ID"

    # Push adapter to Hub
    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='\$OUTPUT_DIR',
    repo_id='\$HUB_MODEL_ID',
    repo_type='model',
    ignore_patterns=['checkpoint-*', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'],
)
print(f'Model uploaded to \$HUB_MODEL_ID')
"

    # Model card
    python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/generate_model_card.py \\
        --model_name "granite-4.0-micro-GRPO-NuminaMath-${CUMULATIVE_LABEL}" \\
        --base_model "$MODEL_NAME" \\
        --dataset "$DATASET_NAME" \\
        --training_method GRPO \\
        --author ermiaazarkhalili \\
        --license cc-by-nc-4.0 \\
        --learning_rate $LEARNING_RATE \\
        --batch_size $BATCH_SIZE \\
        --epochs 1 \\
        --max_length $MAX_COMPLETION_LENGTH \\
        --lora_r $LORA_R \\
        --lora_alpha $LORA_ALPHA \\
        --hardware "NVIDIA H100 MIG" \\
        --output_dir \$OUTPUT_DIR/model_card 2>/dev/null || true

    python -c "
from huggingface_hub import HfApi
import os
api = HfApi()
card = os.path.join('\$OUTPUT_DIR', 'model_card', 'README.md')
if os.path.exists(card):
    api.upload_file(path_or_fileobj=card, path_in_repo='README.md', repo_id='\$HUB_MODEL_ID')
    print('Model card uploaded')
" 2>/dev/null || true

    # Evaluation
    echo ""
    echo "=== Evaluation ==="
    pip install -q lm-eval 2>/dev/null
    python -c "
import subprocess
cmd = ['lm_eval', '--model', 'hf', '--model_args', 'pretrained=\$HUB_MODEL_ID,trust_remote_code=True',
       '--tasks', 'gsm8k,minerva_math', '--batch_size', 'auto',
       '--output_path', '\$OUTPUT_DIR/eval_results', '--log_samples']
print(f'Running: {\" \".join(cmd)}')
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0: print(f'Eval issues: {result.stderr[:500]}')
" 2>/dev/null || true

    # GGUF
    echo ""
    echo "=== GGUF Conversion ==="
    python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/convert_gguf.py \\
        --model \$HUB_MODEL_ID \\
        --base_model $MODEL_NAME \\
        --output_repo \$GGUF_REPO_ID \\
        --quantizations "Q4_K_M,Q5_K_M,Q8_0" \\
        --output_dir \$OUTPUT_DIR/gguf 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "Stage $STAGE Complete!"
echo "=========================================="
echo "End time: \$(date)"
echo "Output: \$OUTPUT_DIR"
echo "Adapter: \$CHAIN_DIR/stage-${STAGE}.adapter"
STAGE_EOF

    chmod +x "$STAGE_SCRIPT"

    # Submit with dependency on previous job
    if [[ -n "$PREV_JOB_ID" ]]; then
        SUBMIT_OUT=$(sbatch --parsable --dependency=afterok:$PREV_JOB_ID "$STAGE_SCRIPT" 2>&1)
    else
        SUBMIT_OUT=$(sbatch --parsable "$STAGE_SCRIPT" 2>&1)
    fi

    # Extract job ID (last line, numeric)
    JOB_ID=$(echo "$SUBMIT_OUT" | grep -oE '[0-9]+' | tail -1)

    if [[ -z "$JOB_ID" ]]; then
        echo "  ERROR: Failed to submit stage $STAGE"
        echo "  sbatch output: $SUBMIT_OUT"
        exit 1
    fi

    echo "  Submitted: job $JOB_ID" \
         $([[ -n "$PREV_JOB_ID" ]] && echo "(after $PREV_JOB_ID)" || echo "(no dependency)")

    PREV_JOB_ID="$JOB_ID"
done

echo ""
echo "=========================================="
echo "Chain Summary"
echo "=========================================="
echo "Chain state: $CHAIN_DIR/"
echo "Total samples: $CUMULATIVE_LABEL"
echo ""
echo "Monitor: squeue -u \$USER --format='%.10i %.30j %.8T %.10M %.20E'"
echo "Logs:    ls -t logs/granite-chain-stage-*.out"
