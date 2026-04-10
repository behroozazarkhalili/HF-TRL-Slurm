#!/bin/bash
# =============================================================================
# Smoke test for the chain script — 2 stages, 15 steps each, ~35 min total
# =============================================================================

set -euo pipefail

CHAIN_DIR="/scratch/$USER/outputs/granite-chain-smoke"
rm -rf "$CHAIN_DIR"
mkdir -p "$CHAIN_DIR" logs

echo "=== CHAIN SMOKE TEST (3 stages) ==="
echo "Stage 1: 15 steps, seed=42, from scratch"
echo "Stage 2: 15 steps, seed=43, from stage 1 adapter"
echo "Stage 3: 15 steps, seed=44, from stage 2 adapter"
echo ""

# --- Stage 1 job script ---
cat > "$CHAIN_DIR/stage-1.sh" << 'STAGE1_EOF'
#!/bin/bash
#SBATCH --job-name=chain-smoke-stage-1
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/granite-chain-smoke/stage1-$SLURM_JOB_ID"
CHAIN_DIR="$SCRATCH/outputs/granite-chain-smoke"

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR

echo "=== CHAIN SMOKE: STAGE 1 ==="
echo "Output: $OUTPUT_DIR"

PYTHONUNBUFFERED=1 python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path ibm-granite/granite-4.0-micro \
    --dataset_name AI-MO/NuminaMath-CoT \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --report_to none \
    --run_name "chain-smoke-stage1" \
    --streaming \
    --max_samples 100 \
    --seed 42 \
    --max_completion_length 2048 \
    --max_prompt_length 512 \
    --num_generations 4 \
    --reward_type combined \
    --max_steps 15

echo "$OUTPUT_DIR" > "$CHAIN_DIR/stage-1.adapter"
echo "Stage 1 adapter saved to: $CHAIN_DIR/stage-1.adapter"
echo "=== STAGE 1 COMPLETE ==="
STAGE1_EOF

# --- Stage 2 job script ---
cat > "$CHAIN_DIR/stage-2.sh" << 'STAGE2_EOF'
#!/bin/bash
#SBATCH --job-name=chain-smoke-stage-2
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/granite-chain-smoke/stage2-$SLURM_JOB_ID"
CHAIN_DIR="$SCRATCH/outputs/granite-chain-smoke"

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR

# Read adapter path from stage 1
ADAPTER_FILE="$CHAIN_DIR/stage-1.adapter"
if [[ ! -f "$ADAPTER_FILE" ]]; then
    echo "ERROR: Stage 1 adapter file not found: $ADAPTER_FILE"
    exit 1
fi
ADAPTER_DIR=$(cat "$ADAPTER_FILE")

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
    echo "ERROR: Adapter not found at $ADAPTER_DIR/adapter_model.safetensors"
    ls -la "$ADAPTER_DIR/" 2>/dev/null
    exit 1
fi

echo "=== CHAIN SMOKE: STAGE 2 ==="
echo "Adapter from stage 1: $ADAPTER_DIR"
echo "Output: $OUTPUT_DIR"

PYTHONUNBUFFERED=1 python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path ibm-granite/granite-4.0-micro \
    --dataset_name AI-MO/NuminaMath-CoT \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --report_to none \
    --run_name "chain-smoke-stage2" \
    --streaming \
    --max_samples 100 \
    --seed 43 \
    --max_completion_length 2048 \
    --max_prompt_length 512 \
    --num_generations 4 \
    --reward_type combined \
    --max_steps 15 \
    --continue_from_adapter $ADAPTER_DIR

echo "$OUTPUT_DIR" > "$CHAIN_DIR/stage-2.adapter"
echo "Stage 2 adapter saved to: $CHAIN_DIR/stage-2.adapter"
echo "=== STAGE 2 COMPLETE ==="
STAGE2_EOF

# --- Stage 3 job script ---
cat > "$CHAIN_DIR/stage-3.sh" << 'STAGE3_EOF'
#!/bin/bash
#SBATCH --job-name=chain-smoke-stage-3
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/granite-chain-smoke/stage3-$SLURM_JOB_ID"
CHAIN_DIR="$SCRATCH/outputs/granite-chain-smoke"

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR

# Read adapter path from stage 2
ADAPTER_FILE="$CHAIN_DIR/stage-2.adapter"
if [[ ! -f "$ADAPTER_FILE" ]]; then
    echo "ERROR: Stage 2 adapter file not found: $ADAPTER_FILE"
    exit 1
fi
ADAPTER_DIR=$(cat "$ADAPTER_FILE")

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
    echo "ERROR: Adapter not found at $ADAPTER_DIR/adapter_model.safetensors"
    ls -la "$ADAPTER_DIR/" 2>/dev/null
    exit 1
fi

echo "=== CHAIN SMOKE: STAGE 3 ==="
echo "Adapter from stage 2: $ADAPTER_DIR"
echo "Output: $OUTPUT_DIR"

PYTHONUNBUFFERED=1 python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path ibm-granite/granite-4.0-micro \
    --dataset_name AI-MO/NuminaMath-CoT \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --report_to none \
    --run_name "chain-smoke-stage3" \
    --streaming \
    --max_samples 100 \
    --seed 44 \
    --max_completion_length 2048 \
    --max_prompt_length 512 \
    --num_generations 4 \
    --reward_type combined \
    --max_steps 15 \
    --continue_from_adapter $ADAPTER_DIR

echo "$OUTPUT_DIR" > "$CHAIN_DIR/stage-3.adapter"
echo "Stage 3 adapter saved to: $CHAIN_DIR/stage-3.adapter"
echo "=== STAGE 3 COMPLETE ==="
STAGE3_EOF

# --- Submit chain ---
JOB1=$(sbatch --parsable "$CHAIN_DIR/stage-1.sh")
echo "Stage 1 submitted: job $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$CHAIN_DIR/stage-2.sh")
echo "Stage 2 submitted: job $JOB2 (after $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$CHAIN_DIR/stage-3.sh")
echo "Stage 3 submitted: job $JOB3 (after $JOB2)"

echo ""
echo "Chain: $JOB1 → $JOB2 → $JOB3"
echo "Monitor: squeue -u \$USER | grep chain-smoke"
echo "Logs:    ls -t logs/chain-smoke-stage-*.out"
