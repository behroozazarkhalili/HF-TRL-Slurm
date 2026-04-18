#!/bin/bash
# =============================================================================
# Batch GGUF conversion for all Unsloth-trained models
#
# Runs AFTER training completes. Takes merged models from Hub,
# produces additional quantization formats beyond what Unsloth built-in provides.
#
# Unsloth built-in: q4_k_m, q5_k_m, q8_0
# This script adds: Q2_K, Q3_K_M, Q6_K (wider range for different use cases)
# =============================================================================

#SBATCH --job-name=batch-gguf-unsloth
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "/project/6014832/ermia/HF-TRL/.env"
fi

mkdir -p logs

SCRIPT_DIR="/project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts"
if [[ ! -f "$SCRIPT_DIR/convert_gguf.py" ]]; then
    echo "[ERROR] convert_gguf.py not found at $SCRIPT_DIR"
    exit 1
fi
WORK_DIR="$SCRATCH/outputs/batch-gguf-unsloth-$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo "Batch GGUF: Unsloth-trained models"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | Start: $(date)"
echo "Work dir: $WORK_DIR"
echo ""

# Format: HUB_ID|BASE_MODEL|TYPE|METHOD|DATASET|LICENSE|SHORT_NAME|OUTPUT_REPO
# All models are full (merged) — no LoRA merge needed
MODELS=(
    # ── SFT Distillation (Claude Reasoning) ──
    "ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Opus-Reasoning-Unsloth|Qwen/Qwen3.5-0.8B|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|qwen35-08b-sft-claude-unsloth|ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Opus-Reasoning-Unsloth|LiquidAI/LFM2.5-1.2B-Instruct|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|lfm25-12b-sft-claude-unsloth|ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Opus-Reasoning-Unsloth|google/gemma-4-E2B-it|full|SFT-Distillation|Claude-Opus-Reasoning|gemma|gemma4-e2b-sft-claude-unsloth|ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"

    # ── xLAM Function Calling ──
    "ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth|Qwen/Qwen3.5-0.8B|full|SFT|xLAM-60K|apache-2.0|qwen35-08b-xlam-unsloth|ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth|Qwen/Qwen3.5-2B|full|SFT|xLAM-60K|apache-2.0|qwen35-2b-xlam-unsloth|ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth|LiquidAI/LFM2.5-1.2B-Instruct|full|SFT|xLAM-60K|apache-2.0|lfm25-12b-xlam-unsloth|ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth|google/gemma-4-E2B-it|full|SFT|xLAM-60K|gemma|gemma4-e2b-xlam-unsloth|ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth|google/gemma-4-E4B-it|full|SFT|xLAM-60K|gemma|gemma4-e4b-xlam-unsloth|ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth-GGUF"
)

# Additional quantizations beyond Unsloth's built-in q4_k_m, q5_k_m, q8_0
EXTRA_QUANTS="Q2_K,Q3_K_M,Q6_K"

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0
SKIPPED=0

for i in "${!MODELS[@]}"; do
    IFS='|' read -r HUB_ID BASE_MODEL MODEL_TYPE METHOD DATASET LICENSE SHORT_NAME OUTPUT_REPO <<< "${MODELS[$i]}"
    MODEL_DIR="$WORK_DIR/$(echo $SHORT_NAME | tr '/' '-')"
    mkdir -p "$MODEL_DIR"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/$TOTAL] $SHORT_NAME"
    echo "=========================================="
    echo "  Source:    $HUB_ID"
    echo "  Output:    $OUTPUT_REPO"
    echo "  Quants:    $EXTRA_QUANTS"
    echo "  Start:     $(date +%H:%M:%S)"

    # Check if source model exists on Hub
    python3 -c "
from huggingface_hub import HfApi
try:
    HfApi().repo_info('$HUB_ID', repo_type='model')
    print('[OK] Model exists on Hub')
except:
    print('[SKIP] Model not yet on Hub')
    exit(1)
" 2>/dev/null
    if [[ $? -ne 0 ]]; then
        echo "  [SKIP] Model not yet available — training may still be running"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # GGUF Conversion (subshell isolates OOM/failures)
    (
        PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/convert_gguf.py" \
            --model "$HUB_ID" \
            --base_model "$BASE_MODEL" \
            --output_repo "$OUTPUT_REPO" \
            --quantizations "$EXTRA_QUANTS" \
            --output_dir "$MODEL_DIR/gguf" 2>&1
    )
    GGUF_EXIT=$?

    if [[ $GGUF_EXIT -eq 0 ]]; then
        echo "  [$(date +%H:%M:%S)] GGUF SUCCESS: $OUTPUT_REPO"
        PASSED=$((PASSED + 1))
    elif [[ $GGUF_EXIT -eq 137 ]]; then
        echo "  [$(date +%H:%M:%S)] GGUF KILLED (OOM)"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  [$(date +%H:%M:%S)] GGUF FAILED (exit $GGUF_EXIT)"
        FAILED=$((FAILED + 1))
    fi

    # Clear memory between models
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo ""
done

echo ""
echo "=========================================="
echo "Batch Complete! $(date +%H:%M:%S)"
echo "=========================================="
echo "Total:   $TOTAL"
echo "Passed:  $PASSED"
echo "Failed:  $FAILED"
echo "Skipped: $SKIPPED (not yet on Hub)"
echo ""
