#!/bin/bash
# =============================================================================
# Batch GGUF conversion for all TRL-trained models
#
# Runs AFTER training completes. Takes merged models from Hub,
# produces all 6 quantization formats and pushes to -GGUF repos.
#
# TRL notebooks' built-in GGUF failed (missing llama-cpp-python).
# This script uses convert_gguf.py with llama.cpp directly.
# =============================================================================

#SBATCH --job-name=batch-gguf-trl
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
WORK_DIR="$SCRATCH/outputs/batch-gguf-trl-$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo "Batch GGUF: TRL-trained models"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | Start: $(date)"
echo "Work dir: $WORK_DIR"
echo ""

# Format: HUB_ID|BASE_MODEL|TYPE|METHOD|DATASET|LICENSE|SHORT_NAME|OUTPUT_REPO
MODELS=(
    # ── SFT Distillation (Claude Reasoning) ──
    "ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Reasoning|Qwen/Qwen3.5-0.8B|full|SFT-Distillation|Claude-Reasoning|apache-2.0|qwen35-08b-sft-claude-trl|ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Reasoning-GGUF"
    "ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Reasoning|LiquidAI/LFM2.5-1.2B-Instruct|full|SFT-Distillation|Claude-Reasoning|apache-2.0|lfm25-12b-sft-claude-trl|ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Reasoning-GGUF"
    "ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Reasoning|google/gemma-4-E2B-it|full|SFT-Distillation|Claude-Reasoning|gemma|gemma4-e2b-sft-claude-trl|ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Reasoning-GGUF"

    # ── xLAM Function Calling ──
    "ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM|Qwen/Qwen3.5-0.8B|full|SFT|xLAM-60K|apache-2.0|qwen35-08b-xlam-trl|ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-GGUF"
    "ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM|Qwen/Qwen3.5-2B|full|SFT|xLAM-60K|apache-2.0|qwen35-2b-xlam-trl|ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-GGUF"
    "ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM|LiquidAI/LFM2.5-1.2B-Instruct|full|SFT|xLAM-60K|apache-2.0|lfm25-12b-xlam-trl|ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-GGUF"
    "ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM|google/gemma-4-E2B-it|full|SFT|xLAM-60K|gemma|gemma4-e2b-xlam-trl|ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-GGUF"
    "ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM|google/gemma-4-E4B-it|full|SFT|xLAM-60K|gemma|gemma4-e4b-xlam-trl|ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-GGUF"
)

# All 6 quantization formats
ALL_QUANTS="Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0"

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
    echo "  Quants:    $ALL_QUANTS"
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
            --quantizations "$ALL_QUANTS" \
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
