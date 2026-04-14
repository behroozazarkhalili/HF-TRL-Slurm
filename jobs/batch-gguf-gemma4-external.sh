#!/bin/bash
#SBATCH --job-name=batch-gguf-gemma4
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

# ─────────────────────────────────────────────────────────────────────────────
# Batch GGUF conversion: Gemma-4 31B distillations + Google Gemma-4 series
#
# Note: GGUF conversion (convert_hf_to_gguf.py) runs on CPU — no GPU VRAM
# needed. The 31B models need ~70GB system RAM in bf16. We request 160GB
# to handle two 31B models loaded sequentially without issue.
#
# Output repos: ermiaazarkhalili/<model>-GGUF
# ─────────────────────────────────────────────────────────────────────────────

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p logs

SCRIPT_DIR="/project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts"
WORK_DIR="$SCRATCH/outputs/batch-gguf-gemma4-$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo "Batch GGUF: Gemma-4 31B Distillations + Google Gemma-4 Series"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | Start: $(date)"
echo "Work dir: $WORK_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Format: HUB_ID | BASE_MODEL | TYPE | METHOD | DATASET | LICENSE | SHORT_NAME | OUTPUT_REPO
#
# "full" = download model directly and convert (no LoRA merge needed)
# OUTPUT_REPO = where to push GGUF files on ermiaazarkhalili's HF hub
# ─────────────────────────────────────────────────────────────────────────────
MODELS=(
    # ── Gemma-4 Claude Opus 4.6 reasoning distillations (third-party) ───────────
    #   EganAI:      full SFT on gemma-4-31b-it
    #   Darwin:      evolutionary merge (DARE-TIES), multilingual
    #   kai-os:      LoRA adapter on gemma-4-31b-it (Opus-4.6-Reasoning-2100x)
    #   arsovskidev: full SFT on gemma-4-E4B-it (4B model)
    # ── 31B models commented out for now (need 160G RAM) ──────────────────────
    # "EganAI/gemma-4-31B-Claude-4.6-Opus-Reasoning-Distilled|google/gemma-4-31B-it|full|Distillation-SFT|Claude-4.6-Opus|apache-2.0|gemma-4-31b-it-claude-opus-4.6-sft-reasoning|ermiaazarkhalili/gemma-4-31b-it-claude-opus-4.6-sft-reasoning-GGUF"
    # "FINAL-Bench/Darwin-31B-Opus|google/gemma-4-31B-it|full|Evolutionary-Merge|Opus-4.6-Reasoning|apache-2.0|gemma-4-31b-it-claude-opus-4.6-merge-reasoning-multilingual|ermiaazarkhalili/gemma-4-31b-it-claude-opus-4.6-merge-reasoning-multilingual-GGUF"
    # "kai-os/gemma4-31b-Opus-4.6-reasoning|google/gemma-4-31B-it|adapter|LoRA-Distillation|Opus-4.6-Reasoning-2100x|apache-2.0|gemma-4-31b-it-claude-opus-4.6-lora-reasoning|ermiaazarkhalili/gemma-4-31b-it-claude-opus-4.6-lora-reasoning-GGUF"
    "arsovskidev/Gemma-4-E4B-Claude-4.6-Opus-Reasoning-Distilled|google/gemma-4-E4B-it|full|Distillation-SFT|Opus-4.6-Reasoning-3000x|apache-2.0|gemma-4-4b-it-claude-opus-4.6-sft-reasoning|ermiaazarkhalili/gemma-4-4b-it-claude-opus-4.6-sft-reasoning-GGUF"

    # ── Google Gemma 4 official series (sub-10B only for now) ──────────────────
    "google/gemma-4-2b-it|google/gemma-4-2b|full|Instruction-Tuning|Google|gemma|gemma-4-2b-it|ermiaazarkhalili/gemma-4-2b-it-GGUF"
    "google/gemma-4-9b-it|google/gemma-4-9b|full|Instruction-Tuning|Google|gemma|gemma-4-9b-it|ermiaazarkhalili/gemma-4-9b-it-GGUF"
    # ── 27B models commented out for now (need more RAM) ──────────────────────
    # "google/gemma-4-27b-it|google/gemma-4-27b|full|Instruction-Tuning|Google|gemma|gemma-4-27b-it|ermiaazarkhalili/gemma-4-27b-it-GGUF"
    # "google/gemma-4-27b|google/gemma-4-27b|full|Pretraining|Google|gemma|gemma-4-27b-base|ermiaazarkhalili/gemma-4-27b-base-GGUF"
)

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
    echo "  Base:      $BASE_MODEL"
    echo "  Method:    $METHOD | Dataset: $DATASET"
    echo "  Output:    $OUTPUT_REPO"
    echo "  Start:     $(date +%H:%M:%S)"

    # --- GGUF Conversion (subshell isolates OOM/failures) ---
    (
        PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/convert_gguf.py" \
            --model "$HUB_ID" \
            --base_model "$BASE_MODEL" \
            --output_repo "$OUTPUT_REPO" \
            --quantizations "Q4_K_M,Q5_K_M,Q8_0" \
            --output_dir "$MODEL_DIR/gguf" 2>&1
    )
    GGUF_EXIT=$?

    if [[ $GGUF_EXIT -eq 0 ]]; then
        echo "  [$(date +%H:%M:%S)] GGUF conversion SUCCESS: $OUTPUT_REPO"
        PASSED=$((PASSED + 1))
    elif [[ $GGUF_EXIT -eq 137 ]]; then
        echo "  [$(date +%H:%M:%S)] GGUF KILLED (OOM)"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  [$(date +%H:%M:%S)] GGUF FAILED (exit $GGUF_EXIT)"
        FAILED=$((FAILED + 1))
    fi

    # Clear GPU memory between models
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
echo "Skipped: $SKIPPED (OOM)"
echo ""
