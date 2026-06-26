#!/bin/bash
#SBATCH --job-name=batch-gguf-cards
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_unsloth/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p logs

SCRIPT_DIR="/project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts"
WORK_DIR="$SCRATCH/outputs/batch-gguf-$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo "Batch GGUF Conversion + Model Cards"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | Start: $(date)"
echo "Work dir: $WORK_DIR"
echo ""

# ─────────────────────────────────────────────
# Model definitions: HUB_ID | BASE_MODEL | TYPE | TRAINING_METHOD | DATASET | LICENSE | SHORT_NAME
#   TYPE: "full" = merged model (convert directly), "adapter" = LoRA adapter (needs base_model merge)
#
# EXCLUDED (empty repos, no model files):
#   - granite-4.0-micro-GRPO-NuminaMath-20K (still training)
#   - Qwen2.5-7B-SFT-UltraChat (placeholder only)
#   - qwen2-7b-instruct-trl-sft-ChartQA (placeholder only)
#   - qwen2.5-7b-instruct-trl-sft-ChartQA-oop (placeholder only)
# ─────────────────────────────────────────────
MODELS=(
    # Full merged models (xLAM function calling) — no base_model needed for GGUF
    "ermiaazarkhalili/Qwen2.5-0.5B-Instruct_Function_Calling_xLAM|Qwen/Qwen2.5-0.5B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|apache-2.0|Qwen2.5-0.5B-Function-Calling-xLAM"
    "ermiaazarkhalili/Qwen2.5-1.5B-Instruct_Function_Calling_xLAM|Qwen/Qwen2.5-1.5B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|apache-2.0|Qwen2.5-1.5B-Function-Calling-xLAM"
    "ermiaazarkhalili/Qwen2.5-3B-Instruct_Function_Calling_xLAM|Qwen/Qwen2.5-3B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|apache-2.0|Qwen2.5-3B-Function-Calling-xLAM"
    "ermiaazarkhalili/Qwen2.5-7B-Instruct_Function_Calling_xLAM|Qwen/Qwen2.5-7B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|apache-2.0|Qwen2.5-7B-Function-Calling-xLAM"
    "ermiaazarkhalili/Qwen2.5-14B-Instruct_Function_Calling_xLAM|Qwen/Qwen2.5-14B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|apache-2.0|Qwen2.5-14B-Function-Calling-xLAM"
    "ermiaazarkhalili/Llama-3.2-1B-Instruct_Function_Calling_xLAM|meta-llama/Llama-3.2-1B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|llama3.2|Llama-3.2-1B-Function-Calling-xLAM"
    "ermiaazarkhalili/Llama-3.2-3B-Instruct_Function_Calling_xLAM|meta-llama/Llama-3.2-3B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|llama3.2|Llama-3.2-3B-Function-Calling-xLAM"
    "ermiaazarkhalili/Llama-3.1-8B-Instruct_Function_Calling_xLAM|meta-llama/Llama-3.1-8B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|llama3.1|Llama-3.1-8B-Function-Calling-xLAM"
    "ermiaazarkhalili/Llama-3-8B-Instruct_Function_Calling_xLAM|meta-llama/Meta-Llama-3-8B-Instruct|full|SFT|Salesforce/xlam-function-calling-60k|llama3|Llama-3-8B-Function-Calling-xLAM"
    # LoRA adapters — need base_model for merge + GGUF
    "ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara|Qwen/Qwen2.5-7B|adapter|SFT|trl-lib/Capybara|apache-2.0|Qwen2.5-7B-SFT-Capybara"
    "ermiaazarkhalili/qwen2.5-7b-instruct-trl-sft-ChartQA|Qwen/Qwen2.5-7B-Instruct|adapter|SFT|HuggingFaceM4/ChartQA|apache-2.0|Qwen2.5-7B-SFT-ChartQA"
)

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0
SKIPPED=0

for i in "${!MODELS[@]}"; do
    IFS='|' read -r HUB_ID BASE_MODEL MODEL_TYPE METHOD DATASET LICENSE SHORT_NAME <<< "${MODELS[$i]}"
    GGUF_REPO="${HUB_ID}-GGUF"
    MODEL_DIR="$WORK_DIR/$(echo $SHORT_NAME | tr '/' '-')"
    mkdir -p "$MODEL_DIR"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/$TOTAL] $SHORT_NAME"
    echo "=========================================="
    echo "  Hub ID:    $HUB_ID"
    echo "  Base:      $BASE_MODEL"
    echo "  Type:      $MODEL_TYPE"
    echo "  Method:    $METHOD | Dataset: $DATASET"
    echo "  License:   $LICENSE"
    echo "  GGUF repo: $GGUF_REPO"
    echo "  Start:     $(date +%H:%M:%S)"

    # --- Phase 1: Model Card (in subshell so failures don't kill the loop) ---
    echo ""
    echo "  --- Model Card ---"
    (
        PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/generate_model_card.py" \
            --model_name "$SHORT_NAME" \
            --base_model "$BASE_MODEL" \
            --dataset "$DATASET" \
            --training_method "$METHOD" \
            --author ermiaazarkhalili \
            --license "$LICENSE" \
            --hardware "NVIDIA H100 MIG" \
            --output_dir "$MODEL_DIR/model_card" 2>&1
    )
    CARD_EXIT=$?

    if [[ $CARD_EXIT -eq 0 && -f "$MODEL_DIR/model_card/README.md" ]]; then
        echo "  Model card generated"
        python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.upload_file(
        path_or_fileobj='$MODEL_DIR/model_card/README.md',
        path_in_repo='README.md',
        repo_id='$HUB_ID',
    )
    print('  Model card uploaded to $HUB_ID')
except Exception as e:
    print(f'  WARNING: Card upload failed: {e}')
" 2>&1
    else
        echo "  WARNING: Model card generation failed (exit $CARD_EXIT) — continuing"
    fi

    # --- Phase 2: GGUF Conversion (in subshell to isolate OOM crashes) ---
    echo ""
    echo "  --- GGUF Conversion ---"
    (
        if [[ "$MODEL_TYPE" == "full" ]]; then
            # Full model — convert directly from HUB_ID, pass base_model for reference only
            PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/convert_gguf.py" \
                --model "$HUB_ID" \
                --base_model "$BASE_MODEL" \
                --output_repo "$GGUF_REPO" \
                --quantizations "Q4_K_M,Q5_K_M,Q8_0" \
                --output_dir "$MODEL_DIR/gguf" 2>&1
        else
            # LoRA adapter — needs base_model to merge first
            PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/convert_gguf.py" \
                --model "$HUB_ID" \
                --base_model "$BASE_MODEL" \
                --output_repo "$GGUF_REPO" \
                --quantizations "Q4_K_M,Q5_K_M,Q8_0" \
                --output_dir "$MODEL_DIR/gguf" 2>&1
        fi
    )
    GGUF_EXIT=$?

    if [[ $GGUF_EXIT -eq 0 ]]; then
        echo "  GGUF conversion SUCCESS"
        PASSED=$((PASSED + 1))
    elif [[ $GGUF_EXIT -eq 137 ]]; then
        echo "  GGUF conversion KILLED (OOM) — needs full H100 80GB"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  GGUF conversion FAILED (exit $GGUF_EXIT)"
        FAILED=$((FAILED + 1))
    fi

    echo "  End: $(date +%H:%M:%S)"

    # Clear GPU memory between models
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "Batch Complete! $(date)"
echo "=========================================="
echo "Total:   $TOTAL"
echo "Passed:  $PASSED"
echo "Failed:  $FAILED"
echo "Skipped: $SKIPPED (OOM — need full H100)"
echo ""
