#!/bin/bash
# =============================================================================
# Dispatcher: submit ONE gguf-single-model.sh SLURM job per model.
#
# Replaces the old batch-gguf-unsloth-models.sh which ran all models in a
# single 12h GPU job using convert_gguf.py (rebuilt llama.cpp per model,
# no fc10713 exclude, one crash killed the rest). The new pipeline:
#   - CPU-only SLURM jobs (cpubase_bycore_b2, 96G RAM)
#   - Cached llama.cpp toolchain at /scratch/$USER/tools/llama.cpp
#   - fc10713 excluded per job
#   - Per-model failure isolation (one model dies → others continue)
#   - Individual jobs pack into backfill slots faster than one big 12h slot
#
# Run from login node, NOT as a SLURM job itself — this is a submitter.
#
# Usage:
#   bash jobs/dispatch-gguf-unsloth.sh          # submits all available models
#   FILTER=qwen35 bash jobs/dispatch-gguf-unsloth.sh  # only matching SHORT_NAMEs
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
SINGLE_MODEL_WRAPPER="$PROJECT_DIR/jobs/gguf-single-model.sh"
VENV="/scratch/ermia/venvs/hf_unsloth"
EXTRA_QUANTS="${EXTRA_QUANTS:-Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0}"
FILTER="${FILTER:-}"

# Format: HUB_ID|BASE_MODEL|TYPE|METHOD|DATASET|LICENSE|SHORT_NAME|OUTPUT_REPO
# Full merged models — no LoRA adapter merge needed in conversion.
MODELS=(
    # ── SFT Distillation (Claude-Opus Reasoning) ──
    "ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Opus-Reasoning-Unsloth|Qwen/Qwen3.5-0.8B|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|qwen35-08b-sft-claude-unsloth|ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Opus-Reasoning-Unsloth|LiquidAI/LFM2.5-1.2B-Instruct|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|lfm25-12b-sft-claude-unsloth|ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/LFM2.5-350M-SFT-Claude-Opus-Reasoning-Unsloth|LiquidAI/LFM2.5-350M|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|lfm25-350m-sft-claude-unsloth|ermiaazarkhalili/LFM2.5-350M-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Opus-Reasoning-Unsloth|google/gemma-4-E2B-it|full|SFT-Distillation|Claude-Opus-Reasoning|gemma|gemma4-e2b-sft-claude-unsloth|ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Gemma4-E4B-SFT-Claude-Opus-Reasoning-Unsloth|google/gemma-4-E4B-it|full|SFT-Distillation|Claude-Opus-Reasoning|gemma|gemma4-e4b-sft-claude-unsloth|ermiaazarkhalili/Gemma4-E4B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3-4B-SFT-Claude-Opus-Reasoning-Unsloth|unsloth/Qwen3-4B|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|qwen3-4b-sft-claude-unsloth|ermiaazarkhalili/Qwen3-4B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3-8B-SFT-Claude-Opus-Reasoning-Unsloth|unsloth/Qwen3-8B|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|qwen3-8b-sft-claude-unsloth|ermiaazarkhalili/Qwen3-8B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"

    # ── xLAM Function Calling ──
    "ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth|Qwen/Qwen3.5-0.8B|full|SFT|xLAM-60K|apache-2.0|qwen35-08b-xlam-unsloth|ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth|Qwen/Qwen3.5-2B|full|SFT|xLAM-60K|apache-2.0|qwen35-2b-xlam-unsloth|ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth|LiquidAI/LFM2.5-1.2B-Instruct|full|SFT|xLAM-60K|apache-2.0|lfm25-12b-xlam-unsloth|ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/LFM2.5-350M-Function-Calling-xLAM-Unsloth|LiquidAI/LFM2.5-350M|full|SFT|xLAM-60K|apache-2.0|lfm25-350m-xlam-unsloth|ermiaazarkhalili/LFM2.5-350M-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth|google/gemma-4-E2B-it|full|SFT|xLAM-60K|gemma|gemma4-e2b-xlam-unsloth|ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth|google/gemma-4-E4B-it|full|SFT|xLAM-60K|gemma|gemma4-e4b-xlam-unsloth|ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3-4B-Function-Calling-xLAM-Unsloth|unsloth/Qwen3-4B|full|SFT|xLAM-60K|apache-2.0|qwen3-4b-xlam-unsloth|ermiaazarkhalili/Qwen3-4B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3-8B-Function-Calling-xLAM-Unsloth|unsloth/Qwen3-8B|full|SFT|xLAM-60K|apache-2.0|qwen3-8b-xlam-unsloth|ermiaazarkhalili/Qwen3-8B-Function-Calling-xLAM-Unsloth-GGUF"

    # ── Qwen3.5 series (VLM-arch, text-only SFT via Jackrong recipe) ──
    "ermiaazarkhalili/Qwen3.5-4B-Function-Calling-xLAM-Unsloth|unsloth/Qwen3.5-4B|full|SFT|xLAM-60K|apache-2.0|qwen35-4b-xlam-unsloth|ermiaazarkhalili/Qwen3.5-4B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3.5-9B-Function-Calling-xLAM-Unsloth|unsloth/Qwen3.5-9B|full|SFT|xLAM-60K|apache-2.0|qwen35-9b-xlam-unsloth|ermiaazarkhalili/Qwen3.5-9B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3.5-4B-SFT-Claude-Opus-Reasoning-Unsloth|unsloth/Qwen3.5-4B|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|qwen35-4b-sft-claude-unsloth|ermiaazarkhalili/Qwen3.5-4B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Qwen3.5-9B-SFT-Claude-Opus-Reasoning-Unsloth|unsloth/Qwen3.5-9B|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|qwen35-9b-sft-claude-unsloth|ermiaazarkhalili/Qwen3.5-9B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"

    # ── Granite 4.1 series (IBM dense decoder) ──
    "ermiaazarkhalili/Granite-4.1-3B-SFT-Claude-Opus-Reasoning-Unsloth|ibm-granite/granite-4.1-3b|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|granite41-3b-sft-claude-unsloth|ermiaazarkhalili/Granite-4.1-3B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Granite-4.1-8B-SFT-Claude-Opus-Reasoning-Unsloth|ibm-granite/granite-4.1-8b|full|SFT-Distillation|Claude-Opus-Reasoning|apache-2.0|granite41-8b-sft-claude-unsloth|ermiaazarkhalili/Granite-4.1-8B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/Granite-4.1-3B-Function-Calling-xLAM-Unsloth|ibm-granite/granite-4.1-3b|full|SFT|xLAM-60K|apache-2.0|granite41-3b-xlam-unsloth|ermiaazarkhalili/Granite-4.1-3B-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/Granite-4.1-8B-Function-Calling-xLAM-Unsloth|ibm-granite/granite-4.1-8b|full|SFT|xLAM-60K|apache-2.0|granite41-8b-xlam-unsloth|ermiaazarkhalili/Granite-4.1-8B-Function-Calling-xLAM-Unsloth-GGUF"

    # ── VibeThinker-3B (WeiboAI, Qwen2 reasoning) ──
    "ermiaazarkhalili/VibeThinker-3B-SFT-Claude-Opus-Reasoning-Unsloth|WeiboAI/VibeThinker-3B|full|SFT-Distillation|Claude-Opus-Reasoning|mit|vibethinker-3b-sft-claude-unsloth|ermiaazarkhalili/VibeThinker-3B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/VibeThinker-3B-Function-Calling-xLAM-Unsloth|WeiboAI/VibeThinker-3B|full|SFT|xLAM-60K|mit|vibethinker-3b-xlam-unsloth|ermiaazarkhalili/VibeThinker-3B-Function-Calling-xLAM-Unsloth-GGUF"

    # ── FastContext-4B (Microsoft, Qwen3; SFT_base + RL_base variants) ──
    "ermiaazarkhalili/FastContext-4B-SFT_base-SFT-Claude-Opus-Reasoning-Unsloth|microsoft/FastContext-1.0-4B-SFT|full|SFT-Distillation|Claude-Opus-Reasoning|mit|fastcontext-4b-sftbase-sft-claude-unsloth|ermiaazarkhalili/FastContext-4B-SFT_base-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/FastContext-4B-SFT_base-Function-Calling-xLAM-Unsloth|microsoft/FastContext-1.0-4B-SFT|full|SFT|xLAM-60K|mit|fastcontext-4b-sftbase-xlam-unsloth|ermiaazarkhalili/FastContext-4B-SFT_base-Function-Calling-xLAM-Unsloth-GGUF"
    "ermiaazarkhalili/FastContext-4B-RL_base-SFT-Claude-Opus-Reasoning-Unsloth|microsoft/FastContext-1.0-4B-RL|full|SFT-Distillation|Claude-Opus-Reasoning|mit|fastcontext-4b-rlbase-sft-claude-unsloth|ermiaazarkhalili/FastContext-4B-RL_base-SFT-Claude-Opus-Reasoning-Unsloth-GGUF"
    "ermiaazarkhalili/FastContext-4B-RL_base-Function-Calling-xLAM-Unsloth|microsoft/FastContext-1.0-4B-RL|full|SFT|xLAM-60K|mit|fastcontext-4b-rlbase-xlam-unsloth|ermiaazarkhalili/FastContext-4B-RL_base-Function-Calling-xLAM-Unsloth-GGUF"
)

if [[ ! -x "$SINGLE_MODEL_WRAPPER" ]]; then
    echo "[ERROR] $SINGLE_MODEL_WRAPPER not found or not executable"
    exit 1
fi

echo "=========================================="
echo "GGUF dispatcher — cached toolchain pipeline"
echo "=========================================="
echo "Wrapper:   $SINGLE_MODEL_WRAPPER"
echo "Quants:    $EXTRA_QUANTS"
echo "Filter:    ${FILTER:-<none>}"
echo ""

# Check each model's existence on the Hub once before submitting; skip the
# missing ones so we don't queue dozens of jobs that will exit [SKIP] in 5s.
source "$VENV/bin/activate"

TOTAL=${#MODELS[@]}
SUBMITTED=0
SKIPPED_MISSING=0
SKIPPED_FILTER=0
SUBMITTED_IDS=()

for i in "${!MODELS[@]}"; do
    IFS='|' read -r HUB_ID BASE_MODEL MODEL_TYPE METHOD DATASET LICENSE SHORT_NAME OUTPUT_REPO <<< "${MODELS[$i]}"

    # FILTER: substring match against SHORT_NAME
    if [[ -n "$FILTER" && "$SHORT_NAME" != *"$FILTER"* ]]; then
        SKIPPED_FILTER=$((SKIPPED_FILTER + 1))
        continue
    fi

    # Check Hub existence (fast: 1 HTTP HEAD equivalent via HfApi)
    EXISTS=$(python3 -c "
from huggingface_hub import HfApi
try:
    HfApi().repo_info('$HUB_ID', repo_type='model')
    print('yes')
except Exception:
    print('no')
" 2>/dev/null)

    if [[ "$EXISTS" != "yes" ]]; then
        printf "  [skip] %-50s  not yet on Hub\n" "$SHORT_NAME"
        SKIPPED_MISSING=$((SKIPPED_MISSING + 1))
        continue
    fi

    # Export QUANTS in the shell environment rather than on the sbatch command
    # line — SLURM's --export= uses commas as separators and would truncate the
    # comma-separated quant list to only the first quant.
    export HUB_ID BASE_MODEL OUTPUT_REPO
    export QUANTS="$EXTRA_QUANTS"
    JOB_ID=$(sbatch --parsable \
        --job-name="gguf-$SHORT_NAME" \
        --export=ALL \
        "$SINGLE_MODEL_WRAPPER" 2>&1 | tail -n1)

    if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        printf "  [OK  ] %-50s  job=%s\n" "$SHORT_NAME" "$JOB_ID"
        SUBMITTED_IDS+=("$JOB_ID")
        SUBMITTED=$((SUBMITTED + 1))
    else
        printf "  [FAIL] %-50s  sbatch error: %s\n" "$SHORT_NAME" "$JOB_ID"
    fi
done

echo ""
echo "=========================================="
echo "Dispatcher complete"
echo "  Total models:       $TOTAL"
echo "  Submitted:          $SUBMITTED"
echo "  Skipped (filter):   $SKIPPED_FILTER"
echo "  Skipped (not on Hub): $SKIPPED_MISSING"
echo "=========================================="
if (( SUBMITTED > 0 )); then
    echo "Job IDs: ${SUBMITTED_IDS[*]}"
    echo "Monitor: squeue -u \$USER | grep gguf-"
fi
