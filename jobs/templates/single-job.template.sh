#!/bin/bash
# =============================================================================
# BLUEPRINT: Single-job GRPO training on DRAC Fir (SLURM + TRL)
# =============================================================================
#
# What this template does:
#   ONE SLURM job — runs a full GRPO training end-to-end in a single allocation.
#   No stage chaining, no adapter resume. The whole run lives inside one
#   SBATCH submission, bounded by --time.
#
# When to use this template:
#   - Training fits comfortably in a single time window (≤ 7 days on b5 partition).
#   - No mid-run checkpointing/resume required.
#   - Simpler operationally than chain-training — fewer moving parts.
#   - Use chain-training.template.sh instead if wall-time constraints force you
#     to split one long run across multiple SLURM jobs.
#
# How to use:
#   1. Copy: cp jobs/templates/single-job.template.sh jobs/my-experiment.sh
#   2. Edit the "EDIT ME" blocks below (marked with # >>>>>).
#   3. Submit:  sbatch jobs/my-experiment.sh
#   4. Monitor: squeue -u $USER | grep <EXP_NAME>
#
# What gets derived automatically from EXP_NAME:
#   - Job name           → <EXP_NAME>
#   - OUTPUT_DIR         → /scratch/$USER/outputs/$EXP_NAME-$SLURM_JOB_ID
#   - Hub model ID       → $HF_USER/<EXP_NAME>        (if HF push enabled)
#
# Prerequisites:
#   - $PROJECT_DIR/jobs/lib/chain_utils.sh  (bootstrap_training_env helper)
#   - venv at /scratch/ermia/venvs/hf_env
#   - train_grpo.py at TRAIN_GRPO_SCRIPT path below
#   - $PROJECT_DIR/.env with HF_TOKEN etc.
#
# Safety:
#   - No destructive `rm -rf` — the OUTPUT_DIR is job-ID-scoped so reruns
#     never overwrite prior runs.
#
# Partition / resource guide (DRAC Fir H100):
#   - gpubase_bygpu_b1  →  ≤  3h  (use for smoke tests)
#   - gpubase_bygpu_b2  →  ≤ 12h
#   - gpubase_bygpu_b3  →  ≤ 24h
#   - gpubase_bygpu_b4  →  ≤ 72h
#   - gpubase_bygpu_b5  →  ≤ 7d   (use for full runs)
#
# To convert this to a SMOKE TEST:
#   - Set MAX_SAMPLES=10, MAX_STEPS=5.
#   - Change --partition=gpubase_bygpu_b1 and --time=0-00:30:00.
#   - Change EXP_NAME to "smoke-<your-exp>".
#
# Failure recovery:
#   - If the job OOMs:    bump #SBATCH --mem=<higher> (or --gres to a larger GPU),
#                         then re-submit:  sbatch jobs/my-experiment.sh
#                         (A new OUTPUT_DIR is created per SLURM_JOB_ID, no clobber.)
#   - If the job TIMEOUTs: bump #SBATCH --time and optionally --partition.
#   - If Hub push fails:   training already completed. Retry push manually:
#                            huggingface-cli upload $HUB_MODEL_ID $OUTPUT_DIR
#   - Duplicate submission:  refused by the concurrent-run guard below
#                            (prevents 2 jobs racing to push the same Hub repo).
#   - Logs: $LOGS_DIR/<job-name>-<job-id>.out and .err.
#   - Monitor: squeue -u $USER | grep <EXP_NAME>
#
# Overrides without editing the file:
#   EXP_NAME=foo sbatch jobs/my-experiment.sh    # change experiment name
#   PROJECT_DIR=/path/to/repo sbatch ...          # run with a different repo clone
#   VENV_USER=ermia sbatch ...                    # use someone else's venv
# =============================================================================

# >>>>> EDIT ME: SBATCH directives >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# NOTE: SBATCH directives are parsed from the script BEFORE any shell runs,
# so they cannot reference shell variables. If you change EXP_NAME below,
# manually update #SBATCH --job-name to match (or use
# `sbatch --job-name="$EXP_NAME" <this-script>` to override at submit time).
#SBATCH --job-name=granite-4.0-micro-grpo-numinamath-10k
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --exclude=fc10713  # Broken MIG config on this node; remove once DRAC repairs
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

set -euo pipefail

# ── Configuration (readonly constants) ───────────────────────────────

# >>>>> EDIT ME: project identity (usually invariant per site) >>>>>>>>
# PROJECT_DIR must match the --output/--error SBATCH paths above.
readonly PROJECT_DIR="${PROJECT_DIR:-/project/6014832/ermia/HF-TRL}"
readonly CHAIN_UTILS="$PROJECT_DIR/jobs/lib/chain_utils.sh"
readonly TRAIN_GRPO_SCRIPT="$PROJECT_DIR/.claude/skills/slurm-model-trainer/scripts/train_grpo.py"
readonly LOGS_DIR="$PROJECT_DIR/logs"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> EDIT ME: experiment identity (single source of truth) >>>>>>>>>
# Keep in sync with #SBATCH --job-name above (cannot derive automatically).
# Env override: EXP_NAME=foo sbatch my-experiment.sh
: "${EXP_NAME:=granite-4.0-micro-grpo-numinamath-10k}"
readonly EXP_NAME
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> EDIT ME: model + dataset + sample budget >>>>>>>>>>>>>>>>>>>>>>
readonly MODEL_NAME='ibm-granite/granite-4.0-micro'
readonly DATASET_NAME='AI-MO/NuminaMath-CoT'
readonly MAX_SAMPLES=10000
readonly SEED=42
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> EDIT ME: training hyperparameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
readonly BATCH_SIZE=2
readonly GRAD_ACCUM=8
readonly LEARNING_RATE=1e-06
readonly NUM_EPOCHS=1
readonly LORA_R=16
readonly LORA_ALPHA=32
readonly MAX_COMPLETION_LENGTH=2048
readonly MAX_PROMPT_LENGTH=512
readonly NUM_GENERATIONS=4
readonly REWARD_TYPE='combined'
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> EDIT ME (optional): HuggingFace Hub push >>>>>>>>>>>>>>>>>>>>>>
# Set PUSH_TO_HUB=false to skip Hub upload. HF_USER is your namespace.
readonly PUSH_TO_HUB="true"
readonly HF_USER="ermiaazarkhalili"
readonly HUB_MODEL_ID="$HF_USER/$EXP_NAME"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ── Output paths (job-ID-scoped; reruns never clobber) ───────────────
readonly OUTPUT_DIR="/scratch/$USER/outputs/${EXP_NAME}-${SLURM_JOB_ID:-local}"

# ── Load shared helpers ──────────────────────────────────────────────
[[ -r "$CHAIN_UTILS" ]] || { printf 'ERROR: chain_utils.sh not found at %s\n' "$CHAIN_UTILS" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CHAIN_UTILS"

# Outer logging (no STAGE_NUM context — single job)
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ── EXP_NAME format validation (defense vs. typos / shell meta) ──────
[[ "$EXP_NAME" =~ ^[a-zA-Z0-9._-]+$ ]] \
    || die "EXP_NAME must match [a-zA-Z0-9._-]+ (no spaces/pipes/shell meta): '$EXP_NAME'"

# ── Concurrent-run guard (Hub push collision prevention) ─────────────
# If another job with same --job-name is RUNNING/PENDING, refuse to
# submit a duplicate (would race on the same HUB_MODEL_ID).
if command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    DUPES=$(squeue -u "$USER" -h --format="%i %j" 2>/dev/null \
                | awk -v me="$SLURM_JOB_ID" -v jn="$EXP_NAME" \
                      '$1 != me && $2 == jn {c++} END {print c+0}')
    if (( DUPES > 0 )); then
        die "Duplicate job detected: $DUPES other job(s) with --job-name=$EXP_NAME already queued/running.
  Cancel the duplicates first, or rename EXP_NAME to run a parallel experiment."
    fi
fi

# ── Banner ───────────────────────────────────────────────────────────
echo "=========================================="
echo "JOB: ${SLURM_JOB_NAME:-$EXP_NAME} (${SLURM_JOB_ID:-local})"
echo "=========================================="
echo "  Model:    $MODEL_NAME"
echo "  Dataset:  $DATASET_NAME  (max_samples=$MAX_SAMPLES)"
echo "  Node:     ${SLURMD_NODENAME:-local}"
echo "  Output:   $OUTPUT_DIR"
echo "  Logs:     $LOGS_DIR"
if [[ "$PUSH_TO_HUB" == "true" ]]; then
    echo "  HubPush:  enabled → $HUB_MODEL_ID"
else
    echo "  HubPush:  disabled"
fi
echo "  Start:    $(date)"
echo "=========================================="

# ── Environment bootstrap ────────────────────────────────────────────
# Loads modules, activates venv, exports HF_HOME & PYTHONUNBUFFERED, reads .env
bootstrap_training_env hf_env

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# ── Build training args (array form — robust to spaces & empties) ────
TRAIN_ARGS=(
    --model_name_or_path      "$MODEL_NAME"
    --dataset_name            "$DATASET_NAME"
    --output_dir              "$OUTPUT_DIR"
    --num_train_epochs        "$NUM_EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate           "$LEARNING_RATE"
    --bf16
    --gradient_checkpointing
    --lora_r                  "$LORA_R"
    --lora_alpha              "$LORA_ALPHA"
    --lora_dropout            0.05
    --save_strategy           steps
    --save_steps              500
    --save_total_limit        3
    --logging_steps           10
    --streaming
    --max_samples             "$MAX_SAMPLES"
    --seed                    "$SEED"
    --max_completion_length   "$MAX_COMPLETION_LENGTH"
    --max_prompt_length       "$MAX_PROMPT_LENGTH"
    --num_generations         "$NUM_GENERATIONS"
    --reward_type             "$REWARD_TYPE"
)

if [[ "$PUSH_TO_HUB" == "true" ]]; then
    TRAIN_ARGS+=(--push_to_hub --hub_model_id "$HUB_MODEL_ID")
fi

# ── Train ────────────────────────────────────────────────────────────
python "$TRAIN_GRPO_SCRIPT" "${TRAIN_ARGS[@]}"

# ── Completion banner (reached only if training exited 0 — set -e) ───
echo ""
echo "=========================================="
echo "  Training complete!"
echo "  End:      $(date)"
echo "  Output:   $OUTPUT_DIR"
[[ "$PUSH_TO_HUB" == "true" ]] && echo "  Pushed:   https://huggingface.co/$HUB_MODEL_ID"
echo "=========================================="
