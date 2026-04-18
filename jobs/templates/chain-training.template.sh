#!/bin/bash
# =============================================================================
# BLUEPRINT: Multi-stage GRPO chain training on DRAC Fir (SLURM + TRL)
# =============================================================================
#
# What this template does:
#   Generates and submits N sequential SLURM jobs (stages) that chain a
#   GRPO training run — each stage continues from the previous stage's LoRA
#   adapter, effectively splitting one long training into resumable segments.
#
# When to use this template:
#   - GRPO/RLHF training with LoRA adapters on DRAC Fir.
#   - When wall-time limits force a single long run into multiple sub-runs.
#   - Any experiment where stage N+1 needs stage N's adapter.
#
# How to use:
#   1. Copy this file to the target experiment name:
#        cp jobs/templates/chain-training.template.sh jobs/my-experiment.sh
#   2. Edit the "EDIT ME" block below (marked with # >>>>>).
#   3. Run locally to submit the chain:
#        bash jobs/my-experiment.sh
#   4. Monitor: squeue -u $USER | grep <EXP_NAME>
#
# What gets derived automatically from EXP_NAME:
#   - CHAIN_DIR         → /scratch/$USER/outputs/$EXP_NAME
#   - SBATCH job-name   → <EXP_NAME>-stage-<N>
#   - OUTPUT_DIR        → $CHAIN_DIR/stage<N>-<JOB_ID>
#   - Monitor grep      → grep <EXP_NAME>
#
# Prerequisites:
#   - $PROJECT_DIR/jobs/lib/chain_utils.sh exists (provides bootstrap_training_env,
#     submit_job, resolve_prev_adapter_arg, save_stage_adapter, logging helpers).
#   - Python venv at /scratch/ermia/venvs/hf_env (override via bootstrap_training_env arg).
#   - train_grpo.py at TRAIN_GRPO_SCRIPT path below.
#   - $PROJECT_DIR/.env with HF_TOKEN etc.
#
# Safety:
#   - `rm -rf "$CHAIN_DIR"` is guarded by 3 invariant checks (USER non-empty,
#     EXP_NAME non-empty, CHAIN_DIR under /scratch/$USER/outputs/).
#   - Each generated stage script is `bash -n`-checked before submission.
#
# Architecture:
#   outer generator (this script)
#     ├─ sources chain_utils.sh                 # submit_job, logging
#     ├─ for each stage:
#     │     generate_stage_script()              # heredoc + sed placeholder injection
#     │     submit_job() with afterok dependency # from chain_utils.sh
#     └─ prints stage→job mapping for monitoring
#
#   each generated stage script
#     ├─ sources chain_utils.sh
#     ├─ bootstrap_training_env()                # module load + venv + HF_HOME + .env
#     ├─ resolve_prev_adapter_arg()              # reads stage-N-1.adapter pointer
#     ├─ python train_grpo.py --flags...         # training
#     └─ save_stage_adapter()                    # writes stage-N.adapter pointer
#
# Extending the template:
#   - Add a new CLI flag: edit the heredoc + add the placeholder to `subs` array.
#   - Change SBATCH resources: edit the SLURM_* constants below (no heredoc edit).
#   - Different venv (Unsloth, TRL): bootstrap_training_env hf_unsloth
#   - Different user's venv: VENV_USER=ermia bash my-chain.sh (or second arg to
#     bootstrap_training_env inside the heredoc).
#
# Failure recovery:
#   - If stage N OOMs:   bump SLURM_MEM or SLURM_GRES, then FORCE=1 bash my-chain.sh
#                        (re-runs from stage 1) — or manually resubmit stage-N.sh:
#                          sbatch --dependency=afterok:<prev-job-id> \
#                                 "$CHAIN_DIR/stage-N.sh"
#   - If stage N TIMEOUT: bump SLURM_TIME (move to b4/b5 partition if needed),
#                         FORCE=1 bash my-chain.sh.
#   - If stage N crashes but adapter saved:  skip to the next stage manually via
#                         sbatch "$CHAIN_DIR/stage-(N+1).sh" — adapter pointer
#                         stage-N.adapter is already written.
#   - Logs: $LOGS_DIR/<job-name>-<job-id>.out and .err.
#   - Monitor: squeue -u $USER | grep $EXP_NAME
#
# Re-run safety:
#   - `rm -rf "$CHAIN_DIR"` is guarded: refuses non-empty CHAIN_DIR unless FORCE=1.
#   - Three invariant checks (USER non-empty, EXP_NAME regex, CHAIN_DIR prefix).
#   - EXP_NAME format restricted to [a-zA-Z0-9._-]+ to prevent shell-meta bugs.
# =============================================================================

set -euo pipefail

# ── Configuration (readonly constants) ───────────────────────────────

# >>>>> EDIT ME: project + cluster identity (usually invariant per site) >>>>>
# PROJECT_DIR: absolute path to your clone of the HF-TRL repo.
# CHAIN_UTILS: path to the shared module. Leave as-is unless you move the file.
readonly PROJECT_DIR="${PROJECT_DIR:-/project/6014832/ermia/HF-TRL}"
readonly CHAIN_UTILS="$PROJECT_DIR/jobs/lib/chain_utils.sh"
readonly LOGS_DIR="$PROJECT_DIR/logs"
readonly TRAIN_GRPO_SCRIPT="$PROJECT_DIR/.claude/skills/slurm-model-trainer/scripts/train_grpo.py"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> EDIT ME: experiment identity (single source of truth) >>>>>>>>>
# Rename EXP_NAME alone to retarget the whole chain — CHAIN_DIR,
# SBATCH --job-name, OUTPUT_DIR, and the monitor command all follow.
# Env override supported: EXP_NAME=foo bash my-chain.sh
: "${EXP_NAME:=granite-chain-5sample}"
readonly EXP_NAME
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

readonly CHAIN_DIR="/scratch/$USER/outputs/$EXP_NAME"
readonly JOB_NAME_PREFIX="$EXP_NAME"

# >>>>> EDIT ME: SLURM resource defaults (used inside the heredoc) >>>>
# Passed to every stage's SBATCH directives via sed injection.
readonly SLURM_ACCOUNT="def-maxwl_gpu"
readonly SLURM_PARTITION="gpubase_bygpu_b3"   # b1 ≤3h (smoke), b3 ≤24h (prod), b5 ≤7d
readonly SLURM_TIME="0-06:00:00"              # per-stage wall time
readonly SLURM_GRES="gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
readonly SLURM_MEM="32G"
readonly SLURM_CPUS=4
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> EDIT ME: model + dataset + chain structure >>>>>>>>>>>>>>>>>>>>
readonly MODEL_NAME='ibm-granite/granite-4.0-micro'
readonly DATASET_NAME='AI-MO/NuminaMath-CoT'
readonly NUM_STAGES=3
readonly BASE_SEED=42
readonly STAGE_SAMPLES=2000       # samples per stage (production default; 5-50 for smoke)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> EDIT ME: training hyperparameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
readonly BATCH_SIZE=2
readonly GRAD_ACCUM=1
readonly LEARNING_RATE=1e-06
readonly LORA_R=16
readonly LORA_ALPHA=32
readonly MAX_COMPLETION_LENGTH=512
readonly MAX_PROMPT_LENGTH=256
readonly NUM_GENERATIONS=2
readonly REWARD_TYPE='combined'
readonly NUM_TRAIN_EPOCHS=1
readonly SAVE_STRATEGY='steps'    # 'no' | 'steps' | 'epoch' — 'no' is UNSAFE for >100 samples
readonly SAVE_STEPS=100
readonly SAVE_TOTAL_LIMIT=2
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ── CLI flags (unofficial; set FORCE=1 to allow destructive re-runs) ─
FORCE="${FORCE:-0}"

# ── Load shared chain helpers (submit_job, log helpers) ──────────────
[[ -r "$CHAIN_UTILS" ]] || { printf 'ERROR: chain_utils.sh not found at %s\n' "$CHAIN_UTILS" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CHAIN_UTILS"

# Outer-generator logging (chain_utils uses STAGE_NUM tag; we have none here)
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Generate one stage SLURM script via quoted-heredoc + sed injection
# Usage: generate_stage_script <stage_num> <seed> <output_path>
generate_stage_script() {
    local stage="$1"
    local seed="$2"
    local out="$3"

    # Quoted heredoc — no escaping needed inside (all $VAR stay literal until sed)
    cat > "$out" << 'STAGE_EOF'
#!/bin/bash
#SBATCH --job-name=__JOB_NAME_PREFIX__-stage-__STAGE__
#SBATCH --account=__SLURM_ACCOUNT__
#SBATCH --time=__SLURM_TIME__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=__SLURM_CPUS__
#SBATCH --mem=__SLURM_MEM__
#SBATCH --gres=__SLURM_GRES__
#SBATCH --partition=__SLURM_PARTITION__
#SBATCH --exclude=fc10713  # Broken MIG config on this node; remove once DRAC repairs
#SBATCH --output=__LOGS_DIR__/%x-%j.out
#SBATCH --error=__LOGS_DIR__/%x-%j.err

set -euo pipefail

# ── Runtime constants (sed-injected) ─────────────────────────────────
STAGE_NUM=__STAGE__
NUM_STAGES=__NUM_STAGES__
CHAIN_DIR="__CHAIN_DIR__"

# ── Load reusable helpers ────────────────────────────────────────────
# shellcheck disable=SC1091
source "__CHAIN_UTILS__"

log_stage_info "=== CHAIN STAGE $STAGE_NUM / $NUM_STAGES (Job $SLURM_JOB_ID) ==="
log_stage_info "Node: $SLURMD_NODENAME | $(date)"

bootstrap_training_env hf_env

export OUTPUT_DIR="$CHAIN_DIR/stage${STAGE_NUM}-$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR" "__LOGS_DIR__"

# Resolve adapter from previous stage (empty arg for stage 1)
ADAPTER_ARG=$(resolve_prev_adapter_arg "$CHAIN_DIR" "$STAGE_NUM")

log_stage_info "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log_stage_info "Output: $OUTPUT_DIR"

# shellcheck disable=SC2086  # $ADAPTER_ARG intentionally word-splits to "" or "--continue_from_adapter PATH"
python __TRAIN_GRPO_SCRIPT__ \
    --model_name_or_path __MODEL_NAME__ \
    --dataset_name __DATASET_NAME__ \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs __NUM_TRAIN_EPOCHS__ \
    --per_device_train_batch_size __BATCH_SIZE__ \
    --gradient_accumulation_steps __GRAD_ACCUM__ \
    --learning_rate __LEARNING_RATE__ \
    --bf16 \
    --gradient_checkpointing \
    --lora_r __LORA_R__ \
    --lora_alpha __LORA_ALPHA__ \
    --lora_dropout 0.05 \
    --save_strategy __SAVE_STRATEGY__ \
    --save_steps __SAVE_STEPS__ \
    --save_total_limit __SAVE_TOTAL_LIMIT__ \
    --logging_steps 1 \
    --streaming \
    --max_samples __STAGE_SAMPLES__ \
    --seed __SEED__ \
    --max_completion_length __MAX_COMPLETION_LENGTH__ \
    --max_prompt_length __MAX_PROMPT_LENGTH__ \
    --num_generations __NUM_GENERATIONS__ \
    --reward_type __REWARD_TYPE__ \
    $ADAPTER_ARG

# set -e above aborts on training failure; record success pointer for next stage
save_stage_adapter "$CHAIN_DIR" "$STAGE_NUM" "$OUTPUT_DIR"
log_stage_info "Stage $STAGE_NUM complete!"
STAGE_EOF

    # Inject build-time constants — __KEY__ placeholders -> values
    # Add a new config here and in the heredoc above; nothing else to touch.
    local -A subs=(
        [STAGE]="$stage"                    [NUM_STAGES]="$NUM_STAGES"
        [CHAIN_DIR]="$CHAIN_DIR"            [CHAIN_UTILS]="$CHAIN_UTILS"
        [LOGS_DIR]="$LOGS_DIR"
        [JOB_NAME_PREFIX]="$JOB_NAME_PREFIX"
        [TRAIN_GRPO_SCRIPT]="$TRAIN_GRPO_SCRIPT"
        [SLURM_ACCOUNT]="$SLURM_ACCOUNT"    [SLURM_PARTITION]="$SLURM_PARTITION"
        [SLURM_TIME]="$SLURM_TIME"          [SLURM_GRES]="$SLURM_GRES"
        [SLURM_MEM]="$SLURM_MEM"            [SLURM_CPUS]="$SLURM_CPUS"
        [SEED]="$seed"                      [STAGE_SAMPLES]="$STAGE_SAMPLES"
        [MODEL_NAME]="$MODEL_NAME"          [DATASET_NAME]="$DATASET_NAME"
        [BATCH_SIZE]="$BATCH_SIZE"          [GRAD_ACCUM]="$GRAD_ACCUM"
        [LEARNING_RATE]="$LEARNING_RATE"
        [LORA_R]="$LORA_R"                  [LORA_ALPHA]="$LORA_ALPHA"
        [MAX_COMPLETION_LENGTH]="$MAX_COMPLETION_LENGTH"
        [MAX_PROMPT_LENGTH]="$MAX_PROMPT_LENGTH"
        [NUM_GENERATIONS]="$NUM_GENERATIONS"
        [REWARD_TYPE]="$REWARD_TYPE"
        [NUM_TRAIN_EPOCHS]="$NUM_TRAIN_EPOCHS"
        [SAVE_STRATEGY]="$SAVE_STRATEGY"
        [SAVE_STEPS]="$SAVE_STEPS"
        [SAVE_TOTAL_LIMIT]="$SAVE_TOTAL_LIMIT"
    )
    local -a sed_args=()
    local key
    for key in "${!subs[@]}"; do
        sed_args+=(-e "s|__${key}__|${subs[$key]}|g")
    done
    sed -i "${sed_args[@]}" "$out"

    bash -n "$out" || die "syntax error in $out"
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
    # Note: $CHAIN_UTILS was already validated and sourced at top of file.

    # Safety guards — refuse to `rm -rf` unless CHAIN_DIR is unambiguously ours.
    [[ "$USER" == *[!/]* ]] || die "USER env is empty/invalid: '$USER'"
    [[ -n "$EXP_NAME" && "$EXP_NAME" != "/" ]] \
        || die "EXP_NAME empty or dangerous: '$EXP_NAME'"
    [[ "$EXP_NAME" =~ ^[a-zA-Z0-9._-]+$ ]] \
        || die "EXP_NAME must match [a-zA-Z0-9._-]+ (no spaces, pipes, or shell meta): '$EXP_NAME'"
    [[ "$CHAIN_DIR" == "/scratch/$USER/outputs/"* ]] \
        || die "CHAIN_DIR sanity check failed (must be under /scratch/\$USER/outputs/): $CHAIN_DIR"

    # Refuse to clobber a populated CHAIN_DIR unless FORCE=1 is explicitly set.
    if [[ -d "$CHAIN_DIR" ]] && [[ -n "$(ls -A "$CHAIN_DIR" 2>/dev/null)" ]]; then
        if [[ "$FORCE" != "1" ]]; then
            die "CHAIN_DIR is non-empty: $CHAIN_DIR
  Refusing to rm -rf existing chain state.
  To intentionally re-run from stage 1 (loses all prior adapters), set FORCE=1:
    FORCE=1 bash $0"
        fi
        echo "[WARN] FORCE=1 set — deleting existing $CHAIN_DIR"
    fi

    rm -rf "$CHAIN_DIR"
    mkdir -p "$CHAIN_DIR" "$LOGS_DIR"

    echo "=== CHAIN JOB SUBMISSION: $NUM_STAGES stages x $STAGE_SAMPLES samples ==="
    echo "    EXP_NAME:   $EXP_NAME"
    echo "    CHAIN_DIR:  $CHAIN_DIR"
    echo "    LOGS_DIR:   $LOGS_DIR"
    echo "    Partition:  $SLURM_PARTITION  |  Time/stage: $SLURM_TIME"
    echo ""

    local prev_job_id=""
    local stage seed stage_script job_id dep_note

    for ((stage=1; stage<=NUM_STAGES; stage++)); do
        seed=$((BASE_SEED + stage))
        stage_script="$CHAIN_DIR/stage-${stage}.sh"

        generate_stage_script "$stage" "$seed" "$stage_script"

        job_id=$(submit_job "$stage_script" "$prev_job_id")
        dep_note=$([[ -n "$prev_job_id" ]] && printf '(after %s)' "$prev_job_id" || printf '(no dep)')

        echo "Stage $stage: job $job_id $dep_note"
        prev_job_id="$job_id"
    done

    echo ""
    echo "Chain submitted: $NUM_STAGES stages x $STAGE_SAMPLES samples"
    echo "Monitor: squeue -u \$USER | grep $JOB_NAME_PREFIX"
}

main "$@"
