#!/bin/bash
# =============================================================================
# chain_utils.sh — Reusable helpers for SLURM chain-training stage scripts
#
# Source from any stage script:
#   source /project/6014832/ermia/HF-TRL/jobs/lib/chain_utils.sh
#
# Exposed functions:
#   resolve_prev_adapter_arg <chain_dir> <stage_num>
#       Prints "--continue_from_adapter <path>" to stdout for stage > 1,
#       empty string for stage 1. Exits non-zero on chain breakage.
#       Accepts both safetensors (modern) and bin (legacy) adapter formats.
#
#   load_env_file <env_file>
#       Sources a KEY=VAL .env file (quote-tolerant, comment-skipping).
#       No-op if file missing/unreadable.
#
#   save_stage_adapter <chain_dir> <stage_num> <adapter_dir>
#       Writes adapter pointer file for next stage to read.
#
#   log_stage_info / log_stage_error / stage_die
#       Tagged logging helpers (use env var STAGE_NUM for tag).
#       All logs go to stderr so function return values stay clean on stdout.
# =============================================================================

# Guard against double-sourcing
[[ -n "${_CHAIN_UTILS_SOURCED:-}" ]] && return 0
readonly _CHAIN_UTILS_SOURCED=1

# ── Logging ──────────────────────────────────────────────────────────
# All logs go to stderr — keeps stdout clean for functions that return values
# via command substitution (e.g. resolve_prev_adapter_arg).
log_stage_info()  { printf '[stage-%s] %s\n' "${STAGE_NUM:-?}" "$*" >&2; }
log_stage_error() { printf '[stage-%s ERROR] %s\n' "${STAGE_NUM:-?}" "$*" >&2; }
stage_die()       { log_stage_error "$@"; exit 1; }

# ── Env file loader (quote-tolerant) ─────────────────────────────────
# Usage: load_env_file <path>
# No-op if file missing; skips comments and blank lines; strips quotes.
load_env_file() {
    local env_file="${1:?env_file arg required}"
    [[ -r "$env_file" ]] || return 0

    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "$env_file"
}

# ── Adapter chain resolution ─────────────────────────────────────────
# Usage: resolve_prev_adapter_arg <chain_dir> <stage_num>
# Prints the --continue_from_adapter flag (or empty for stage 1).
# Exits on chain breakage with clear diagnostic.
#
# Accepts both safetensors (modern PEFT) and bin (legacy PEFT) adapter files.
resolve_prev_adapter_arg() {
    local chain_dir="${1:?chain_dir arg required}"
    local stage_num="${2:?stage_num arg required}"

    # Stage 1 has no previous adapter — return empty arg
    (( stage_num > 1 )) || { printf ''; return 0; }

    local prev=$((stage_num - 1))
    local pointer="$chain_dir/stage-${prev}.adapter"

    [[ -r "$pointer" ]] || stage_die "Previous adapter pointer missing: $pointer"

    local adapter_dir
    adapter_dir=$(<"$pointer")

    if [[ ! -f "$adapter_dir/adapter_model.safetensors" && \
          ! -f "$adapter_dir/adapter_model.bin" ]]; then
        log_stage_error "No adapter weights in: $adapter_dir"
        log_stage_error "  Expected: adapter_model.safetensors or adapter_model.bin"
        ls -la "$adapter_dir" >&2 2>/dev/null \
            || log_stage_error "  (directory missing)"
        exit 1
    fi

    log_stage_info "Loading adapter from stage $prev: $adapter_dir"
    printf -- '--continue_from_adapter %s' "$adapter_dir"
}

# ── Stage completion marker ──────────────────────────────────────────
# Usage: save_stage_adapter <chain_dir> <stage_num> <output_dir>
# Persists the output directory path so the next stage can find it.
save_stage_adapter() {
    local chain_dir="${1:?chain_dir arg required}"
    local stage_num="${2:?stage_num arg required}"
    local output_dir="${3:?output_dir arg required}"

    [[ -d "$chain_dir" ]] || mkdir -p "$chain_dir"
    printf '%s\n' "$output_dir" > "$chain_dir/stage-${stage_num}.adapter"
    log_stage_info "Adapter pointer saved: $chain_dir/stage-${stage_num}.adapter"
}

# ── DRAC training environment bootstrap ──────────────────────────────
# Usage: bootstrap_training_env [venv_name] [venv_user]
# Loads modules, activates venv, exports HF caches, loads project .env.
#
# Default venv: hf_env. Pass hf_unsloth for Unsloth jobs, hf_trl_env for TRL.
# Default venv_user resolution order (first non-empty wins):
#   1) 2nd positional arg
#   2) $VENV_USER env var
#   3) $USER (i.e. the running user's own /scratch/$USER/venvs/...)
# Override with: VENV_USER=ermia bootstrap_training_env hf_env
#            or: bootstrap_training_env hf_env ermia
bootstrap_training_env() {
    local venv="${1:-hf_env}"
    local venv_user="${2:-${VENV_USER:-$USER}}"
    local venv_path="/scratch/${venv_user}/venvs/${venv}"

    [[ -d "$venv_path" ]] || stage_die "venv not found: $venv_path (tried venv_user='$venv_user')"

    module load gcc arrow python/3.11.5
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"

    export SCRATCH="${SCRATCH:-/scratch/$USER}"
    export HF_HOME="$SCRATCH/.cache/huggingface"
    export TRANSFORMERS_CACHE="$HF_HOME/hub"
    export PYTHONUNBUFFERED=1

    # Honor PROJECT_DIR if set by caller; otherwise fall back to the original HF-TRL path.
    load_env_file "${PROJECT_DIR:-/project/6014832/ermia/HF-TRL}/.env"
    log_stage_info "Training env bootstrapped (venv=$venv, user=$venv_user)"
}

# ── SLURM job submission ─────────────────────────────────────────────
# Usage: job_id=$(submit_job <script> [<prev_job_id>])
# Submits via sbatch --parsable with optional afterok dependency.
# Returns job ID on stdout; errors to stderr; non-zero exit on failure.
submit_job() {
    local script="${1:?script path required}"
    local dep="${2:-}"
    local -a args=(--parsable)

    [[ -r "$script" ]] || { printf 'script not readable: %s\n' "$script" >&2; return 1; }
    [[ -n "$dep" ]] && args+=("--dependency=afterok:$dep")
    args+=("$script")

    local job_id
    job_id=$(sbatch "${args[@]}" 2>&1 | grep -oE '[0-9]+$') \
        || { printf 'sbatch failed for %s\n' "$script" >&2; return 1; }
    [[ -n "$job_id" ]] || { printf 'empty job ID from sbatch for %s\n' "$script" >&2; return 1; }

    printf '%s' "$job_id"
}
