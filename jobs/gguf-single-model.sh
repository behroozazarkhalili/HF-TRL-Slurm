#!/bin/bash
# =============================================================================
# GGUF conversion for a SINGLE model using cached llama.cpp toolchain.
# No rebuild, no GPU. Submits one CPU job per invocation.
#
# Usage:
#   sbatch --export=ALL,HUB_ID=...,BASE_MODEL=...,OUTPUT_REPO=...,QUANTS=... \
#          jobs/gguf-single-model.sh
#
# Env knobs (all optional except HUB_ID and OUTPUT_REPO):
#   HUB_ID        — HF repo or local path to convert (required)
#   OUTPUT_REPO   — HF repo to upload GGUFs to (required)
#   BASE_MODEL    — base model ID if HUB_ID is a LoRA adapter
#   QUANTS        — comma-sep quant types (default: Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0)
#   LLAMA_CPP_DIR — cached toolchain (default: /scratch/$USER/tools/llama.cpp)
# =============================================================================
#SBATCH --job-name=gguf-single
#SBATCH --account=def-maxwl_cpu
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
SCRIPT_DIR="$PROJECT_DIR/.claude/skills/slurm-model-trainer/scripts"
VENV="/scratch/ermia/venvs/hf_env"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/scratch/ermia/tools/llama.cpp}"
QUANTS="${QUANTS:-Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0}"

: "${HUB_ID:?HUB_ID must be set}"
: "${OUTPUT_REPO:?OUTPUT_REPO must be set}"

echo "=========================================="
echo "GGUF (cached toolchain) — single model"
echo "=========================================="
echo "Node:        $SLURMD_NODENAME"
echo "Job:         $SLURM_JOB_ID"
echo "Start:       $(date -Iseconds)"
echo "Model:       $HUB_ID"
echo "Base:        ${BASE_MODEL:-<none>}"
echo "Output:      $OUTPUT_REPO"
echo "Quants:      $QUANTS"
echo "Toolchain:   $LLAMA_CPP_DIR"
echo "=========================================="

module load StdEnv/2023 gcc arrow python/3.11.5
source "$VENV/bin/activate"

export SCRATCH="${SCRATCH:-/scratch/$USER}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "$PROJECT_DIR/.env"
fi

WORK_DIR="$SCRATCH/gguf-work/$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

ARGS=(
    --model "$HUB_ID"
    --output_repo "$OUTPUT_REPO"
    --quantizations "$QUANTS"
    --llama_cpp_dir "$LLAMA_CPP_DIR"
    --work_dir "$WORK_DIR"
)
if [[ -n "${BASE_MODEL:-}" ]]; then
    ARGS+=(--base_model "$BASE_MODEL")
fi

PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/convert_gguf_direct.py" "${ARGS[@]}"
EXIT=$?

echo "=========================================="
echo "End:         $(date -Iseconds)"
echo "Exit:        $EXIT"
# Cleanup scratch work dir unless debug requested
if [[ $EXIT -eq 0 && -z "${KEEP_WORK:-}" ]]; then
    rm -rf "$WORK_DIR"
    echo "Work dir:    removed ($WORK_DIR)"
else
    echo "Work dir:    preserved ($WORK_DIR)"
fi
echo "=========================================="
exit $EXIT
