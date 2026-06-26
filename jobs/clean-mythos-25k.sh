#!/bin/bash
#SBATCH --job-name=clean-mythos-25k
#SBATCH --account=def-maxwl_cpu
#SBATCH --time=0-00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Clean Dataset A (mythos-25K synthetic) + push PRIVATE — full re-run on a
# compute node (login-node hf_xet push hit RLIMIT_NPROC thread-cap).
# Aggressive: strip canned preamble/closing, drop stub/fake-code, dedup.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
VPY="/project/6014832/ermia/transformer-circuits/transformer-crosscoders/.venv/bin/python"

echo "=========================================="
echo "CLEAN: mythos-25K → private clean dataset"
echo "Node:  $SLURMD_NODENAME   Job: $SLURM_JOB_ID   Start: $(date -Iseconds)"
echo "=========================================="

# Thread-pin (RLIMIT_NPROC) + disable hf_xet fast-path (its thread-spawn is what
# failed on the login node).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=2 POLARS_MAX_THREADS=2
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="/scratch/$USER/.cache/huggingface"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "$PROJECT_DIR/.env"
fi

PYTHONUNBUFFERED=1 "$VPY" "$PROJECT_DIR/scripts/clean_fable_dataset.py" \
    --source A \
    --out_jsonl "/scratch/$USER/fable-clean/A_Mythos25K_clean.jsonl" \
    --out_repo "ermiaazarkhalili/Mythos-25K-Clean" \
    --push
EXIT=$?

echo "=========================================="
if (( EXIT == 0 )); then echo "[PASS] A clean+push complete"; else echo "[FAIL] exit=$EXIT"; fi
echo "End: $(date -Iseconds)"
exit $EXIT
