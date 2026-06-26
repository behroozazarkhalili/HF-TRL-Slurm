#!/bin/bash
#SBATCH --job-name=clean-fable-2m
#SBATCH --account=def-maxwl_cpu
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Clean Dataset D (Complete-FABLE.5-traces-2M) → standalone PRIVATE clean dataset.
# Quality-routed (evidence-derived keep-set, 8 genuine sources + TheFusionCube
# non-decoy). Full labeled superset (~824K rows) with source_dataset column.
# Uses the crosscoders .venv (pyarrow+polars+datasets) — thread-pinned.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
VPY="/project/6014832/ermia/transformer-circuits/transformer-crosscoders/.venv/bin/python"

echo "=========================================="
echo "CLEAN: Fable-5 2M → private clean dataset"
echo "=========================================="
echo "Node:  $SLURMD_NODENAME"
echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date -Iseconds)"

# Thread-pin BEFORE any numpy/polars import (login+compute RLIMIT_NPROC) — and
# polars uses rayon, so pin RAYON/POLARS too.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=4 POLARS_MAX_THREADS=4
export HF_HUB_DISABLE_PROGRESS_BARS=1
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
    --source D \
    --out_jsonl "/scratch/$USER/fable-clean/D_Complete2M_clean.jsonl" \
    --out_repo "ermiaazarkhalili/Fable-5-Complete-2M-Clean" \
    --push
EXIT=$?

echo "=========================================="
if (( EXIT == 0 )); then echo "[PASS] D clean+push complete"; else echo "[FAIL] exit=$EXIT"; fi
echo "End: $(date -Iseconds)"
exit $EXIT
