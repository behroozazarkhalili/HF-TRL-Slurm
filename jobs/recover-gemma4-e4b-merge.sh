#!/bin/bash
#SBATCH --job-name=recover-gemma4-e4b-merge
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err

# =============================================================================
# Recover the Gemma4-E4B-SFT merge that Unsloth's in-place addmm_ crashed on
# (job 45310123). Training succeeded; adapter is intact. Merge via PEFT
# merge_and_unload on the fp16 base (routes around Unsloth's broken merge),
# then push merged model to the Hub.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
VENV="/scratch/ermia/venvs/hf_unsloth"

echo "=========================================="
echo "RECOVER MERGE: Gemma4-E4B-SFT"
echo "=========================================="
echo "Node:  $SLURMD_NODENAME"
echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date -Iseconds)"

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6
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

PYTHONUNBUFFERED=1 python "$PROJECT_DIR/scripts/tmp/recover_gemma4_merge.py"
EXIT=$?

echo "=========================================="
if (( EXIT == 0 )); then echo "[PASS] merge recovery complete"; else echo "[FAIL] exit=$EXIT"; fi
echo "End: $(date -Iseconds)"
exit $EXIT
