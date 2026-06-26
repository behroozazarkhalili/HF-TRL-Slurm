#!/bin/bash
#SBATCH --job-name=train-fastcontext-4b-sft-sft
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Real (non-smoke) training: FastContext-4B-SFT_base SFT distillation
# Notebook has SMOKE_TEST=False committed; runs full epoch.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-train-fastcontext-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TRAIN: FastContext-4B-SFT_base SFT distillation (full epoch)"
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

NB_INPUT="$NB_DIR/sft_distillation_fastcontext-4b-sft_unsloth.ipynb"
NB_OUTPUT="$OUTPUT_DIR/sft_fastcontext_4b_sft_train_output.ipynb"

if [[ ! -f "$NB_INPUT" ]]; then
    echo "[FAIL] Notebook not found: $NB_INPUT" >&2
    exit 1
fi

# Verify SMOKE_TEST is False in committed notebook (safety check)
python3 - <<PY
import json, sys
nb = json.load(open("$NB_INPUT"))
for c in nb["cells"]:
    s = c.get("source", "")
    if isinstance(s, list): s = "".join(s)
    if "SMOKE_TEST: bool = True" in s:
        print("[FAIL] Notebook has SMOKE_TEST=True committed — refusing to launch real run", file=sys.stderr)
        sys.exit(1)
print("[OK] SMOKE_TEST=False confirmed")
PY

echo ""
echo "[....] Running via papermill..."
PYTHONUNBUFFERED=1 papermill "$NB_INPUT" "$NB_OUTPUT" \
    --kernel python3 --progress-bar --log-output --cwd "$NB_DIR"
EXIT=$?

echo ""
echo "=========================================="
if (( EXIT == 0 )); then
    echo "[PASS] training complete"
else
    echo "[FAIL] training exit=$EXIT"
fi
echo "Output notebook: $NB_OUTPUT"
echo "End: $(date -Iseconds)"
exit $EXIT
