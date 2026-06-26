#!/bin/bash
#SBATCH --job-name=train-fastcontext-4b-sft-fable-glint
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

# Real training: FastContext-4B-SFT on FABLE-5 (Fable5-Glint, 4,350 rows, 3 epochs).
# Notebook ships SMOKE_TEST=False; assistant-only masking applied inline. Checkpoints
# at SAVE_STEPS (resume on requeue). Walltime scaled to dataset size.

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/fable-train-fastcontext_4b_sft-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TRAIN: FastContext-4B-SFT on FABLE-5 (Fable5-Glint)"
echo "Node:  $SLURMD_NODENAME   Job: $SLURM_JOB_ID   Start: $(date -Iseconds)"
echo "=========================================="

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

NB_INPUT="$NB_DIR/fable_distillation_fastcontext-4b-sft_fable-glint_unsloth.ipynb"
NB_OUTPUT="$OUTPUT_DIR/fable_fastcontext_4b_sft_b_train_output.ipynb"
[[ -f "$NB_INPUT" ]] || { echo "[FAIL] Notebook not found: $NB_INPUT" >&2; exit 1; }

# Safety: refuse to launch a real run if SMOKE_TEST=True is committed.
python3 - <<PYEOF
import json, sys
nb = json.load(open("$NB_INPUT"))
for c in nb["cells"]:
    s = c.get("source", "")
    if isinstance(s, list): s = "".join(s)
    if "SMOKE_TEST: bool = True" in s:
        print("[FAIL] SMOKE_TEST=True committed — refusing real run", file=sys.stderr); sys.exit(1)
print("[OK] SMOKE_TEST=False confirmed")
PYEOF

echo "[....] Running via papermill..."
PYTHONUNBUFFERED=1 papermill "$NB_INPUT" "$NB_OUTPUT" \
    --kernel python3 --progress-bar --log-output --cwd "$NB_DIR"
EXIT=$?

echo "=========================================="
if (( EXIT == 0 )); then echo "[PASS] training complete"; else echo "[FAIL] training exit=$EXIT"; fi
echo "Output notebook: $NB_OUTPUT"
echo "End: $(date -Iseconds)"
exit $EXIT
