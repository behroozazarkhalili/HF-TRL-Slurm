#!/bin/bash
#SBATCH --job-name=train-gemma4-e4b-sft
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
# Real (non-smoke) training: Gemma4-E4B SFT distillation (Claude reasoning).
# Fills the never-trained fleet gap (the Apr 11 attempt died in the old hf_trl
# preprocessing path on a torchvision/VideoProcessor import; the modern
# notebook+Unsloth path bypasses it — Apr 24 smoke PASSED).
# Walltime 2h: SFT-Claude (10,477 samples) lands <1h empirically; 2x headroom.
# Gemma4 is multimodal -> notebook uses FastModel (NOT FastLanguageModel).
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-train-gemma4-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TRAIN: Gemma4-E4B SFT distillation (full epoch)"
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

NB_INPUT="$NB_DIR/sft_distillation_gemma4-e4b_unsloth.ipynb"
NB_OUTPUT="$OUTPUT_DIR/sft_gemma4_e4b_train_output.ipynb"

if [[ ! -f "$NB_INPUT" ]]; then
    echo "[FAIL] Notebook not found: $NB_INPUT" >&2
    exit 1
fi

# Safety: refuse to launch a "real" run if SMOKE_TEST=True is committed.
python3 - <<PY
import json, sys
nb = json.load(open("$NB_INPUT"))
for c in nb["cells"]:
    s = c.get("source", "")
    if isinstance(s, list): s = "".join(s)
    if "SMOKE_TEST: bool = True" in s:
        print("[FAIL] Notebook has SMOKE_TEST=True committed — refusing real run", file=sys.stderr)
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
