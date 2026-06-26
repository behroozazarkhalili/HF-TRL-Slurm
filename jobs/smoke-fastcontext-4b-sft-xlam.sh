#!/bin/bash
#SBATCH --job-name=smoke-fastcontext-4b-sft-xlam
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-smoke-fastcontext-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "SMOKE: FastContext-4B-SFT_base xLAM function calling (compat validation)"
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

export NB_INPUT="$NB_DIR/xlam_function_calling_fastcontext-4b-sft_unsloth.ipynb"
export NB_SMOKE="$OUTPUT_DIR/xlam_fastcontext_4b_sft_smoke_input.ipynb"
export NB_OUTPUT="$OUTPUT_DIR/xlam_fastcontext_4b_sft_smoke_output.ipynb"

# Flip SMOKE_TEST → True (handle both str and list[str] source, per memory)
python3 - <<'PY'
import json, os
src = os.environ["NB_INPUT"]
dst = os.environ["NB_SMOKE"]
nb = json.load(open(src))
patched = 0
for c in nb['cells']:
    s = c.get('source', '')
    if isinstance(s, str):
        if 'SMOKE_TEST: bool = False' in s:
            c['source'] = s.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True')
            patched += 1
    elif isinstance(s, list):
        new = [l.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True') for l in s]
        if new != s:
            c['source'] = new
            patched += 1
json.dump(nb, open(dst, 'w'), indent=1)
print(f"[OK] Patched SMOKE_TEST in {patched} cell(s)")
assert patched == 1, f"Expected exactly 1 patch, got {patched}"
PY

echo ""
echo "[....] Running via papermill..."
PYTHONUNBUFFERED=1 papermill "$NB_SMOKE" "$NB_OUTPUT" \
    --kernel python3 --progress-bar --log-output --cwd "$NB_DIR"
EXIT=$?

echo ""
echo "=========================================="
if [[ $EXIT -eq 0 ]]; then
    echo "[PASS] smoke complete"
else
    echo "[FAIL] smoke exit=$EXIT"
fi
echo "End: $(date -Iseconds)"
exit $EXIT
