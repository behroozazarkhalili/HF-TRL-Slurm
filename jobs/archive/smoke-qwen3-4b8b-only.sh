#!/bin/bash
# Submit ONLY the 4 Qwen3-4B + Qwen3-8B smoke tests
set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/smoke-qwen3-4b8b-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR" logs

NOTEBOOKS=(
    "xlam_function_calling_qwen3-4b_unsloth|smoke-unsloth-xlam-qwen3-4b|32G|0-01:00:00"
    "sft_distillation_qwen3-4b_unsloth|smoke-unsloth-sft-qwen3-4b|32G|0-01:00:00"
    "xlam_function_calling_qwen3-8b_unsloth|smoke-unsloth-xlam-qwen3-8b|40G|0-01:30:00"
    "sft_distillation_qwen3-8b_unsloth|smoke-unsloth-sft-qwen3-8b|40G|0-01:30:00"
)

echo "=== QWEN3 4B + 8B SMOKE TESTS (4 jobs) ==="
JOB_IDS=()

for entry in "${NOTEBOOKS[@]}"; do
    IFS='|' read -r nb_name job_name mem walltime <<< "$entry"
    nb_input="$NB_DIR/${nb_name}.ipynb"
    nb_output="$OUTPUT_DIR/${nb_name}_smoke_output.ipynb"

    [[ -f "$nb_input" ]] || { echo "[ERROR] Notebook not found: $nb_input"; exit 1; }

    SCRIPT="$OUTPUT_DIR/${job_name}.sh"
    cat > "$SCRIPT" << 'JOB_EOF'
#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=__WALLTIME__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=__MEM__
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=========================================="
echo "UNSLOTH SMOKE TEST: __NB_NAME__"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | Start: $(date) | Job: $SLURM_JOB_ID"

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6
source __VENV__/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

if [[ -f "__PROJECT_DIR__/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "__PROJECT_DIR__/.env"
fi

NB_INPUT="__NB_INPUT__"
NB_SMOKE="__OUTPUT_DIR__/__NB_NAME___smoke_input.ipynb"
NB_OUTPUT="__NB_OUTPUT__"

python3 -c "
import json
nb = json.load(open('$NB_INPUT'))
for c in nb['cells']:
    src = c.get('source', '')
    if isinstance(src, str) and 'SMOKE_TEST: bool = False' in src:
        c['source'] = src.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True')
    elif isinstance(src, list):
        c['source'] = [l.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True') for l in src]
json.dump(nb, open('$NB_SMOKE', 'w'), indent=1)
print('[OK] Patched SMOKE_TEST=True')
"

PYTHONUNBUFFERED=1 papermill "$NB_SMOKE" "$NB_OUTPUT" \
    --kernel python3 --progress-bar --log-output --cwd __NB_DIR__
EXIT_CODE=$?

echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[OK] SMOKE TEST PASSED: __NB_NAME__"
else
    echo "[FAIL] SMOKE TEST FAILED: __NB_NAME__ (exit $EXIT_CODE)"
fi
echo "End: $(date) | Output: $NB_OUTPUT"

exit $EXIT_CODE
JOB_EOF

    sed -i \
        -e "s|__JOB_NAME__|${job_name}|g" \
        -e "s|__WALLTIME__|${walltime}|g" \
        -e "s|__MEM__|${mem}|g" \
        -e "s|__NB_NAME__|${nb_name}|g" \
        -e "s|__NB_INPUT__|${nb_input}|g" \
        -e "s|__NB_OUTPUT__|${nb_output}|g" \
        -e "s|__NB_DIR__|${NB_DIR}|g" \
        -e "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" \
        -e "s|__VENV__|${VENV}|g" \
        -e "s|__PROJECT_DIR__|${PROJECT_DIR}|g" \
        "$SCRIPT"

    bash -n "$SCRIPT" || { echo "[ERROR] Syntax error in $SCRIPT"; exit 1; }

    JOB_ID=$(sbatch --parsable "$SCRIPT" 2>&1 | grep -oE '[0-9]+$')
    JOB_IDS+=("$JOB_ID")
    printf "  %-45s  job %s  (%s, %s)\n" "$nb_name" "$JOB_ID" "$mem" "$walltime"
done

echo ""
echo "=== Submitted ${#JOB_IDS[@]} smoke tests ==="
echo "Job IDs: ${JOB_IDS[*]}"
