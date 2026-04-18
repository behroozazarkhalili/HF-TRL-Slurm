#!/bin/bash
# =============================================================================
# Smoke test: All 8 Unsloth notebooks via papermill
# Submits 8 independent SLURM jobs, one per notebook, SMOKE_TEST=True
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-smoke-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR" logs

# ── Notebook list ──
# Format: "notebook_basename|job_name|mem|time"
# Smoke tests: short time, b1 partition
NOTEBOOKS=(
    "xlam_function_calling_qwen3.5-0.8b_unsloth|smoke-unsloth-xlam-qwen35-08b|32G|0-01:00:00"
    "xlam_function_calling_qwen3.5-2b_unsloth|smoke-unsloth-xlam-qwen35-2b|32G|0-01:00:00"
    "xlam_function_calling_lfm2.5-1.2b_unsloth|smoke-unsloth-xlam-lfm25-12b|32G|0-01:00:00"
    "xlam_function_calling_gemma4-e2b_unsloth|smoke-unsloth-xlam-gemma4-e2b|32G|0-01:00:00"
    "xlam_function_calling_gemma4-e4b_unsloth|smoke-unsloth-xlam-gemma4-e4b|40G|0-01:30:00"
    # gemma4-26b-a4b REMOVED 2026-04-17: MoE 26B does not fit 3g.40gb MIG slice
    # (bnb-4bit dispatches to CPU). Needs full H100 80GB or future `--gres=nvidia_h100_80gb_hbm3:1`.
    "xlam_function_calling_gemma4-31b_unsloth|smoke-unsloth-xlam-gemma4-31b|64G|0-01:30:00"
    "xlam_function_calling_carnice-9b_unsloth|smoke-unsloth-xlam-carnice-9b|40G|0-01:30:00"
    "sft_distillation_qwen3.5_unsloth|smoke-unsloth-sft-qwen35|32G|0-01:00:00"
    "sft_distillation_lfm2.5_unsloth|smoke-unsloth-sft-lfm25|32G|0-01:00:00"
    "sft_distillation_gemma4_unsloth|smoke-unsloth-sft-gemma4|32G|0-01:00:00"
    # sft gemma4-26b-a4b REMOVED 2026-04-17: see xlam entry comment above
    "sft_distillation_gemma4-31b_unsloth|smoke-unsloth-sft-gemma4-31b|64G|0-01:30:00"
    "sft_distillation_carnice-9b_unsloth|smoke-unsloth-sft-carnice-9b|40G|0-01:30:00"
)

echo "=== UNSLOTH NOTEBOOK SMOKE TESTS ==="
echo "Output dir: $OUTPUT_DIR"
echo ""

JOB_IDS=()

for entry in "${NOTEBOOKS[@]}"; do
    IFS='|' read -r nb_name job_name mem walltime <<< "$entry"

    nb_input="$NB_DIR/${nb_name}.ipynb"
    nb_output="$OUTPUT_DIR/${nb_name}_smoke_output.ipynb"

    if [[ ! -f "$nb_input" ]]; then
        echo "[ERROR] Notebook not found: $nb_input"
        continue
    fi

    # Create per-job SLURM script
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
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6
source __VENV__/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

# Load HF token
if [[ -f "__PROJECT_DIR__/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "__PROJECT_DIR__/.env"
fi

# Prepare smoke test copy: force SMOKE_TEST = True
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

echo ""
echo "[....] Running notebook via papermill ..."
echo ""

PYTHONUNBUFFERED=1 papermill \
    "$NB_SMOKE" \
    "$NB_OUTPUT" \
    --kernel python3 \
    --progress-bar \
    --log-output \
    --cwd __NB_DIR__

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[OK] SMOKE TEST PASSED: __NB_NAME__"
else
    echo "[FAIL] SMOKE TEST FAILED: __NB_NAME__ (exit code $EXIT_CODE)"
fi
echo "Output notebook: $NB_OUTPUT"
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE
JOB_EOF

    # Inject values via sed
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

    # Verify syntax
    bash -n "$SCRIPT" || { echo "[ERROR] Syntax error in $SCRIPT"; continue; }

    # Submit
    JOB_ID=$(sbatch --parsable "$SCRIPT" 2>&1 | grep -oE '[0-9]+$')
    JOB_IDS+=("$JOB_ID")

    printf "  %-45s  job %s  (%s, %s)\n" "$nb_name" "$JOB_ID" "$mem" "$walltime"
done

echo ""
echo "=== Submitted ${#JOB_IDS[@]} smoke tests ==="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor: squeue -u \$USER | grep smoke-unsloth"
echo "Outputs: $OUTPUT_DIR"
