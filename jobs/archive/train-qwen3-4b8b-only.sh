#!/bin/bash
# Submit 4 Qwen3-4B + Qwen3-8B full training jobs (xLAM + SFT Claude-Opus)
# Partition walltime caps: b1=3h, b2=12h, b3=24h, b4=72h, b5=168h
set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-train-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

NOTEBOOKS=(
    "xlam_function_calling_qwen3-4b_unsloth|unsloth-xlam-qwen3-4b|32G|gpubase_bygpu_b3|0-16:00:00"
    "sft_distillation_qwen3-4b_unsloth|unsloth-sft-qwen3-4b|32G|gpubase_bygpu_b2|0-04:00:00"
    "xlam_function_calling_qwen3-8b_unsloth|unsloth-xlam-qwen3-8b|40G|gpubase_bygpu_b4|1-08:00:00"
    "sft_distillation_qwen3-8b_unsloth|unsloth-sft-qwen3-8b|40G|gpubase_bygpu_b2|0-06:00:00"
)

echo "=== QWEN3 4B + 8B FULL TRAINING (4 jobs) ==="
echo "Output dir: $OUTPUT_DIR"
JOB_IDS=()

for entry in "${NOTEBOOKS[@]}"; do
    IFS='|' read -r nb_name job_name mem partition walltime <<< "$entry"
    nb_input="$NB_DIR/${nb_name}.ipynb"
    nb_output="$OUTPUT_DIR/${nb_name}_train_output.ipynb"

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
#SBATCH --partition=__PARTITION__
#SBATCH --exclude=fc10713
#SBATCH --output=__LOG_DIR__/%x-%j.out
#SBATCH --error=__LOG_DIR__/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "UNSLOTH FULL TRAINING: __NB_NAME__"
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
NB_OUTPUT="__NB_OUTPUT__"

echo "[OK] SMOKE_TEST=False (full training)"
echo "[....] Running notebook via papermill ..."

PYTHONUNBUFFERED=1 papermill \
    "$NB_INPUT" \
    "$NB_OUTPUT" \
    --kernel python3 \
    --progress-bar \
    --log-output \
    --cwd __NB_DIR__
EXIT_CODE=$?

echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[OK] TRAINING COMPLETE: __NB_NAME__"
else
    echo "[FAIL] TRAINING FAILED: __NB_NAME__ (exit $EXIT_CODE)"
fi
echo "Output: $NB_OUTPUT | End: $(date)"

exit $EXIT_CODE
JOB_EOF

    sed -i \
        -e "s|__JOB_NAME__|${job_name}|g" \
        -e "s|__WALLTIME__|${walltime}|g" \
        -e "s|__MEM__|${mem}|g" \
        -e "s|__PARTITION__|${partition}|g" \
        -e "s|__NB_NAME__|${nb_name}|g" \
        -e "s|__NB_INPUT__|${nb_input}|g" \
        -e "s|__NB_OUTPUT__|${nb_output}|g" \
        -e "s|__NB_DIR__|${NB_DIR}|g" \
        -e "s|__LOG_DIR__|${LOG_DIR}|g" \
        -e "s|__VENV__|${VENV}|g" \
        -e "s|__PROJECT_DIR__|${PROJECT_DIR}|g" \
        "$SCRIPT"

    bash -n "$SCRIPT" || { echo "[ERROR] Syntax error in $SCRIPT"; exit 1; }

    JOB_ID=$(sbatch --parsable "$SCRIPT" 2>&1 | grep -oE '[0-9]+$')
    if [[ -z "$JOB_ID" ]]; then
        echo "[ERROR] sbatch failed for $nb_name"
        sbatch "$SCRIPT"
        exit 1
    fi
    JOB_IDS+=("$JOB_ID")
    printf "  %-45s  job %s  (%s, %s, %s)\n" "$nb_name" "$JOB_ID" "$mem" "$partition" "$walltime"
done

echo ""
echo "=== Submitted ${#JOB_IDS[@]} full training jobs ==="
echo "Job IDs: ${JOB_IDS[*]}"
