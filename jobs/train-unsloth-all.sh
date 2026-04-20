#!/bin/bash
# =============================================================================
# Full training: All 8 Unsloth notebooks via papermill
# Submits 8 independent SLURM jobs, SMOKE_TEST=False (full datasets)
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-train-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR" logs

# ── Notebook list ──
# Format: "notebook_basename|job_name|mem|partition|time"
# Walltimes derived from 7 runtime-verified 1000-step runs (trainer_stats.metrics['train_runtime']):
#   LFM2.5 0.382 s/step, Gemma4-E2B 1.763, Gemma4-E4B 2.075, Qwen3.5-* ~3.0, Carnice ~13 (weak)
# Formula: steps × s/step × 1.30 (30% headroom) + 25-45min overhead (load/merge/push/GGUF/validation)
# xLAM: 60k samples → N=1 epoch = 7500 steps @ EB=8
# SFT distillation: ~10k samples → N=1 epoch = ~1200 steps @ EB=8
# Carnice-9B xLAM excluded from batch — needs throughput probe first (estimate 22-54h range).
NOTEBOOKS=(
    "xlam_function_calling_qwen3.5-0.8b_unsloth|unsloth-xlam-qwen35-08b|32G|gpubase_bygpu_b2|0-12:00:00"
    "xlam_function_calling_qwen3.5-2b_unsloth|unsloth-xlam-qwen35-2b|32G|gpubase_bygpu_b2|0-12:00:00"
    "xlam_function_calling_lfm2.5-1.2b_unsloth|unsloth-xlam-lfm25-12b|32G|gpubase_bygpu_b2|0-03:00:00"
    "xlam_function_calling_gemma4-e2b_unsloth|unsloth-xlam-gemma4-e2b|32G|gpubase_bygpu_b2|0-07:00:00"
    "xlam_function_calling_gemma4-e4b_unsloth|unsloth-xlam-gemma4-e4b|40G|gpubase_bygpu_b2|0-09:00:00"
    # Carnice-9B is Qwen3.5-9B GQA — estimate ~11 s/step × 7500 × 1.3 + overhead = 32h, b5 (probe running in parallel)
    "xlam_function_calling_carnice-9b_unsloth|unsloth-xlam-carnice-9b|40G|gpubase_bygpu_b5|1-08:00:00"
    "sft_distillation_qwen3.5_unsloth|unsloth-sft-qwen35|32G|gpubase_bygpu_b2|0-03:00:00"
    "sft_distillation_lfm2.5_unsloth|unsloth-sft-lfm25|32G|gpubase_bygpu_b1|0-01:00:00"
    "sft_distillation_gemma4_unsloth|unsloth-sft-gemma4|32G|gpubase_bygpu_b2|0-02:00:00"
    # Carnice-9B SFT: 1200 steps × 11 s/step × 1.3 + 45min overhead = 5h45m → 6h safety margin
    "sft_distillation_carnice-9b_unsloth|unsloth-sft-carnice-9b|40G|gpubase_bygpu_b2|0-06:00:00"
)

echo "=== UNSLOTH NOTEBOOK FULL TRAINING ==="
echo "Output dir: $OUTPUT_DIR"
echo ""

JOB_IDS=()

for entry in "${NOTEBOOKS[@]}"; do
    IFS='|' read -r nb_name job_name mem partition walltime <<< "$entry"

    nb_input="$NB_DIR/${nb_name}.ipynb"
    nb_output="$OUTPUT_DIR/${nb_name}_train_output.ipynb"

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
#SBATCH --partition=__PARTITION__
#SBATCH --exclude=fc10713
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "UNSLOTH FULL TRAINING: __NB_NAME__"
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

# The committed notebooks already have SMOKE_TEST=False — run directly
NB_INPUT="__NB_INPUT__"
NB_OUTPUT="__NB_OUTPUT__"

echo "[OK] SMOKE_TEST=False (full training)"
echo ""
echo "[....] Running notebook via papermill ..."
echo ""

PYTHONUNBUFFERED=1 papermill \
    "$NB_INPUT" \
    "$NB_OUTPUT" \
    --kernel python3 \
    --progress-bar \
    --log-output \
    --cwd __NB_DIR__

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[OK] TRAINING COMPLETE: __NB_NAME__"
else
    echo "[FAIL] TRAINING FAILED: __NB_NAME__ (exit code $EXIT_CODE)"
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
        -e "s|__PARTITION__|${partition}|g" \
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

    printf "  %-45s  job %s  (%s, %s, %s)\n" "$nb_name" "$JOB_ID" "$mem" "$partition" "$walltime"
done

echo ""
echo "=== Submitted ${#JOB_IDS[@]} full training jobs ==="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor: squeue -u \$USER | grep unsloth-"
echo "Outputs: $OUTPUT_DIR"
