#!/bin/bash
# =============================================================================
# Model Evaluation Job Template (lm-eval-harness)
# Fir Cluster (Digital Research Alliance of Canada)
# =============================================================================
# Usage:
#   sbatch sbatch_eval.sh
#   sbatch --export=ALL,MODEL=username/my-model,TASKS=comprehensive sbatch_eval.sh
# =============================================================================

#SBATCH --job-name=hf-trl-eval
#SBATCH --account=def-maxwl_gpu    # Replace with your account
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# =============================================================================
# Configuration (override via --export)
# =============================================================================
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
TASKS=${TASKS:-comprehensive}  # Options: reasoning, general, coding, comprehensive
BATCH_SIZE=${BATCH_SIZE:-auto}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_results}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs $OUTPUT_DIR

PROJECT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
SKILL_DIR=$(dirname $(dirname $(realpath $0)))

# Load modules
module load python/3.11.5 cuda/12.2 arrow/17.0.0

# Activate virtual environment
source /scratch/ermia/venvs/hf_env/bin/activate

# Set cache directories
export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Load HF token
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' $PROJECT_DIR/.env | xargs)
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Tasks: $TASKS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# Print GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# =============================================================================
# Run Evaluation
# =============================================================================
cd $PROJECT_DIR

python $SKILL_DIR/scripts/evaluate_model.py \
    --model $MODEL \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR/$SLURM_JOB_ID \
    ${PUSH_TO_HUB:+--push_to_hub}

# =============================================================================
# Post-Evaluation
# =============================================================================
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "End time: $(date)"
echo "=========================================="

echo "Results saved to: $OUTPUT_DIR/$SLURM_JOB_ID"

# Display summary
if [[ -f "$OUTPUT_DIR/$SLURM_JOB_ID/results.json" ]]; then
    echo ""
    echo "Results Summary:"
    python -c "
import json
with open('$OUTPUT_DIR/$SLURM_JOB_ID/results.json') as f:
    results = json.load(f)
for task, metrics in results.get('results', {}).items():
    acc = metrics.get('acc,none', metrics.get('acc_norm,none', 'N/A'))
    print(f'  {task}: {acc}')
"
fi
