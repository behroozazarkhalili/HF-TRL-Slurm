#!/bin/bash
# =============================================================================
# GGUF Conversion Job Template
# Fir Cluster (Digital Research Alliance of Canada)
# =============================================================================
# Usage:
#   sbatch sbatch_convert.sh
#   sbatch --export=ALL,MODEL=username/my-model,OUTPUT_REPO=username/my-model-gguf sbatch_convert.sh
# =============================================================================

#SBATCH --job-name=hf-trl-gguf
#SBATCH --account=def-maxwl_gpu    # Replace with your account
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=cpubase_bycore_b3  # CPU-only, no GPU needed
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# =============================================================================
# Configuration (override via --export)
# =============================================================================
MODEL=${MODEL:-username/my-model}
BASE_MODEL=${BASE_MODEL:-}  # Optional: if MODEL is a LoRA adapter
OUTPUT_REPO=${OUTPUT_REPO:-${MODEL}-gguf}
QUANTIZATIONS=${QUANTIZATIONS:-Q4_K_M,Q5_K_M,Q8_0}

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

PROJECT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
SKILL_DIR=$(dirname $(dirname $(realpath $0)))

# Load modules
module load python/3.11.5

# Activate virtual environment
source /scratch/ermia/venvs/hf_env/bin/activate

# Set cache directories
export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

# Load HF token
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' $PROJECT_DIR/.env | xargs)
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Base Model: ${BASE_MODEL:-<same as model>}"
echo "  Output Repo: $OUTPUT_REPO"
echo "  Quantizations: $QUANTIZATIONS"
echo ""

# =============================================================================
# Run Conversion
# =============================================================================
cd $PROJECT_DIR

python $SKILL_DIR/scripts/convert_gguf.py \
    --model $MODEL \
    ${BASE_MODEL:+--base_model $BASE_MODEL} \
    --output_repo $OUTPUT_REPO \
    --quantizations $QUANTIZATIONS

# =============================================================================
# Post-Conversion
# =============================================================================
echo ""
echo "=========================================="
echo "Conversion Complete!"
echo "End time: $(date)"
echo "=========================================="

echo "GGUF models uploaded to: https://huggingface.co/$OUTPUT_REPO"
echo ""
echo "To use with Ollama:"
echo "  ollama pull hf.co/$OUTPUT_REPO:Q4_K_M"
echo "  ollama run ${OUTPUT_REPO##*/}"
