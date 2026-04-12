#!/bin/bash
#SBATCH --job-name=build-claude-distillation
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL

# ─────────────────────────────────────────────────────────────────────────────
# Claude Reasoning Distillation Dataset Builder
#
# CPU-only job — no GPU needed for data pipeline.
# Downloads 5 source datasets from HF Hub, normalizes, deduplicates, and
# pushes to ermiaazarkhalili/claude-reasoning-distillation (two configs: sft, grpo)
# ─────────────────────────────────────────────────────────────────────────────

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p logs

echo "=========================================="
echo "Claude Reasoning Distillation Build"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | Start: $(date)"
echo ""

SCRIPT_DIR="/project/6014832/ermia/HF-TRL/datasets/claude-reasoning-distillation"

cd "$SCRIPT_DIR"

# Install dependencies (datasketch may not be in hf_env)
pip install -q datasketch 2>/dev/null  # datasets/pyarrow provided by hf_env + arrow module

echo "[$(date +%H:%M:%S)] Starting pipeline ..."

python build_dataset.py \
    --output_repo "ermiaazarkhalili/claude-reasoning-distillation" \
    --dedup_threshold 0.85 \
    --min_answer_len 50 \
    --test_size 0.05

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Build Complete: $(date +%H:%M:%S)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
