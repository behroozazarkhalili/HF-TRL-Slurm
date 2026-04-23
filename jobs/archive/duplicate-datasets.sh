#!/bin/bash
#SBATCH --job-name=duplicate-reasoning-datasets
#SBATCH --account=def-maxwl_cpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "=========================================="
echo "Dataset duplication (Kimi-K2.5 + GLM-5.1)"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | Job: $SLURM_JOB_ID | Start: $(date)"
echo "=========================================="

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "/project/6014832/ermia/HF-TRL/.env"
fi

export HF_HOME=${SCRATCH:-/scratch/$USER}/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets

cd /scratch/ermia
python /project/6014832/ermia/HF-TRL/scripts/duplicate_datasets.py
EXIT=$?

echo "=========================================="
if [[ $EXIT -eq 0 ]]; then
    echo "[OK] All datasets duplicated"
else
    echo "[FAIL] exit=$EXIT"
fi
echo "End: $(date)"
echo "=========================================="
exit $EXIT
