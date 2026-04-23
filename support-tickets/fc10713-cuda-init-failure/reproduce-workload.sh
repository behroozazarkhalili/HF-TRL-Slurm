#!/bin/bash
# =============================================================================
# fc10713 CUDA init failure — REAL WORKLOAD reproducer
#
# Purpose: Reproduces the exact failure observed in production jobs on
#   fc10713, where the Unsloth ML training framework's import chain
#   calls torch.cuda.is_available() which returns False, triggering
#   NotImplementedError.
#
# This matches the stack trace from SLURM job 36164743 (see evidence-logs/).
#
# Prerequisites:
#   - A Python venv with unsloth installed (pip install unsloth)
#   - The SBATCH resources here match what that job used: 3g.40gb MIG slice
#
# Expected outcome:
#   - Job lands on fc10713 (pinned)
#   - Exits non-zero within ~10 seconds
#   - stderr contains:
#     1. "CUDA initialization: CUDA unknown error - this may be due to an
#        incorrectly set up environment, e.g. changing env variable
#        CUDA_VISIBLE_DEVICES after program start"
#     2. "NotImplementedError: Unsloth cannot find any torch accelerator?
#        You need a GPU."
# =============================================================================

#SBATCH --job-name=repro-fc10713-workload
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodelist=fc10713
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

echo "=========================================="
echo "fc10713 CUDA init — Unsloth workload reproducer"
echo "=========================================="
echo "Node:                 $SLURMD_NODENAME"
echo "Job ID:               $SLURM_JOB_ID"
echo "Start:                $(date -Iseconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

# Set VENV_PATH to a Python venv that has `unsloth` installed.
# A minimal venv can be built with:
#   python3 -m venv /tmp/unsloth-repro-venv
#   source /tmp/unsloth-repro-venv/bin/activate
#   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
#   pip install unsloth
VENV_PATH="${VENV_PATH:-/scratch/$USER/venvs/hf_unsloth}"

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6

if [[ -d "$VENV_PATH" ]]; then
    source "$VENV_PATH/bin/activate"
    echo "[INFO] Using venv: $VENV_PATH"
else
    echo "[WARN] VENV_PATH=$VENV_PATH not found; using system python."
    echo "[WARN] Reproducer requires 'unsloth' on PYTHONPATH. See header for setup."
fi

# This is the exact import chain that fails in production.
# Unsloth's device_type.py line 218 raises NotImplementedError when
# torch.cuda.is_available() returns False — which happens on fc10713
# within 8 seconds of job start.
python3 -c '
import sys
import torch
print(f"[INFO] torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"[INFO] torch.cuda.device_count: {torch.cuda.device_count()}")
# The failing import chain:
from unsloth import FastLanguageModel  # noqa: F401 — import triggers failure
print("[PASS] Unsloth imported successfully (fc10713 may be fixed)")
'

EXIT=$?

echo "=========================================="
echo "End:   $(date -Iseconds)"
echo "Exit:  $EXIT"
if [[ $EXIT -ne 0 ]]; then
    echo "Verdict: REPRODUCED — Unsloth import fails because torch.cuda.is_available() == False"
else
    echo "Verdict: PASSED — fc10713 may be healthy"
fi
echo "=========================================="

exit $EXIT
