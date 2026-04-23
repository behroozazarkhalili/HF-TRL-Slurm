#!/bin/bash
# =============================================================================
# fc10713 CUDA init failure — MINIMAL reproducer
#
# Purpose: Demonstrates that GPU workloads fail on node fc10713 with a
#   CUDA initialization error, despite SLURM allocating a healthy MIG slice
#   and `scontrol show node fc10713` reporting the node as healthy.
#
# Expected outcome:
#   - Job lands on fc10713 (pinned via --nodelist)
#   - Python exits non-zero within ~5 seconds
#   - stderr contains: "CUDA unknown error - this may be due to an
#     incorrectly set up environment, e.g. changing env variable
#     CUDA_VISIBLE_DEVICES after program start"
#   - stderr also contains: "torch.cuda.is_available() == False"
#
# Control test: Submit the same script with
#   --exclude=fc10713 --nodelist="" and it succeeds on any other node in
#   the same partition with the same MIG slice type.
# =============================================================================

#SBATCH --job-name=repro-fc10713-cuda-init
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodelist=fc10713
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

echo "=========================================="
echo "fc10713 CUDA init reproducer"
echo "=========================================="
echo "Node:                 $SLURMD_NODENAME"
echo "Job ID:               $SLURM_JOB_ID"
echo "Start:                $(date -Iseconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU allocated:        $(scontrol show job $SLURM_JOB_ID | grep -oP 'gres/gpu[^,]+' | head -1)"
echo "=========================================="

# Use whatever torch/cuda modules the site prefers. On DRAC Fir cluster:
module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6

# Minimal Python: torch CUDA sanity check. No venv required beyond
# a stock python+torch install (pip install torch==2.4.0 --index-url
# https://download.pytorch.org/whl/cu121 works standalone).
python3 -c '
import sys
import torch

print(f"[INFO] python:             {sys.version.split()[0]}")
print(f"[INFO] torch:              {torch.__version__}")
print(f"[INFO] torch.version.cuda: {torch.version.cuda}")
print(f"[INFO] cuda.is_available:  {torch.cuda.is_available()}")
print(f"[INFO] cuda.device_count:  {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("[FAIL] CUDA not available on this node — reproduces fc10713 bug")
    sys.exit(1)

# If we get here on fc10713, the bug has been fixed. Try an actual GPU op.
try:
    device = torch.device("cuda:0")
    x = torch.randn(1024, 1024, device=device)
    y = x @ x
    torch.cuda.synchronize()
    print(f"[PASS] GPU matmul succeeded, result shape: {y.shape}, "
          f"device: {y.device}, dtype: {y.dtype}")
    print(f"[PASS] GPU name: {torch.cuda.get_device_name(0)}")
    sys.exit(0)
except Exception as e:
    print(f"[FAIL] GPU operation failed on available device: {e}")
    sys.exit(2)
'

EXIT=$?

echo "=========================================="
echo "End:   $(date -Iseconds)"
echo "Exit:  $EXIT"
if [[ $EXIT -eq 1 ]]; then
    echo "Verdict: REPRODUCED — fc10713 CUDA init bug is present"
elif [[ $EXIT -eq 0 ]]; then
    echo "Verdict: PASSED — fc10713 is healthy (bug may be fixed)"
else
    echo "Verdict: UNEXPECTED failure (exit $EXIT) — see stderr"
fi
echo "=========================================="

exit $EXIT
