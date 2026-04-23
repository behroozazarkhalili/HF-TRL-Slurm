#!/bin/bash
# =============================================================================
# fc10713 CUDA init failure — CONTROL test
#
# Purpose: Demonstrates that the SAME workload as reproduce-minimal.sh
#   SUCCEEDS when run on any other node in the same partition with the same
#   MIG slice type. Proves the failure is node-specific, not a script bug.
#
# Expected outcome:
#   - Job lands on any fc106xx/fc107xx node EXCEPT fc10713
#   - Exits 0 within ~10 seconds
#   - stdout contains: "[PASS] GPU matmul succeeded"
# =============================================================================

#SBATCH --job-name=repro-fc10713-control
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

echo "=========================================="
echo "fc10713 CUDA init CONTROL test (any non-fc10713 node)"
echo "=========================================="
echo "Node:                 $SLURMD_NODENAME"
echo "Job ID:               $SLURM_JOB_ID"
echo "Start:                $(date -Iseconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6

python3 -c '
import sys
import torch

print(f"[INFO] python:             {sys.version.split()[0]}")
print(f"[INFO] torch:              {torch.__version__}")
print(f"[INFO] torch.version.cuda: {torch.version.cuda}")
print(f"[INFO] cuda.is_available:  {torch.cuda.is_available()}")
print(f"[INFO] cuda.device_count:  {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("[FAIL] CUDA not available — unexpected for a non-fc10713 node")
    sys.exit(1)

device = torch.device("cuda:0")
x = torch.randn(1024, 1024, device=device)
y = x @ x
torch.cuda.synchronize()
print(f"[PASS] GPU matmul succeeded, result shape: {y.shape}, "
      f"device: {y.device}, dtype: {y.dtype}")
print(f"[PASS] GPU name: {torch.cuda.get_device_name(0)}")
'

EXIT=$?
echo "=========================================="
echo "End:   $(date -Iseconds)"
echo "Exit:  $EXIT"
echo "=========================================="

exit $EXIT
