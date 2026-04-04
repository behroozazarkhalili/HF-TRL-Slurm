#!/bin/bash
# =============================================================================
# GPU Diagnostic: Test MIG CUDA behavior for pin_memory / bf16 fixes
# Quick job to validate assumptions before modifying training scripts
# =============================================================================

#SBATCH --job-name=gpu-diagnostic
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

echo "=========================================="
echo "GPU Diagnostic — MIG CUDA Behavior Test"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================="

echo ""
echo "=== Test 1: Raw SLURM environment ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-not set}"

echo ""
echo "=== Test 2: nvidia-smi output ==="
nvidia-smi -L 2>&1
echo ""
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv 2>&1

echo ""
echo "=== Test 3: PyTorch with SLURM's MIG UUID ==="
python -c "
import os, torch
cuda_dev = os.environ.get('CUDA_VISIBLE_DEVICES', '')
print(f'CUDA_VISIBLE_DEVICES = {cuda_dev}')
print(f'Contains MIG UUID: {\"MIG\" in cuda_dev.upper()}')
print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
print(f'torch.cuda.device_count() = {torch.cuda.device_count()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'Device: {props.name} ({props.total_mem / 1024**3:.1f} GB)')
    print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
else:
    print('GPU NOT DETECTED by PyTorch')
"

echo ""
echo "=== Test 4: PyTorch with CUDA_VISIBLE_DEVICES=0 (subprocess) ==="
CUDA_VISIBLE_DEVICES=0 python -c "
import os, torch
cuda_dev = os.environ.get('CUDA_VISIBLE_DEVICES', '')
print(f'CUDA_VISIBLE_DEVICES = {cuda_dev}')
print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
print(f'torch.cuda.device_count() = {torch.cuda.device_count()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'Device: {props.name} ({props.total_mem / 1024**3:.1f} GB)')
    print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
    print(f'Compute capability: {props.major}.{props.minor}')
else:
    print('GPU NOT DETECTED even with CUDA_VISIBLE_DEVICES=0')
"

echo ""
echo "=== Test 5: Pin memory test with MIG UUID ==="
python -c "
import os, torch
from torch.utils.data import DataLoader, TensorDataset

data = TensorDataset(torch.randn(100, 10))
try:
    loader = DataLoader(data, batch_size=10, pin_memory=True, num_workers=0)
    batch = next(iter(loader))
    print(f'pin_memory=True, workers=0, MIG UUID: OK (is_pinned={batch[0].is_pinned()})')
except Exception as e:
    print(f'pin_memory=True, workers=0, MIG UUID: FAILED — {e}')

try:
    loader = DataLoader(data, batch_size=10, pin_memory=True, num_workers=2)
    batch = next(iter(loader))
    print(f'pin_memory=True, workers=2, MIG UUID: OK (is_pinned={batch[0].is_pinned()})')
except Exception as e:
    print(f'pin_memory=True, workers=2, MIG UUID: FAILED — {e}')

try:
    loader = DataLoader(data, batch_size=10, pin_memory=False, num_workers=0)
    batch = next(iter(loader))
    print(f'pin_memory=False: OK')
except Exception as e:
    print(f'pin_memory=False: FAILED — {e}')
"

echo ""
echo "=== Test 6: Pin memory test with CUDA_VISIBLE_DEVICES=0 ==="
CUDA_VISIBLE_DEVICES=0 python -c "
import os, torch
from torch.utils.data import DataLoader, TensorDataset

data = TensorDataset(torch.randn(100, 10))
try:
    loader = DataLoader(data, batch_size=10, pin_memory=True, num_workers=2)
    batch = next(iter(loader))
    print(f'pin_memory=True, workers=2, CVD=0: OK (is_pinned={batch[0].is_pinned()})')
except Exception as e:
    print(f'pin_memory=True, workers=2, CVD=0: FAILED — {e}')
"

echo ""
echo "=== Test 7: Model load + forward pass on MIG ==="
python -c "
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
print('Loading Qwen3-0.6B with 4-bit quant...')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', quantization_config=bnb_config, trust_remote_code=True)
print(f'Model loaded on: {next(model.parameters()).device}')

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
inputs = tokenizer('Hello', return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model(**inputs)
print(f'Forward pass OK, logits dtype: {out.logits.dtype}')
" 2>&1 || echo "Model loading test failed"

echo ""
echo "=========================================="
echo "Diagnostic complete: $(date)"
echo "=========================================="
