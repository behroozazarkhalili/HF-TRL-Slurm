#!/bin/bash
#SBATCH --job-name=smoke-load-large
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

echo "=========================================="
echo "Smoke Test: Load Large Models on MIG 40GB"
echo "=========================================="
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo ""

PYTHONUNBUFFERED=1 python -c "
import torch
import gc
import time

models = [
    ('Qwen2.5-7B xLAM',  'ermiaazarkhalili/Qwen2.5-7B-Instruct_Function_Calling_xLAM',  'full',    None),
    ('Llama-3.1-8B xLAM', 'ermiaazarkhalili/Llama-3.1-8B-Instruct_Function_Calling_xLAM', 'full',    None),
    ('Llama-3-8B xLAM',   'ermiaazarkhalili/Llama-3-8B-Instruct_Function_Calling_xLAM',   'full',    None),
    ('Qwen2.5-14B xLAM',  'ermiaazarkhalili/Qwen2.5-14B-Instruct_Function_Calling_xLAM',  'full',    None),
    ('Qwen2.5-7B Capybara','ermiaazarkhalili/Qwen2.5-7B-SFT-Capybara',                    'adapter', 'Qwen/Qwen2.5-7B'),
    ('Qwen2.5-7B ChartQA', 'ermiaazarkhalili/qwen2.5-7b-instruct-trl-sft-ChartQA',        'adapter', 'Qwen/Qwen2.5-7B-Instruct'),
]

results = []

for name, repo_id, model_type, base_model in models:
    print(f'\\n{\"=\"*50}')
    print(f'{name} ({model_type})')
    print(f'  Repo: {repo_id}')
    print(f'  Base: {base_model or \"(self)\"}')
    print(f'  VRAM before: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

    start = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_type == 'full':
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                trust_remote_code=True,
            )
        else:
            # Load base + adapter
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, repo_id)

        params = sum(p.numel() for p in model.parameters()) / 1e9
        vram = torch.cuda.memory_allocated() / 1024**3
        elapsed = time.time() - start
        print(f'  LOADED: {params:.1f}B params, {vram:.1f} GB VRAM, {elapsed:.0f}s')
        results.append((name, 'PASS', f'{params:.1f}B, {vram:.1f}GB'))

        del model
        if model_type == 'adapter':
            del base
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        elapsed = time.time() - start
        err = str(e)[:150]
        print(f'  FAILED ({elapsed:.0f}s): {err}')
        if 'out of memory' in err.lower() or 'CUDA' in err:
            results.append((name, 'OOM', err[:80]))
        else:
            results.append((name, 'ERROR', err[:80]))

        gc.collect()
        torch.cuda.empty_cache()

print(f'\\n{\"=\"*50}')
print(f'SUMMARY')
print(f'{\"=\"*50}')
for name, status, detail in results:
    print(f'  {name:25s} {status:6s} {detail}')
"

echo ""
echo "=== COMPLETE $(date) ==="
