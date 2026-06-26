#!/bin/bash
#SBATCH --job-name=probe-new-models-unsloth
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err

# =============================================================================
# Probe Unsloth FastLanguageModel compat for 3 new sub-5B models:
#   - WeiboAI/VibeThinker-3B          (Qwen2 arch, reasoning model)
#   - microsoft/FastContext-1.0-4B-SFT (Qwen3 arch)
#   - microsoft/FastContext-1.0-4B-RL  (Qwen3 arch)
# Verifies: model loads in 4-bit, tokenizer renders chat template,
# get_peft_model accepts standard LoRA target_modules.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
VENV="/scratch/ermia/venvs/hf_unsloth"

echo "=========================================="
echo "PROBE: Unsloth compat for 3 new models"
echo "=========================================="
echo "Node:  $SLURMD_NODENAME"
echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date -Iseconds)"

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6
source "$VENV/bin/activate"

export SCRATCH="${SCRATCH:-/scratch/$USER}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "$PROJECT_DIR/.env"
fi

PYTHONUNBUFFERED=1 python - <<'PY'
import sys, traceback
print("[probe] importing unsloth...", flush=True)
import unsloth
from unsloth import FastLanguageModel
print(f"[probe] unsloth version: {unsloth.__version__}", flush=True)

MODELS = [
    "WeiboAI/VibeThinker-3B",
    "microsoft/FastContext-1.0-4B-SFT",
    "microsoft/FastContext-1.0-4B-RL",
]

results = {}
for mid in MODELS:
    print(f"\n[probe] ===== {mid} =====", flush=True)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=mid,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        print(f"[probe][OK] model class: {type(model).__name__}", flush=True)
        print(f"[probe][OK] tokenizer class: {type(tokenizer).__name__}", flush=True)
        # Detect VL-processor wrapping
        is_processor = hasattr(tokenizer, 'tokenizer') and not callable(getattr(tokenizer, '__call__', None))
        print(f"[probe][OK] is_VL_processor: {is_processor}", flush=True)
        # Render chat template
        has_template = bool(getattr(tokenizer, 'chat_template', None))
        print(f"[probe][OK] has chat_template: {has_template}", flush=True)
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f"[probe][OK] chat render (first 200): {text[:200]!r}", flush=True)
        # LoRA wrap
        print(f"[probe] testing get_peft_model with standard targets...", flush=True)
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=16, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        print(f"[probe][OK] peft wrapped: {type(model).__name__}", flush=True)
        # Free for next iter
        del model, tokenizer
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        results[mid] = "PASS"
    except Exception as e:
        print(f"[probe][FAIL] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results[mid] = f"FAIL: {type(e).__name__}"

print("\n[probe] ===== SUMMARY =====")
for mid, verdict in results.items():
    print(f"  {verdict:30s}  {mid}")
n_pass = sum(1 for v in results.values() if v == "PASS")
print(f"\n[probe] {n_pass}/{len(results)} models passed compat check")
sys.exit(0 if n_pass == len(results) else 1)
PY

echo ""
echo "End: $(date -Iseconds)"
echo "=========================================="
