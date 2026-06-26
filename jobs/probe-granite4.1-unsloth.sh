#!/bin/bash
#SBATCH --job-name=probe-granite4.1-unsloth
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err

# =============================================================================
# Probe whether Unsloth's FastLanguageModel can load Granite 4.1.
# Tests 3B; if 3B works the 8B path is reasonably safe.
# Failure of this probe means we adjust the plan BEFORE writing 12 files.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
VENV="/scratch/ermia/venvs/hf_unsloth"

echo "=========================================="
echo "PROBE: Unsloth + Granite 4.1-3B compat check"
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

echo ""
echo "[....] Probing FastLanguageModel.from_pretrained('ibm-granite/granite-4.1-3b')..."
echo ""

PYTHONUNBUFFERED=1 python - <<'PY'
import sys, traceback
print("[probe] importing unsloth...", flush=True)
import unsloth
from unsloth import FastLanguageModel
print(f"[probe] unsloth version: {unsloth.__version__}", flush=True)

print("[probe] FastLanguageModel.from_pretrained('ibm-granite/granite-4.1-3b', load_in_4bit=True)...", flush=True)
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ibm-granite/granite-4.1-3b",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    print(f"[probe][OK] model class: {type(model).__name__}")
    print(f"[probe][OK] tokenizer class: {type(tokenizer).__name__}")
    has_template = bool(getattr(tokenizer, 'chat_template', None))
    print(f"[probe][OK] has chat_template: {has_template}")
    msgs = [{"role":"user","content":"hello"}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"[probe][OK] chat_template render (first 300):\n{text[:300]!r}")
    # Check LoRA target_modules can be discovered
    print("[probe] testing FastLanguageModel.get_peft_model with default targets...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print(f"[probe][OK] get_peft_model returned: {type(model).__name__}")
    print(f"[probe][PASS] all checks succeeded for Granite 4.1-3B")
except Exception as e:
    print(f"[probe][FAIL] {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
PY

echo ""
echo "End: $(date -Iseconds)"
echo "=========================================="
