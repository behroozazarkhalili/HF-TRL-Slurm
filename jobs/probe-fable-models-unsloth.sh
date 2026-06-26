#!/bin/bash
#SBATCH --job-name=probe-fable-models-unsloth
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err

# =============================================================================
# Probe Unsloth compat for the 5 NEW fable-fleet base models, before the
# generator hardcodes per-family LoRA targets / loader path / chat template:
#   - unsloth/Qwen3.5-2B            (Qwen3.5 dense — FastLanguageModel)
#   - unsloth/Qwen3.5-4B-Instruct   (Qwen3.5 dense — FastLanguageModel)
#   - LiquidAI/LFM2.5-1.2B-Instruct  (LFM2 hybrid — FastLanguageModel)
#   - unsloth/gemma-4-E2B-it         (Gemma4 — FastModel, VL-family)
#   - unsloth/gemma-4-12B-it         (Gemma4 — FastModel, VL-family)
# Reports per model: loader path that works, model/tokenizer class, VL-unwrap
# need, chat_template presence, get_peft_model accept. Drives the registry.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
VENV="/scratch/ermia/venvs/hf_unsloth"

echo "=========================================="
echo "PROBE: Unsloth compat for 5 new fable models"
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
import sys, gc, traceback
print("[probe] importing unsloth...", flush=True)
import unsloth, torch
from unsloth import FastLanguageModel, FastModel
print(f"[probe] unsloth version: {unsloth.__version__}", flush=True)

# (model_id, loader): Gemma4 needs FastModel; dense Qwen/LFM use FastLanguageModel.
MODELS = [
    ("unsloth/Qwen3.5-2B",           "FastLanguageModel"),
    ("unsloth/Qwen3.5-4B-Instruct",  "FastLanguageModel"),
    ("LiquidAI/LFM2.5-1.2B-Instruct", "FastLanguageModel"),
    ("unsloth/gemma-4-E2B-it",       "FastModel"),
    ("unsloth/gemma-4-12B-it",       "FastModel"),
]

def load(mid, loader):
    cls = FastModel if loader == "FastModel" else FastLanguageModel
    return cls.from_pretrained(model_name=mid, max_seq_length=4096, load_in_4bit=True)

def peft(model, loader):
    cls = FastModel if loader == "FastModel" else FastLanguageModel
    return cls.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
    )

results = {}
for mid, loader in MODELS:
    print(f"\n[probe] ===== {mid}  (loader={loader}) =====", flush=True)
    try:
        model, tok = load(mid, loader)
        print(f"[probe][OK] model class: {type(model).__name__}", flush=True)
        print(f"[probe][OK] tokenizer/processor class: {type(tok).__name__}", flush=True)
        # VL-processor detection: a processor wraps an inner .tokenizer
        inner = getattr(tok, "tokenizer", None)
        is_vl = inner is not None
        print(f"[probe][OK] is_VL_processor: {is_vl}"
              + (f" (inner={type(inner).__name__})" if is_vl else ""), flush=True)
        real_tok = inner if is_vl else tok
        has_tmpl = bool(getattr(real_tok, "chat_template", None))
        print(f"[probe][OK] has chat_template: {has_tmpl}", flush=True)
        msgs = [{"role":"user","content":"What is 2+2?"}]
        try:
            text = real_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            print(f"[probe][OK] chat render (first 240): {text[:240]!r}", flush=True)
        except Exception as e:
            print(f"[probe][WARN] chat render failed: {e}", flush=True)
        model = peft(model, loader)
        print(f"[probe][OK] peft wrapped: {type(model).__name__}", flush=True)
        del model, tok; gc.collect(); torch.cuda.empty_cache()
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
