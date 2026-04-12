#!/bin/bash
# =============================================================================
# SMOKE TEST: Gemma4-E4B SFT Distillation on Claude Reasoning
# 100 samples, <1h, validates: preprocessing adapter, SFT training, inference
# =============================================================================

#SBATCH --job-name=gemma4-e4b-sft-claude
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME='google/gemma-4-E4B-it'
MODEL_FAMILY='gemma4'
DATASET_REPO='ermiaazarkhalili/claude-reasoning-distillation'
MAX_SAMPLES=100

HUB_MODEL_ID='ermiaazarkhalili/Gemma4-E4B-SFT-Claude-Reasoning-smoke-100'

BATCH_SIZE=2
GRAD_ACCUM=4
LEARNING_RATE=2e-5
NUM_EPOCHS=1
LORA_R=16
LORA_ALPHA=32
MAX_LENGTH=2048

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "SMOKE: Gemma4-E4B SFT Claude Reasoning"
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

mkdir -p logs

module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_trl/bin/activate


export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/gemma4-e4b-sft-claude-$SLURM_JOB_ID"
export PREPROCESSED_DIR="$OUTPUT_DIR/preprocessed"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        export "$key=$value"
    done < "/project/6014832/ermia/HF-TRL/.env"
fi

mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Model Family: $MODEL_FAMILY"
echo "  Dataset: $DATASET_REPO"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Batch Size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "  Max Length: $MAX_LENGTH"
echo "  Output: $OUTPUT_DIR"
echo ""

# =============================================================================
# Phase 0: GPU Profiler (background)
# =============================================================================
VRAM_LOG="$OUTPUT_DIR/vram_usage.log"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "$VRAM_LOG" 2>/dev/null &
NVIDIA_PID=$!

# =============================================================================
# Phase 1: Preprocess Dataset
# =============================================================================
echo "=========================================="
echo "Phase 1: Preprocessing ($MODEL_FAMILY adapter)"
echo "=========================================="

python /project/6014832/ermia/HF-TRL/datasets/claude-reasoning-distillation/preprocess_distillation.py \
    --model_family $MODEL_FAMILY \
    --model_name_or_path $MODEL_NAME \
    --dataset_repo $DATASET_REPO \
    --max_samples $MAX_SAMPLES \
    --output_dir $PREPROCESSED_DIR

PREPROCESS_EXIT=$?
if [[ $PREPROCESS_EXIT -ne 0 ]]; then
    echo "ERROR: Preprocessing failed (exit $PREPROCESS_EXIT)"
    kill $NVIDIA_PID 2>/dev/null
    exit $PREPROCESS_EXIT
fi

# =============================================================================
# Phase 2: SFT Training
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 2: SFT Training"
echo "=========================================="

TRAIN_START=$(date +%s)

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_sft_new.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $PREPROCESSED_DIR \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --bf16 \
    --use_4bit \
    --gradient_checkpointing \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --max_length $MAX_LENGTH \
    --save_strategy no \
    --logging_steps 5 \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy end \
    --report_to trackio \
    --project "gemma4-e4b-sft-claude-reasoning" \
    --run_name "gemma4-e4b-sft-claude-smoke-$SLURM_JOB_ID"

TRAIN_EXIT=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Training failed (exit $TRAIN_EXIT)"
    kill $NVIDIA_PID 2>/dev/null
    exit $TRAIN_EXIT
fi

echo "Training completed in ${TRAIN_DURATION}s"

# =============================================================================
# Phase 3: Quick Inference Test
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 3: Inference Test"
echo "=========================================="

python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('$MODEL_NAME', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
model = PeftModel.from_pretrained(model, '$OUTPUT_DIR')

prompts = [
    'What is 15 + 27?',
    'Explain why the sky is blue in simple terms.',
    'Write a Python function to check if a number is prime.',
]

for p in prompts:
    messages = [{'role': 'user', 'content': p}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    print(f'Q: {p}')
    print(f'A: {response[:200]}')
    print('---')
" 2>&1 || echo "WARNING: Inference test failed (non-critical for smoke)"

# =============================================================================
# Phase 4: Metrics Report
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 4: Metrics Report"
echo "=========================================="

kill $NVIDIA_PID 2>/dev/null

echo "Training duration: ${TRAIN_DURATION}s"
echo "Samples: $MAX_SAMPLES"
echo "Throughput: $(python -c "print(f'{$MAX_SAMPLES / max(1, $TRAIN_DURATION):.2f} samples/sec')")"

if [[ -f "$VRAM_LOG" ]]; then
    PEAK_VRAM=$(tail -n +2 "$VRAM_LOG" | awk -F', ' '{gsub(/ MiB/,"",$2); print $2}' | sort -n | tail -1)
    echo "Peak GPU vRAM: ${PEAK_VRAM} MiB"
fi

echo ""
echo "=========================================="
echo "Smoke Test Complete! $(date)"
echo "=========================================="
