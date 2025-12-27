#!/bin/bash
# =============================================================================
# Qwen2.5-14B SFT Training on UltraChat-200k
# Full Pipeline: Train → Eval (MMLU, GSM8K) → GGUF Conversion → HF Upload
# =============================================================================

#SBATCH --job-name=qwen2.5-14b-sft-ultrachat
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME="Qwen/Qwen2.5-14B"
DATASET_NAME="HuggingFaceH4/ultrachat_200k"
HUB_MODEL_ID="ermiaazarkhalili/Qwen2.5-14B-SFT-UltraChat"
GGUF_REPO_ID="ermiaazarkhalili/Qwen2.5-14B-SFT-UltraChat-GGUF"

# Training parameters (Large model config - 4-bit QLoRA, optimized for 40GB MIG)
# Reduced LoRA rank and grad_accum to prevent OOM
BATCH_SIZE=1
GRAD_ACCUM=4
LEARNING_RATE=2e-4
NUM_EPOCHS=1
MAX_SEQ_LENGTH=2048
LORA_R=32
LORA_ALPHA=64

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

# Load modules
module load gcc arrow python/3.11.5

# Activate virtual environment
source /scratch/ermia/venvs/hf_env/bin/activate

# Set environment variables
export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR=$SCRATCH/outputs/qwen2.5-14b-sft-$SLURM_JOB_ID

# Load HF token
if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Hub Model ID: $HUB_MODEL_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LoRA Rank: $LORA_R"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# =============================================================================
# Phase 1: Training
# =============================================================================
echo "=========================================="
echo "Phase 1: SFT Training"
echo "=========================================="

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_sft.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_split "train_sft" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_SEQ_LENGTH \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --test_size 0.05 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --push_to_hub \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy end \
    --report_to trackio \
    --trackio_dir $OUTPUT_DIR/trackio \
    --project "qwen-sft-ultrachat" \
    --run_name "qwen2.5-14b-sft-$SLURM_JOB_ID"

TRAIN_EXIT_CODE=$?

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully!"

# =============================================================================
# Phase 2: Generate Model Card
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 2: Generating Model Card"
echo "=========================================="

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/generate_model_card.py \
    --model_name "Qwen2.5-14B-SFT-UltraChat" \
    --base_model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --training_method SFT \
    --author ermiaazarkhalili \
    --license cc-by-nc-4.0 \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --max_length $MAX_SEQ_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --hardware "NVIDIA H100 40GB MIG" \
    --output_dir $OUTPUT_DIR/model_card

# Push model card to Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='$OUTPUT_DIR/model_card/README.md',
    path_in_repo='README.md',
    repo_id='$HUB_MODEL_ID'
)
print('Model card uploaded to $HUB_MODEL_ID')
"

# =============================================================================
# Phase 3: Evaluation (MMLU and IFEval)
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 3: Evaluation (MMLU, IFEval)"
echo "=========================================="

pip install -q lm-eval

python -c "
import subprocess
import json

model_id = '$HUB_MODEL_ID'
output_dir = '$OUTPUT_DIR/eval_results'

cmd = [
    'lm_eval',
    '--model', 'hf',
    '--model_args', f'pretrained={model_id},trust_remote_code=True',
    '--tasks', 'mmlu,ifeval',
    '--batch_size', 'auto',
    '--output_path', output_dir,
    '--log_samples'
]

print(f'Running: {\" \".join(cmd)}')
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f'Warning: Evaluation had issues: {result.stderr}')
"

python -c "
import json
import os
try:
    import trackio
    trackio.init(project='qwen-sft-ultrachat', name='eval-$SLURM_JOB_ID')

    eval_dir = '$OUTPUT_DIR/eval_results'
    for f in os.listdir(eval_dir) if os.path.exists(eval_dir) else []:
        if f.endswith('.json'):
            with open(os.path.join(eval_dir, f)) as fp:
                results = json.load(fp)
                if 'results' in results:
                    for task, metrics in results['results'].items():
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                trackio.log({f'{task}/{metric}': value})
    trackio.finish()
    print('Evaluation results logged to trackio')
except Exception as e:
    print(f'Could not log to trackio: {e}')
"

# =============================================================================
# Phase 4: GGUF Conversion
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 4: GGUF Conversion (Q4_K_M)"
echo "=========================================="

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/convert_gguf.py \
    --model $HUB_MODEL_ID \
    --base_model $MODEL_NAME \
    --output_repo $GGUF_REPO_ID \
    --quantizations "Q4_K_M,Q5_K_M,Q8_0" \
    --output_dir $OUTPUT_DIR/gguf

GGUF_EXIT_CODE=$?

if [[ $GGUF_EXIT_CODE -ne 0 ]]; then
    echo "WARNING: GGUF conversion had issues (exit code $GGUF_EXIT_CODE)"
else
    echo "GGUF conversion completed successfully!"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Training Output: $OUTPUT_DIR"
echo "  Model on Hub: https://huggingface.co/$HUB_MODEL_ID"
echo "  GGUF on Hub: https://huggingface.co/$GGUF_REPO_ID"
echo ""
echo "Evaluation Results: $OUTPUT_DIR/eval_results"
echo ""
echo "To use with Ollama:"
echo "  ollama pull hf.co/$GGUF_REPO_ID:Q4_K_M"
echo ""
