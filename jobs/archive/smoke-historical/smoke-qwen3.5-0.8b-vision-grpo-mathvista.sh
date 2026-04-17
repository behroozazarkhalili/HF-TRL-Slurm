#!/bin/bash
# =============================================================================
# SMOKE TEST: Qwen3.5-0.8B Vision GRPO on MathVista
# Multimodal math reasoning — Unsloth-based VLM training
# 50 samples, <1h, validates: model load, vision GRPO, reward functions
# =============================================================================

#SBATCH --job-name=qwen3.5-0.8b-vision-grpo
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
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "SMOKE: Qwen3.5-0.8B Vision GRPO on MathVista"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================="

mkdir -p logs

# Load modules and activate environment
module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

# Environment
export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR=$SCRATCH/outputs/qwen3.5-0.8b-vision-grpo-mathvista-$SLURM_JOB_ID

# Log GPU info
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load HF token
if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR

# Install Unsloth if needed (one-time on compute node)
pip install --quiet --upgrade unsloth unsloth_zoo 2>/dev/null || true

echo ""
echo "Starting Qwen3.5-0.8B Vision GRPO training..."
echo ""

# Run the training script
python -c "
import os, re, torch
from unsloth import FastVisionModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

print('=' * 60)
print('Qwen3.5-0.8B Vision GRPO — Smoke Test')
print('=' * 60)

# ===== Model Loading =====
print('Loading model: Qwen3.5-0.8B...')
model, tokenizer = FastVisionModel.from_pretrained(
    model_name='unsloth/Qwen3.5-0.8B',
    max_seq_length=4096,
    load_in_4bit=False,       # bf16 only for VLMs
    fast_inference=False,      # No vLLM support yet
)
print('Model loaded successfully')

# ===== LoRA Setup =====
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
)
print('LoRA configured')

# ===== Dataset =====
print('Loading MathVista dataset...')
dataset = load_dataset('AI4Math/MathVista', split='testmini')

# Filter to numeric answers only
def is_numeric(example):
    try:
        float(example['answer'])
        return True
    except:
        return False

dataset = dataset.filter(is_numeric)
print(f'Filtered to {len(dataset)} numeric-answer problems')

# Resize and convert images
def preprocess_image(example):
    img = example['decoded_image']
    img = img.resize((512, 512))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    example['decoded_image'] = img
    return example

dataset = dataset.map(preprocess_image)

# Limit to 50 samples for smoke test
dataset = dataset.select(range(min(50, len(dataset))))
print(f'Using {len(dataset)} samples for smoke test')

# Format as conversation
REASON_S, REASON_E = '<REASONING>', '</REASONING>'
ANSWER_S, ANSWER_E = '<SOLUTION>', '</SOLUTION>'

def make_conversation(example):
    text = (
        f\"{example['question']}. First provide reasoning between {REASON_S} and {REASON_E}, \"
        f\"then your numeric answer between {ANSWER_S} and {ANSWER_E}.\"
    )
    prompt = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
    return {'prompt': prompt, 'image': example['decoded_image'], 'answer': example['answer']}

dataset = dataset.map(make_conversation)
dataset = dataset.remove_columns('image') if 'image' in dataset.column_names and 'decoded_image' in dataset.column_names else dataset
if 'decoded_image' in dataset.column_names:
    dataset = dataset.rename_column('decoded_image', 'image')
print(f'Dataset prepared: {dataset.column_names}')

# ===== Reward Functions =====
def format_reward(completions, **kwargs):
    scores = []
    for c in completions:
        if isinstance(c, list):
            c = c[0]['content'] if c else ''
        score = 0
        if len(re.findall(f'{REASON_S}(.*?){REASON_E}', c, re.DOTALL)) == 1:
            score += 1.0
        if len(re.findall(f'{ANSWER_S}(.*?){ANSWER_E}', c, re.DOTALL)) == 1:
            score += 1.0
        scores.append(score)
    return scores

def accuracy_reward(prompts, completions, answer, **kwargs):
    pattern = f'{ANSWER_S}(.*?){ANSWER_E}'
    completions = [(c[0]['content'] if c else '') if isinstance(c, list) else c for c in completions]
    responses = [re.findall(pattern, c, re.DOTALL) for c in completions]
    return [
        2.0 if len(r) == 1 and a == r[0].strip() else 0.0
        for r, a in zip(responses, answer)
    ]

# ===== Training =====
output_dir = os.environ.get('OUTPUT_DIR', './outputs')
training_args = GRPOConfig(
    learning_rate=5e-6,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    optim='adamw_8bit',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_prompt_length=1024,
    max_completion_length=1024,
    max_steps=25,
    save_steps=25,
    logging_steps=1,
    max_grad_norm=0.1,
    report_to='none',
    output_dir=output_dir,
    loss_type='dr_grpo',
    mask_truncated_completions=False,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[format_reward, accuracy_reward],
    tokenizer=tokenizer,
)

print()
print('Starting Vision GRPO training...')
trainer.train()

# Save
print('Saving model...')
trainer.save_model()
print(f'Model saved to: {output_dir}')

print()
print('=' * 60)
print('Vision GRPO Smoke Test Complete!')
print('=' * 60)
"

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "SMOKE TEST RESULTS"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS: Vision GRPO training completed"
else
    echo "FAILED: Training exited with code $TRAIN_EXIT_CODE"
fi
