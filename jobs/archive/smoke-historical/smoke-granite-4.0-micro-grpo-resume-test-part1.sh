#!/bin/bash
#SBATCH --job-name=granite-resume-test-p1
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=0-00:30:00
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
export OUTPUT_DIR=$SCRATCH/outputs/granite-resume-test

# Load HF token
if [[ -f "/project/6014832/ermia/HF-TRL/.env" ]]; then
    export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
fi

mkdir -p $OUTPUT_DIR logs

echo "=== PART 1: Train steps 1-25, save checkpoint every 10 steps ==="

PYTHONUNBUFFERED=1 python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path ibm-granite/granite-4.0-micro \
    --dataset_name AI-MO/NuminaMath-CoT \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --report_to none \
    --run_name "granite-resume-test-p1" \
    --streaming \
    --max_samples 100 \
    --seed 42 \
    --max_completion_length 2048 \
    --max_prompt_length 512 \
    --num_generations 4 \
    --reward_type combined \
    --max_steps 15

echo ""
echo "=== PART 1 COMPLETE ==="
echo "Checkpoints:"
ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null
echo "Exit code: $?"
