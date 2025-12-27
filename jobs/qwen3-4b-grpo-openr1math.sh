#!/bin/bash
# Qwen3-4B GRPO Training on OpenR1-Math-220k

#SBATCH --job-name=qwen3-4b-grpo-openr1math
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

MODEL_NAME="Qwen/Qwen3-4B"
DATASET_NAME="open-r1/OpenR1-Math-220k"
HUB_MODEL_ID="ermiaazarkhalili/Qwen3-4B-GRPO-OpenR1Math"
GGUF_REPO_ID="ermiaazarkhalili/Qwen3-4B-GRPO-OpenR1Math-GGUF"

BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE=1e-6
NUM_EPOCHS=1
MAX_COMPLETION_LENGTH=512
MAX_PROMPT_LENGTH=256
NUM_GENERATIONS=4
REWARD_TYPE="combined"
LORA_R=16
LORA_ALPHA=32

echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) | Start: $(date)"
mkdir -p logs
module load gcc arrow python/3.11.5
source /scratch/ermia/venvs/hf_env/bin/activate

export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export OUTPUT_DIR=$SCRATCH/outputs/qwen3-4b-grpo-$SLURM_JOB_ID
[[ -f "/project/6014832/ermia/HF-TRL/.env" ]] && export $(grep -v '^#' /project/6014832/ermia/HF-TRL/.env | xargs)
mkdir -p $OUTPUT_DIR

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/train_grpo.py \
    --model_name_or_path $MODEL_NAME --dataset_name $DATASET_NAME --dataset_split "train" \
    --output_dir $OUTPUT_DIR --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE --max_length $MAX_COMPLETION_LENGTH --max_prompt_length $MAX_PROMPT_LENGTH \
    --num_generations $NUM_GENERATIONS --reward_type $REWARD_TYPE \
    --bf16 --gradient_checkpointing --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout 0.05 \
    --save_strategy steps --save_steps 500 --save_total_limit 3 --logging_steps 10 \
    --push_to_hub --hub_model_id $HUB_MODEL_ID --hub_strategy end \
    --report_to trackio --trackio_dir $OUTPUT_DIR/trackio --project "grpo-openr1math" --run_name "qwen3-4b-grpo-$SLURM_JOB_ID"

[[ $? -ne 0 ]] && exit 1

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/generate_model_card.py \
    --model_name "Qwen3-4B-GRPO-OpenR1Math" --base_model "$MODEL_NAME" --dataset "$DATASET_NAME" \
    --training_method GRPO --author ermiaazarkhalili --license apache-2.0 \
    --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE --epochs $NUM_EPOCHS --max_length $MAX_COMPLETION_LENGTH \
    --lora_r $LORA_R --lora_alpha $LORA_ALPHA --hardware "NVIDIA H100 40GB MIG" --output_dir $OUTPUT_DIR/model_card

python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$OUTPUT_DIR/model_card/README.md', path_in_repo='README.md', repo_id='$HUB_MODEL_ID')"

python /project/6014832/ermia/HF-TRL/.claude/skills/slurm-model-trainer/scripts/convert_gguf.py \
    --model $HUB_MODEL_ID --base_model $MODEL_NAME --output_repo $GGUF_REPO_ID \
    --quantizations "Q4_K_M,Q5_K_M,Q8_0" --output_dir $OUTPUT_DIR/gguf

echo "Complete! https://huggingface.co/$HUB_MODEL_ID"
