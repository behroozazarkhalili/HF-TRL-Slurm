#!/bin/bash
#SBATCH --job-name=granite-inference
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

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print('Loading Granite 4.0-micro + GRPO adapter on GPU...')
base = AutoModelForCausalLM.from_pretrained(
    'ibm-granite/granite-4.0-micro',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, 'ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-10K')
tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-4.0-micro', trust_remote_code=True)
model.eval()
print('Model loaded')

problems = [
    'The arithmetic mean, geometric mean, and harmonic mean of a, b, c are 7, 6, 5 respectively. What is the value of a^2+b^2+c^2?',
    'In triangle ABC, cos(C/2) = sqrt(5)/3 and a*cos(B) + b*cos(A) = 2. Find the maximum area of triangle ABC.',
    'Find all positive integers n such that n^2 + 1 is divisible by n + 1.',
]

for i, problem in enumerate(problems):
    print(f'')
    print(f'={\"=\"*60}')
    print(f'Problem {i+1}: {problem}')
    print(f'={\"=\"*60}')

    messages = [{'role': 'user', 'content': problem}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f'Response:')
    print(response)
    print()
"
