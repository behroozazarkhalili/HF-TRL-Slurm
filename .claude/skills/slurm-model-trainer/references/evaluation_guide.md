# Evaluation Guide with lm-eval-harness

## Overview

lm-eval-harness is the standard for LLM evaluation:
- 60+ benchmarks
- Reproducible evaluations
- HF Hub integration
- vLLM support for speed

## Installation

```bash
pip install lm-eval[hf]      # For HuggingFace models
pip install lm-eval[vllm]    # For vLLM (faster)
```

## Benchmark Suites

### Reasoning Benchmarks

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| GSM8K | Grade school math | Accuracy |
| MATH | Competition math | Accuracy |
| ARC-Challenge | Science reasoning | Accuracy |
| ARC-Easy | Basic science | Accuracy |

### General Knowledge

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| MMLU | 57 subjects | Accuracy |
| HellaSwag | Commonsense | Accuracy |
| TruthfulQA | Truthfulness | MC2 Score |
| Winogrande | Coreference | Accuracy |

### Coding

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| HumanEval | Function completion | pass@1 |
| MBPP | Python problems | pass@1 |

## Quick Start

### Evaluate Your Model

```bash
# Basic evaluation
lm-eval --model hf \
    --model_args pretrained=ermiaazarkhalili/my-model \
    --tasks mmlu,hellaswag,gsm8k \
    --batch_size auto

# With chat template (for instruct models)
lm-eval --model hf \
    --model_args pretrained=ermiaazarkhalili/my-model \
    --tasks mmlu,hellaswag \
    --apply_chat_template \
    --batch_size auto
```

### Using the Script

```bash
python scripts/evaluate_model.py \
    --model ermiaazarkhalili/my-model \
    --tasks comprehensive \
    --output_dir ./eval_results \
    --push_to_hub
```

## Slurm Job

### SBATCH Template

```bash
#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00

source $PROJECT/envs/hf-trl/bin/activate

lm-eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks $TASKS \
    --batch_size auto \
    --output_path ./results_$SLURM_JOB_ID
```

### Submit Evaluation

```bash
sbatch --export=ALL,MODEL=ermiaazarkhalili/my-model,TASKS=comprehensive \
    templates/sbatch_eval.sh
```

## Task Configurations

### Available Suites

```python
# In scripts/evaluate_model.py

BENCHMARK_SUITES = {
    "reasoning": ["gsm8k", "arc_challenge", "arc_easy"],
    "general": ["mmlu", "hellaswag", "truthfulqa_mc2", "winogrande"],
    "coding": ["humaneval", "mbpp"],
    "comprehensive": [...],  # All of the above
    "quick": ["arc_easy", "hellaswag"],  # Fast test
}
```

### Using Config Files

```bash
# Use predefined config
lm-eval --model hf \
    --model_args pretrained=ermiaazarkhalili/my-model \
    --tasks configs/eval_tasks/reasoning.yaml
```

## Output Format

### JSON Results

```json
{
  "results": {
    "mmlu": {
      "acc,none": 0.652,
      "acc_stderr,none": 0.012
    },
    "hellaswag": {
      "acc_norm,none": 0.781,
      "acc_norm_stderr,none": 0.008
    }
  },
  "config": {
    "model": "ermiaazarkhalili/my-model",
    "tasks": ["mmlu", "hellaswag"],
    "batch_size": 16
  }
}
```

### Markdown Summary

```markdown
# Evaluation Results

| Task | Accuracy |
|------|----------|
| MMLU | 65.2% |
| HellaSwag | 78.1% |
| GSM8K | 45.3% |
```

## Push Results to Hub

### Automatic (via Script)

```bash
python scripts/evaluate_model.py \
    --model ermiaazarkhalili/my-model \
    --tasks comprehensive \
    --push_to_hub
```

### Manual

```python
from huggingface_hub import HfApi, ModelCard

# Load results
import json
with open("results.json") as f:
    results = json.load(f)

# Update model card
card = ModelCard.load("ermiaazarkhalili/my-model")
card.text += f"""

## Evaluation Results

| Benchmark | Score |
|-----------|-------|
| MMLU | {results['results']['mmlu']['acc,none']:.1%} |
| HellaSwag | {results['results']['hellaswag']['acc_norm,none']:.1%} |
"""
card.push_to_hub("ermiaazarkhalili/my-model")
```

## Advanced Usage

### vLLM for Faster Evaluation

```bash
pip install lm-eval[vllm]

lm-eval --model vllm \
    --model_args pretrained=ermiaazarkhalili/my-model,tensor_parallel_size=2 \
    --tasks mmlu \
    --batch_size auto
```

### Multi-GPU Evaluation

```bash
# With vLLM tensor parallelism
lm-eval --model vllm \
    --model_args pretrained=my-model,tensor_parallel_size=4 \
    --tasks comprehensive
```

### Few-Shot Settings

```bash
# Override default few-shot
lm-eval --model hf \
    --model_args pretrained=my-model \
    --tasks mmlu \
    --num_fewshot 5
```

### Custom Tasks

```yaml
# my_task.yaml
task: my_custom_task
dataset_path: my_org/my_dataset
output_type: generate_until
doc_to_text: "{{question}}"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
```

## Expected Scores

### Model Size Reference

| Model Size | MMLU | HellaSwag | GSM8K |
|------------|------|-----------|-------|
| 0.5B | ~35% | ~50% | ~5% |
| 3B | ~45% | ~65% | ~15% |
| 7B | ~55% | ~75% | ~35% |
| 13B | ~60% | ~80% | ~45% |
| 70B | ~70% | ~85% | ~60% |

### After Fine-tuning

- SFT on math → GSM8K +10-20%
- DPO on preferences → TruthfulQA +5-10%
- Domain SFT → Domain-specific benchmarks improve

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 1

# Use quantization
--model_args pretrained=model,load_in_4bit=True
```

### Slow Evaluation

1. Use vLLM instead of HF
2. Use tensor parallelism
3. Reduce number of tasks

### Task Not Found

```bash
# List available tasks
lm-eval --tasks list
```

## Quick Reference

### Common Commands

```bash
# Quick test
lm-eval --model hf --model_args pretrained=model --tasks arc_easy

# Full evaluation
lm-eval --model hf --model_args pretrained=model --tasks mmlu,hellaswag,gsm8k,truthfulqa

# With chat template
lm-eval --model hf --model_args pretrained=model --tasks mmlu --apply_chat_template

# Save results
lm-eval --model hf --model_args pretrained=model --tasks mmlu --output_path ./results
```
