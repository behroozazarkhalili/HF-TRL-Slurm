# Reasoning Models Guide

Best practices and lessons learned for training reasoning models (math, logic, code).

## Key Principles

### 1. Dataset Selection - Token Efficiency Matters

| Dataset | Size | Token Range | Recommended Max Length | Best For |
|---------|------|-------------|------------------------|----------|
| `nvidia/OpenMathInstruct-2` | 14M | 99.9% < 1024 | **2048** | Most efficient, general math |
| `open-r1/OpenR1-Math-220k` | 220K | 4K-8K | **8192** | Long reasoning chains |
| `nlile/NuminaMath-1.5-RL-Verifiable` | 131K | 1K-4K | **4096** | RL with verifiable answers |
| `AI-MO/NuminaMath-CoT` | 860K | 1K-3K | **4096** | Olympiad-level (IMO/AIME) |
| `EleutherAI/hendrycks_math` | 12.5K | 200-500 | **1024** | Competition MATH benchmark |

**Rule of thumb**: Match `max_completion_length` to 99th percentile of your dataset. Over-allocating wastes compute; under-allocating truncates solutions.

### 2. Streaming Mode for Large Datasets

For datasets with millions of samples, use streaming to avoid memory issues:

```python
from datasets import load_dataset

# Streaming with reproducibility
dataset = load_dataset(
    "nvidia/OpenMathInstruct-2",
    split="train",
    streaming=True
).shuffle(seed=42, buffer_size=10000).take(500000)
```

Key parameters:
- `streaming=True` - Don't load entire dataset into memory
- `seed=42` - Reproducible shuffling
- `buffer_size=10000` - Shuffle buffer (larger = more random)
- `.take(N)` - Limit samples for training

### 3. Model Selection - Instruct Models Required

**GRPO requires instruction-tuned models** as the starting point:

| Model Family | Instruct Version | Notes |
|--------------|------------------|-------|
| Qwen2.5 | `Qwen/Qwen2.5-*B-Instruct` | Explicit "-Instruct" suffix |
| Qwen3 | `Qwen/Qwen3-*B` | Already instruction-tuned (no suffix) |
| DeepSeek-R1-Distill | `deepseek-ai/DeepSeek-R1-Distill-Qwen-*B` | Pre-tuned for reasoning |
| Llama | `meta-llama/Llama-*-Instruct` | Explicit suffix |

### 4. Learning Rate - Lower for RL

| Training Method | Typical Learning Rate | Notes |
|-----------------|----------------------|-------|
| SFT | 2e-5 | Standard fine-tuning |
| GRPO (small models) | 1e-6 | 20x lower than SFT |
| GRPO (large 7B+) | 5e-7 | 40x lower than SFT |
| DPO | 5e-7 | Similar to GRPO |

**Why lower?** RL methods can be unstable. Lower rates prevent:
- Catastrophic forgetting of base capabilities
- Policy collapse
- Reward hacking

### 5. Memory Requirements Scale Non-linearly

GRPO generates multiple completions per prompt (`num_generations`):

```
Memory ≈ Model Size × (1 + num_generations × completion_length_ratio)
```

| Model Size | GPU Memory | Quantization | num_generations |
|------------|------------|--------------|-----------------|
| 0.5B-1.5B | 10GB MIG | bf16 | 4 |
| 1.5B-4B | 40GB MIG | bf16 | 4 |
| 7B-14B | 40GB MIG | **4-bit** | 4 |
| 14B+ | Full H100 | 4-bit | 2-4 |

### 6. Reward Function Design

For math reasoning, use a combined reward:

```python
def compute_rewards(completions, ground_truths):
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        reward = 0.0

        # 50% - Accuracy (most important)
        if extract_boxed_answer(completion) == truth:
            reward += 0.5

        # 25% - Format (proper \boxed{} usage)
        if "\\boxed{" in completion and "}" in completion:
            reward += 0.25

        # 15% - Length penalty (discourage padding)
        if len(completion) < 2000:
            reward += 0.15

        # 10% - Reasoning steps (encourage work)
        if completion.count("Step") >= 2:
            reward += 0.10

        rewards.append(reward)
    return rewards
```

### 7. Dataset Format for GRPO

nvidia/OpenMathInstruct-2 format:
```json
{
  "problem": "Solve for x: 2x + 5 = 13",
  "generated_solution": "Step 1: Subtract 5...\n\\boxed{4}",
  "expected_answer": "4",
  "problem_source": "gsm8k"
}
```

Map to GRPO format (prompts only):
```python
def format_for_grpo(example):
    return {
        "prompt": example["problem"]
    }
# Store expected_answer separately for reward computation
```

## Configuration Templates

### Small Models (0.5B-1.5B) - 10GB MIG
```bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

BATCH_SIZE=2
GRAD_ACCUM=8
NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=2048
LEARNING_RATE=1e-6
LORA_R=16
LORA_ALPHA=32
```

### Medium Models (1.5B-4B) - 40GB MIG
```bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

BATCH_SIZE=2
GRAD_ACCUM=8
NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=2048
LEARNING_RATE=1e-6
LORA_R=16
LORA_ALPHA=32
```

### Large Models (7B-14B) - 40GB MIG with 4-bit
```bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00

BATCH_SIZE=1
GRAD_ACCUM=16
NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=2048
LEARNING_RATE=5e-7
LORA_R=16
LORA_ALPHA=32
USE_4BIT=true
```

## Common Issues

### 1. OOM During Generation
- Reduce `num_generations` from 4 to 2
- Enable `--use_4bit` for 7B+ models
- Reduce `max_completion_length`

### 2. Training Instability
- Lower learning rate (try 5e-7)
- Increase `gradient_accumulation_steps`
- Check reward function isn't sparse (too many 0s)

### 3. Reward Hacking
- Model outputs short/empty responses for length bonus
- Solution: Make accuracy reward dominant (>50%)

### 4. Slow Convergence
- Increase `num_generations` to get better reward signal
- Use larger effective batch size

## Recommended Datasets by Task

### General Math
- `nvidia/OpenMathInstruct-2` (best efficiency)
- `AI-MO/NuminaMath-CoT` (competition-level)

### Verifiable RL
- `nlile/NuminaMath-1.5-RL-Verifiable`

### Long Reasoning
- `open-r1/OpenR1-Math-220k`

### Benchmark Evaluation
- `EleutherAI/hendrycks_math`
