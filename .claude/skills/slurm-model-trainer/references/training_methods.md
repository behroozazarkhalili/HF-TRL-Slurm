# Training Methods Guide

## Overview

| Method | Purpose | Data Format | Learning Rate | Use When |
|--------|---------|-------------|---------------|----------|
| **SFT** | Instruction tuning | messages/text | 2e-5 | Starting point, new capabilities |
| **DPO** | Preference alignment | prompt/chosen/rejected | 5e-7 | After SFT, improve quality |
| **GRPO** | Online RL | prompt only | 1e-6 | Reasoning, math, complex tasks |

## SFT (Supervised Fine-Tuning)

### When to Use
- Teaching new tasks or domains
- Adapting to specific formats
- First step before alignment

### Dataset Formats

**Chat Format (Recommended)**
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

**Text Format**
```json
{
  "text": "<|user|>Hello<|assistant|>Hi there!"
}
```

**Prompt-Completion Format**
```json
{
  "prompt": "Translate to French: Hello",
  "completion": "Bonjour"
}
```

### Key Parameters
```python
SFTConfig(
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)
```

### TRL Datasets
- `trl-lib/Capybara` - General instruction following
- `HuggingFaceH4/ultrachat_200k` - Multi-turn conversations

## DPO (Direct Preference Optimization)

### When to Use
- After SFT to improve quality
- When you have preference data
- Aligning with human preferences

### Dataset Format
```json
{
  "prompt": "Write a poem about the ocean",
  "chosen": "The waves crash upon the shore...",
  "rejected": "Ocean is wet and has fish..."
}
```

### Key Parameters
```python
DPOConfig(
    beta=0.1,           # KL penalty (higher = stay closer to reference)
    learning_rate=5e-7, # Much lower than SFT!
    num_train_epochs=1, # Usually just 1 epoch
    loss_type="sigmoid",
)
```

### Important Notes
1. **Base model should be instruction-tuned** (use *-Instruct models)
2. **Lower learning rate** than SFT (5e-7 vs 2e-5)
3. **Fewer epochs** (usually 1)
4. **Beta parameter** controls how much the model can deviate

### TRL Datasets
- `trl-lib/ultrafeedback_binarized` - General preferences
- `Anthropic/hh-rlhf` - Helpfulness & harmlessness

## GRPO (Group Relative Policy Optimization)

> **See also**: [Reasoning Models Guide](reasoning_models_guide.md) for detailed lessons learned

### When to Use
- Reasoning tasks (math, logic)
- When you can compute rewards online
- Improving specific capabilities with feedback

### Critical Requirements
1. **Must use Instruct models** - GRPO needs instruction-tuned base models
2. **Lower learning rate** - 1e-6 for small, 5e-7 for large (7B+)
3. **Streaming for large datasets** - Use streaming mode with seed for reproducibility

### Dataset Format
```json
{
  "prompt": "Solve: 2x + 5 = 13"
}
```
Only prompts needed - model generates responses during training.

### Key Parameters
```python
GRPOConfig(
    num_generations=4,   # Responses per prompt
    learning_rate=1e-6,  # Very low (5e-7 for 7B+)
    num_train_epochs=1,
    max_completion_length=2048,  # Match to dataset distribution
)
```

### Streaming Mode (for large datasets)
```python
dataset = load_dataset(
    "nvidia/OpenMathInstruct-2",
    split="train",
    streaming=True
).shuffle(seed=42, buffer_size=10000).take(500000)
```

### Reward Functions
```python
def math_reward(completions, ground_truths):
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        reward = 0.0
        # 50% accuracy, 25% format, 15% length, 10% reasoning
        if extract_answer(completion) == truth:
            reward += 0.5
        if "\\boxed{" in completion:
            reward += 0.25
        if len(completion) < 2000:
            reward += 0.15
        if completion.count("Step") >= 2:
            reward += 0.10
        rewards.append(reward)
    return rewards
```

### Recommended Math Datasets
| Dataset | Size | Max Length | Best For |
|---------|------|------------|----------|
| `nvidia/OpenMathInstruct-2` | 14M | 2048 | Most efficient |
| `open-r1/OpenR1-Math-220k` | 220K | 8192 | Long reasoning |
| `AI-MO/NuminaMath-CoT` | 860K | 4096 | Olympiad level |

### Memory Requirements
| Model Size | GPU | Quantization |
|------------|-----|--------------|
| 0.5B-1.5B | 10GB MIG | bf16 |
| 1.5B-4B | 40GB MIG | bf16 |
| 7B-14B | 40GB MIG | 4-bit |

## Method Selection Guide

```
                    ┌─────────────────────┐
                    │ What's your goal?   │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
  ┌───────────┐         ┌───────────┐         ┌───────────┐
  │ New task  │         │ Improve   │         │ Reasoning │
  │ or domain │         │ quality   │         │ tasks     │
  └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │   SFT   │          │   DPO   │          │  GRPO   │
   └─────────┘          └─────────┘          └─────────┘
```

## Typical Pipeline

```
Base Model
    │
    ▼ (SFT)
Instruction-Tuned Model
    │
    ▼ (DPO)
Aligned Model
    │
    ▼ (Optional: GRPO for specific skills)
Final Model
    │
    ▼ (Evaluation)
Validated Model
    │
    ▼ (GGUF Conversion)
Deployable Model
```

## Comparison Table

| Aspect | SFT | DPO | GRPO |
|--------|-----|-----|------|
| **Input** | Examples | Preferences | Prompts + Rewards |
| **Learning Rate** | 2e-5 | 5e-7 | 1e-6 |
| **Epochs** | 1-3 | 1 | 1 |
| **Memory** | Moderate | Higher (pairs) | Higher (generations) |
| **Speed** | Fast | Moderate | Slow |
| **Complexity** | Low | Low | Medium |
| **When** | First | After SFT | For specific skills |
