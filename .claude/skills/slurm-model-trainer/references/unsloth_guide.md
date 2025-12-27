# Unsloth Optimization Guide

## Overview

Unsloth provides:
- **2-3x faster training** than standard TRL
- **80% less VRAM** usage
- Optimized Triton kernels
- Native TRL integration

## Installation

```bash
pip install unsloth
```

## Supported Models

### Fully Supported
- Llama 2, Llama 3, Llama 3.1, Llama 3.2
- Mistral, Mixtral
- Qwen, Qwen2, Qwen2.5
- Phi-2, Phi-3
- Gemma, Gemma 2
- DeepSeek

### Partial Support
- Other transformer architectures (may work with reduced speedup)

## Basic Usage

### Loading Model with Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B",
    max_seq_length=2048,
    dtype=None,  # Auto-detect (bf16 for H100)
    load_in_4bit=True,  # 4-bit quantization
)
```

### Applying LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Optimized for 0
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth optimized
    random_state=42,
)
```

## SFT with Unsloth

```python
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# Training config
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Higher LR for Unsloth
    bf16=True,
    optim="adamw_8bit",  # 8-bit optimizer
    logging_steps=10,
    report_to="trackio",
    push_to_hub=True,
    hub_model_id="ermiaazarkhalili/my-unsloth-model",
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## DPO with Unsloth

```python
from unsloth import FastLanguageModel
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

# Training
training_args = DPOConfig(
    output_dir="./output",
    beta=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    bf16=True,
    optim="adamw_8bit",
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
    processing_class=tokenizer,
)
trainer.train()
```

## Key Optimizations

### 1. Gradient Checkpointing

```python
# Use Unsloth's optimized gradient checkpointing
use_gradient_checkpointing="unsloth"  # NOT True/False
```

### 2. 8-bit Optimizer

```python
optim="adamw_8bit"  # Reduces memory for optimizer states
```

### 3. Zero Dropout

```python
lora_dropout=0  # Unsloth optimizes for zero dropout
```

### 4. Automatic Dtype

```python
dtype=None  # Let Unsloth auto-detect (bf16 for H100/A100)
```

## Memory Comparison

| Model | Standard | Unsloth | Savings |
|-------|----------|---------|---------|
| 7B | ~28 GB | ~6 GB | -78% |
| 13B | ~52 GB | ~10 GB | -80% |
| 34B | ~136 GB | ~24 GB | -82% |

## Speed Comparison

| Model | Standard | Unsloth | Speedup |
|-------|----------|---------|---------|
| 7B | 1x | 2.5x | 150% faster |
| 13B | 1x | 2.2x | 120% faster |
| 34B | 1x | 2.0x | 100% faster |

## Best Practices

### 1. Use 4-bit Quantization

```python
load_in_4bit=True
```

### 2. Use Packing

```python
SFTConfig(
    packing=True,  # Combine short sequences
)
```

### 3. Higher Learning Rate

```python
# Unsloth can handle higher LR
learning_rate=2e-4  # vs 2e-5 for standard
```

### 4. Proper Target Modules

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]
```

## Saving Models

### Save LoRA Adapter

```python
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
```

### Merge and Save

```python
# Merge LoRA weights into base model
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

### Push to Hub

```python
model.push_to_hub("ermiaazarkhalili/my-model")
tokenizer.push_to_hub("ermiaazarkhalili/my-model")
```

## Troubleshooting

### "Unsloth not installed"

```bash
pip install unsloth
```

### "Model not supported"

Check supported models list. If not supported:
1. Use standard TRL (`train_sft.py` instead of `train_sft_unsloth.py`)
2. Check Unsloth GitHub for updates

### "CUDA error"

1. Reduce batch size
2. Use smaller sequence length
3. Ensure 4-bit quantization enabled

### Training Slower Than Expected

1. Verify `use_gradient_checkpointing="unsloth"`
2. Check `lora_dropout=0`
3. Use `optim="adamw_8bit"`

## Quick Reference

### Load Model
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="MODEL",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### Apply LoRA
```python
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
)
```

### Training Config
```python
SFTConfig(
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_8bit",
)
```
