# Trackio Monitoring Guide

## Overview

Trackio is a lightweight experiment tracking library from Hugging Face:
- Local-first: Logs stored in SQLite
- HF Hub integration: Sync to Spaces/Datasets
- Native TRL support: Built-in with TRL trainers
- Real-time dashboards: Visualize training progress

## Setup

### Installation

```bash
pip install trackio
```

### Basic Configuration

```python
import trackio

trackio.init(
    project="my-training-project",
    run_name="sft-qwen-lr2e-5",
    space_id="ermiaazarkhalili/trackio",  # Your HF Space
    config={
        "model": "Qwen/Qwen2.5-0.5B",
        "dataset": "trl-lib/Capybara",
        "learning_rate": 2e-5,
    }
)
```

## TRL Integration

### SFT Training

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./output",

    # Trackio settings
    report_to="trackio",
    run_name="sft-qwen-capybara",

    # Other settings...
    num_train_epochs=3,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### DPO Training

```python
from trl import DPOConfig, DPOTrainer

training_args = DPOConfig(
    output_dir="./output",
    report_to="trackio",
    run_name="dpo-qwen-preferences",
    # ...
)
```

## Environment Variables

Set these in your `.env` or SBATCH script:

```bash
# Project name (appears in dashboard)
export TRACKIO_PROJECT="llm-finetuning"

# Your Trackio Space
export TRACKIO_SPACE_ID="ermiaazarkhalili/trackio"

# Run grouping (for experiments)
export TRACKIO_GROUP="baseline-experiments"
```

## Slurm Integration

### SBATCH Script

```bash
#!/bin/bash
#SBATCH --job-name=trackio-train
#SBATCH --gres=gpu:h100:1

# Load environment
source /project/6014832/ermia/HF-TRL/.env
source $PROJECT/envs/hf-trl/bin/activate

# Set Trackio variables
export TRACKIO_PROJECT="sft-experiments"
export TRACKIO_SPACE_ID="ermiaazarkhalili/trackio"

# Run training
python train.py \
    --report_to trackio \
    --run_name "run-$SLURM_JOB_ID"
```

### Offline Mode

Compute nodes often lack internet. Trackio handles this:

1. **During training**: Logs saved locally to SQLite
2. **After job**: Sync from login node

```bash
# After job completes, on login node:
python -c "
import trackio
trackio.sync_to_hub('ermiaazarkhalili/trackio')
print('Synced!')
"
```

## Creating Trackio Space

### Automatic Creation

Trackio auto-creates the Space if it doesn't exist:

```python
trackio.init(
    project="my-project",
    space_id="ermiaazarkhalili/trackio",  # Auto-created
)
```

### Manual Creation

```bash
# Create Space
huggingface-cli repo create trackio --type space

# Or via Python
from huggingface_hub import create_repo
create_repo("ermiaazarkhalili/trackio", repo_type="space", space_sdk="gradio")
```

## Dashboard Features

### Metrics Tracked

- **Training metrics**: loss, learning_rate, gradient_norm
- **Evaluation metrics**: eval_loss, accuracy, perplexity
- **System metrics**: GPU memory, GPU utilization, throughput
- **Custom metrics**: Any logged values

### Viewing Dashboard

1. Go to: `https://huggingface.co/spaces/ermiaazarkhalili/trackio`
2. Select your project
3. Compare runs, view charts

### Local Dashboard

For immediate viewing without Hub:

```bash
trackio dashboard --port 7860
# Open http://localhost:7860
```

## Logging Custom Metrics

```python
import trackio

# Log single value
trackio.log({"custom_metric": 0.95})

# Log with step
trackio.log({"accuracy": 0.92}, step=500)

# Log multiple
trackio.log({
    "bleu_score": 0.45,
    "rouge_l": 0.62,
    "human_eval": 0.38,
})
```

## Comparing Runs

### Grouping Experiments

```python
trackio.init(
    project="sft-experiments",
    run_name="lr-2e-5-bs-8",
    group="learning-rate-sweep",  # Group related runs
)
```

### In Dashboard

1. Filter by group
2. Select runs to compare
3. View overlaid charts

## Best Practices

### 1. Descriptive Run Names

```python
# Good
run_name = f"sft-{model_short}-lr{lr}-bs{batch_size}"

# Bad
run_name = "run1"
```

### 2. Log Hyperparameters

```python
trackio.init(
    config={
        "model": args.model_name,
        "dataset": args.dataset_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.num_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
)
```

### 3. Use Groups for Experiments

```python
# Baseline
trackio.init(group="baseline", run_name="sft-default")

# Experiment 1
trackio.init(group="lr-sweep", run_name="lr-1e-5")
trackio.init(group="lr-sweep", run_name="lr-5e-5")
```

### 4. Sync After Training

```python
# At end of training script
trackio.finish()

# If offline, sync explicitly
trackio.sync_to_hub()
```

## Troubleshooting

### "Space not found"

```python
# Create Space first
from huggingface_hub import create_repo
create_repo("ermiaazarkhalili/trackio", repo_type="space", space_sdk="gradio")
```

### Metrics Not Appearing

1. Check `report_to="trackio"` in config
2. Verify `logging_steps` is set
3. Call `trackio.finish()` at end

### Sync Failed

```bash
# Check HF token
huggingface-cli whoami

# Manual sync with debug
python -c "
import trackio
import logging
logging.basicConfig(level=logging.DEBUG)
trackio.sync_to_hub('ermiaazarkhalili/trackio')
"
```

## Quick Reference

### Initialize
```python
trackio.init(project="name", run_name="run", space_id="user/space")
```

### Log Metrics
```python
trackio.log({"metric": value})
```

### Finish Run
```python
trackio.finish()
```

### Sync to Hub
```python
trackio.sync_to_hub("user/space")
```
