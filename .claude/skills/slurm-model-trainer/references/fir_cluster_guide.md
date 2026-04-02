# Fir Cluster Guide (DRAC)

## Overview

The Fir cluster is part of the Digital Research Alliance of Canada (DRAC) infrastructure, featuring NVIDIA H100 GPUs with support for Multi-Instance GPU (MIG) configurations.

## Hardware

### GPU Types

| Type | VRAM | GRES Request | Best For |
|------|------|--------------|----------|
| **H100 (full)** | 80GB | `gpu:h100:1` | Large models (7B-70B), multi-GPU |
| H100 MIG 3g.40gb | 40GB | `gpu:nvidia_h100_80gb_hbm3_3g.40gb:1` | Medium models (3B-13B) |
| H100 MIG 2g.20gb | 20GB | `gpu:nvidia_h100_80gb_hbm3_2g.20gb:1` | Small models (0.5B-3B) |
| H100 MIG 1g.10gb | 10GB | `gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` | Testing, tiny models (<1B) |

### Partitions

#### Single/Multiple GPU Partitions (`gpubase_bygpu_*`)

| Partition | Time Limit | Priority | Use Case |
|-----------|------------|----------|----------|
| `gpubase_bygpu_b1` | 3 hours | Highest | Quick tests, debugging |
| `gpubase_bygpu_b2` | 12 hours | High | Short training runs |
| `gpubase_bygpu_b3` | 1 day | Medium | Standard training |
| `gpubase_bygpu_b4` | 3 days | Medium-Low | Extended training |
| `gpubase_bygpu_b5` | 7 days | Lower | Long GRPO/large model runs |

#### Full Node Partitions (`gpubase_bynode_*`) - 4x H100 80GB

| Partition | Time Limit | Use Case |
|-----------|------------|----------|
| `gpubase_bynode_b1` | 3 hours | Multi-GPU testing |
| `gpubase_bynode_b2` | 12 hours | Short multi-GPU training |
| `gpubase_bynode_b3` | 1 day | Standard multi-GPU |
| `gpubase_bynode_b4` | 3 days | Extended multi-GPU |
| `gpubase_bynode_b5` | 7 days | Long multi-GPU runs |

#### Special Partitions

| Partition | Time Limit | Notes |
|-----------|------------|-------|
| `gpubackfill` | 1 day | May be preempted, good for cost savings |
| `gpupreempt` | 122 days | Will be preempted anytime, lowest priority |
| `gpubase_interac` | 3 hours | Interactive sessions (MIG only) |

## Storage

### Directories

| Path | Quota | Purge Policy | Use For |
|------|-------|--------------|---------|
| `$HOME` | ~50GB | None | Scripts, configs |
| `$SCRATCH` | Large | 60 days unused | Training outputs, cache |
| `$PROJECT` | Varies | None | Shared project files |

### Recommended Setup

```bash
# Set cache directories (add to ~/.bashrc)
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/.cache/torch

# Create directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $TORCH_HOME
```

## Network

### Important: Compute Nodes Have No Internet

Compute nodes **cannot access the internet**. You must:

1. **Pre-download models/datasets** on login node:
   ```bash
   # On login node
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('MODEL_NAME')"
   python -c "from datasets import load_dataset; load_dataset('DATASET_NAME')"
   ```

2. **Configure cache directories** before job submission
3. **Include HF_TOKEN** in job exports for Hub push

### Workaround for Hub Push

Since compute nodes can't reach the internet directly, Hub push works through:
- Internal proxy (if configured)
- Checkpoint saving locally + post-job upload

## Modules

### Required Modules

```bash
module load python/3.11.5
module load cuda/12.2
module load arrow/17.0.0  # For datasets
```

### Check Available Modules

```bash
module spider python
module spider cuda
```

## Job Submission

### Basic GPU Job

```bash
#!/bin/bash
#SBATCH --account=def-YOUR_ACCOUNT
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpubase_bygpu_b3

module load python/3.11.5 cuda/12.2
source $PROJECT/envs/hf-trl/bin/activate

python train.py
```

### Multi-GPU Job

```bash
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=gpubase_bynode_b3
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32

accelerate launch --num_processes 4 train.py
```

## Best Practices

### 1. Pre-download Everything

```bash
# Create download script
cat > download_models.py << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Download model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Download dataset
dataset = load_dataset("trl-lib/Capybara")

print("Downloads complete!")
EOF

# Run on login node
python download_models.py
```

### 2. Use Checkpointing

Enable checkpoint saving in case of job timeout:
```python
SFTConfig(
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
)
```

### 3. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job JOBID

# View job output
tail -f logs/job-JOBID.out

# Check past jobs
sacct -u $USER --starttime=2024-01-01
```

### 4. Resource Estimation

Use the walltime estimator:
```bash
python utils/estimate_time.py --model Qwen/Qwen2.5-7B --dataset_size 10000 --epochs 3
```

## Troubleshooting

### Job Pending Too Long

```bash
# Check why job is pending
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Common reasons:
# - Resources: Request smaller GPU or different partition
# - Priority: Use gpubackfill for lower priority
```

### Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing
3. Use LoRA/PEFT
4. Request larger GPU

### Job Killed (Timeout)

1. Increase `--time` in SBATCH
2. Enable checkpointing
3. Use longer partition (b4 or b5)

## Useful Commands

```bash
# Your allocations
sacctmgr show assoc where user=$USER

# Cluster status
sinfo -s

# GPU availability
sinfo -o "%P %G %D %a" | grep gpu

# Your usage
sreport cluster UserUtilizationByAccount user=$USER

# Job efficiency
seff JOBID
```
