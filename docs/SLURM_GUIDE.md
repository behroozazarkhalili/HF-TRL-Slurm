# SLURM Guide for HF-TRL on Fir Cluster (DRAC)

## Overview

This guide covers SLURM commands, job dependencies, monitoring, and best practices for training language models on the Fir cluster.

---

## Job Dependencies

The `--dependency` flag tells SLURM to hold a job until conditions on other jobs are met. Essential for building training pipelines.

### Dependency Types

| Flag | Meaning |
|------|---------|
| `--dependency=afterok:JOBID` | Run only if JOBID **succeeded** (exit code 0) |
| `--dependency=afternotok:JOBID` | Run only if JOBID **failed** (non-zero exit) |
| `--dependency=afterany:JOBID` | Run after JOBID finishes, **regardless** of exit code |
| `--dependency=after:JOBID` | Run after JOBID **starts** (not finishes) |
| `--dependency=singleton` | Only one job with this `--job-name` runs at a time |

### Chaining Multiple Jobs

```bash
# Chain: train → evaluate → push to hub
JOB1=$(sbatch --parsable train.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 evaluate.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 push_to_hub.sh)
echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
```

> `--parsable` makes sbatch output just the job ID (no text), so you can capture it in a variable.

### Multiple Dependencies

```bash
# Wait for BOTH jobs to succeed
sbatch --dependency=afterok:111:222 final.sh

# Wait for job1 to succeed AND job2 to finish (any status)
sbatch --dependency=afterok:111,afterany:222 final.sh
```

### The `singleton` Pattern

Prevents duplicate runs of the same job:

```bash
#SBATCH --job-name=nightly-eval
#SBATCH --dependency=singleton
```

If submitted twice, the second waits until the first finishes. Great for cron-like scheduled jobs where you don't want overlap.

---

## Continual Training Pipeline

Chain training runs beyond the 7-day SLURM limit using `--continue_from_adapter`:

```bash
# Stage 1: Train 10K samples
JOB1=$(sbatch --parsable jobs/granite-10k-part1.sh)

# Stage 2: Continue from Stage 1's adapter on NEW data
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/granite-10k-part2.sh)

# Stage 3: Continue from Stage 2's adapter
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 jobs/granite-10k-part3.sh)

echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
```

Each stage uses:
- `--continue_from_adapter <previous_output_dir>` — loads and merges prior adapter
- `--seed <different_seed>` — ensures new data samples from the streaming dataset
- A separate `--output_dir` — keeps metrics comparable between stages

> Use `afterok` so that if Stage N fails, Stage N+1 doesn't try to load a non-existent adapter.

---

## Essential Commands

### Job Submission

| Command | Description |
|---------|-------------|
| `sbatch script.sh` | Submit a job |
| `sbatch --parsable script.sh` | Submit and return only the job ID |
| `sbatch --dependency=afterok:JOBID script.sh` | Submit with dependency |
| `sbatch --hold script.sh` | Submit in held state (release later) |
| `scontrol release JOBID` | Release a held job |

### Monitoring

| Command | Description |
|---------|-------------|
| `squeue -u $USER` | List your queued/running jobs |
| `squeue -u $USER --start` | Estimated start times for pending jobs |
| `scontrol show job JOBID` | Full job details (partition, dependency, resources) |
| `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed` | Post-mortem on finished jobs |
| `sacct -j JOBID --format=JobID,MaxRSS,MaxVMSize,Elapsed` | Memory usage of finished jobs |

### Job Control

| Command | Description |
|---------|-------------|
| `scancel JOBID` | Cancel a specific job |
| `scancel -u $USER` | Cancel ALL your jobs |
| `scancel --state=PENDING -u $USER` | Cancel only pending jobs |
| `scontrol hold JOBID` | Hold a pending job |
| `scontrol release JOBID` | Release a held job |
| `scontrol update JobId=JOBID TimeLimit=2-00:00:00` | Extend time limit (if allowed) |

### Cluster Info

| Command | Description |
|---------|-------------|
| `sinfo -p gpubase_bygpu_b1 --Node` | Available nodes in partition |
| `sinfo -p gpubase_bygpu_b1 -o "%n %G %C %m"` | Node GPU/CPU/memory details |
| `sacctmgr show qos format=Name,MaxWall` | Max wall time per QOS |
| `sshare -u $USER` | Your fairshare priority |

---

## Output Formatting

Customize `squeue` output with `--format`:

```bash
# Compact view with job name, state, time, and node
squeue -u $USER --format="%.10i %.9P %.55j %.8T %.10M %.6D %R"

# Show dependencies
squeue -u $USER --format="%.10i %.30j %.8T %.20E"
```

Customize `sacct` output:

```bash
# Detailed job accounting
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AllocGRES

# Jobs from last 7 days
sacct --starttime=$(date -d '7 days ago' +%Y-%m-%d) -u $USER --format=JobID,JobName,State,Elapsed
```

---

## GPU Resources on Fir

### Available GPU Types

| GRES String | GPU | Memory | Notes |
|-------------|-----|--------|-------|
| `gpu:nvidia_h100_80gb_hbm3:1` | H100 (full) | 80 GB | Best for large models |
| `gpu:nvidia_h100_80gb_hbm3_3g.40gb:1` | H100 MIG 3g | 40 GB | Sufficient for <4B models |

### MIG Considerations

- MIG slices (3g.40gb) have 40 GB VRAM — enough for most <4B parameter models with LoRA
- Some MIG nodes have broken CUDA detection; the training scripts handle this via `is_mig_gpu()`
- MIG GPUs should use bf16 (not fp16) for H100 architecture

---

## Best Practices

### Always Use PYTHONUNBUFFERED

Without it, stdout is buffered and metrics are lost until job completion or timeout:

```bash
# In job scripts, before the python command:
PYTHONUNBUFFERED=1 python train_grpo.py ...
```

### Smoke Test Before Full Runs

Always submit a short test (<1 hour) before committing to multi-day runs:

```bash
# Quick smoke test: 15 steps, 100 samples
sbatch jobs/smoke-granite-4.0-micro-grpo-numina-10k.sh

# Only after smoke passes:
sbatch jobs/granite-4.0-micro-grpo-numina-10k.sh
```

### Reading Metrics from Checkpoints

When stdout is buffered (forgot `PYTHONUNBUFFERED=1`), read metrics from checkpoint files:

```python
import json

with open('/scratch/ermia/outputs/{job}/checkpoint-{step}/trainer_state.json') as f:
    state = json.load(f)

for entry in state['log_history']:
    step = entry.get('step', 0)
    loss = entry.get('loss', 'N/A')
    reward = entry.get('reward', 'N/A')
    clipped = entry.get('completions/clipped_ratio', 'N/A')
    print(f"Step {step}: loss={loss}, reward={reward}, clipped={clipped}")
```

### Log File Locations

```bash
# SLURM stdout/stderr (set in job script)
logs/{job-name}-{job-id}.out
logs/{job-name}-{job-id}.err

# Training outputs and checkpoints
/scratch/ermia/outputs/{run-name}/
├── checkpoint-{step}/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── trainer_state.json
├── adapter_model.safetensors      # Final adapter
└── tokenizer/
```

### Time Estimation

For Granite 4.0-micro on MIG 3g.40gb with GRPO (batch=2, grad_accum=8, num_gen=4):

| Max Completion Length | Time/Step | Steps for 10K samples | Total Time |
|----------------------|-----------|----------------------|------------|
| 512 | ~40s | 625 | ~7 hours |
| 2048 | ~84s | 625 | ~14.5 hours |

Formula: `total_steps = max_samples / (per_device_batch * grad_accum * num_generations)`

---

## Common Patterns

### Retry on Failure

```bash
# Submit a retry job that only runs if the original fails
MAIN=$(sbatch --parsable jobs/train.sh)
sbatch --dependency=afternotok:$MAIN jobs/train.sh  # same script, retries on failure
```

### Fan-Out / Fan-In

```bash
# Train 3 models in parallel, then run comparison
M1=$(sbatch --parsable jobs/model-a.sh)
M2=$(sbatch --parsable jobs/model-b.sh)
M3=$(sbatch --parsable jobs/model-c.sh)

# Compare after ALL three finish
sbatch --dependency=afterok:$M1:$M2:$M3 jobs/compare-models.sh
```

### Preemption-Safe Jobs

```bash
#SBATCH --signal=B:USR1@120   # Send USR1 signal 120 seconds before timeout
#SBATCH --requeue              # Automatically requeue on preemption
```

The training scripts save checkpoints periodically, so `--resume_from_checkpoint latest` can pick up after preemption.
