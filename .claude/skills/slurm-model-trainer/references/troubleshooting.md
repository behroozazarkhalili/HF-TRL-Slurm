# Troubleshooting Guide

## Common Issues and Solutions

### 1. Out of Memory (OOM)

**Symptoms:**
- `CUDA out of memory`
- `RuntimeError: CUDA error: out of memory`

**Solutions (try in order):**

1. **Reduce batch size:**
   ```python
   per_device_train_batch_size=1
   gradient_accumulation_steps=16  # Keep effective batch size
   ```

2. **Enable gradient checkpointing:**
   ```python
   gradient_checkpointing=True
   ```

3. **Use LoRA (if not already):**
   ```python
   from peft import LoraConfig
   peft_config = LoraConfig(r=16, lora_alpha=32)
   ```

4. **Use 4-bit quantization:**
   ```python
   --use_4bit  # In script arguments
   ```

5. **Request larger GPU:**
   ```bash
   --gres=gpu:h100:1  # Instead of MIG
   ```

### 2. Job Timeout

**Symptoms:**
- Job killed unexpectedly
- Training incomplete
- No error, just stops

**Solutions:**

1. **Increase walltime:**
   ```bash
   #SBATCH --time=48:00:00  # 2 days
   ```

2. **Use longer partition:**
   - b3 → b4 (1 day → 3 days)
   - b4 → b5 (3 days → 7 days)

3. **Enable checkpointing:**
   ```python
   save_strategy="steps"
   save_steps=100
   hub_strategy="every_save"  # Save to Hub
   ```

4. **Reduce training:**
   ```python
   num_train_epochs=1  # Fewer epochs
   max_steps=1000      # Or limit steps
   ```

### 3. Model/Dataset Not Found

**Symptoms:**
- `HTTPError: 404`
- `Connection timeout`
- `OSError: Can't load model`

**Solutions:**

1. **Pre-download on login node:**
   ```bash
   # Run BEFORE submitting job
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('MODEL')"
   python -c "from datasets import load_dataset; load_dataset('DATASET')"
   ```

2. **Check cache directory:**
   ```bash
   export HF_HOME=$SCRATCH/.cache/huggingface
   ls -la $HF_HOME/hub/
   ```

3. **Verify download completed:**
   ```bash
   # Check model files exist
   find $HF_HOME -name "*.safetensors" | head
   ```

### 4. Hub Push Failed

**Symptoms:**
- Training completes but model not on Hub
- `AuthenticationError`
- `HTTPError: 403`

**Solutions:**

1. **Check HF_TOKEN:**
   ```bash
   # In .env file
   HF_TOKEN=hf_xxxxx

   # Verify it's loaded
   echo $HF_TOKEN
   ```

2. **Ensure token has write permissions:**
   - Go to https://huggingface.co/settings/tokens
   - Create token with "Write" permission

3. **Add to SBATCH:**
   ```bash
   #SBATCH --export=ALL
   # Or explicitly:
   --export=ALL,HF_TOKEN=$HF_TOKEN
   ```

4. **Enable checkpoint push:**
   ```python
   hub_strategy="every_save"
   ```

### 5. Training Hangs

**Symptoms:**
- No progress for long time
- GPU utilization at 0%
- No error messages

**Causes & Solutions:**

1. **Eval dataset missing:**
   ```python
   # Either provide eval_dataset:
   eval_dataset=dataset_split["test"]

   # Or disable evaluation:
   eval_strategy="no"
   ```

2. **Dataloader issue:**
   ```python
   dataloader_num_workers=0  # Try disabling multiprocessing
   ```

3. **NCCL timeout (multi-GPU):**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_TIMEOUT=1800  # Increase timeout
   ```

### 6. Dataset Format Error

**Symptoms:**
- `KeyError: 'messages'`
- `ValueError: Expected column X`
- Training starts but loss is NaN

**Solutions:**

1. **Validate dataset first:**
   ```bash
   python scripts/validate_dataset.py --dataset YOUR_DATASET --method sft
   ```

2. **Check column names:**
   ```python
   from datasets import load_dataset
   ds = load_dataset("YOUR_DATASET", split="train")
   print(ds.column_names)
   print(ds[0])  # First example
   ```

3. **Apply mapping if needed:**
   ```python
   def format_for_sft(example):
       return {"text": example["instruction"] + example["output"]}
   dataset = dataset.map(format_for_sft)
   ```

### 7. Import Errors

**Symptoms:**
- `ModuleNotFoundError: No module named 'xxx'`
- `ImportError`

**Solutions:**

1. **Check environment activation:**
   ```bash
   source $PROJECT/envs/hf-trl/bin/activate
   which python  # Should be in your venv
   ```

2. **Install missing package:**
   ```bash
   pip install missing_package
   ```

3. **Re-run setup:**
   ```bash
   source utils/setup_env.sh
   ```

### 8. NCCL Errors (Multi-GPU)

**Symptoms:**
- `NCCL error`
- `Watchdog timeout`
- Training hangs on multi-GPU

**Solutions:**

1. **Set NCCL variables:**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=0
   export NCCL_NET_GDR_LEVEL=2
   ```

2. **Check network interface:**
   ```bash
   export NCCL_SOCKET_IFNAME=ib0  # or eth0
   ```

3. **Increase timeout:**
   ```bash
   export NCCL_TIMEOUT=3600
   ```

### 9. Trackio Not Working

**Symptoms:**
- No metrics in dashboard
- Trackio errors in logs

**Solutions:**

1. **Check Trackio installed:**
   ```bash
   pip show trackio
   ```

2. **Initialize properly:**
   ```python
   import trackio
   trackio.init(project="my-project", run_name="my-run")
   ```

3. **For offline compute nodes:**
   ```bash
   # Sync after job completes (on login node)
   python -c "import trackio; trackio.sync_to_hub('username/trackio')"
   ```

### 10. Unsloth Not Working

**Symptoms:**
- `ModuleNotFoundError: No module named 'unsloth'`
- `TypeError` from Unsloth functions

**Solutions:**

1. **Install Unsloth:**
   ```bash
   pip install unsloth
   ```

2. **Check compatibility:**
   - Unsloth supports specific model architectures
   - Check: https://github.com/unslothai/unsloth

3. **Fall back to TRL:**
   - Use `train_sft.py` instead of `train_sft_unsloth.py`

## Getting Help

### Check Logs

```bash
# Job output
cat logs/job-JOBID.out

# Job errors
cat logs/job-JOBID.err

# Slurm job info
scontrol show job JOBID
```

### Resource Usage

```bash
# Job efficiency
seff JOBID

# GPU utilization
nvidia-smi

# Memory usage
free -h
```

### DRAC Support

- Documentation: https://docs.alliancecan.ca/
- Support: support@tech.alliancecan.ca
