# DRAC support ticket draft: fc10713 GPU initialization failures

**Recipient:** `support@tech.alliancecan.ca` (or via https://support.alliancecan.ca web portal)

**Cluster:** Fir (login1.int.fir.alliancecan.ca)
**User:** ermia
**Account:** def-maxwl_gpu
**Node in question:** fc10713

---

## Subject
fc10713: consistent GPU/CUDA initialization failures on MIG 3g.40gb slices despite node reporting healthy

## Summary
Since ~2026-04-13, SLURM jobs that land on **fc10713** fail within 8–10 seconds with a CUDA initialization error, even though the node reports healthy state (MIXED/ALLOCATED, no drain flags, `BootTime=2026-03-25T15:17:05` stable). The same jobs succeed when scheduled on other fc-\* nodes (e.g. fc10606 confirmed working today). Our specific workload stack: StdEnv/2023 + cuda/12.6 + python/3.11.5 + torch/unsloth in a user venv.

## Error signature (from job 36164743, 2026-04-20 09:19)
```
/scratch/ermia/venvs/hf_unsloth/lib/python3.11/site-packages/torch/cuda/__init__.py:182:
UserWarning: CUDA initialization: CUDA unknown error - this may be due to an
incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES
after program start. Setting the available devices to be zero.
  return torch._C._cuda_getDeviceCount() > 0
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/scratch/ermia/venvs/hf_unsloth/lib/python3.11/site-packages/unsloth/__init__.py", line 105, in <module>
    import unsloth_zoo
  File ".../unsloth_zoo/device_type.py", line 231, in <module>
    DEVICE_TYPE : str = get_device_type()
  File ".../unsloth_zoo/device_type.py", line 218, in get_device_type
    raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

Job logs (full): `/project/6014832/ermia/HF-TRL/logs/gguf-recover-xlam-gemma4-e4b-36164743.{out,err}`

## Empirical failure rate on fc10713 (our jobs only)
Via `sacct -u ermia -S 2026-04-13 --format=NodeList,State`:

| JobID    | JobName                                | State          | Date       |
|----------|----------------------------------------|----------------|------------|
| 35349576 | smoke-carnice-sft                      | FAILED         | 2026-04-13 |
| 35349578 | smoke-carnice-xlam                     | FAILED         | 2026-04-13 |
| 35592455 | verify-unsloth-gemma4                  | FAILED         | 2026-04-15 |
| 35598063 | smoke-unsloth-sft-gemma4-26b-a4b       | FAILED         | 2026-04-15 |
| 35598064 | smoke-unsloth-xlam-gemma4-26b-a4b      | FAILED (13s)   | 2026-04-17 |
| 35598065 | smoke-unsloth-sft-gemma4-31b           | FAILED (18s)   | 2026-04-17 |
| 36164743 | gguf-recover-xlam-gemma4-e4b           | FAILED (8s)    | 2026-04-20 |

**7 consecutive failures on fc10713** over 3 days (2026-04-17 to 2026-04-20). All died in <30s at CUDA init. Same jobs succeed on other nodes (same stack, same venv, same gres request).

## Diagnostic probe
To narrow the root cause, we ran a minimal probe forced to `--nodelist=fc10713` (SLURM job **36221819** submitted 2026-04-20). Output at:
- `/project/6014832/ermia/HF-TRL/logs/probe-fc10713-gpu-36221819.out`
- `/project/6014832/ermia/HF-TRL/logs/probe-fc10713-gpu-36221819.err`

The probe captures: SLURM env, cgroup GPU device list, `nvidia-smi` host view, `nvidia-smi -L`, `nvidia-smi mig -lgi/-lci`, module stack, and a minimal `torch.cuda` test.

## Diagnostic results (from probe 36221819, 2026-04-20 11:50:55 PDT, 49s runtime)

**Key findings:**

1. **`nvidia-smi` reports a healthy GPU:** NVIDIA H100 80GB HBM3, driver 580.65.06, CUDA 13.0, 22°C, 66W. One MIG 3g.40gb instance allocated and visible (UUID: `MIG-5809cf1f-584f-5b41-8799-e0ef738266e0`). Memory-Usage: 44 MiB / 40448 MiB.

2. **SLURM/cgroup correctly sets `CUDA_VISIBLE_DEVICES`:**
   ```
   CUDA_VISIBLE_DEVICES=MIG-5809cf1f-584f-5b41-8799-e0ef738266e0
   NVIDIA_VISIBLE_DEVICES=<unset>   ← possibly a problem
   ```
   Notable: `NVIDIA_VISIBLE_DEVICES` is unset. On other cluster nodes (fc10606, fc11009, fc11016) where the same workload succeeds, this env var is typically set by the SLURM prolog. This may or may not be relevant but is worth checking.

3. **`/dev/nvidia*` devices are present and world-accessible:**
   ```
   crw-rw-rw- /dev/nvidia0..3, /dev/nvidiactl, /dev/nvidia-modeset, /dev/nvidia-uvm, /dev/nvidia-uvm-tools
   drwxr-xr-x /dev/nvidia-caps/
   cr--r--r-- /dev/nvidia-caps/nvidia-cap156, 157, 165, 166, 21, 2   ← readable
   cr-------- /dev/nvidia-caps/nvidia-cap1                           ← root-only (this is the MIG fabric cap)
   ```

4. **`nvidia-smi mig -lgi`  → "Insufficient Permissions"** (from inside the user namespace):
   ```
   Failed to display GPU instances: Insufficient Permissions
   No GPU instances found: Insufficient Permissions
   ```
   This is the critical signal. On other working nodes the same command from inside a user SLURM job either returns the MIG layout or is read-only but accessible. Here the MIG capability binding is missing for the user context.

5. **`torch.cuda.is_available()` → `False`** with the same error we saw on 36164743:
   ```
   RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment,
   e.g. changing env variable CUDA_VISIBLE_DEVICES after program start.
   ```

**Interpretation:** The GPU hardware is fine. The MIG *instance* exists on the device. SLURM correctly requests the slice. But inside the user's job namespace, the MIG capability (`/dev/nvidia-caps/nvidia-capX` for our compute-instance) is not properly bound/exposed. `nvidia-smi` can read the host-level MIG view but `nvidia-smi mig -lgi` and `torch` cannot reach the per-CI capability needed to actually use the MIG slice. This looks like a **MIG prolog/namespace configuration issue on fc10713**, not hardware, not driver version, and not our software stack.

**Hypotheses** (in order of likelihood):

1. **MIG compute-instance capability binding broken on fc10713.** The SLURM prolog that should bind per-CI capability files into the user cgroup/namespace isn't running correctly. Rebooting the node or re-running `nvidia-smi mig -dc && nvidia-smi mig -cgi ... && nvidia-smi mig -cci` might reset this.

2. **Missing `NVIDIA_VISIBLE_DEVICES` env var.** On other nodes this is set alongside `CUDA_VISIBLE_DEVICES` by the prolog. Its absence here suggests the fc10713 prolog has drifted.

3. **`/dev/nvidia-caps/nvidia-cap1` being root-only** (line above shows `cr--------`) — if the user's MIG CI needs access to the fabric-level cap, this is blocked. Other nodes likely have this readable.

## Impact
We currently use `#SBATCH --exclude=fc10713` on all training/inference jobs as a workaround. This works for us but:
- Reduces cluster capacity for other users who may hit the same issue silently
- Jobs that accidentally land there waste a full queue cycle (we had a multi-hour queue wait followed by 8s failure on 36164743)

## Request
Please investigate fc10713's MIG capability binding. Based on the probe output, most likely causes in order:

1. **MIG CI capability not bound in user namespace.** Check the SLURM prolog/epilog scripts for MIG handling on fc10713 vs other working nodes (fc10606, fc11009, fc11016). Likely a prolog drift or a broken `nvidia-smi mig -cci` on this node.

2. **Reconfigure MIG partitions:** `nvidia-smi mig -dci && nvidia-smi mig -dgi` followed by re-creating the 3g.40gb + 2g.20gb + 1g.10gb layout shown in `scontrol show node fc10713`.

3. **Reboot the node** (last boot: 2026-03-25, ~26 days ago). This usually resets MIG capability state as a last-resort fix.

4. **Check why `NVIDIA_VISIBLE_DEVICES` is unset in SLURM jobs on fc10713 but set on other nodes.** Compare `/etc/slurm/prolog.d/` scripts or whatever mechanism propagates this var.

Happy to run additional probes or provide more data as needed.

Thanks,
Ermia
