# fc10713 CUDA initialization failure — reproducer package

**Contact:** Behrooz Azarkhalili (ermia) <ermiaazarkhalili@gmail.com>
**Affected node:** `fc10713` on Fir cluster (DRAC)
**Account:** `def-maxwl_gpu`
**First observed:** 2026-04-17
**Most recent confirmed failure:** 2026-04-20 (SLURM job 36164743)
**Total documented failures:** 7 jobs across 4 days

---

## Summary

Node `fc10713` reports as healthy to SLURM (`State=MIXED`, no drain flags,
`scontrol show node fc10713` reports normal MIG layout) but **every GPU
workload that lands on it fails within 8–30 seconds** with a CUDA
initialization error. The failure is deterministic and node-specific —
the same job script succeeds on any other node in the same partition
with the same MIG slice type (`nvidia_h100_80gb_hbm3_3g.40gb`).

---

## Expected error

When torch is imported and `torch.cuda.is_available()` is called on
fc10713, PyTorch emits:

```
UserWarning: CUDA initialization: CUDA unknown error - this may be due
to an incorrectly set up environment, e.g. changing env variable
CUDA_VISIBLE_DEVICES after program start. Setting the available devices
to be zero. (Triggered internally at /tmp/build_wheels_tmp.6038/
python-3.11/torch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
```

and `torch.cuda.is_available()` returns `False` despite `CUDA_VISIBLE_DEVICES`
being set correctly by SLURM.

For downstream libraries like Unsloth that require a CUDA device at import
time, this manifests as:

```
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

See `evidence-logs/job-36164743.err` for the full stack trace.

---

## How to reproduce

Three SLURM scripts are provided:

| Script | Pins to | Expected | Purpose |
|---|---|---|---|
| `reproduce-minimal.sh` | fc10713 | FAIL in ~5s | Minimal torch CUDA probe — no venv needed |
| `reproduce-workload.sh` | fc10713 | FAIL in ~10s | Matches the exact Unsloth import chain from job 36164743 |
| `reproduce-control.sh` | any node EXCEPT fc10713 | PASS in ~10s | Proves the failure is node-specific |

### Running

```bash
cd /project/6014832/ermia/HF-TRL/support-tickets/fc10713-cuda-init-failure
sbatch reproduce-minimal.sh     # pinned to fc10713 — expect FAIL
sbatch reproduce-control.sh     # any non-fc10713 node — expect PASS
sbatch reproduce-workload.sh    # pinned to fc10713 — expect FAIL (needs unsloth venv)
```

Note: the `--nodelist=fc10713` constraint on the failing-case scripts may
take several hours to schedule because it pins to one specific node. The
control script (`--exclude=fc10713`) will run as soon as any other
MIG 3g.40gb slice is available.

---

## Evidence of the bug pattern (7 failures)

Logs from 7 jobs that failed on fc10713 between 2026-04-17 and 2026-04-20
are in `evidence-logs/`. All share the same failure signature:

| Job ID | Elapsed | Date | Workload |
|---|---|---|---|
| 35349576 | 24s | 2026-04-17 | Unsloth Carnice-9B SFT smoke |
| 35349578 | 30s | 2026-04-17 | Unsloth Carnice-9B xLAM smoke |
| 35592455 | 18s | 2026-04-17 | Unsloth Gemma4 verify |
| 35598063 | 16s | 2026-04-17 | Unsloth Gemma4-26B SFT smoke |
| 35598064 | 13s | 2026-04-17 | Unsloth Gemma4-26B xLAM smoke |
| 35598065 | 18s | 2026-04-17 | Unsloth Gemma4-31B SFT smoke |
| 36164743 | 8s  | 2026-04-20 | GGUF recovery (Gemma4-E4B xLAM) |

All seven crashed at CUDA init, before any real work could start.

### Successful control runs on non-fc10713 nodes

The same job scripts (with `--exclude=fc10713` added) subsequently ran
successfully on `fc10609` and other nodes. Most recently:

- Job **36446084** (LFM2.5-350M xLAM smoke) — COMPLETED 04:19 on fc10609 (2026-04-22 05:22)
- Job **36446085** (LFM2.5-350M SFT smoke)  — COMPLETED 01:27 on fc10609 (2026-04-22 05:28)

---

## What I have already checked

- `scontrol show node fc10713` reports `State=MIXED`, no drain flags,
  `AvailableFeatures=h100mig`, `ActiveFeatures=h100mig`, MIG layout normal
- Node has been up continuously: `BootTime=2026-03-25T15:17:03`
- No obvious hardware-level alarm in `sinfo -p gpubase_bygpu_b1`
- The issue is not my Python environment: the same venv works on fc10609
  and other nodes without any changes
- The issue is not my SLURM request: `--gres` and `CUDA_VISIBLE_DEVICES`
  look identical between fc10713 and successful nodes
- A single brief probe (SLURM job 36221819, 2026-04-20 11:50) did COMPLETE
  in 49 seconds, suggesting the failure may be intermittent — but the
  subsequent real workload (job 36164743 at 09:19) and all earlier jobs
  failed deterministically

---

## Files in this directory

```
README.md                       ← this file
reproduce-minimal.sh            ← minimal torch CUDA reproducer (pinned to fc10713)
reproduce-control.sh            ← same workload, excluding fc10713 (should pass)
reproduce-workload.sh           ← Unsloth workload reproducer (pinned to fc10713)
evidence-logs/                  ← stdout/stderr from 7 historical failures
    job-35349576.{out,err}
    job-35349578.{out,err}
    job-35592455.{out,err}
    job-35598063.{out,err}
    job-35598064.{out,err}
    job-35598065.{out,err}
    job-36164743.{out,err}
```

All scripts are self-contained, use only public modules (`StdEnv/2023`,
`gcc`, `arrow`, `python/3.11.5`, `cuda/12.6`), and do not require any of
my personal tokens or credentials.

---

## What would help resolve this

1. Confirm the node-level health of fc10713's MIG slices (dmesg, nvidia-smi
   on the compute node, GPU ECC state, driver restart, MIG reconfig)
2. If the issue is transient (intermittent driver state), please share
   the node's recent reboot or CUDA driver-reset history
3. If a fix is applied, I'm happy to re-run the reproducer scripts to
   confirm
