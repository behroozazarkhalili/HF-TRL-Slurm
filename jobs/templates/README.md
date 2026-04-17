# Job Templates

Blueprints for SLURM training jobs on DRAC Fir. Copy, edit the `# EDIT ME` blocks, submit.

## Which template do I use?

| Template | When to use | Structure | Submits |
|----------|-------------|-----------|---------|
| `single-job.template.sh` | Training fits in a single SLURM time window (≤ 7d on b5). No adapter resume needed. | ONE SLURM script that IS the training job. | 1 job via `sbatch` |
| `chain-training.template.sh` | Training needs to be split across multiple SLURM jobs with LoRA adapter resume between stages. | A meta-script (generator) that produces N stage scripts and submits them with `afterok` dependencies. | N jobs via `sbatch --parsable` |

**Rule of thumb:** start with `single-job.template.sh`. Move to `chain-training.template.sh` only when wall-time forces the split.

## Workflow

```bash
# Single-job case
cp jobs/templates/single-job.template.sh jobs/my-experiment.sh
# edit EDIT-ME blocks in my-experiment.sh
sbatch jobs/my-experiment.sh

# Chain case
cp jobs/templates/chain-training.template.sh jobs/my-chain.sh
# edit EDIT-ME blocks
bash jobs/my-chain.sh     # the generator submits the whole chain
```

## What both templates share

Both source `jobs/lib/chain_utils.sh` for:
- `bootstrap_training_env [venv]` — module load + venv + HF_HOME + `.env` loader
- `load_env_file <path>` — quote-tolerant `.env` reader
- `log_stage_info` / `log_stage_error` / `stage_die` — tagged logging

The chain template additionally uses:
- `submit_job <script> [dep]` — sbatch `--parsable` + `afterok` wrapper
- `resolve_prev_adapter_arg <chain_dir> <stage>` — reads stage-N-1 adapter pointer
- `save_stage_adapter <chain_dir> <stage> <output>` — writes pointer for next stage

## What each template does NOT do

- **Not a smoke test by default.** The single-job template ships with production-ish defaults (`MAX_SAMPLES=10000`, 4-day wall-time, b5 partition). The chain template is pre-sized for a 5-sample smoke test (`STAGE_SAMPLES=5`). Convert as needed — each template documents the conversion at the top.
- **Not a data loader.** Templates assume `train_grpo.py` handles dataset loading. Changes to dataset format require editing `train_grpo.py`, not the template.
- **Not GGUF conversion.** Use `jobs/batch-gguf-*.sh` for post-training GGUF export.

## Adding a new template

Conventions:
- Put it under `jobs/templates/<name>.template.sh`
- Start with a big docstring block: what / when / how / prereqs / safety
- Mark every user-editable field with `# >>>>> EDIT ME` and `# <<<<<` sentinels
- Derive everything possible from `EXP_NAME` (single source of truth)
- Source `chain_utils.sh` for shared bootstrap/logging
- Add a row to the table above
