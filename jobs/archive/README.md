# Archived Job Scripts

Historical SLURM job scripts that are no longer part of the active training pipeline.
Kept for provenance and as reference for reproducing past experiments.

Nothing here is submitted by any current orchestrator (`train-unsloth-all.sh`,
`smoke-unsloth-all.sh`, `batch-gguf-*.sh`, or the chain generators). None of
these scripts have job names matching anything currently in `squeue`.

## Contents

| Subdirectory | Topic | Count | Status |
|--------------|-------|-------|--------|
| `grpo-openmath-experiments/` | Early GRPO runs on `open-r1/OpenR1-Math-220k` and `NuminaMath-OpenMath2` datasets (Dec 2025 – Jan 2026) | 24 | Superseded by `*-grpo-numina-*.sh` (NuminaMath-CoT). |
| `sft-ultrachat/` | SFT runs on `HuggingFaceH4/ultrachat_200k` (late Dec 2025) | 16 | Superseded by Claude reasoning distillation SFT notebooks under `notebooks/`. |
| `sft-openhermes/` | SFT runs on `teknium/OpenHermes-2.5` (late Dec 2025) | 6 | Abandoned — OpenHermes dropped in favor of Claude-generated data. |
| `smoke-historical/` | Historical smoke-test scripts for one-model experiments (vision, claude reasoning, math variants, etc.) — Mar–early-Apr 2026 | 36 | Not referenced by current squeue or orchestrators (`smoke-unsloth-all.sh` covers today's smoke suite). |
| `full-runs-historical/` | Full-scale GRPO NuminaMath training scripts (single-model, per-size), Mar–early-Apr 2026 | 35 | Superseded by per-model Unsloth notebooks + `train-unsloth-all.sh`. |
| `misc/` | One-offs: granite inference sample, early test harness | 2 | Reference only. |

**Total archived: 119 scripts** across 6 subdirs.

## What remains active in `jobs/`

Only 11 scripts live at the top of `jobs/` now — these are strictly:

1. Scripts whose SBATCH `--job-name` matches a currently-queued job (`batch-gguf-*`), OR
2. Orchestrators / chain generators that PRODUCE queued jobs at runtime (`*-chain.sh`, `*-unsloth-all.sh`, `build-claude-reasoning-distillation.sh`), OR
3. Utility (`gpu-diagnostic.sh`) kept for on-demand debug.

If you find yourself needing to archive another script later, the rule is:
> Is it referenced by `squeue -u $USER` OR by one of the 11 active orchestrators? If no → archive it.

## Reviving a script

If you need to re-run one of these:

```bash
# Move back to jobs/
git mv jobs/archive/<subdir>/<script>.sh jobs/

# Update paths/constants to current conventions (chain_utils.sh, templates,
# production SBATCH partitions, etc.), then submit as usual:
sbatch jobs/<script>.sh
```

Expect the older scripts to need light modernization — they pre-date
`jobs/lib/chain_utils.sh` and `jobs/templates/` and may hardcode paths
that have since been parameterized.
