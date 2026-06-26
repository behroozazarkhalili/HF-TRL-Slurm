#!/usr/bin/env python3
"""Generate the fable-fleet SLURM wrappers (16 models x 2 datasets x {smoke,train}
= 64 wrappers) from the registry.

Mirrors the validated jobs/{smoke,train}-qwen35-9b-sft.sh structure:
  - module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6
  - venv /scratch/ermia/venvs/hf_unsloth
  - MIG gpu nvidia_h100_80gb_hbm3_3g.40gb:1, --exclude=fc10713
  - papermill drives the notebook; smoke patches SMOKE_TEST=True, train asserts False
  - .env loaded via the safe key-value loop (compute node, outside the read-guard hook)

Resources keyed by (size_tag, dataset, phase):
  smoke      : b1, 1.5h, mem by size                       (100-row, minutes)
  train on B : b2, 2-3h, mem by size                       (4,350 rows)
  train on D : b2, walltime scaled (397K rows ~= 90x B),   (checkpoint/resume on)
               mem by size; capped <= partition max.
"""
from __future__ import annotations
import json, stat
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
JOBS = ROOT / "jobs"
REG = json.loads((ROOT / "scripts/tmp/fable_fleet_registry.json").read_text())
MODELS = REG["models"]
DATASETS = REG["datasets"]

# mem (GB) + cpus by size_tag — matches the existing fleet's per-size resourcing.
MEM = {"3b": 40, "4b": 40, "8b": 48, "9b": 48}
CPUS = {"3b": 8, "4b": 8, "8b": 8, "9b": 8}
# train walltime (H) by (size_tag, dataset). D is ~90x B → much longer.
TRAIN_H = {
    ("3b", "b"): 2, ("4b", "b"): 2, ("8b", "b"): 3, ("9b", "b"): 3,
    ("3b", "d"): 12, ("4b", "d"): 12, ("8b", "d"): 16, ("9b", "d"): 16,
}
SMOKE_H = 2  # generous; smoke is 100 rows / 5 steps


def hhmm(h: int) -> str:
    return f"0-{h:02d}:00:00"


def smoke_wrapper(m: dict, ds_key: str) -> str:
    ds = DATASETS[ds_key]
    nb = f"fable_distillation_{m['nickname']}_{ds['name_tag']}_unsloth.ipynb"
    job = f"smoke-{m['nickname']}-{ds['name_tag']}"
    out_pref = f"fable_{m['dir']}_{ds_key}"
    size = m["size_tag"]
    return f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account=def-maxwl_gpu
#SBATCH --time={hhmm(SMOKE_H)}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={CPUS[size]}
#SBATCH --mem={MEM[size]}G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/fable-smoke-{m['dir']}-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "SMOKE: {m['name']} on FABLE-5 ({ds['suffix']})"
echo "Node:  $SLURMD_NODENAME   Job: $SLURM_JOB_ID   Start: $(date -Iseconds)"
echo "=========================================="

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6
source "$VENV/bin/activate"
export SCRATCH="${{SCRATCH:-/scratch/$USER}}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${{value%\\"}}"; value="${{value#\\"}}"
        value="${{value%\\'}}"; value="${{value#\\'}}"
        export "$key=$value"
    done < "$PROJECT_DIR/.env"
fi

export NB_INPUT="$NB_DIR/{nb}"
export NB_SMOKE="$OUTPUT_DIR/{out_pref}_smoke_input.ipynb"
export NB_OUTPUT="$OUTPUT_DIR/{out_pref}_smoke_output.ipynb"

python3 - <<'PYEOF'
import json, os
nb = json.load(open(os.environ["NB_INPUT"]))
patched = 0
for c in nb['cells']:
    s = c.get('source', '')
    if isinstance(s, str):
        if 'SMOKE_TEST: bool = False' in s:
            c['source'] = s.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True'); patched += 1
    elif isinstance(s, list):
        new = [l.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True') for l in s]
        if new != s: c['source'] = new; patched += 1
json.dump(nb, open(os.environ["NB_SMOKE"], 'w'), indent=1)
print(f"[OK] Patched SMOKE_TEST in {{patched}} cell(s)")
assert patched == 1, f"Expected exactly 1 patch, got {{patched}}"
PYEOF

echo "[....] Running via papermill..."
PYTHONUNBUFFERED=1 papermill "$NB_SMOKE" "$NB_OUTPUT" \\
    --kernel python3 --progress-bar --log-output --cwd "$NB_DIR"
EXIT=$?

echo "=========================================="
if (( EXIT == 0 )); then echo "[PASS] smoke complete"; else echo "[FAIL] smoke exit=$EXIT"; fi
echo "End: $(date -Iseconds)"
exit $EXIT
'''


def train_wrapper(m: dict, ds_key: str) -> str:
    ds = DATASETS[ds_key]
    nb = f"fable_distillation_{m['nickname']}_{ds['name_tag']}_unsloth.ipynb"
    job = f"train-{m['nickname']}-{ds['name_tag']}"
    out_pref = f"fable_{m['dir']}_{ds_key}"
    size = m["size_tag"]
    wall = hhmm(TRAIN_H[(size, ds_key)])
    return f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account=def-maxwl_gpu
#SBATCH --time={wall}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={CPUS[size]}
#SBATCH --mem={MEM[size]}G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Real training: {m['name']} on FABLE-5 ({ds['suffix']}, {ds['rows']:,} rows, {ds['epochs']} epochs).
# Notebook ships SMOKE_TEST=False; assistant-only masking applied inline. Checkpoints
# at SAVE_STEPS (resume on requeue). Walltime scaled to dataset size.

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/fable-train-{m['dir']}-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TRAIN: {m['name']} on FABLE-5 ({ds['suffix']})"
echo "Node:  $SLURMD_NODENAME   Job: $SLURM_JOB_ID   Start: $(date -Iseconds)"
echo "=========================================="

module load StdEnv/2023 gcc arrow python/3.11.5 cuda/12.6
source "$VENV/bin/activate"
export SCRATCH="${{SCRATCH:-/scratch/$USER}}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${{value%\\"}}"; value="${{value#\\"}}"
        value="${{value%\\'}}"; value="${{value#\\'}}"
        export "$key=$value"
    done < "$PROJECT_DIR/.env"
fi

NB_INPUT="$NB_DIR/{nb}"
NB_OUTPUT="$OUTPUT_DIR/{out_pref}_train_output.ipynb"
[[ -f "$NB_INPUT" ]] || {{ echo "[FAIL] Notebook not found: $NB_INPUT" >&2; exit 1; }}

# Safety: refuse to launch a real run if SMOKE_TEST=True is committed.
python3 - <<PYEOF
import json, sys
nb = json.load(open("$NB_INPUT"))
for c in nb["cells"]:
    s = c.get("source", "")
    if isinstance(s, list): s = "".join(s)
    if "SMOKE_TEST: bool = True" in s:
        print("[FAIL] SMOKE_TEST=True committed — refusing real run", file=sys.stderr); sys.exit(1)
print("[OK] SMOKE_TEST=False confirmed")
PYEOF

echo "[....] Running via papermill..."
PYTHONUNBUFFERED=1 papermill "$NB_INPUT" "$NB_OUTPUT" \\
    --kernel python3 --progress-bar --log-output --cwd "$NB_DIR"
EXIT=$?

echo "=========================================="
if (( EXIT == 0 )); then echo "[PASS] training complete"; else echo "[FAIL] training exit=$EXIT"; fi
echo "Output notebook: $NB_OUTPUT"
echo "End: $(date -Iseconds)"
exit $EXIT
'''


def main():
    written = []
    for m in MODELS:
        for ds_key in ("d", "b"):
            ds = DATASETS[ds_key]
            for phase, fn in (("smoke", smoke_wrapper), ("train", train_wrapper)):
                path = JOBS / f"{phase}-{m['nickname']}-{ds['name_tag']}.sh"
                path.write_text(fn(m, ds_key))
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                written.append(path.name)
    print(f"[OK] wrote {len(written)} fable wrappers")
    for n in sorted(written):
        print("   ", n)
    assert len(written) == len(MODELS) * 2 * 2, f"expected {len(MODELS)*4}, got {len(written)}"


if __name__ == "__main__":
    main()
