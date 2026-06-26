#!/usr/bin/env python3
"""Generate 8 SLURM wrappers for the 4 Granite 4.1 notebooks (smoke + train)."""
from __future__ import annotations
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
JOBS = ROOT / "jobs"

# (kind, size, partition_smoke, time_smoke, partition_train, time_train, mem)
SPECS = [
    {"kind": "sft", "size": "3b", "param": "3B",
     "nb": "sft_distillation_granite4.1-3b_unsloth.ipynb",
     "slug_smoke": "smoke-granite41-3b-sft",
     "slug_train": "train-granite41-3b-sft",
     "out_prefix": "sft_granite41_3b",
     "title_kind": "SFT distillation"},
    {"kind": "sft", "size": "8b", "param": "8B",
     "nb": "sft_distillation_granite4.1-8b_unsloth.ipynb",
     "slug_smoke": "smoke-granite41-8b-sft",
     "slug_train": "train-granite41-8b-sft",
     "out_prefix": "sft_granite41_8b",
     "title_kind": "SFT distillation"},
    {"kind": "xlam", "size": "3b", "param": "3B",
     "nb": "xlam_function_calling_granite4.1-3b_unsloth.ipynb",
     "slug_smoke": "smoke-granite41-3b-xlam",
     "slug_train": "train-granite41-3b-xlam",
     "out_prefix": "xlam_granite41_3b",
     "title_kind": "xLAM function calling"},
    {"kind": "xlam", "size": "8b", "param": "8B",
     "nb": "xlam_function_calling_granite4.1-8b_unsloth.ipynb",
     "slug_smoke": "smoke-granite41-8b-xlam",
     "slug_train": "train-granite41-8b-xlam",
     "out_prefix": "xlam_granite41_8b",
     "title_kind": "xLAM function calling"},
]

# 3B can use b1 (3h cap), 8B uses b2 (12h cap)
TRAIN_RESOURCES = {
    "3b": {"partition": "gpubase_bygpu_b2", "time": "0-04:00:00", "mem": "32G", "cpus": 8},
    "8b": {"partition": "gpubase_bygpu_b2", "time": "0-08:00:00", "mem": "48G", "cpus": 8},
}

SMOKE_RESOURCES = {
    "3b": {"partition": "gpubase_bygpu_b1", "time": "0-01:00:00", "mem": "24G", "cpus": 4},
    "8b": {"partition": "gpubase_bygpu_b1", "time": "0-01:00:00", "mem": "32G", "cpus": 6},
}


SMOKE_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={slug}
#SBATCH --account=def-maxwl_gpu
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition={partition}
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-smoke-granite41-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "SMOKE: Granite 4.1-{param} {title_kind} (compat validation)"
echo "=========================================="
echo "Node:  $SLURMD_NODENAME"
echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date -Iseconds)"

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

export NB_INPUT="$NB_DIR/{notebook}"
export NB_SMOKE="$OUTPUT_DIR/{out_prefix}_smoke_input.ipynb"
export NB_OUTPUT="$OUTPUT_DIR/{out_prefix}_smoke_output.ipynb"

# Flip SMOKE_TEST → True (handle both str and list[str] source, per memory)
python3 - <<'PY'
import json, os
src = os.environ["NB_INPUT"]
dst = os.environ["NB_SMOKE"]
nb = json.load(open(src))
patched = 0
for c in nb['cells']:
    s = c.get('source', '')
    if isinstance(s, str):
        if 'SMOKE_TEST: bool = False' in s:
            c['source'] = s.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True')
            patched += 1
    elif isinstance(s, list):
        new = [l.replace('SMOKE_TEST: bool = False', 'SMOKE_TEST: bool = True') for l in s]
        if new != s:
            c['source'] = new
            patched += 1
json.dump(nb, open(dst, 'w'), indent=1)
print(f"[OK] Patched SMOKE_TEST in {{patched}} cell(s)")
assert patched == 1, f"Expected exactly 1 patch, got {{patched}}"
PY

echo ""
echo "[....] Running via papermill..."
PYTHONUNBUFFERED=1 papermill "$NB_SMOKE" "$NB_OUTPUT" \\
    --kernel python3 --progress-bar --log-output --cwd "$NB_DIR"
EXIT=$?

echo ""
echo "=========================================="
if [[ $EXIT -eq 0 ]]; then
    echo "[PASS] smoke complete"
else
    echo "[FAIL] smoke exit=$EXIT"
fi
echo "End: $(date -Iseconds)"
exit $EXIT
"""


TRAIN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={slug}
#SBATCH --account=def-maxwl_gpu
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --partition={partition}
#SBATCH --exclude=fc10713
#SBATCH --output=/project/6014832/ermia/HF-TRL/logs/%x-%j.out
#SBATCH --error=/project/6014832/ermia/HF-TRL/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Real (non-smoke) training: Granite 4.1-{param} {title_kind}
# Notebook has SMOKE_TEST=False committed; runs full epoch.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-train-granite41-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TRAIN: Granite 4.1-{param} {title_kind} (full epoch)"
echo "=========================================="
echo "Node:  $SLURMD_NODENAME"
echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date -Iseconds)"

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

NB_INPUT="$NB_DIR/{notebook}"
NB_OUTPUT="$OUTPUT_DIR/{out_prefix}_train_output.ipynb"

if [[ ! -f "$NB_INPUT" ]]; then
    echo "[FAIL] Notebook not found: $NB_INPUT" >&2
    exit 1
fi

# Verify SMOKE_TEST is False in committed notebook (safety check)
python3 - <<PY
import json, sys
nb = json.load(open("$NB_INPUT"))
for c in nb["cells"]:
    s = c.get("source", "")
    if isinstance(s, list): s = "".join(s)
    if "SMOKE_TEST: bool = True" in s:
        print("[FAIL] Notebook has SMOKE_TEST=True committed — refusing to launch real run", file=sys.stderr)
        sys.exit(1)
print("[OK] SMOKE_TEST=False confirmed")
PY

echo ""
echo "[....] Running via papermill..."
PYTHONUNBUFFERED=1 papermill "$NB_INPUT" "$NB_OUTPUT" \\
    --kernel python3 --progress-bar --log-output --cwd "$NB_DIR"
EXIT=$?

echo ""
echo "=========================================="
if (( EXIT == 0 )); then
    echo "[PASS] training complete"
else
    echo "[FAIL] training exit=$EXIT"
fi
echo "Output notebook: $NB_OUTPUT"
echo "End: $(date -Iseconds)"
exit $EXIT
"""


def main():
    written = 0
    for spec in SPECS:
        sm = SMOKE_RESOURCES[spec["size"]]
        tr = TRAIN_RESOURCES[spec["size"]]

        smoke = SMOKE_TEMPLATE.format(
            slug=spec["slug_smoke"],
            param=spec["param"],
            title_kind=spec["title_kind"],
            notebook=spec["nb"],
            out_prefix=spec["out_prefix"],
            partition=sm["partition"], time=sm["time"], mem=sm["mem"], cpus=sm["cpus"],
        )
        train = TRAIN_TEMPLATE.format(
            slug=spec["slug_train"],
            param=spec["param"],
            title_kind=spec["title_kind"],
            notebook=spec["nb"],
            out_prefix=spec["out_prefix"],
            partition=tr["partition"], time=tr["time"], mem=tr["mem"], cpus=tr["cpus"],
        )

        smoke_path = JOBS / f"{spec['slug_smoke']}.sh"
        train_path = JOBS / f"{spec['slug_train']}.sh"
        smoke_path.write_text(smoke)
        train_path.write_text(train)
        smoke_path.chmod(0o755)
        train_path.chmod(0o755)
        print(f"[OK] {smoke_path.name}")
        print(f"[OK] {train_path.name}")
        written += 2
    print(f"\n[DONE] wrote {written} wrappers")


if __name__ == "__main__":
    main()
