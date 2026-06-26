#!/usr/bin/env python3
"""Generate 12 SLURM wrappers (6 smoke + 6 train) for the new-model notebooks.

Mirrors build_granite41_wrappers.py but parameterizes the header label and the
scratch OUTPUT_DIR family tag per model. 3B → 4B all use b2 for training; smokes
on b1. 4B gets slightly more mem/time than 3B.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
JOBS = ROOT / "jobs"

# One SPEC per (model, dataset). size_tag selects resources; family is the
# scratch-output tag; label is the human header.
SPECS = [
    # VibeThinker-3B
    {"size_tag": "3b", "param": "3B", "family": "vibethinker", "label": "VibeThinker-3B",
     "kind": "sft", "nb": "sft_distillation_vibethinker-3b_unsloth.ipynb",
     "slug_smoke": "smoke-vibethinker-3b-sft", "slug_train": "train-vibethinker-3b-sft",
     "out_prefix": "sft_vibethinker_3b", "title_kind": "SFT distillation"},
    {"size_tag": "3b", "param": "3B", "family": "vibethinker", "label": "VibeThinker-3B",
     "kind": "xlam", "nb": "xlam_function_calling_vibethinker-3b_unsloth.ipynb",
     "slug_smoke": "smoke-vibethinker-3b-xlam", "slug_train": "train-vibethinker-3b-xlam",
     "out_prefix": "xlam_vibethinker_3b", "title_kind": "xLAM function calling"},
    # FastContext-4B-SFT
    {"size_tag": "4b", "param": "4B", "family": "fastcontext", "label": "FastContext-4B-SFT_base",
     "kind": "sft", "nb": "sft_distillation_fastcontext-4b-sft_unsloth.ipynb",
     "slug_smoke": "smoke-fastcontext-4b-sft-sft", "slug_train": "train-fastcontext-4b-sft-sft",
     "out_prefix": "sft_fastcontext_4b_sft", "title_kind": "SFT distillation"},
    {"size_tag": "4b", "param": "4B", "family": "fastcontext", "label": "FastContext-4B-SFT_base",
     "kind": "xlam", "nb": "xlam_function_calling_fastcontext-4b-sft_unsloth.ipynb",
     "slug_smoke": "smoke-fastcontext-4b-sft-xlam", "slug_train": "train-fastcontext-4b-sft-xlam",
     "out_prefix": "xlam_fastcontext_4b_sft", "title_kind": "xLAM function calling"},
    # FastContext-4B-RL
    {"size_tag": "4b", "param": "4B", "family": "fastcontext", "label": "FastContext-4B-RL_base",
     "kind": "sft", "nb": "sft_distillation_fastcontext-4b-rl_unsloth.ipynb",
     "slug_smoke": "smoke-fastcontext-4b-rl-sft", "slug_train": "train-fastcontext-4b-rl-sft",
     "out_prefix": "sft_fastcontext_4b_rl", "title_kind": "SFT distillation"},
    {"size_tag": "4b", "param": "4B", "family": "fastcontext", "label": "FastContext-4B-RL_base",
     "kind": "xlam", "nb": "xlam_function_calling_fastcontext-4b-rl_unsloth.ipynb",
     "slug_smoke": "smoke-fastcontext-4b-rl-xlam", "slug_train": "train-fastcontext-4b-rl-xlam",
     "out_prefix": "xlam_fastcontext_4b_rl", "title_kind": "xLAM function calling"},
]

# Walltime is driven by DATASET size, not model size (empirical, Granite 4.1 runs
# 2026-05-03): SFT-Claude (10,477 samples) finishes <1h regardless of 3B/8B;
# xLAM (60,000 samples, 5.7x larger) takes ~3h. Caps below carry ~2x / ~1.5x
# headroom over the observed worst case. Mem still scales with model size.
TRAIN_RESOURCES = {
    # keyed by (size_tag, kind)
    ("3b", "sft"):  {"partition": "gpubase_bygpu_b2", "time": "0-02:00:00", "mem": "32G", "cpus": 8},
    ("3b", "xlam"): {"partition": "gpubase_bygpu_b2", "time": "0-05:00:00", "mem": "32G", "cpus": 8},
    ("4b", "sft"):  {"partition": "gpubase_bygpu_b2", "time": "0-02:00:00", "mem": "40G", "cpus": 8},
    ("4b", "xlam"): {"partition": "gpubase_bygpu_b2", "time": "0-05:00:00", "mem": "40G", "cpus": 8},
}
SMOKE_RESOURCES = {
    "3b": {"partition": "gpubase_bygpu_b1", "time": "0-01:00:00", "mem": "24G", "cpus": 4},
    "4b": {"partition": "gpubase_bygpu_b1", "time": "0-01:00:00", "mem": "32G", "cpus": 6},
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
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-smoke-{family}-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "SMOKE: {label} {title_kind} (compat validation)"
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
# Real (non-smoke) training: {label} {title_kind}
# Notebook has SMOKE_TEST=False committed; runs full epoch.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/6014832/ermia/HF-TRL"
NB_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="/scratch/$USER/outputs/unsloth-train-{family}-$(date +%Y%m%d)"
VENV="/scratch/ermia/venvs/hf_unsloth"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TRAIN: {label} {title_kind} (full epoch)"
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
        sm = SMOKE_RESOURCES[spec["size_tag"]]
        tr = TRAIN_RESOURCES[(spec["size_tag"], spec["kind"])]
        smoke = SMOKE_TEMPLATE.format(
            slug=spec["slug_smoke"], label=spec["label"], family=spec["family"],
            title_kind=spec["title_kind"], notebook=spec["nb"], out_prefix=spec["out_prefix"],
            partition=sm["partition"], time=sm["time"], mem=sm["mem"], cpus=sm["cpus"],
        )
        train = TRAIN_TEMPLATE.format(
            slug=spec["slug_train"], label=spec["label"], family=spec["family"],
            title_kind=spec["title_kind"], notebook=spec["nb"], out_prefix=spec["out_prefix"],
            partition=tr["partition"], time=tr["time"], mem=tr["mem"], cpus=tr["cpus"],
        )
        sp = JOBS / f"{spec['slug_smoke']}.sh"
        tp = JOBS / f"{spec['slug_train']}.sh"
        sp.write_text(smoke); tp.write_text(train)
        sp.chmod(0o755); tp.chmod(0o755)
        print(f"[OK] {sp.name}")
        print(f"[OK] {tp.name}")
        written += 2
    print(f"\n[DONE] wrote {written} wrappers")


if __name__ == "__main__":
    main()
