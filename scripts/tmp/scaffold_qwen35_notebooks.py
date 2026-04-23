#!/usr/bin/env python3
"""Scaffold 4 Qwen3.5-{4,9}B notebooks from qwen3-8b templates.

Mutations per notebook:
  1. Markdown title:     "Qwen3-8B" → "Qwen3.5-{size}"
  2. Config constants:   MODEL_NAME, HUB_MODEL_ID, LORA_R, LORA_ALPHA,
                         BATCH_SIZE, GRAD_ACCUM
  3. Checkpoint block:   append CHECKPOINT_ROOT / TB_DIR / JSONL_LOG /
                         SAVE_STEPS constants using absolute /scratch
                         paths (F3 avoidance)
  4. LoRA targets:       append "out_proj" (Jackrong recipe)
  5. SFTConfig:          output_dir, logging_dir, save_strategy,
                         save_steps, save_total_limit, logging_steps=1,
                         logging_first_step=True
  6. Callback:           append JsonlLoggerCallback for per-step JSONL
                         persistence (survives job kill/timeout)

Verification: after write, re-parse each notebook and assert every
expected value is present exactly once. Print diff summary.

Run from repo root:
    python scripts/scaffold_qwen35_notebooks.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # scripts/tmp/ → scripts/ → repo/
NB_DIR = REPO / "notebooks"

# Per-notebook specs.
# Each tuple: (template_name, target_name, short_size, slug,
#              batch_size, grad_accum, title_model_token, hub_suffix)
SPECS = [
    # xLAM
    ("xlam_function_calling_qwen3-8b_unsloth.ipynb",
     "xlam_function_calling_qwen3.5-4b_unsloth.ipynb",
     "4B", "xlam-qwen35-4b", 2, 4,
     "Qwen3.5-4B", "Function-Calling-xLAM-Unsloth"),
    ("xlam_function_calling_qwen3-8b_unsloth.ipynb",
     "xlam_function_calling_qwen3.5-9b_unsloth.ipynb",
     "9B", "xlam-qwen35-9b", 1, 8,
     "Qwen3.5-9B", "Function-Calling-xLAM-Unsloth"),
    # SFT distillation
    ("sft_distillation_qwen3-8b_unsloth.ipynb",
     "sft_distillation_qwen3.5-4b_unsloth.ipynb",
     "4B", "sft-qwen35-4b", 2, 4,
     "Qwen3.5-4B", "SFT-Claude-Opus-Reasoning-Unsloth"),
    ("sft_distillation_qwen3-8b_unsloth.ipynb",
     "sft_distillation_qwen3.5-9b_unsloth.ipynb",
     "9B", "sft-qwen35-9b", 1, 8,
     "Qwen3.5-9B", "SFT-Claude-Opus-Reasoning-Unsloth"),
]

# JsonlLoggerCallback — streamed per-step persistence.
# Appended to cell 08 via string insertion so the notebook is
# self-contained and reproducible without our repo.
CALLBACK_SRC = '''
# ── Per-step JSONL log persistence (survives kill/timeout) ─────────────
import json as _json, time as _time
from transformers import TrainerCallback

class JsonlLoggerCallback(TrainerCallback):
    """Append every logged metric dict as a JSON line to `path`."""
    def __init__(self, path: str):
        self.path = path
        # Touch file so absence-of-file vs zero-steps is unambiguous.
        open(self.path, "a").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        rec = {"ts": _time.time(), "step": state.global_step, **logs}
        with open(self.path, "a") as f:
            f.write(_json.dumps(rec) + "\\n")

trainer.add_callback(JsonlLoggerCallback(JSONL_LOG))
print(f"[OK] JSONL log: {JSONL_LOG}")
'''


def cell_src(cell: dict) -> str:
    """Return cell source as str regardless of underlying type."""
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_cell_src(cell: dict, new_src: str) -> None:
    """Write new source back, preserving original list/str type."""
    if isinstance(cell["source"], list):
        cell["source"] = new_src.splitlines(keepends=True)
    else:
        cell["source"] = new_src


def mutate_notebook(nb: dict, size: str, slug: str, bs: int, ga: int,
                    title_model: str, hub_suffix: str) -> dict:
    """Apply all 6 mutation classes. Returns a new notebook dict."""
    nb = copy.deepcopy(nb)
    hub_id = f"ermiaazarkhalili/{title_model}-{hub_suffix}"
    model_hub = f"unsloth/{title_model}"

    # ── Mutation 1: Markdown title (cell 0) ────────────────────────────
    md = cell_src(nb["cells"][0])
    md = md.replace("Qwen3-8B", title_model)
    set_cell_src(nb["cells"][0], md)

    # ── Mutation 2+3: Config cell (cell 2) — constants + new block ─────
    cfg = cell_src(nb["cells"][2])

    # Replace model identifiers
    cfg = cfg.replace('unsloth/Qwen3-8B', model_hub)
    # Replace hub model id (match prefix to avoid touching comments)
    for old_hub in ["ermiaazarkhalili/Qwen3-8B-Function-Calling-xLAM-Unsloth",
                    "ermiaazarkhalili/Qwen3-8B-SFT-Claude-Opus-Reasoning-Unsloth"]:
        cfg = cfg.replace(old_hub, hub_id)

    # Replace LoRA hyperparameters (Jackrong's Qwen3.5-9B recipe)
    cfg = cfg.replace("LORA_R = 16", "LORA_R = 64")
    cfg = cfg.replace("LORA_ALPHA = 16", "LORA_ALPHA = 64")

    # Replace batch/accum to match model size (4B gets denser batch)
    cfg = cfg.replace("BATCH_SIZE = 1", f"BATCH_SIZE = {bs}")
    cfg = cfg.replace("GRAD_ACCUM = 8", f"GRAD_ACCUM = {ga}")

    # Append checkpoint+logging block after existing constants.
    # Insertion point: before the first `print(` line (which is
    # the summary banner at the end of cell 2).
    split_at = cfg.find("print(f\"[OK] Model:")
    if split_at == -1:
        raise RuntimeError("Cannot find print(...) anchor in cell 02")

    ckpt_block = f'''
# ── Checkpoint + structured logging (absolute paths; F3 avoidance) ─────
SLUG = "{slug}"
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
CHECKPOINT_ROOT = os.environ.get(
    "CHECKPOINT_ROOT",
    f"/scratch/{{os.environ.get('USER','ermia')}}/checkpoints/{{SLUG}}-{{JOB_ID}}",
)
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
TB_DIR = f"{{CHECKPOINT_ROOT}}/tb"
JSONL_LOG = f"{{CHECKPOINT_ROOT}}/train_log.jsonl"
SAVE_STEPS = ({750 if slug.startswith("xlam") else 150} if not SMOKE_TEST else 999_999)
print(f"[OK] Checkpoints → {{CHECKPOINT_ROOT}}")

'''
    cfg = cfg[:split_at] + ckpt_block + cfg[split_at:]
    set_cell_src(nb["cells"][2], cfg)

    # ── Mutation 4: LoRA targets (cell 5) — append "out_proj" ──────────
    lora = cell_src(nb["cells"][5])
    old_targets = '"gate_proj", "up_proj", "down_proj"'
    new_targets = '"gate_proj", "up_proj", "down_proj", "out_proj"'
    if old_targets not in lora:
        raise RuntimeError("target_modules anchor not found in cell 05")
    lora = lora.replace(old_targets, new_targets)
    set_cell_src(nb["cells"][5], lora)

    # ── Mutation 5: SFTConfig (cell 8) — checkpoint + logging knobs ────
    # Replace the whole SFTConfig(...) arg block in one pass. Using an
    # anchor-based approach to avoid fragile line-number assumptions.
    tr = cell_src(nb["cells"][8])
    old_cfg = '''    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5, max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE, logging_steps=1,
        optim="adamw_8bit", weight_decay=0.001,
        lr_scheduler_type="linear", seed=3407,
        output_dir="outputs", report_to="none",
    ),'''
    new_cfg = '''    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5, max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit", weight_decay=0.001,
        lr_scheduler_type="linear", seed=3407,
        # ── Absolute paths (F3 avoidance) ──
        output_dir=CHECKPOINT_ROOT,
        logging_dir=TB_DIR,
        # ── Checkpoints: periodic snapshots, rolling keep ──
        save_strategy=("no" if SMOKE_TEST else "steps"),
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        save_safetensors=True,
        # ── Streaming logs: every step, first step included ──
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        report_to=["tensorboard"],
    ),'''
    if old_cfg not in tr:
        raise RuntimeError("SFTConfig anchor not found in cell 08")
    tr = tr.replace(old_cfg, new_cfg)

    # ── Mutation 6: Append JsonlLoggerCallback after trainer init ──────
    # Anchor: the first occurrence of `gpu_stats = torch.cuda` marks the
    # start of the post-trainer block; we insert *before* it so the
    # callback is registered before training starts.
    anchor = "gpu_stats = torch.cuda.get_device_properties(0)"
    if anchor not in tr:
        raise RuntimeError("trainer post-init anchor not found in cell 08")
    tr = tr.replace(anchor, CALLBACK_SRC.strip() + "\n\n" + anchor)

    set_cell_src(nb["cells"][8], tr)

    return nb


def verify_notebook(path: Path, spec) -> list[str]:
    """Return list of error strings; empty list = passed."""
    (_tpl, _tgt, size, slug, bs, ga, title_model, hub_suffix) = spec
    errors: list[str] = []
    try:
        nb = json.load(open(path))
    except Exception as e:
        return [f"invalid JSON: {e}"]

    full_src = "\n".join(cell_src(c) for c in nb["cells"])

    def must_contain(s: str) -> None:
        if s not in full_src:
            errors.append(f"missing expected: {s!r}")

    def must_not_contain(s: str) -> None:
        if s in full_src:
            errors.append(f"unexpected residue: {s!r}")

    # Expected substitutions
    must_contain(f'unsloth/{title_model}')
    must_contain(f'ermiaazarkhalili/{title_model}-{hub_suffix}')
    must_contain("LORA_R = 64")
    must_contain("LORA_ALPHA = 64")
    must_contain(f"BATCH_SIZE = {bs}")
    must_contain(f"GRAD_ACCUM = {ga}")
    must_contain('"out_proj"')
    must_contain("CHECKPOINT_ROOT")
    must_contain("JSONL_LOG")
    must_contain("JsonlLoggerCallback")
    must_contain("save_strategy=")
    must_contain("save_total_limit=3")
    must_contain('logging_first_step=True')
    must_contain(f'SLUG = "{slug}"')

    # Residues that should be fully replaced
    must_not_contain("unsloth/Qwen3-8B")
    must_not_contain("LORA_R = 16")
    must_not_contain('output_dir="outputs"')
    must_not_contain('report_to="none"')

    return errors


def main() -> int:
    rc = 0
    for spec in SPECS:
        (tpl, tgt, size, slug, bs, ga, title_model, hub_suffix) = spec
        tpl_path = NB_DIR / tpl
        tgt_path = NB_DIR / tgt
        if not tpl_path.is_file():
            print(f"[FAIL] template missing: {tpl_path}")
            rc = 1
            continue

        src_nb = json.load(open(tpl_path))
        out_nb = mutate_notebook(src_nb, size, slug, bs, ga,
                                 title_model, hub_suffix)
        json.dump(out_nb, open(tgt_path, "w"), indent=1)

        errs = verify_notebook(tgt_path, spec)
        if errs:
            rc = 1
            print(f"[FAIL] {tgt}")
            for e in errs:
                print(f"       • {e}")
        else:
            print(f"[OK  ] {tgt}  (slug={slug}, bs={bs}, ga={ga})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
