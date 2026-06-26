#!/usr/bin/env python3
"""Generate the 4 Granite 4.1 training notebooks from Qwen3.5-4B templates.

Granite-specific deltas (probe-verified, job 38230438):
  - model_name: ibm-granite/granite-4.1-{3b,8b}
  - tokenizer is a plain GPT2Tokenizer (no VL processor) — drop the unwrap branch
  - LoRA r=16, alpha=16 (Granite is a dense transformer, not a VLM)
  - target_modules: standard 7-tuple (no out_proj — Granite MLP doesn't have one)
  - chat template ships in the tokenizer (no get_chat_template patch needed)
"""
from __future__ import annotations
import json, copy
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
NB_DIR = ROOT / "notebooks"

TEMPLATES = {
    "sft": NB_DIR / "sft_distillation_qwen3.5-4b_unsloth.ipynb",
    "xlam": NB_DIR / "xlam_function_calling_qwen3.5-4b_unsloth.ipynb",
}

# (kind, size) → notebook config
SPECS = [
    {
        "kind": "sft",
        "size": "3b",
        "param_label": "3B",
        "model_name": "ibm-granite/granite-4.1-3b",
        "hub_id": "ermiaazarkhalili/Granite-4.1-3B-SFT-Claude-Opus-Reasoning-Unsloth",
        "slug": "sft-granite41-3b",
        "save_steps": 150,
        "out": "sft_distillation_granite4.1-3b_unsloth.ipynb",
    },
    {
        "kind": "sft",
        "size": "8b",
        "param_label": "8B",
        "model_name": "ibm-granite/granite-4.1-8b",
        "hub_id": "ermiaazarkhalili/Granite-4.1-8B-SFT-Claude-Opus-Reasoning-Unsloth",
        "slug": "sft-granite41-8b",
        "save_steps": 150,
        "out": "sft_distillation_granite4.1-8b_unsloth.ipynb",
    },
    {
        "kind": "xlam",
        "size": "3b",
        "param_label": "3B",
        "model_name": "ibm-granite/granite-4.1-3b",
        "hub_id": "ermiaazarkhalili/Granite-4.1-3B-Function-Calling-xLAM-Unsloth",
        "slug": "xlam-granite41-3b",
        "save_steps": 750,
        "out": "xlam_function_calling_granite4.1-3b_unsloth.ipynb",
    },
    {
        "kind": "xlam",
        "size": "8b",
        "param_label": "8B",
        "model_name": "ibm-granite/granite-4.1-8b",
        "hub_id": "ermiaazarkhalili/Granite-4.1-8B-Function-Calling-xLAM-Unsloth",
        "slug": "xlam-granite41-8b",
        "save_steps": 750,
        "out": "xlam_function_calling_granite4.1-8b_unsloth.ipynb",
    },
]


def cell_text(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def set_cell(c, new_text: str):
    # Preserve list-of-strings format (line-by-line) like the templates do.
    lines = new_text.splitlines(keepends=True)
    c["source"] = lines
    # Markdown cells must NOT have outputs/execution_count per nbformat schema.
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
    else:
        c.pop("outputs", None)
        c.pop("execution_count", None)
    return c


def transform_markdown_header(text: str, spec: dict) -> str:
    """Rewrite the title/feature table for Granite."""
    kind = spec["kind"]
    label = spec["param_label"]
    model = spec["model_name"]
    if kind == "sft":
        title = f"# SFT Distillation: Granite 4.1-{label} on Claude Reasoning Traces (Unsloth)"
        why = (
            "### Why Granite 4.1?\n"
            "IBM's Granite 4.1 is a dense decoder-only transformer with strong reasoning\n"
            "and instruction-following baselines. Distilling Claude reasoning traces\n"
            "should sharpen its multi-step reasoning while preserving its grounded\n"
            "instruction-following behavior."
        )
        dataset_row = "| Dataset | Claude Reasoning Distillation (~10K samples with thinking traces) |"
    else:
        title = f"# Function Calling Fine-Tuning: Granite 4.1-{label} on xLAM Dataset (Unsloth)"
        why = (
            "### Why Granite 4.1?\n"
            "IBM's Granite 4.1 is a dense decoder-only transformer suitable for tool-use\n"
            "and function-calling. xLAM fine-tuning gives it structured JSON tool-call\n"
            "patterns on top of the base instruction-following behavior."
        )
        dataset_row = "| Dataset | xLAM 60K function calling examples |"

    # Rough size estimates (4-bit footprint): 3B → ~2 GB, 8B → ~5 GB
    if label == "3B":
        size_row = "| Size | ~6 GB fp16 → ~2 GB 4-bit |"
        vram_min = "8 GB"
    else:
        size_row = "| Size | ~16 GB fp16 → ~5 GB 4-bit |"
        vram_min = "12 GB"

    return f"""{title}

**Author:** Behrooz Azarkhalili

Fine-tunes **Granite 4.1-{label}** on
{"[claude-reasoning-distillation](https://huggingface.co/datasets/ermiaazarkhalili/claude-reasoning-distillation)" if kind == "sft" else "[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)"}
using **Unsloth** for 2x faster training.

| Feature | Detail |
|---------|--------|
| Framework | Unsloth (FastLanguageModel) |
| Model | `{model}` |
{size_row}
| Architecture | granite (GraniteForCausalLM, dense transformer) |
{dataset_row}
| Method | LoRA (4-bit QLoRA) |
| GGUF Export | Separate dispatcher (`jobs/gguf-single-model.sh`) |

### Hardware
- **Minimum:** {vram_min} VRAM (QLoRA, batch_size=1)
- **Recommended:** 40+ GB for batch_size > 1

{why}
"""


def transform_hyperparams(text: str, spec: dict) -> str:
    """Replace the MODEL_NAME / HUB_MODEL_ID / LORA / SLUG block."""
    out = text
    # Model name (template has it twice — actual + commented alt)
    out = out.replace('MODEL_NAME = "unsloth/Qwen3.5-4B"', f'MODEL_NAME = "{spec["model_name"]}"')
    out = out.replace('# MODEL_NAME = "unsloth/Qwen3.5-4B"  # Alternative',
                      f'# MODEL_NAME = "{spec["model_name"]}"  # Alternative')
    # Hub id
    if spec["kind"] == "sft":
        old_hub = "ermiaazarkhalili/Qwen3.5-4B-SFT-Claude-Opus-Reasoning-Unsloth"
    else:
        old_hub = "ermiaazarkhalili/Qwen3.5-4B-Function-Calling-xLAM-Unsloth"
    out = out.replace(old_hub, spec["hub_id"])
    # LoRA r/alpha (Jackrong 64/64 → Granite 16/16)
    out = out.replace("LORA_R = 64", "LORA_R = 16")
    out = out.replace("LORA_ALPHA = 64", "LORA_ALPHA = 16")
    # Slug
    if spec["kind"] == "sft":
        out = out.replace('SLUG = "sft-qwen35-4b"', f'SLUG = "{spec["slug"]}"')
    else:
        out = out.replace('SLUG = "xlam-qwen35-4b"', f'SLUG = "{spec["slug"]}"')
    # save_steps
    if spec["kind"] == "sft":
        out = out.replace("SAVE_STEPS = (150 if not SMOKE_TEST else 999_999)",
                          f"SAVE_STEPS = ({spec['save_steps']} if not SMOKE_TEST else 999_999)")
    else:
        out = out.replace("SAVE_STEPS = (750 if not SMOKE_TEST else 999_999)",
                          f"SAVE_STEPS = ({spec['save_steps']} if not SMOKE_TEST else 999_999)")
    return out


def transform_load_model(text: str, spec: dict) -> str:
    """Drop the VL-processor unwrap (Granite returns a plain tokenizer)."""
    # Remove the VL unwrap block — keep the comment minimal.
    old = """# Qwen3.5 VL-family: FastLanguageModel returns the processor, not the tokenizer.
# Keep the processor for chat-template rendering but expose .tokenizer for plain
# text calls so test-generation cells don't hit the image branch.
if hasattr(tokenizer, "tokenizer"):
    processor = tokenizer
    tokenizer = processor.tokenizer
    print("[OK] Extracted tokenizer from processor (VL-arch)")"""
    new = """# Granite 4.1 returns a plain tokenizer (probe-verified GPT2Tokenizer) — no
# VL-processor unwrap needed."""
    out = text.replace(old, new)
    if old in text:
        return out
    # Fallback: ensure VL unwrap is gone even if whitespace shifted slightly.
    if 'if hasattr(tokenizer, "tokenizer"):' in out:
        raise RuntimeError("VL unwrap block did not get removed — substitution failed")
    return out


def transform_lora(text: str, spec: dict) -> str:
    """Drop out_proj from target_modules (Granite MLP has no out_proj)."""
    old = '''target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj", "out_proj"],'''
    new = '''target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],'''
    return text.replace(old, new)


def transform_chat_template_cell(text: str, spec: dict) -> str:
    """Update the chat-template comment for Granite."""
    return text.replace(
        "# Chat Template (Qwen3.5 uses built-in template — no get_chat_template needed)",
        "# Chat Template (Granite 4.1 ships with chat_template.jinja in the tokenizer)",
    ).replace(
        '[OK] Using built-in Qwen3.5 chat template',
        '[OK] Using built-in Granite 4.1 chat template',
    )


def transform_save_export(text: str, spec: dict) -> str:
    """Rename the local LoRA dir from qwen3_8b_distill_* → granite41_*."""
    if spec["kind"] == "sft":
        out = (text
               .replace("qwen3_8b_distill_lora", f"granite41_{spec['size']}_distill_lora")
               .replace("qwen3_8b_distill_merged", f"granite41_{spec['size']}_distill_merged")
               .replace("qwen3_8b_distill_gguf", f"granite41_{spec['size']}_distill_gguf"))
    else:
        out = (text
               .replace("qwen3_8b_distill_lora", f"granite41_{spec['size']}_xlam_lora")
               .replace("qwen3_8b_distill_merged", f"granite41_{spec['size']}_xlam_merged")
               .replace("qwen3_8b_distill_gguf", f"granite41_{spec['size']}_xlam_gguf"))
    return out


def build_one(spec: dict) -> Path:
    template_path = TEMPLATES[spec["kind"]]
    nb = json.loads(template_path.read_text())
    nb = copy.deepcopy(nb)

    transformations = {
        0: transform_markdown_header,   # markdown header
        2: transform_hyperparams,        # MODEL_NAME / HUB / LORA / SLUG
        4: transform_load_model,         # drop VL unwrap
        5: transform_lora,               # drop out_proj
        6: transform_chat_template_cell, # chat template comment
        10: transform_save_export,       # local dir names
    }

    for idx, fn in transformations.items():
        if idx >= len(nb["cells"]):
            continue
        c = nb["cells"][idx]
        old = cell_text(c)
        new = fn(old, spec)
        set_cell(c, new)

    out_path = NB_DIR / spec["out"]
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out_path


def main():
    written = []
    for spec in SPECS:
        p = build_one(spec)
        written.append(p)
        print(f"[OK] {p.name}")
    print(f"\n[DONE] wrote {len(written)} notebooks")


if __name__ == "__main__":
    main()
