#!/usr/bin/env python3
"""Generate 6 new-model training notebooks from the validated Granite 4.1-3B
notebooks (proven end-to-end through GGUF on 2026-05-03).

New models (probe job 45153667):
  - WeiboAI/VibeThinker-3B           (Qwen2ForCausalLM, reasoning model, mit)
  - microsoft/FastContext-1.0-4B-SFT (Qwen3ForCausalLM, mit)
  - microsoft/FastContext-1.0-4B-RL  (Qwen3ForCausalLM, mit)

Why clone Granite-3B rather than Qwen3.5-4B template:
  - Granite-3B notebooks already carry the dense-transformer LoRA config
    (r=16/alpha=16, standard 7-tuple targets, NO out_proj, NO VL-processor
    unwrap) — which is exactly what Qwen2/Qwen3 dense arches need.
  - They are validated end-to-end (train + merge + GGUF all PASS).

Per-model deltas: MODEL_NAME, HUB_MODEL_ID, SLUG, local dir prefix, header text.
LoRA config and target_modules are inherited unchanged from Granite-3B.
"""
from __future__ import annotations
import json, copy
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
NB_DIR = ROOT / "notebooks"

# Source templates: the validated Granite 4.1-3B notebooks.
TEMPLATES = {
    "sft": NB_DIR / "sft_distillation_granite4.1-3b_unsloth.ipynb",
    "xlam": NB_DIR / "xlam_function_calling_granite4.1-3b_unsloth.ipynb",
}

# Granite-3B template tokens we substitute out.
TPL = {
    "model_name": "ibm-granite/granite-4.1-3b",
    "hub_sft": "ermiaazarkhalili/Granite-4.1-3B-SFT-Claude-Opus-Reasoning-Unsloth",
    "hub_xlam": "ermiaazarkhalili/Granite-4.1-3B-Function-Calling-xLAM-Unsloth",
    "slug_sft": "sft-granite41-3b",
    "slug_xlam": "xlam-granite41-3b",
    "dir_prefix": "granite41_3b",  # used in qwen-derived local dir names? no — granite builder set granite41_3b
    "display": "Granite 4.1-3B",
    "family_label": "granite (GraniteForCausalLM, dense transformer)",
}

# Per-model config. `nickname` drives slug + local dir; `display` drives header.
# `model_name`  = exact HF path passed to FastLanguageModel.from_pretrained
# `display`     = human label used in notebook headers
# `repo_stem`   = stem for OUR published repos (decoupled from model_name to avoid
#                 doubled task tags, e.g. FastContext-4B-SFTbase not -SFT-SFT-...)
# `nickname`    = slug / filename segment (kebab)
# `dir_prefix`  = local scratch-dir segment (snake)
MODELS = [
    {
        "model_name": "WeiboAI/VibeThinker-3B",
        "display": "VibeThinker-3B",
        "repo_stem": "VibeThinker-3B",
        "nickname": "vibethinker-3b",
        "dir_prefix": "vibethinker_3b",
        "arch": "qwen2 (Qwen2ForCausalLM, dense transformer)",
        "size_row": "| Size | ~6 GB fp16 → ~2 GB 4-bit |",
        "vram_min": "8 GB",
        "why": (
            "WeiboAI's VibeThinker-3B is a reasoning-specialized dense model "
            "(Qwen2 architecture). Distilling Claude reasoning traces should "
            "reinforce its multi-step chain-of-thought while adding "
            "Claude-style structured reasoning."
        ),
    },
    {
        "model_name": "microsoft/FastContext-1.0-4B-SFT",
        "display": "FastContext-1.0-4B-SFT",
        "repo_stem": "FastContext-4B-SFT_base",
        "nickname": "fastcontext-4b-sft",
        "dir_prefix": "fastcontext_4b_sft",
        "arch": "qwen3 (Qwen3ForCausalLM, dense transformer)",
        "size_row": "| Size | ~8 GB fp16 → ~2.5 GB 4-bit |",
        "vram_min": "10 GB",
        "why": (
            "Microsoft's FastContext-1.0-4B-SFT is a recent (June 2026) "
            "instruction-tuned dense model (Qwen3 architecture). Further "
            "fine-tuning extends its instruction-following with the target "
            "dataset's behavior."
        ),
    },
    {
        "model_name": "microsoft/FastContext-1.0-4B-RL",
        "display": "FastContext-1.0-4B-RL",
        "repo_stem": "FastContext-4B-RL_base",
        "nickname": "fastcontext-4b-rl",
        "dir_prefix": "fastcontext_4b_rl",
        "arch": "qwen3 (Qwen3ForCausalLM, dense transformer)",
        "size_row": "| Size | ~8 GB fp16 → ~2.5 GB 4-bit |",
        "vram_min": "10 GB",
        "why": (
            "Microsoft's FastContext-1.0-4B-RL is the RL-tuned sibling of the "
            "SFT variant (Qwen3 architecture). Training it on the same dataset "
            "as the SFT variant enables a direct SFT-vs-RL base comparison."
        ),
    },
]


def cell_text(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def set_cell(c, new_text: str):
    lines = new_text.splitlines(keepends=True)
    c["source"] = lines
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
    else:
        c.pop("outputs", None)
        c.pop("execution_count", None)
    return c


def hub_id(model: dict, kind: str) -> str:
    # ermiaazarkhalili/<repo_stem>-<Task>-Unsloth
    stem = model["repo_stem"]
    if kind == "sft":
        return f"ermiaazarkhalili/{stem}-SFT-Claude-Opus-Reasoning-Unsloth"
    return f"ermiaazarkhalili/{stem}-Function-Calling-xLAM-Unsloth"


def slug(model: dict, kind: str) -> str:
    return f"{'sft' if kind == 'sft' else 'xlam'}-{model['nickname']}"


def transform_markdown_header(text: str, model: dict, kind: str) -> str:
    out = text
    # Title line
    out = out.replace(f"Granite 4.1-3B", model["display"])
    # The Granite header has a "Why Granite 4.1?" section + body — replace whole block.
    # Easiest robust approach: rebuild the feature table + why from scratch.
    disp = model["display"]
    if kind == "sft":
        title = f"# SFT Distillation: {disp} on Claude Reasoning Traces (Unsloth)"
        ds_link = "[claude-reasoning-distillation](https://huggingface.co/datasets/ermiaazarkhalili/claude-reasoning-distillation)"
        dataset_row = "| Dataset | Claude Reasoning Distillation (~10K samples with thinking traces) |"
    else:
        title = f"# Function Calling Fine-Tuning: {disp} on xLAM Dataset (Unsloth)"
        ds_link = "[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)"
        dataset_row = "| Dataset | xLAM 60K function calling examples |"
    return f"""{title}

**Author:** Behrooz Azarkhalili

Fine-tunes **{disp}** on
{ds_link}
using **Unsloth** for 2x faster training.

| Feature | Detail |
|---------|--------|
| Framework | Unsloth (FastLanguageModel) |
| Model | `{model['model_name']}` |
{model['size_row']}
| Architecture | {model['arch']} |
{dataset_row}
| Method | LoRA (4-bit QLoRA) |
| GGUF Export | Separate dispatcher (`jobs/gguf-single-model.sh`) |

### Hardware
- **Minimum:** {model['vram_min']} VRAM (QLoRA, batch_size=1)
- **Recommended:** 40+ GB for batch_size > 1

### Why {disp}?
{model['why']}
"""


def transform_hyperparams(text: str, model: dict, kind: str) -> str:
    out = text
    out = out.replace(f'MODEL_NAME = "{TPL["model_name"]}"', f'MODEL_NAME = "{model["model_name"]}"')
    old_hub = TPL["hub_sft"] if kind == "sft" else TPL["hub_xlam"]
    out = out.replace(old_hub, hub_id(model, kind))
    old_slug = TPL["slug_sft"] if kind == "sft" else TPL["slug_xlam"]
    out = out.replace(f'SLUG = "{old_slug}"', f'SLUG = "{slug(model, kind)}"')
    return out


def transform_chat_template_cell(text: str, model: dict, kind: str) -> str:
    return text.replace(
        "# Chat Template (Granite 4.1 ships with chat_template.jinja in the tokenizer)",
        f"# Chat Template ({model['display']} ships a built-in chat_template in the tokenizer)",
    ).replace(
        "[OK] Using built-in Granite 4.1 chat template",
        f"[OK] Using built-in {model['display']} chat template",
    )


def transform_save_export(text: str, model: dict, kind: str) -> str:
    # Local scratch-dir names. The Granite SFT notebook uses granite41_3b_distill_*;
    # the Granite xLAM notebook still carries stale qwen3_8b_xlam_* (a no-op leftover
    # in the upstream Granite builder — harmless, local-only). Substitute whichever
    # prefix is actually present so the new notebooks get clean, model-specific names.
    suffix = "distill" if kind == "sft" else "xlam"
    out = text
    candidate_prefixes = (f"{TPL['dir_prefix']}", "qwen3_8b")
    for src_prefix in candidate_prefixes:
        for part in ("lora", "merged", "gguf"):
            out = out.replace(f"{src_prefix}_{suffix}_{part}",
                              f"{model['dir_prefix']}_{suffix}_{part}")
    return out


def out_name(model: dict, kind: str) -> str:
    base = "sft_distillation" if kind == "sft" else "xlam_function_calling"
    return f"{base}_{model['nickname']}_unsloth.ipynb"


def build_one(model: dict, kind: str) -> Path:
    nb = copy.deepcopy(json.loads(TEMPLATES[kind].read_text()))
    transformations = {
        0: transform_markdown_header,
        2: transform_hyperparams,
        6: transform_chat_template_cell,
        10: transform_save_export,
    }
    for idx, fn in transformations.items():
        if idx >= len(nb["cells"]):
            continue
        c = nb["cells"][idx]
        set_cell(c, fn(cell_text(c), model, kind))

    # Safety: ensure no stale Granite token leaked through.
    blob = json.dumps(nb)
    for stale in (TPL["model_name"], TPL["hub_sft"], TPL["hub_xlam"]):
        if stale in blob:
            raise RuntimeError(f"Stale Granite token leaked: {stale} in {out_name(model, kind)}")

    out_path = NB_DIR / out_name(model, kind)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out_path


def main():
    written = []
    for model in MODELS:
        for kind in ("sft", "xlam"):
            p = build_one(model, kind)
            written.append(p)
            print(f"[OK] {p.name}")
    print(f"\n[DONE] wrote {len(written)} notebooks")


if __name__ == "__main__":
    main()
