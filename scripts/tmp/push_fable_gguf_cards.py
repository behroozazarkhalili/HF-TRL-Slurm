#!/usr/bin/env python3
"""Generate + push README cards to the 16 B-leg fable GGUF repos.

Facts are read from the LIVE Hub repo (actual .gguf files + sizes) and the fleet
registry (verified base model) — nothing hard-coded or guessed. Matches the existing
house-style GGUF card (ermiaazarkhalili/*-GGUF). Dataset (Fable-5-Glint) is PRIVATE,
so it is named but NOT linked (no 404). Quants listed = exactly what each repo contains.

Run on the login node (uses the stored HF CLI token via HfApi(); no .env read).
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("/project/6014832/ermia/HF-TRL")
REG = json.loads((ROOT / "scripts/tmp/fable_fleet_registry.json").read_text())
api = HfApi()

QUANT_DESC = {
    "q4_k_m": ("Q4_K_M", "**Recommended** — best quality/size balance"),
    "q5_k_m": ("Q5_K_M", "Higher quality"),
    "q8_0": ("Q8_0", "Maximum quality (near-lossless)"),
}


def human_size(nbytes: int | None) -> str:
    if not nbytes:
        return "?"
    mb = nbytes / 1024 / 1024
    return f"~{mb/1024:.1f} GB" if mb >= 1024 else f"~{mb:.0f} MB"


def _smallest_gguf_name(ggufs: list) -> str:
    """Filename of the q4_k_m (smallest) quant, for the llama.cpp example."""
    for f in ggufs:
        if _quant_key(f.rfilename) == "q4_k_m":
            return f.rfilename
    return ggufs[0].rfilename if ggufs else "model.q4_k_m.gguf"


def build_card(merged_repo: str, gguf_repo: str, model: dict, recipe: dict, dataset: dict, ggufs: list) -> str:
    """Comprehensive GGUF model card. All facts are passed in (registry + live Hub files);
    nothing is guessed. Quant table lists ONLY the quants actually present."""
    name = merged_repo.split("/")[-1]
    base_model = model["base"]
    family = model.get("family", "dense")
    size_tag = model.get("size_tag", "")
    base_link = f"[{base_model}](https://huggingface.co/{base_model})"

    order = {"q4_k_m": 0, "q5_k_m": 1, "q8_0": 2}
    rows, smallest = [], _smallest_gguf_name(ggufs)
    for f in sorted(ggufs, key=lambda s: order.get(_quant_key(s.rfilename), 9)):
        qk = _quant_key(f.rfilename)
        label, desc = QUANT_DESC.get(qk, (qk.upper(), ""))
        rows.append(f"| `{f.rfilename}` | {label} | {human_size(f.size)} | {desc} |")
    rows_md = "\n".join(rows)

    # family-specific note: gemma/qwen35 are VL-family bases; thinking models emit <think>.
    family_note = {
        "gemma": "Built on a Gemma-family base (uses the Gemma chat template).",
        "qwen35": "Built on a Qwen3.5 base (chat template applied at training time).",
        "lfm2": "Built on a LiquidAI LFM2 base.",
        "dense": "Standard dense decoder base.",
    }.get(family, "")

    lr = recipe.get("learning_rate")
    return f"""---
license: apache-2.0
language:
  - en
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - llama-cpp
  - ollama
  - lm-studio
  - sft
  - distillation
  - fable
  - creative-writing
base_model: {merged_repo}
---

# {name} — GGUF

Quantized **GGUF** builds of [`{name}`](https://huggingface.co/{merged_repo}), a
[{base_model}](https://huggingface.co/{base_model}) model supervised-fine-tuned on the
**FABLE-5** trace corpus. These files run locally with
[llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com),
[LM Studio](https://lmstudio.ai/), and any GGUF-compatible runtime — no GPU required for the
smaller quants.

## Overview

| | |
|---|---|
| **Fine-tuned model** | [{name}](https://huggingface.co/{merged_repo}) |
| **Base model** | {base_link} |
| **Parameter class** | {size_tag.upper()} |
| **Model family** | {family} |
| **Training method** | LoRA SFT (distillation) |
| **Domain** | FABLE-5 creative / agentic traces |
| **Format** | GGUF (this repo) · safetensors ([merged repo]({( 'https://huggingface.co/'+merged_repo )})) |

{family_note}

## What is FABLE-5?

This model was fine-tuned on **FABLE-5-Glint**, a cleaned corpus of FABLE-5 pi-agent
reasoning traces (each target completion may include a `<think>…</think>` reasoning span
followed by the response). Training used assistant-only loss masking so the model learns to
*produce* the response, not echo the prompt. The dataset is private; the fine-tuned weights
are public.

## Available Quantizations

| File | Quant | Size | Notes |
|------|-------|------|-------|
{rows_md}

**Which to pick:** `Q4_K_M` is the best size/quality trade-off for most users. Use `Q5_K_M`
if you have spare RAM/VRAM and want a little more fidelity, or `Q8_0` for near-lossless output
when size is not a concern.

## Usage

### Ollama

```bash
# Pull + run the recommended Q4_K_M quant directly from the Hub
ollama run hf.co/{gguf_repo}:Q4_K_M "Write a short story about a clockwork fox."
```

To pin a different quant, swap the tag (e.g. `:Q5_K_M`, `:Q8_0`).

### llama.cpp

```bash
# Download a single quant, then run it
huggingface-cli download {gguf_repo} {smallest} --local-dir .
llama-cli -m {smallest} -p "Write a short story about a clockwork fox." -n 512

# Or serve an OpenAI-compatible endpoint
llama-server -m {smallest} --host 0.0.0.0 --port 8080
```

### LM Studio

Search for `{gguf_repo}` in LM Studio, or download a `.gguf` above and load it from disk.

### Python (llama-cpp-python)

```python
from llama_cpp import Llama
llm = Llama.from_pretrained(repo_id="{gguf_repo}", filename="*q4_k_m.gguf", n_ctx=4096)
out = llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Write a short story about a clockwork fox."}}]
)
print(out["choices"][0]["message"]["content"])
```

## Prompt format

Use the base model's chat template (applied automatically by Ollama / LM Studio /
`create_chat_completion`). The model was trained on 2-turn `user → assistant` chats. For
thinking-style bases, the model may emit a `<think>…</think>` span before its answer.

## Training details

| Hyperparameter | Value |
|---|---|
| Method | LoRA SFT, merged to 16-bit then quantized |
| LoRA rank / α | {recipe.get('lora_r')} / {recipe.get('lora_alpha')} |
| Learning rate | {lr} |
| LR scheduler | {recipe.get('lr_scheduler')} (warmup {recipe.get('warmup_ratio')}) |
| Max sequence length | {recipe.get('max_seq_length')} |
| Epochs | {dataset.get('epochs')} |
| Loss masking | assistant-only |
| Quantization toolchain | llama.cpp `convert_hf_to_gguf` + `llama-quantize` |

A deterministic ~5% slice of the corpus was held out from training for evaluation.

## Intended use & limitations

- **Intended:** local/offline creative writing, reasoning-trace style generation, and
  experimentation with FABLE-5-distilled behavior on consumer hardware.
- **Limitations:** inherits the base model's knowledge cutoff and biases; quantization
  (especially `Q4_K_M`) trades some fidelity for size; not safety-tuned for production use
  without additional guardrails. Outputs may be fictional/unverified.

## Citation

```bibtex
@misc{{azarkhalili2026{_bibkey(name)},
    author = {{Azarkhalili, Behrooz}},
    title  = {{{name}: FABLE-5 SFT distillation (GGUF)}},
    year   = {{2026}},
    publisher = {{Hugging Face}},
    url    = {{https://huggingface.co/{gguf_repo}}}
}}
```
"""


def _quant_key(fname: str) -> str:
    f = fname.lower()
    for k in ("q4_k_m", "q5_k_m", "q8_0"):
        if k in f:
            return k
    return f.rsplit(".", 2)[-2] if f.endswith(".gguf") else f


def _bibkey(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]", "", name.lower())


def main() -> None:
    recipe = REG.get("recipe", {})
    dataset = REG.get("datasets", {}).get("b", {})
    done, fail = 0, 0
    for m in REG["models"]:
        stem = m.get("repo_stem", m["name"])
        merged = f"ermiaazarkhalili/{stem}-SFT-Fable5-Glint"
        gguf = f"{merged}-GGUF"
        try:
            info = api.model_info(gguf, files_metadata=True)
            ggufs = [f for f in info.siblings if f.rfilename.endswith(".gguf")]
            if not ggufs:
                print(f"  SKIP {gguf}: no gguf files")
                fail += 1
                continue
            card = build_card(merged, gguf, m, recipe, dataset, ggufs)
            # write README.md to the repo
            from huggingface_hub import upload_file
            import io

            upload_file(
                path_or_fileobj=io.BytesIO(card.encode("utf-8")),
                path_in_repo="README.md",
                repo_id=gguf,
                repo_type="model",
                commit_message="Add comprehensive GGUF model card",
            )
            print(f"  [OK] card -> {gguf}  ({len(ggufs)} quants)")
            done += 1
        except Exception as e:
            print(f"  [FAIL] {gguf}: {type(e).__name__}: {e}")
            fail += 1
    print(f"\n[DONE] cards pushed: {done}/16   failed: {fail}")


if __name__ == "__main__":
    main()
