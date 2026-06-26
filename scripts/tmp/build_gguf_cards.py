#!/usr/bin/env python3
"""Generate + push consumer-facing model cards for GGUF repos.

The main card generator (generate_unsloth_model_card.py) renders MERGED-model
cards that LINK to the GGUF repo, but never produces a standalone GGUF-repo
README. This fills that: a GGUF-specific card (quant download table from the
LIVE repo, llama.cpp + Ollama usage, pointer back to the full-precision repo),
reusing the MODELS registry for base-model / license / dataset / training metrics.

Usage:
  python build_gguf_cards.py --repo <merged_repo_id>     # one (card pushed to <repo>-GGUF)
  python build_gguf_cards.py --missing                   # all GGUF repos lacking a README
  python build_gguf_cards.py --missing --dry_run         # preview only
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path

from huggingface_hub import HfApi

CARD_GEN = Path("/project/6014832/ermia/HF-TRL/scripts/generate_unsloth_model_card.py")

# Load the registry from the main generator (single source of truth).
_spec = importlib.util.spec_from_file_location("mcg", CARD_GEN)
_mcg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mcg)
MODELS = _mcg.MODELS

api = HfApi()

# Recommended-use blurbs per quant token.
QUANT_NOTES = {
    "q2_k":   "Smallest, lowest quality — quick tests / very constrained devices",
    "q3_k_m": "Small, acceptable quality",
    "q3_k_s": "Small, slightly lower quality than Q3_K_M",
    "q3_k_l": "Small-medium, better quality than Q3_K_M",
    "q4_0":   "Legacy 4-bit, fast",
    "q4_k_s": "Compact 4-bit",
    "q4_k_m": "Recommended — best size/quality balance",
    "q5_0":   "Legacy 5-bit",
    "q5_k_s": "5-bit, compact",
    "q5_k_m": "High quality, larger",
    "q6_k":   "Very high quality, near-fp16",
    "q8_0":   "Near-lossless, largest",
}
QUANT_ORDER = ["q2_k","q3_k_s","q3_k_m","q3_k_l","q4_0","q4_k_s","q4_k_m",
               "q5_0","q5_k_s","q5_k_m","q6_k","q8_0"]


def list_gguf(gguf_repo: str):
    """Return [(filename, quant_token, size_bytes)] for the live GGUF repo."""
    info = api.repo_info(gguf_repo, repo_type="model", files_metadata=True)
    out = []
    for s in info.siblings:
        fn = s.rfilename
        if not fn.endswith(".gguf"):
            continue
        q = fn.rsplit(".", 2)[-2].lower()
        sz = getattr(s, "size", 0) or 0
        if sz == 0 and getattr(s, "lfs", None):
            sz = s.lfs.get("size", 0) if isinstance(s.lfs, dict) else getattr(s.lfs, "size", 0)
        out.append((fn, q, sz))
    out.sort(key=lambda x: QUANT_ORDER.index(x[1]) if x[1] in QUANT_ORDER else 99)
    return out


def has_readme(repo: str) -> bool:
    try:
        info = api.repo_info(repo, repo_type="model")
        return any(s.rfilename == "README.md" for s in info.siblings)
    except Exception:
        return False


def render(merged_repo: str, m: dict) -> str:
    gguf_repo = f"{merged_repo}-GGUF"
    ggufs = list_gguf(gguf_repo)
    is_xlam = m["task"] == "xlam_function_calling"
    task_desc = "function calling" if is_xlam else "reasoning distillation (chain-of-thought)"
    task_tag = "function-calling" if is_xlam else "reasoning"
    base = m["base_model"]
    disp = m["display_name"]
    gguf_name = gguf_repo.split("/")[-1]
    first_q = next((q for _, q, _ in ggufs if q == "q4_k_m"), ggufs[0][1] if ggufs else "q4_k_m")
    Q = first_q.upper()

    # Download table
    rows = []
    for fn, q, sz in ggufs:
        note = QUANT_NOTES.get(q, "")
        rows.append(f"| `{q.upper()}` | {sz/1e9:.2f} GB | {note} | [`{fn}`](https://huggingface.co/{gguf_repo}/blob/main/{fn}) |")
    table = "\n".join(rows) if rows else "| _none found_ | | | |"

    dataset_link = (
        "[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)"
        if is_xlam else
        "[claude-reasoning-distillation](https://huggingface.co/datasets/ermiaazarkhalili/claude-reasoning-distillation)"
    )
    usage_prompt = ("Check if the numbers 8 and 1233 are powers of two."
                    if is_xlam else
                    "Solve step by step: What is the sum of the first 10 prime numbers?")

    # Training outcome (reuse registry metrics if present)
    o = m.get("training_outcome")
    outcome = ""
    if o:
        rt = o.get("runtime_sec", 0); h, rem = divmod(rt, 3600); mi, s = divmod(rem, 60)
        hms = f"{h}h {mi:02d}m {s:02d}s" if h else f"{mi}m {s:02d}s"
        outcome = (
            "\n## Training Outcome\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| SLURM Job ID | `{o.get('job_id','N/A')}` |\n"
            f"| Runtime | {hms} |\n"
            f"| Final Training Loss | {o.get('train_loss','N/A')} |\n"
            f"| Peak VRAM | {o.get('peak_vram_gb','N/A')} GB |\n"
            f"| GPU | {o.get('gpu','N/A')} |\n"
        )

    return f"""---
license: {m['license']}
language:
  - en
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - quantized
  - {m['model_family']}
  - {task_tag}
  - unsloth
  - text-generation
base_model: {merged_repo}
datasets:
  - {m['dataset']}
---

# {gguf_name}

GGUF quantizations of [{disp}](https://huggingface.co/{merged_repo}) for **CPU and edge inference** with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), LM Studio, and other GGUF runtimes.

This model is a fine-tune of [{m['base_model_name']}](https://huggingface.co/{base}) for **{task_desc}**, trained with [Unsloth](https://github.com/unslothai/unsloth) on {dataset_link}.

- **Full-precision model:** [{disp}](https://huggingface.co/{merged_repo})
- **Base model:** [{m['base_model_name']}](https://huggingface.co/{base})
- **Parameters:** {m['params']}

## Available Quantizations

| Quant | Size | Recommended Use | File |
|-------|------|-----------------|------|
{table}

`Q4_K_M` is the recommended default for most users.

## Usage

### Download a single quant

```bash
pip install -U "huggingface_hub[cli]"
hf download {gguf_repo} \\
  --include "*{first_q}*.gguf" --local-dir ./{gguf_name}
```

### llama.cpp

```bash
# build: https://github.com/ggerganov/llama.cpp
./llama-cli -m ./{gguf_name}/{ggufs[0][0] if ggufs else f"{gguf_name}.{first_q}.gguf"} \\
  -p "{usage_prompt}" -n 512
```

### Ollama

```bash
ollama run hf.co/{gguf_repo}:{Q} "{usage_prompt}"
```
{outcome}
## License

{m['license'].upper()} — see the [base model]({f"https://huggingface.co/{base}"}) for full terms.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for 2x faster fine-tuning
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF quantization
- Base model developers ({base.split('/')[0]})
- [Compute Canada / DRAC](https://alliancecan.ca/) for HPC resources
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="merged repo id (card pushed to <repo>-GGUF)")
    ap.add_argument("--missing", action="store_true", help="all GGUF repos lacking a README")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.repo:
        targets = [args.repo]
    elif args.missing:
        targets = []
        for mr in MODELS:
            gr = f"{mr}-GGUF"
            try:
                if api.repo_info(gr, repo_type="model") and not has_readme(gr):
                    targets.append(mr)
            except Exception:
                pass  # GGUF repo absent — skip
    else:
        raise SystemExit("pass --repo <id> or --missing")

    print(f"Targets ({len(targets)}):")
    for t in targets:
        print(f"  {t}-GGUF")
    print()

    for mr in targets:
        if mr not in MODELS:
            print(f"[SKIP] {mr} not in registry"); continue
        gr = f"{mr}-GGUF"
        try:
            card = render(mr, MODELS[mr])
        except Exception as e:
            print(f"[FAIL] render {gr}: {type(e).__name__}: {e}"); continue
        if args.dry_run:
            print(f"--- {gr} ({len(card)} chars) ---")
            i = card.find("## Available")
            print(card[:i+600])
            print("...\n")
        else:
            api.upload_file(path_or_fileobj=card.encode("utf-8"),
                            path_in_repo="README.md", repo_id=gr, repo_type="model")
            print(f"[OK] pushed card -> {gr}")


if __name__ == "__main__":
    main()
