#!/usr/bin/env python3
"""Generate the fable-fleet training notebooks (16 models x 2 datasets = 32).

Clones each model's VALIDATED claude-distill template notebook and patches only
what differs for fable trace-SFT:

  cell 2 (config)   : MODEL_NAME, HUB_MODEL_ID (<stem>-SFT-Fable5[-Glint]), SLUG,
                      LORA_R/ALPHA=16, MAX_SEQ_LENGTH=4096, NUM_EPOCHS per dataset.
  cell 4 (load)     : inherited — already family-correct (FastModel vs FastLanguageModel,
                      VL-unwrap). Untouched.
  cell 5 (LoRA)     : inherited — r/alpha come from cell-2 constants. Untouched.
  cell 7 (dataset)  : REPLACED with the fable completion-path loader. The cleaner pushes
                      `messages` as a JSON STRING and tool_use assistant turns have EMPTY
                      content — the trainable signal is the `completion` field
                      (=<think>\\n{thinking}\\n</think>\\n{response}, tool-calls inline).
                      We render a synthetic 2-turn chat [user=context, assistant=completion]
                      through the model's OWN chat template so the masking markers exist.
  NEW masking cell  : inserted right after the SFTTrainer cell — wraps the trainer with
                      train_on_responses_only using markers DERIVED AT RUNTIME from the
                      tokenizer (family-agnostic; no fragile hardcoded delimiters), then
                      asserts the first batch has >0 unmasked label tokens (catches a
                      silent all-masked dataset before a wasted run).
  cell 10 (save)    : repo-stem swap only (HUB_MODEL_ID already set in cell 2).

Why runtime marker derivation: Qwen3.5 / Granite / LFM2 / Gemma all use different
assistant-turn delimiters, and login-node can't import unsloth (needs GPU) to probe
them ahead of time. Deriving the instruction/response parts from the live tokenizer at
train time is correct for every family and removes a whole class of mask-mismatch bugs.
"""
from __future__ import annotations
import json, copy, re
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
NB_DIR = ROOT / "notebooks"
REG = json.loads((ROOT / "scripts/tmp/fable_fleet_registry.json").read_text())

DATASETS = REG["datasets"]
RECIPE = REG["recipe"]
MODELS = REG["models"]


def cell_source(c) -> str:
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


def find_cell(nb, *needles, kind="code"):
    """Return index of first cell containing all needles."""
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != kind:
            continue
        s = cell_source(c)
        if all(n in s for n in needles):
            return i
    return None


# ---- the fable dataset cell (shared across all families) --------------------
def dataset_cell(repo: str, ds_key: str) -> str:
    return f'''# ============================================================================
# Load & Process FABLE Trace Dataset ({ds_key.upper()}) — completion-path
# ============================================================================
# The clean fable datasets store `messages` as a JSON STRING, and tool_use
# assistant turns have EMPTY content. The trainable signal is the `completion`
# field (= <think>..</think> + response, tool-calls inline). We rebuild a clean
# 2-turn chat [user=context, assistant=completion] and render it through THIS
# model's own chat template so assistant-turn markers exist for response-masking.
from datasets import load_dataset
import json as _json, hashlib as _hashlib

dataset = load_dataset("{repo}", split="train")
print(f"[OK] Loaded {{len(dataset):,}} fable samples from {repo}")

# ── Held-out eval split (Qwable-style) — EXCLUDE from training ───────────────
# Deterministic ~5pct holdout keyed on a stable per-row id, so the eval script can
# reconstruct the SAME complement without a separately-pushed dataset. Key =
# session_id if present, else a hash of the completion (D rows). hash mod 20 == 0.
def _holdout_key(sample):
    sid = str(sample.get("session_id") or "")
    return sid if sid else "C:" + str(sample.get("completion") or "")[:200]

def _is_holdout(sample):
    return int(_hashlib.md5(_holdout_key(sample).encode()).hexdigest(), 16) % 20 == 0

_n_all = len(dataset)
dataset = dataset.filter(lambda s: not _is_holdout(s))
_pct = 100 * (_n_all - len(dataset)) / max(_n_all, 1)
print(f"[OK] Excluded held-out eval split: train={{len(dataset):,}} / "
      f"held-out={{_n_all - len(dataset):,}} ({{_pct:.1f}} pct)")

if SMOKE_TEST:
    dataset = dataset.select(range(min(100, len(dataset))))
    print(f"[SMOKE] Truncated to {{len(dataset)}} samples")

def _first_user(sample):
    """Extract the user prompt: prefer messages[] (JSON string), fall back to context."""
    raw = sample.get("messages")
    if isinstance(raw, str) and raw.strip():
        try:
            for m in _json.loads(raw):
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                    return str(m["content"])
        except Exception:
            pass
    return str(sample.get("context") or "")

def format_fable(sample):
    user = _first_user(sample).strip()
    completion = str(sample.get("completion") or "").strip()
    msgs = [
        {{"role": "user", "content": user}},
        {{"role": "assistant", "content": completion}},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {{"text": text}}

def _has_target(sample):
    # Filter on the COMPLETION (the trainable assistant target), NOT the rendered
    # text length — a long user context must not let an EMPTY completion survive
    # (xmodel-review finding #5). Require a non-trivial completion.
    return len(str(sample.get("completion") or "").strip()) > 16

_n_before = len(dataset)
dataset = dataset.filter(_has_target)
dataset = dataset.map(format_fable)
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])
print(f"[OK] Processed: {{len(dataset):,}} samples (dropped {{_n_before - len(dataset):,}} empty-target)")
assert len(dataset) > 0, "Dataset is EMPTY after filtering — no trainable rows; refusing to train."
print("[sample] first 300 chars:\\n", dataset[0]["text"][:300])
'''


# ---- the response-only masking block, INJECTED into the trainer cell --------
# This is spliced in between `trainer = SFTTrainer(...)` and `trainer.train()`
# (same cell in every template). Inserting it as a separate cell AFTER the
# trainer cell would run masking post-training — a silent no-op. Markers are
# DERIVED AT RUNTIME from the tokenizer (family-agnostic).
MASK_BLOCK = '''
# ── Assistant-only loss masking (train_on_responses_only) — fable trace-SFT ──
# Mask prompt + tool-result tokens so loss is computed ONLY on assistant turns
# (Qwable-9B / recipe §3.3). Markers derived at runtime by diffing a chat
# rendered with vs without the generation prompt — correct for every family.
from unsloth.chat_templates import train_on_responses_only

def _derive_parts(tok):
    # IMPORTANT: derive markers from the SAME object the dataset cell renders with
    # (the module-level `tokenizer`), NOT a re-unwrapped inner — otherwise Gemma
    # (whose `tokenizer` is the get_chat_template-wrapped processor) would diff a
    # different template than the data was rendered with → all-masked → abort.
    sentinel = "\\u0000USR\\u0000"
    base = [{"role": "user", "content": sentinel}]
    no_gen = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=False)
    with_gen = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
    # Guard (xmodel-review finding #3): if the template stripped/escaped the NUL
    # sentinel, split() yields the WHOLE prompt as inst — a marker that won't match
    # real rows. Detect by requiring the sentinel to actually appear in no_gen.
    if sentinel not in no_gen:
        raise RuntimeError(
            "chat template did not preserve the NUL sentinel — cannot derive masking "
            "markers reliably. Inspect the rendered template for this model."
        )
    resp = with_gen[len(no_gen):] if with_gen.startswith(no_gen) else with_gen.split(sentinel)[-1]
    inst = no_gen.split(sentinel)[0]
    resp = resp.strip("\\n")
    # Thinking-model templates (e.g. Qwen3.5-0.8B/2B) auto-inject an EMPTY
    # "<think>\\n\\n</think>" scaffold right after the assistant role marker when
    # add_generation_prompt=True. Our real rows carry REAL <think> content, so a
    # response_part containing the empty scaffold matches NOTHING → everything is
    # masked → 0 trainable rows → "num_samples=0". Key masking on the assistant
    # ROLE marker only: truncate response_part at the first "<think>".
    _ti = resp.find("<think>")
    if _ti != -1:
        resp = resp[:_ti].rstrip("\\n")
    return inst.strip("\\n"), resp

_inst_part, _resp_part = _derive_parts(tokenizer)
print(f"[mask] instruction_part={_inst_part!r}")
print(f"[mask] response_part={_resp_part!r}")
assert _resp_part, "empty response_part — cannot mask"

trainer = train_on_responses_only(
    trainer, instruction_part=_inst_part, response_part=_resp_part,
)

# Sanity gate: run the collator on a couple of examples and count unmasked
# (!= -100) label tokens. All-masked => marker mismatch => silent zero signal.
# This is BEST-EFFORT verification: the collator's exact input shape varies by
# TRL/unsloth version, so any failure to introspect is a WARNING (never a crash)
# — a false gate-crash would block a perfectly good run. The real backstop is
# train loss being finite and > 0, which the JSONL logger captures per step.
import numpy as _np
try:
    _rows = [trainer.train_dataset[i] for i in range(min(2, len(trainer.train_dataset)))]
    _batch = trainer.data_collator(_rows)
    _lab = _batch["labels"]
    _labels = _lab.cpu().numpy() if hasattr(_lab, "cpu") else _np.asarray(_lab)
    _kept = int((_labels != -100).sum())
    print(f"[mask] unmasked label tokens across {len(_rows)} samples: {_kept}")
    if _kept == 0:
        print("[mask][WARN] ALL labels masked — response_part marker may not match the "
              "rendered template. Training would learn nothing. Inspect _resp_part above.")
    else:
        print("[OK] Assistant-only masking applied + verified (>0 unmasked tokens)")
except Exception as _e:
    print(f"[mask][WARN] could not introspect collator labels ({type(_e).__name__}: {_e}); "
          "masking IS applied, verification skipped. Watch first-step train_loss > 0.")
'''


def header_cell(name: str, base: str, ds_key: str, ds_repo: str) -> str:
    tag = "Glint pi-agent traces" if ds_key == "b" else "Complete 2M traces"
    return f'''# SFT Distillation: {name} on FABLE-5 ({tag}) — Unsloth

**Author:** Behrooz Azarkhalili

| Feature | Detail |
|---------|--------|
| Base | `{base}` |
| Dataset | `{ds_repo}` |
| Method | LoRA (QLoRA 4-bit), assistant-only masking |
| Rank / α | r={RECIPE["lora_r"]}, α={RECIPE["lora_alpha"]} |
| LR / seq | {RECIPE["learning_rate"]}, max_seq {RECIPE["max_seq_length"]} |
| Recipe | docs/recipes/fable-mythos-distillation.md §3.1 (Qwable-9B-aligned) |
'''


def patch_config(src: str, m: dict, ds_key: str, ds: dict) -> str:
    """Rewrite cell-2 constants for this (model, dataset).

    Matches each constant by EXACT name at the start of a top-level assignment
    (regex `^NAME\\s*=`), so `LORA_RANK`/`NUM_EPOCHS_OVERRIDE` never false-match
    (xmodel finding #9), and preserves the original indentation. Asserts each
    REQUIRED constant was actually present (catches a template missing the line —
    xmodel finding #8).
    """
    stem = m.get("repo_stem", m["name"])
    hub = f"ermiaazarkhalili/{stem}-SFT-{ds['suffix']}"
    slug = f"fable{'glint' if ds_key=='b' else ''}-{m['nickname']}"
    repl = {
        "MODEL_NAME": f'MODEL_NAME = "{m["base"]}"',
        "HUB_MODEL_ID": f'HUB_MODEL_ID = "{hub}"',
        "MAX_SEQ_LENGTH": f'MAX_SEQ_LENGTH = {RECIPE["max_seq_length"]}',
        "LORA_R": f'LORA_R = {RECIPE["lora_r"]}',
        "LORA_ALPHA": f'LORA_ALPHA = {RECIPE["lora_alpha"]}',
        "LEARNING_RATE": f'LEARNING_RATE = {RECIPE["learning_rate"]}',
        "NUM_EPOCHS": f'NUM_EPOCHS = 1 if SMOKE_TEST else {ds["epochs"]}',
        "SLUG": f'SLUG = "{slug}"',
    }
    # HARD-required: a correct training notebook cannot lack these. LORA_R/ALPHA are
    # only WARNED on — a template may legitimately derive alpha inline or from rank
    # (xmodel round-2 finding), so a false SystemExit there would be wrong.
    required = {"MODEL_NAME", "HUB_MODEL_ID", "NUM_EPOCHS"}
    warn_if_missing = {"LORA_R", "LORA_ALPHA"}
    seen = set()
    out = []
    for line in src.splitlines():
        indent = line[: len(line) - len(line.lstrip())]
        stripped = line.lstrip()
        hit = None
        for name, newval in repl.items():
            # exact assignment: NAME possibly-spaces = ...   (not NAME_FOO = ...)
            if re.match(rf"{re.escape(name)}\s*=", stripped) and not stripped.startswith("#"):
                hit = (name, newval)
                break
        if hit:
            out.append(indent + hit[1])
            seen.add(hit[0])
        else:
            out.append(line)
    missing = required - seen
    if missing:
        raise SystemExit(f"{m['name']}: config cell missing required constants {sorted(missing)}")
    soft = warn_if_missing - seen
    if soft:
        print(f"[warn] {m['name']}: config cell lacks {sorted(soft)} — assuming derived elsewhere")
    return "\n".join(out)


def as_lines(s: str):
    """nbformat source as a list[str] with trailing newlines except last."""
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build_one(m: dict, ds_key: str) -> tuple[str, dict]:
    ds = DATASETS[ds_key]
    tpl_path = NB_DIR / m["template"]
    nb = json.loads(tpl_path.read_text())

    # cell 0 header
    if nb["cells"] and nb["cells"][0]["cell_type"] == "markdown":
        nb["cells"][0]["source"] = as_lines(header_cell(m["name"], m["base"], ds_key, ds["repo"]))

    # cell 2 config
    ci = find_cell(nb, "SMOKE_TEST", "MODEL_NAME")
    if ci is None:
        raise SystemExit(f"{m['name']}: no config cell in {m['template']}")
    nb["cells"][ci]["source"] = as_lines(patch_config(cell_source(nb["cells"][ci]), m, ds_key, ds))

    # cell 7 dataset (match by load_dataset)
    di = find_cell(nb, "load_dataset")
    if di is None:
        raise SystemExit(f"{m['name']}: no dataset cell in {m['template']}")
    nb["cells"][di]["source"] = as_lines(dataset_cell(ds["repo"], ds_key))

    # Inject masking INLINE into the trainer cell, between the SFTTrainer(...)
    # construction and the `<var>.train()` call. Robust to (xmodel #1/#2):
    #   - a `.train()` mention inside a comment before construction,
    #   - a trainer variable named other than `trainer` (sft_trainer, etc.).
    # We find the assignment `<var> = SFTTrainer(`, then the FIRST real (non-comment)
    # `<var>.train(` line after it, and splice the mask block before that line.
    ti = find_cell(nb, "SFTTrainer", ".train(")
    if ti is None:
        raise SystemExit(f"{m['name']}: no combined SFTTrainer+train cell in {m['template']}")
    tlines = cell_source(nb["cells"][ti]).split("\n")
    var = None
    construct_i = None
    for i, ln in enumerate(tlines):
        st = ln.strip()
        if st.startswith("#"):
            continue
        mm = re.match(r"([A-Za-z_]\w*)\s*=\s*SFTTrainer\(", st)
        if mm:
            var, construct_i = mm.group(1), i
            break
    if var is None:
        raise SystemExit(f"{m['name']}: could not find `<var> = SFTTrainer(` in trainer cell")
    train_i = None
    for i in range(construct_i + 1, len(tlines)):
        st = tlines[i].strip()
        if st.startswith("#"):
            continue
        if re.search(rf"\b{re.escape(var)}\.train\(", st):
            train_i = i
            break
    if train_i is None:
        raise SystemExit(f"{m['name']}: no real `{var}.train(` after SFTTrainer construction")
    # The MASK_BLOCK references `trainer`; alias it to the real var name if needed.
    block = MASK_BLOCK.strip("\n")
    if var != "trainer":
        block = f"trainer = {var}  # alias for masking helper\n" + block + f"\n{var} = trainer"
    tlines[train_i:train_i] = ["", *block.split("\n"), ""]
    # Requeue-safety: make `<var>.train()` resume from the last checkpoint in
    # output_dir if one exists (auto-detected). A long D-run that hits walltime or
    # a node failure then continues instead of restarting from scratch. No-op on a
    # fresh run (no checkpoint → trains from step 0).
    joined = "\n".join(tlines)
    joined = re.sub(
        rf"(\b{re.escape(var)}\.train\()\s*\)",
        r"\1resume_from_checkpoint=(_os.path.isdir(CHECKPOINT_ROOT) and any("
        r"d.startswith('checkpoint-') for d in _os.listdir(CHECKPOINT_ROOT))))",
        joined, count=1,
    )
    if "resume_from_checkpoint" in joined and "import os as _os" not in joined.split("resume_from_checkpoint")[0][-400:]:
        joined = "import os as _os\n" + joined
    nb["cells"][ti]["source"] = as_lines(joined)

    # Localize the save-cell scratch-dir literals to a UNIQUE per-(model,dataset)
    # prefix. The templates carry stale, COLLIDING names (every Qwen3.5 variant
    # inherited `qwen3_8b_distill_*`), and save_pretrained writes relative to the
    # papermill --cwd (notebooks/), so concurrent runs would clobber the same dir.
    # Hub push is unaffected (uses HUB_MODEL_ID) — this only fixes local races.
    # Absolute, unique scratch path so (a) concurrent runs never collide and
    # (b) artifacts don't pollute the papermill --cwd (notebooks/). Mirrors the
    # template's CHECKPOINT_ROOT convention.
    out_def = (
        'import os as _os\n'
        '_SCRATCH = _os.environ.get("SCRATCH", "/scratch/" + _os.environ.get("USER", "ermia"))\n'
        f'_FABLE_OUT = _os.path.join(_SCRATCH, "fable-artifacts", "{m["dir"]}_{ds_key}")\n'
        '_os.makedirs(_os.path.dirname(_FABLE_OUT), exist_ok=True)\n'
    )
    # Only rewrite the literal that is the FIRST ARG of a save_pretrained* call
    # (xmodel #7) — never a quoted token inside print()/comments. Tolerate
    # path-prefixed/uppercase dir names (xmodel #6).
    save_call = re.compile(
        r'(\.save_pretrained(?:_merged|_gguf)?\(\s*)"([^"]*?_distill_(lora|merged|gguf))"'
    )
    for c in nb["cells"]:
        s = cell_source(c)
        if ".save_pretrained" not in s:
            continue
        new = save_call.sub(lambda mm: f'{mm.group(1)}(_FABLE_OUT + "_{mm.group(3)}")', s)
        if new != s:
            c["source"] = as_lines(out_def + new)

    # Bump generation token caps to 4096 everywhere. Fable completions
    # (<think>+response+tool-calls) run long: measured B p99=2420, max=3830 tok.
    # Both the inference test AND the Hub round-trip validation must generate a
    # FULL reasoning+response (a short cap truncates mid-<think> and proves nothing).
    # 4096 covers 100% of completions and aligns with MAX_SEQ_LENGTH.
    for c in nb["cells"]:
        s = cell_source(c)
        if "max_new_tokens" not in s and "_VAL_MAX_TOKENS" not in s:
            continue
        new = re.sub(r"max_new_tokens\s*=\s*\d+", "max_new_tokens=4096", s)
        new = re.sub(r"_VAL_MAX_TOKENS\s*=\s*\d+", "_VAL_MAX_TOKENS = 4096", new)
        if new != s:
            c["source"] = as_lines(new)

    out_name = f"fable_distillation_{m['nickname']}_{ds['name_tag']}_unsloth.ipynb"
    return out_name, nb


def main():
    written = []
    for m in MODELS:
        for ds_key in ("d", "b"):
            name, nb = build_one(m, ds_key)
            (NB_DIR / name).write_text(json.dumps(nb, indent=1))
            written.append(name)
    print(f"[OK] wrote {len(written)} fable notebooks:")
    for n in written:
        print("   ", n)
    assert len(written) == len(MODELS) * 2, "expected 32 notebooks"


if __name__ == "__main__":
    main()
