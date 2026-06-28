#!/usr/bin/env python3
"""Generate notebooks/eval/fable_eval_TEMPLATE.ipynb — ONE papermill-parameterized eval
notebook reused for all B-leg fable models.

Evaluates a TRAINED model and its BASE on the deterministic ~5% held-out split (reconstructed
verbatim from the train notebooks' holdout logic), reporting top-1 + top-5 teacher-forced
token-accuracy (assistant-masked, each model's own chat template) plus a generative ROUGE-L
sample. Writes a JSON result.

Parameters are read from env (set by the SLURM wrapper):
  MODEL_REPO, BASE_REPO, FAMILY, DATASET_REPO, OUT_JSON, N_EVAL, N_GEN

Run:  python3 scripts/tmp/build_fable_eval_notebook.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
OUT = ROOT / "notebooks" / "eval"
OUT.mkdir(parents=True, exist_ok=True)


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


# --- cells ---------------------------------------------------------------------

PARAMS = '''
# --- Parameters (papermill / env-driven) -------------------------------------
import os
MODEL_REPO   = os.environ["MODEL_REPO"]                     # trained model
BASE_REPO    = os.environ["BASE_REPO"]                      # its base
FAMILY       = os.environ.get("FAMILY", "dense")            # gemma|qwen35|lfm2|dense
DATASET_REPO = os.environ.get("DATASET_REPO", "ermiaazarkhalili/Fable-5-Glint-Clean")
OUT_JSON     = os.environ["OUT_JSON"]
N_EVAL       = int(os.environ.get("N_EVAL", "0"))           # 0 = full held-out split
N_GEN        = int(os.environ.get("N_GEN", "20"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
LOAD_IN_4BIT   = os.environ.get("LOAD_IN_4BIT", "1") not in ("0", "false", "False")
print(f"MODEL={MODEL_REPO}\\nBASE={BASE_REPO}\\nFAMILY={FAMILY}\\nDATASET={DATASET_REPO}")
print(f"N_EVAL={N_EVAL or 'full'}  N_GEN={N_GEN}")
'''

HOLDOUT = r'''
# --- Reconstruct the held-out split (VERBATIM key from the train notebooks) ----
# Training EXCLUDED rows where md5(key)%20==0; the eval is exactly that complement.
from datasets import load_dataset
import json as _json, hashlib as _hashlib

dataset = load_dataset(DATASET_REPO, split="train")
print(f"[OK] Loaded {len(dataset):,} rows from {DATASET_REPO}")

def _holdout_key(sample):
    sid = str(sample.get("session_id") or "")
    return sid if sid else "C:" + str(sample.get("completion") or "")[:200]

def _is_holdout(sample):
    return int(_hashlib.md5(_holdout_key(sample).encode()).hexdigest(), 16) % 20 == 0

def _has_target(sample):
    return len(str(sample.get("completion") or "").strip()) > 16

def _first_user(sample):
    raw = sample.get("messages")
    if isinstance(raw, str) and raw.strip():
        try:
            for m in _json.loads(raw):
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                    return str(m["content"])
        except Exception:
            pass
    return str(sample.get("context") or "")

_held = dataset.filter(lambda s: _is_holdout(s) and _has_target(s))
print(f"[OK] Held-out eval rows: {len(_held):,} ({100*len(_held)/max(len(dataset),1):.1f}% of corpus)")

# Leakage guard: the training complement must NOT intersect the held-out keys.
_hk = {_holdout_key(s) for s in _held}
_train_sample = dataset.filter(lambda s: not _is_holdout(s)).select(range(min(2000, max(0, len(dataset)-len(_held)))))
_leak = sum(1 for s in _train_sample if _holdout_key(s) in _hk)
assert _leak == 0, f"LEAKAGE: {_leak} train rows share a held-out key"
print(f"[OK] Leakage check passed (0 shared keys over {len(_train_sample)} train rows)")

EVAL = [{"prompt": _first_user(s).strip(), "gold": str(s.get("completion") or "").strip()} for s in _held]
if N_EVAL and N_EVAL < len(EVAL):
    EVAL = EVAL[:N_EVAL]
assert EVAL, "Empty eval set"
print(f"[OK] Eval pairs: {len(EVAL)}")
'''

LOADER_FN = '''
# --- Family-aware loader (mirrors the train notebooks) ------------------------
import torch

def load_model_tok(repo: str, family: str):
    """Return (model, tokenizer) for any fable family. gemma->FastModel, else
    FastLanguageModel; unwrap processor->tokenizer for VL (qwen35)."""
    if family == "gemma":
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(
            model_name=repo, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=LOAD_IN_4BIT,
        )
    else:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=repo, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=LOAD_IN_4BIT,
        )
    if hasattr(tok, "tokenizer"):          # VL-arch: unwrap processor -> tokenizer
        tok = tok.tokenizer
    from unsloth import FastLanguageModel as _FLM
    try:
        _FLM.for_inference(model)
    except Exception:
        pass
    model.eval()
    return model, tok
'''

SCORE_FN = r'''
# --- Token-accuracy (top-1 + top-5, assistant-masked) + generation ------------
import torch, re

def _render(tok, user: str, gold: str):
    """Render [user, assistant=gold] via the model's OWN chat template, with a plain
    fallback if the template is missing. Returns (full_text, assistant_char_start)."""
    msgs_full = [{"role": "user", "content": user}, {"role": "assistant", "content": gold}]
    msgs_prompt = [{"role": "user", "content": user}]
    try:
        full = tok.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
        # assistant span = everything after the prompt prefix (if prompt is a prefix of full)
        if full.startswith(prompt):
            return full, len(prompt)
        # template not prefix-consistent: fall back to locating gold
        idx = full.rfind(gold)
        return full, (idx if idx >= 0 else len(full) - len(gold))
    except Exception:
        full = f"user:\n{user}\nassistant:\n{gold}"
        prompt = f"user:\n{user}\nassistant:\n"
        return full, len(prompt)

@torch.no_grad()
def token_accuracy(model, tok, pairs, max_len=4096):
    """Teacher-forced top-1 + top-5 next-token accuracy over ASSISTANT tokens only."""
    dev = model.device
    n_top1 = n_top5 = n_tok = 0
    for ex in pairs:
        full, a_start = _render(tok, ex["prompt"], ex["gold"])
        enc = tok(full, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"].to(dev)
        if ids.shape[1] < 2:
            continue
        # assistant token start: tokenize the prompt prefix to get its length
        pref_ids = tok(full[:a_start], return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]
        a_tok = pref_ids.shape[1]
        if a_tok >= ids.shape[1]:
            continue
        logits = model(ids).logits  # [1, T, V]
        # predict token t from position t-1; we score positions [a_tok .. T-1]
        pred = logits[0, a_tok-1:ids.shape[1]-1, :]      # [La, V]
        gold_ids = ids[0, a_tok:ids.shape[1]]             # [La]
        if pred.shape[0] == 0:
            continue
        top5 = pred.topk(5, dim=-1).indices               # [La, 5]
        top1 = top5[:, 0]
        n_top1 += (top1 == gold_ids).sum().item()
        n_top5 += (top5 == gold_ids.unsqueeze(-1)).any(dim=-1).sum().item()
        n_tok += gold_ids.shape[0]
    return {
        "top1": n_top1 / max(n_tok, 1),
        "top5": n_top5 / max(n_tok, 1),
        "n_tokens": n_tok,
    }

def _rouge_l(a: str, b: str) -> float:
    """Lightweight ROUGE-L F1 (LCS over whitespace tokens). No external dep."""
    x, y = a.split(), b.split()
    if not x or not y:
        return 0.0
    dp = [[0]*(len(y)+1) for _ in range(len(x)+1)]
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            dp[i][j] = dp[i-1][j-1]+1 if x[i-1] == y[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs/len(x), lcs/len(y)
    return 2*prec*rec/(prec+rec)

@torch.no_grad()
def generate_samples(model, tok, pairs, n, max_new_tokens=512):
    outs = []
    for ex in pairs[:n]:
        msgs = [{"role": "user", "content": ex["prompt"]}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"user:\n{ex['prompt']}\nassistant:\n"
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append({"prompt": ex["prompt"][:300], "gold": ex["gold"][:500], "gen": text[:500],
                     "rougeL": _rouge_l(text, ex["gold"])})
    return outs
'''

RUN = r'''
# --- Run: trained then base (free GPU between loads) --------------------------
import gc, torch, json

def eval_one(repo, family):
    model, tok = load_model_tok(repo, family)
    acc = token_accuracy(model, tok, EVAL, max_len=MAX_SEQ_LENGTH)
    gens = generate_samples(model, tok, EVAL, N_GEN)
    rouge = sum(g["rougeL"] for g in gens) / max(len(gens), 1)
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return acc, rouge, gens

print(f"\\n=== TRAINED: {MODEL_REPO} ===")
acc_t, rouge_t, gens_t = eval_one(MODEL_REPO, FAMILY)
print(f"  top1={acc_t['top1']:.4f}  top5={acc_t['top5']:.4f}  rougeL={rouge_t:.4f}  ntok={acc_t['n_tokens']}")

print(f"\\n=== BASE: {BASE_REPO} ===")
acc_b, rouge_b, gens_b = eval_one(BASE_REPO, FAMILY)
print(f"  top1={acc_b['top1']:.4f}  top5={acc_b['top5']:.4f}  rougeL={rouge_b:.4f}  ntok={acc_b['n_tokens']}")

# pair generative samples for qualitative inspection
samples = []
for gt, gb in zip(gens_t, gens_b):
    samples.append({"prompt": gt["prompt"], "gold": gt["gold"],
                    "trained_gen": gt["gen"], "base_gen": gb["gen"],
                    "rougeL_trained": gt["rougeL"], "rougeL_base": gb["rougeL"]})

result = {
    "model": MODEL_REPO, "base": BASE_REPO, "family": FAMILY, "dataset": DATASET_REPO,
    "n_eval": len(EVAL),
    "top1_trained": acc_t["top1"], "top1_base": acc_b["top1"], "top1_delta": acc_t["top1"]-acc_b["top1"],
    "top5_trained": acc_t["top5"], "top5_base": acc_b["top5"], "top5_delta": acc_t["top5"]-acc_b["top5"],
    "rougeL_trained": rouge_t, "rougeL_base": rouge_b, "rougeL_delta": rouge_t-rouge_b,
    "n_gen": len(samples), "samples": samples,
}
import os as _os
_os.makedirs(_os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print("\\n================= SUMMARY =================")
print(f"{'metric':<10}{'base':>10}{'trained':>10}{'delta':>10}")
for k in ("top1", "top5"):
    print(f"{k:<10}{result[k+'_base']:>10.4f}{result[k+'_trained']:>10.4f}{result[k+'_delta']:>+10.4f}")
print(f"{'rougeL':<10}{rouge_b:>10.4f}{rouge_t:>10.4f}{result['rougeL_delta']:>+10.4f}")
print(f"\\n[OK] wrote {OUT_JSON}")
'''


def build():
    cells = [
        md("# Fable held-out eval — trained vs base\n\nReconstructs the deterministic ~5% held-out "
           "split and scores the trained model and its base: top-1 + top-5 teacher-forced "
           "token-accuracy (assistant-masked, each model's own chat template) plus a generative "
           "ROUGE-L sample. Parameterized by env vars; driven by `jobs/eval-fable-single.sh`."),
        code(PARAMS),
        md("## 1 · Held-out split (verbatim holdout key, leakage-guarded)"),
        code(HOLDOUT),
        md("## 2 · Family-aware loader"),
        code(LOADER_FN),
        md("## 3 · Scoring functions (token-acc top1/top5 + ROUGE-L generation)"),
        code(SCORE_FN),
        md("## 4 · Evaluate trained + base, write JSON"),
        code(RUN),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = OUT / "fable_eval_TEMPLATE.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    # AST-validate code cells (strip magics — none here, but be consistent)
    bad = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        py = "\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in src.splitlines())
        try:
            ast.parse(py)
        except SyntaxError as e:
            bad.append((i, e))
    if bad:
        for i, e in bad:
            print(f"  [SYNTAX] cell {i}: {e}")
        raise SystemExit("[FAIL] eval notebook has syntax errors")
    print(f"[OK] {path}: {len(cells)} cells, all code cells AST-clean")


if __name__ == "__main__":
    build()
