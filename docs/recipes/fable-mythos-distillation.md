# Fable-5 / Mythos Distillation — Preprocessing & Training Recipes

> Durable reference for cleaning Fable-5 / Mythos agent-trace datasets and training
> small models on them. Distilled from the **trending fable models' own dataset cards
> and tooling** (survey 2026-06-24). Reuse this for any future trace-distillation work.

---

## 0. TL;DR

- **Preprocess** raw agent traces with **`teich`** (harness→OpenAI-chat) → then a
  **`clean_fable5`-style normalizer** (PII anon, ANSI strip, 3-mode schema).
- **Train** with LoRA **r=16 / α=16, LR 2e-4, 2–3 epochs**, **assistant-only loss masking**
  (`train_on_responses_only`), **preserve tool-calls**, `max_seq ~4096–8192`.
- Output ONE clean JSONL per source supporting **3 training shapes** (full-SFT /
  reasoning-split / instruction-only). Publish **private**.

---

## 1. Source datasets (real > synthetic)

| Tag | Repo | Size | Type | Notes |
|---|---|---:|---|---|
| A | `WithinUsAI/claude_mythos_distilled_25k` | 25K | ⚠️ synthetic mirror | canned preambles, fake code stubs, dup benchmarks. NO trending model used it. |
| B | `Glint-Research/Fable-5-traces` (`pi_agent/train`) | 4,665 | ✅ real pi-agent | ~83% tool-use. `messages[]` w/ `reasoning_content` + structured `tool_calls`. |
| C | `armand0e/claude-fable-5-claude-code` | raw | ✅ real anon | TeichAI lineage (Qwen3.6-27B, gpt-oss-120b trained on it). |
| D | `Crownelius/Complete-FABLE.5-traces-2M` | 2.0M | ✅ meta-aggregation | 981 MB. content in `row_json` string; route by `first_source_dataset`. |

**Provenance verdict:** every trending fable model trained on a *real-trace* source
(B/C/D). The #1 model (`yuxinlu1/gemma-4-12B-coder-fable5`, 495K dl) used a "composer2.5"
blend + Opus-rebuilt reasoning + test-gated Python CoT. None used the synthetic mythos-25K.

---

## 2. Preprocessing pipeline

### 2.1 `teich` — the standard trace extractor

```bash
pip install -U teich
teich extract claude --model fable-5     # Claude Code sessions
# teich extract pi     --model fable-5   # Pi-agent
# teich extract hermes --model fable-5
# teich extract cursor --model fable-5
```

- Converts raw session JSONLs → OpenAI-style `messages:[{role,content,tool_calls}]`.
- Knows each harness's exact tool-schemas/descriptions → tool calls round-trip faithfully.
- Auto-filters: session-limit hits, model-switch lines (`/model → Set model to Fable 5`),
  harness artifacts.

### 2.2 `clean_fable5`-style normalization (replicate kelexine/clean_fable5.py)

Cleaning steps:
1. Strip ANSI escapes (`[1m` …) + command-injection / `<local-command-caveat>` blocks.
2. **PII path anonymization:** `/home/<username>/` → `/home/user/` (+ Windows equiv).
3. Parse `context` → OpenAI-format `messages`.
4. Normalize `response`; standardize tool-use format.
5. Build `completion` with explicit think tags.
6. Classify `task_type` by `output_type` + CoT length (**>450 chars ⇒ reasoning**).
7. **Dedup** by normalized row/session content. Drop empty-assistant / truncated rows.

### 2.3 Output schema — ONE JSONL, THREE training shapes

| Mode | Fields used | Purpose |
|---|---|---|
| **Full SFT** | `messages` OR `context`+`completion` | thinking + response end-to-end |
| **Reasoning-split** | `context` + `thinking` + `response` | separate CoT / output targets |
| **Instruction-only** | `context` + `response` | response only, no CoT |

Columns per row:
`model`, `origin`(local/hf), `task_type`(agentic/reasoning), `output_type`(tool_use/text),
`context_truncated`(bool), **`messages`** (OpenAI list w/ `tool_calls` + `role:"tool"` stdout),
`context` (cleaned+anon), `thinking` (raw CoT, **no** `<think>` tags), `response` (clean),
`output` (dict `{tool,input}`), `completion` (= `<think>\n{thinking}\n</think>\n{response}`),
`cot_length`, `context_length`, `response_length` (int).

Tool-call representation:
```
<tool_call>{"name":"Bash","arguments":{...}}</tool_call>
```
inline in `completion`, plus structured `messages[].tool_calls` + `role:"tool"` for stdout.

### 2.4 Per-source specifics

- **B (Glint pi_agent):** already structured — use `messages[]`, DROP the flattened `prompt`
  column, keep `reasoning_content`→thinking, keep `tool_calls`. Tool/text ≈ 83/17.
- **C (armand0e raw):** `teich extract claude --model fable-5` then normalize.
- **D (2M):** **polars LAZY scan** (981 MB → CPU SLURM job, never login-node pandas).
  Parse `row_json` per row; route extraction by `first_source_dataset`; **filter out
  synthetic-mirror sources**; then normalize. Already row-dedup'd + session-limit-rows removed.
- **A (mythos-25K):** not a trace set. Strip canned preamble/closing tag, dedup templated rows,
  drop stub/fake-code responses. Expect survivors << 25K (audit before training).

---

## 3. Training recipe (proven by trending models)

### 3.1 LoRA SFT — community sweet spot (our default)

From `hotdogs/qwen3.6-27b-fable5-lora` (loss 0.239) and aligned with our existing fleet:

| Param | Value |
|---|---|
| Method | LoRA (4-bit QLoRA, NF4, bf16 compute) |
| Rank / α | **r=16, α=16** (hotdogs used r=8; 16 matches our fleet) |
| Target modules | `q,k,v,o,gate,up,down_proj` |
| LR | **2e-4**, cosine |
| Epochs | **2–3** |
| Max seq | **4096–8192** (MIG fleet; filter longer sessions) |
| Batch | bs=2, grad_accum=4 (per our MIG 3g.40gb fleet) |
| Loss | **assistant-only masking** via `train_on_responses_only` |

### 3.2 Full fine-tune — reference (Qwable-9B, empero-ai)

| Param | Value |
|---|---|
| Method | full-param SFT, TRL |
| Base | Qwen3.5-9B (text backbone; vision tower frozen) |
| Epochs | 2 |
| Eff. batch | 16 |
| Max seq | 76,800 (no truncation — needs big VRAM) |
| LR | 1e-5, cosine, 3% warmup |
| Optim | AdamW-8bit, bf16 |
| Loss | chunked NLL, **assistant-only masking** |
| Data mix | ~97% Fable traces + ~3% terminal; 100-ex holdout |

### 3.3 Key rules for trace data

- **Always** mask the prompt + tool-results from the loss — score only assistant turns.
- **Preserve tool-calls** as structured turns; do not flatten to plain text.
- Pick `max_seq` to your VRAM: full sessions (76K) need full GPUs; LoRA on MIG → 4–8K + filter.
- `train_on_responses_only` (TRL/Unsloth) handles masking when fed OpenAI `messages[]`.

---

## 4. Our deliverable & conventions

- Clean **each** source (A/B/C/D) → its **own standalone PRIVATE dataset** in
  `ermiaazarkhalili/…` (create with `private=True`). Reusable for future training.
- Preserve license per source: B/C agpl-3.0, D mit, A apache-2.0. Derived sets carry
  upstream license terms.
- Then train the small-model fleet (Granite/VibeThinker/FastContext/Qwen3.5/Gemma4) on
  each clean dataset **independently** (not combined), via the §3.1 LoRA recipe + our
  proven smoke→train→GGUF→card pipeline.

---

## 5. Environment (DRAC)

- **Data wrangling (CPU) — READY:** crosscoders `.venv`
  (`/project/6014832/ermia/transformer-circuits/transformer-crosscoders/.venv`, a **uv** venv).
  Installed 2026-06-25: pyarrow 24.0.0, polars 1.42.0, pandas 3.0.3, datasets 4.8.5,
  **teich 0.2.8**, hf_hub 1.14.0. (DRAC NOTE: these projects pip/uv-install pyarrow from PyPI
  into a uv venv — they do NOT rely on `module load arrow`. To add pkgs:
  `PATH=$HOME/.local/bin:$PATH; VIRTUAL_ENV=<.venv> uv pip install <pkg>`.)
  ALWAYS pin `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before numpy import
  (login-node RLIMIT_NPROC). Run the 2M scan as a **CPU SLURM job**, not on login.
  **teich CLI commands:** `extract` (local sessions), `anonymize` (PII), `convert`
  (traces→training JSONL), `generate`, `init`, `studio`. teich covers extract+anonymize+convert
  natively — may not need a hand-written clean_fable5.py.
- **Training (GPU):** `hf_unsloth` venv (`module load StdEnv/2023 gcc arrow python/3.11.5
  cuda/12.6`), MIG `nvidia_h100_80gb_hbm3_3g.40gb`, partition `gpubase_bygpu_b2`,
  `--exclude=fc10713`. Same as the existing fleet pipeline.
