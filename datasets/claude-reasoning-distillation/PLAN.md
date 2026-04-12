# Plan: claude-reasoning-distillation Dataset

## Context

Build a clean, deduplicated, model-agnostic reasoning distillation dataset by merging 5 source datasets of Claude Opus/Sonnet reasoning traces. Goal: a single high-quality dataset that directly trains Qwen3.5, LFM2.5, and Gemma4 models via SFT distillation or GRPO reinforcement.

Researched using: chain-of-thoughts (15-step sequential reasoning), flow-of-thoughts (multi-path decomposition), pro-plan orchestration, and two parallel Explore agents on source schemas + training code.

---

## Source Datasets

| Source | Size | Model | Format | Schema Type | License |
|--------|------|-------|--------|-------------|---------|
| Roman1111111/claude-opus-4.6-10000x | ~10,000 | Claude Opus 4.6 | JSON | `conversations` column, `<think>` embedded in assistant content | MIT |
| nohurry/Opus-4.6-Reasoning-3000x-filtered | ~3,000 | Claude Opus 4.6 | JSON | Same as Roman — `conversations` column, `<think>` embedded | Apache 2.0 |
| TeichAI/Claude-Sonnet-4.6-Reasoning-1100x | 1,096 | Claude Sonnet 4.6 | Parquet | Separate `thinking`/`reasoning_content` + `content` columns | Apache 2.0 |
| TeichAI/claude-4.5-opus-high-reasoning-250x | 250 | Claude Opus 4.5 | JSON | Unknown until runtime — flexible normalizer handles variants | — |
| TeichAI/claude-sonnet-4.5-high-reasoning-250x | 250 | Claude Sonnet 4.5 | JSON | Unknown until runtime — flexible normalizer handles variants | Apache 2.0 |

**Total raw: ~14,596 → expected ~12,500–13,500 after dedup**

---

## Canonical Target Schema (model-agnostic)

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "thinking": "..." }
  ],
  "source": "roman-opus-4.6 | nohurry-opus-4.6 | teichai-sonnet-4.6 | teichai-opus-4.5 | teichai-sonnet-4.5",
  "domain": "coding | math | science | logic | humanities | general",
  "model": "claude-opus-4.6 | claude-sonnet-4.6 | claude-opus-4.5 | claude-sonnet-4.5"
}
```

**Why `thinking` (not `reasoning_content`):**
- Model-agnostic naming; training-time adapters handle per-model conversion (see below)
- Qwen3.5: rename `thinking` → `reasoning_content` before `apply_chat_template(enable_thinking=True)`
- LFM2.5: embed `<think>{thinking}</think>\n` into content (no `enable_thinking` param)
- Gemma4: uses `<|channel>thought\n...<channel|>` tokens via apply_chat_template + `<|think|>` system prompt

**Two HF Hub configs:**
1. **`sft`** — Full `messages` + `thinking` field → SFT/distillation training
2. **`grpo`** — `prompt` column (list of all non-assistant turns: system + user) → GRPOTrainer

---

## File Structure

```
/project/6014832/ermia/HF-TRL/
├── datasets/claude-reasoning-distillation/
│   ├── build_dataset.py       # Main pipeline (~450 lines, argparse CLI)
│   ├── requirements.txt       # datasets, datasketch, tqdm
│   ├── PLAN.md                # This file
│   └── TRAINING_GUIDE.md      # Per-model training adapters
└── jobs/
    └── build-claude-reasoning-distillation.sh   # SLURM submit script
```

---

## Pipeline: build_dataset.py

### Phase 0: Argparse CLI
```python
--output_repo     ermiaazarkhalili/claude-reasoning-distillation
--dedup_threshold 0.85
--min_answer_len  50
--test_size       0.05
--skip_near_dedup # optional flag to skip MinHash (faster debug runs)
```

### Phase 1: Load & Inspect
- Load each source via `datasets.load_dataset()` with `HF_TOKEN` env var
- Print column names + 2 sample rows per source (fast-fail if wrong schema)
- Roman/nohurry: `conversations` column with `role`/`content` dicts

### Phase 2: Normalize (per-source functions, all output identical schema)

**`extract_thinking(text) → (thinking: str, answer: str)`**
```python
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def extract_thinking(text):
    if "<think>" in text and "</think>" not in text:
        return None, None  # Truncated — caller drops sample
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text.strip()
```

**`normalize_conversations(sample)` — for Roman & nohurry**
- Parse `conversations` column (handle role aliases: `human`/`gpt` in addition to `user`/`assistant`)
- Extract `<think>` from LAST assistant turn only (not intermediate turns)
- Return `None` to drop truncated samples (`<think>` without `</think>`)
- Preserve system turn if present at index 0

**`normalize_flexible(sample, source_name, model_name)` — for TeichAI datasets**
- Flexible column-name fallbacks (defensive programming for unknown 4.5 schemas):
  ```python
  user = sample.get("prompt") or sample.get("question") or sample.get("instruction") or sample.get("input")
  thinking = sample.get("thinking") or sample.get("reasoning_content") or sample.get("reasoning")
  answer = sample.get("content") or sample.get("response") or sample.get("output") or sample.get("answer")
  ```
- If `"conversations"` column found, delegate to `normalize_conversations()`
- Drop sample if `user` or `answer` is None

### Phase 3: Quality Filter (before dedup — faster)
```
drop if: answer is empty or None
drop if: len(answer) < min_answer_len (default 50)
drop if: re.search(r"I (cannot|can't|am unable|don't feel comfortable)", answer, re.I)
```

### Phase 4: Deduplicate (ordered pipeline)

**Step 1 — Exact dedup (SHA256 on stripped user content):**
```python
h = hashlib.sha256(user_content.strip().encode("utf-8")).hexdigest()
```
- Hash raw (NOT lowercased) to preserve code/math case sensitivity

**Step 2 — Near-dedup (MinHash LSH, datasketch):**
- 128 permutations, threshold=0.85
- 3-word shingles on lowercased content
- Skip MinHash for prompts < 20 words (rely on exact dedup only)
- When near-dup pair found: **keep the sample with more thinking content** (longer reasoning trace = higher value)

### Phase 5: Domain Tagging (ordered priority — coding checked first)
```python
DOMAIN_RULES = [
    ("coding",     re.compile(r"```|\bcode\b|\bfunction\b|\bclass\b|\balgorithm\b|\bimplement\b|\bpython\b|\bjavascript\b|\bsql\b", re.I)),
    ("math",       re.compile(r"\b(calculat|solv|proof|theorem|equation|integral|derivative|matrix|algebra|geometry|probabilit)\w*\b|\$.*?\$|\\frac|\\sum|\\int|\d+\s*[+\-*/^]\s*\d+", re.I)),
    ("science",    re.compile(r"\b(physics|chemistry|biology|molecule|atom|force|energy|reaction|evolution|genetics|neuron)\w*\b", re.I)),
    ("logic",      re.compile(r"\b(logic|reasoning|argument|fallacy|deductive|inductive|syllogism|valid|sound|premise)\w*\b", re.I)),
    ("humanities", re.compile(r"\b(histor|philosoph|ethic|moral|econom|polic|society|culture|literature|art|music|religion)\w*\b", re.I)),
]
```
Fallback: `"general"`

### Phase 6: Split & Push
- 95/5 train/test stratified split by domain
- Fallback: random split if any domain stratum has < 2 samples
- Build GRPO config: `prompt` = all non-assistant messages (preserves system turns)
- Run verification assertions before pushing
- Push both configs:
  ```python
  sft_dd.push_to_hub("ermiaazarkhalili/claude-reasoning-distillation", config_name="sft")
  grpo_dd.push_to_hub("ermiaazarkhalili/claude-reasoning-distillation", config_name="grpo")
  ```
- Auto-generate dataset card with: total count, per-source breakdown, domain distribution, % with thinking, training-time adapter note

---

## SLURM Job: jobs/build-claude-reasoning-distillation.sh

```bash
#SBATCH --job-name=build-claude-distillation
#SBATCH --account=def-maxwl_gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8     # parallel dataset.map()
#SBATCH --mem=32G              # 14k large-text samples
# No --gres= needed: pure CPU data pipeline
```

---

## Model Compatibility + Training-Time Adapters

| Model Family | Training | Config | Adapter required |
|---|---|---|---|
| Qwen3.5-0.6B/1.7B/7B | SFT distillation | `sft` | Rename `thinking` → `reasoning_content` in assistant dict; then `apply_chat_template(enable_thinking=True)` |
| Qwen3.5 | GRPO | `grpo` | None — prompt only |
| LFM2.5-1.2B (Instruct/Thinking) | SFT distillation | `sft` | Embed `<think>{thinking}</think>\n` into content before tokenizing; standard ChatML template |
| LFM2.5 | GRPO | `grpo` | None |
| Gemma4-E2B/E4B/26B/31B | SFT distillation | `sft` | Add `<|think|>` system prompt; custom data collator for `mm_token_type_ids` (required even text-only) |
| Gemma4 | GRPO | `grpo` | None |

---

## Bugs Prevented by CoT/FoT Analysis

1. **Truncated `<think>` blocks**: Drop samples with `<think>` but no `</think>` (not silently kept as empty)
2. **MinHash on short prompts**: Skip MinHash for < 20 words (unreliable; exact dedup already covers)
3. **Near-dup selection**: Keep richer sample (longer thinking), not random first-seen
4. **GRPO prompt includes system turn**: Not user-only — preserves full non-assistant context
5. **Exact hash not lowercased**: Preserves code/math case sensitivity
6. **Role aliases**: Handle `human`/`gpt` in addition to `user`/`assistant`
7. **Multi-turn `<think>` extraction**: Only from LAST assistant turn, not all turns
8. **Stratified split fallback**: Random split if any stratum < 2 samples

---

## License

Apache 2.0 (most restrictive of source licenses; MIT is compatible)

---

## Verification Suite

```python
from datasets import load_dataset

# SFT config
ds = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "sft")
train = ds["train"]

# Schema checks
assert "messages" in train.column_names
assert "source" in train.column_names
assert "domain" in train.column_names
assert "model" in train.column_names

# Sample integrity (check 200 random samples)
for i in range(min(200, len(train))):
    s = train[i]
    roles = [m["role"] for m in s["messages"]]
    assert "user" in roles and "assistant" in roles
    asst = s["messages"][-1]
    assert asst["role"] == "assistant"
    assert len(asst["content"]) >= 50

# Exact dedup check
user_prompts = [m["content"] for s in train for m in s["messages"] if m["role"] == "user"]
assert len(user_prompts) == len(set(user_prompts)), "Exact duplicates found!"

print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")

# GRPO config
ds_g = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "grpo")
assert "prompt" in ds_g["train"].column_names
s0 = ds_g["train"][0]
assert isinstance(s0["prompt"], list)
assert s0["prompt"][-1]["role"] == "user"
assert all(m["role"] != "assistant" for m in s0["prompt"])
print("All assertions passed!")
```

---

## Effort Estimate

| Task | Human | AI-assisted | Compression |
|------|-------|-------------|-------------|
| Write build_dataset.py | 1 day | ~35 min | ~20x |
| Write SLURM job | 30 min | ~5 min | ~6x |
| Run pipeline + verify | 30 min | ~10 min | ~3x |
| Push to HF Hub | 15 min | ~5 min | ~3x |
| **Total** | **~2 days** | **~55 min** | **~35x** |
