# Training Guide: claude-reasoning-distillation

Dataset: `ermiaazarkhalili/claude-reasoning-distillation`

This dataset has two configs:
- **`sft`** — Full messages with separate `thinking` field → SFT / distillation training
- **`grpo`** — Prompt-only (system + user turns) → GRPO reinforcement training

---

## Schema

### SFT config

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "thinking": "..."}
  ],
  "source": "roman-opus-4.6 | nohurry-opus-4.6 | teichai-sonnet-4.6 | teichai-opus-4.5 | teichai-sonnet-4.5",
  "domain": "coding | math | science | logic | humanities | general",
  "model": "claude-opus-4.6 | claude-sonnet-4.6 | claude-opus-4.5 | claude-sonnet-4.5"
}
```

### GRPO config

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "source": "...",
  "domain": "...",
  "model": "..."
}
```

---

## Model-Specific Training Adapters

The `thinking` field in the SFT config is stored in a model-agnostic way.
Each model family needs a small training-time adapter before tokenizing.

### Qwen3.5 (0.6B / 1.7B / 7B / 14B / 32B)

Qwen3.5's chat template reads `reasoning_content` (not `thinking`) in the assistant message dict.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "sft")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def adapt_for_qwen3(sample):
    messages = []
    for msg in sample["messages"]:
        m = dict(msg)
        if m["role"] == "assistant" and m.get("thinking"):
            m["reasoning_content"] = m.pop("thinking")
        elif m["role"] == "assistant":
            m.pop("thinking", None)
        messages.append(m)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return {"text": text}

ds_processed = ds.map(adapt_for_qwen3)
# Then pass ds_processed to TRL SFTTrainer with dataset_text_field="text"
```

**GRPO training:**
```python
ds_grpo = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "grpo")
# Pass ds_grpo["train"]["prompt"] directly to GRPOTrainer
# The `prompt` column is already in the list-of-messages format TRL expects
```

---

### LFM2.5 (1.2B-Instruct / 1.2B-Thinking)

LFM2.5 uses ChatML format. Thinking must be embedded as `<think>...</think>` at the
start of the assistant content. There is no `enable_thinking` parameter.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "sft")

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")

def adapt_for_lfm2(sample):
    messages = []
    for msg in sample["messages"]:
        m = dict(msg)
        if m["role"] == "assistant":
            thinking = m.pop("thinking", None)
            if thinking:
                m["content"] = f"<think>{thinking}</think>\n{m['content']}"
        messages.append(m)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

ds_processed = ds.map(adapt_for_lfm2)
```

---

### Gemma4 (E2B / E4B / 26B-A4B / 31B)

Gemma4 is multimodal-native and requires `mm_token_type_ids` in the batch even for
text-only training. Thinking is controlled via `<|think|>` in the system prompt.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
import torch

ds = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "sft")

processor = AutoProcessor.from_pretrained("google/gemma-4-9b-it")

GEMMA_THINK_SYSTEM = "<|think|>"

def adapt_for_gemma4(sample):
    messages = list(sample["messages"])
    # Ensure system prompt with think trigger
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": GEMMA_THINK_SYSTEM}] + messages
    elif GEMMA_THINK_SYSTEM not in messages[0]["content"]:
        messages[0] = {"role": "system", "content": GEMMA_THINK_SYSTEM + "\n" + messages[0]["content"]}

    # Remove thinking field (Gemma4 handles thinking via its own template)
    clean_messages = []
    for msg in messages:
        m = dict(msg)
        m.pop("thinking", None)
        clean_messages.append(m)

    text = processor.apply_chat_template(
        clean_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

ds_processed = ds.map(adapt_for_gemma4)
```

**IMPORTANT**: Gemma4 SFT training requires a custom data collator that pads
`mm_token_type_ids`. See the Gemma4 fine-tuning documentation for the full
`DataCollatorForGemma4` implementation.

---

## Quick Load

```python
from datasets import load_dataset

# For SFT training
sft = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "sft")
print(sft["train"][0])

# For GRPO training
grpo = load_dataset("ermiaazarkhalili/claude-reasoning-distillation", "grpo")
print(grpo["train"][0])

# Filter by domain
math_only = sft["train"].filter(lambda x: x["domain"] == "math")

# Filter by source model
opus_only = sft["train"].filter(lambda x: "opus-4.6" in x["model"])

# Filter samples WITH thinking traces only
with_thinking = sft["train"].filter(
    lambda x: x["messages"][-1].get("thinking") is not None
)
print(f"Samples with thinking: {len(with_thinking)} / {len(sft['train'])}")
```

---

## Dataset Statistics

*(Populated after build_dataset.py completes)*

| Metric | Value |
|--------|-------|
| Total samples (train) | — |
| Total samples (test) | — |
| % with thinking traces | — |
| Domain: coding | — |
| Domain: math | — |
| Domain: science | — |
| Domain: logic | — |
| Domain: humanities | — |
| Domain: general | — |
| Source: roman-opus-4.6 | — |
| Source: nohurry-opus-4.6 | — |
| Source: teichai-sonnet-4.6 | — |
| Source: teichai-opus-4.5 | — |
| Source: teichai-sonnet-4.5 | — |
