#!/usr/bin/env python3
"""
build_dataset.py — Claude Reasoning Distillation Dataset Pipeline

Merges 5 source datasets of Claude Opus/Sonnet reasoning traces into a single
deduplicated, model-agnostic dataset for SFT distillation and GRPO training.

Output: ermiaazarkhalili/claude-reasoning-distillation (two configs: sft, grpo)

Usage:
    python build_dataset.py [--output_repo REPO] [--dedup_threshold 0.85]
                            [--min_answer_len 50] [--test_size 0.05]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

REFUSAL_RE = re.compile(
    r"\bI (cannot|can't|am unable|don't feel comfortable|must (decline|refuse))\b",
    re.I,
)

DOMAIN_RULES = [
    ("coding", re.compile(
        r"```|\bcode\b|\bfunction\b|\bclass\b|\balgorithm\b|\bimplement\b"
        r"|\bpython\b|\bjavascript\b|\bsql\b|\bprogramm\w*\b",
        re.I,
    )),
    ("math", re.compile(
        r"\b(calculat|solv|proof|theorem|equation|integral|derivative|matrix"
        r"|algebra|geometry|probabilit|statistic|combinatori)\w*\b"
        r"|\$.*?\$|\\frac|\\sum|\\int|\d+\s*[+\-*/^]\s*\d+",
        re.I,
    )),
    ("science", re.compile(
        r"\b(physics|chemistry|biology|molecule|atom|force|energy|reaction"
        r"|evolution|genetics|neuron|quantum|thermodynam)\w*\b",
        re.I,
    )),
    ("logic", re.compile(
        r"\b(logic|reasoning|argument|fallacy|deductive|inductive|syllogism"
        r"|valid|sound|premise|paradox)\w*\b",
        re.I,
    )),
    ("humanities", re.compile(
        r"\b(histor|philosoph|ethic|moral|econom|polic|society|culture"
        r"|literature|art|music|religion|psychology)\w*\b",
        re.I,
    )),
]

SOURCE_CONFIGS = [
    {
        "repo": "Roman1111111/claude-opus-4.6-10000x",
        "source": "roman-opus-4.6",
        "model": "claude-opus-4.6",
        # JSONL file: `messages` column stored as JSON STRING, `reasoning` field (not `thinking`)
        # Must load via hf_hub_download + load_dataset("json") to bypass broken dataset card
        "schema": "roman",
    },
    {
        "repo": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "source": "nohurry-opus-4.6",
        "model": "claude-opus-4.6",
        # Flat schema: problem / thinking / solution columns
        "schema": "flexible",
    },
    {
        "repo": "TeichAI/Claude-Sonnet-4.6-Reasoning-1100x",
        "source": "teichai-sonnet-4.6",
        "model": "claude-sonnet-4.6",
        "schema": "flexible",
    },
    {
        "repo": "TeichAI/claude-4.5-opus-high-reasoning-250x",
        "source": "teichai-opus-4.5",
        "model": "claude-opus-4.5",
        "schema": "flexible",
    },
    {
        "repo": "TeichAI/claude-sonnet-4.5-high-reasoning-250x",
        "source": "teichai-sonnet-4.5",
        "model": "claude-sonnet-4.5",
        "schema": "flexible",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def extract_thinking(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract <think> block from text.
    Returns (thinking, answer).
    Returns (None, None) if text is truncated (has <think> but no </think>).
    Returns (None, text) if no <think> block found.
    """
    if "<think>" in text and "</think>" not in text:
        return None, None  # Truncated — caller must drop sample
    m = THINK_RE.search(text)
    if m:
        thinking = m.group(1).strip()
        answer = text[m.end():].strip()
        return (thinking if thinking else None), answer
    return None, text.strip()


def get_user_content(messages: list[dict]) -> Optional[str]:
    """Return the first user turn content, or None."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def make_hash(text: str) -> str:
    """SHA256 hash of stripped text (NOT lowercased to preserve code/math case)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def tag_domain(user_content: str) -> str:
    """Heuristic domain tagging. Order matters — coding checked first."""
    for domain, pattern in DOMAIN_RULES:
        if pattern.search(user_content):
            return domain
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Normalizers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_messages_with_thinking_field(sample: dict, source: str, model: str) -> Optional[dict]:
    """
    Normalize Roman1111111/nohurry-style datasets.
    These use a `messages` column where the assistant message already has a
    separate `thinking` field (NOT embedded <think> tags in content).

    Input schema:
      {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "thinking": "...", "content": "..."}
      ]}
    """
    msgs = sample.get("messages", [])
    if not msgs:
        return None

    ROLE_MAP = {"human": "user", "user": "user", "system": "system",
                "assistant": "assistant", "gpt": "assistant", "bot": "assistant"}

    normalized = []
    for turn in msgs:
        role_raw = turn.get("role") or turn.get("from") or ""
        role = ROLE_MAP.get(role_raw.lower(), role_raw.lower())
        content = turn.get("content") or turn.get("value") or ""
        thinking = turn.get("thinking") or turn.get("reasoning_content") or None

        if role == "assistant":
            # If content itself embeds <think> tags (fallback for mixed-format sources)
            if "<think>" in content:
                if "</think>" not in content:
                    return None  # Truncated — drop
                extracted_thinking, extracted_answer = extract_thinking(content)
                if extracted_thinking:
                    thinking = extracted_thinking
                content = extracted_answer or content
            normalized.append({
                "role": "assistant",
                "content": content.strip(),
                "thinking": thinking.strip() if thinking else None,
            })
        else:
            normalized.append({"role": role, "content": content})

    roles = {m["role"] for m in normalized}
    if "user" not in roles or "assistant" not in roles:
        return None

    return {
        "messages": normalized,
        "source": source,
        "model": model,
        "domain": "general",
    }


def normalize_roman(sample: dict, source: str, model: str) -> Optional[dict]:
    """
    Normalize Roman1111111/claude-opus-4.6-10000x samples.
    Schema: messages column is a JSON STRING (not parsed list).
    Each message has: role, content, reasoning (not thinking).
    reasoning field may be None for non-assistant turns.
    """
    import ast

    msgs_raw = sample.get("messages", "")
    if not msgs_raw:
        return None

    # Parse the JSON string — it uses single quotes (Python repr), so use ast.literal_eval
    try:
        if isinstance(msgs_raw, str):
            msgs = ast.literal_eval(msgs_raw)
        else:
            msgs = msgs_raw  # already parsed (shouldn't happen but guard)
    except Exception:
        return None

    if not isinstance(msgs, list) or not msgs:
        return None

    ROLE_MAP = {"human": "user", "user": "user", "system": "system",
                "assistant": "assistant", "gpt": "assistant", "bot": "assistant"}

    normalized = []
    for turn in msgs:
        role_raw = turn.get("role") or turn.get("from") or ""
        role = ROLE_MAP.get(role_raw.lower(), role_raw.lower())
        content = turn.get("content") or turn.get("value") or ""
        # Roman uses `reasoning` field (not `thinking`)
        reasoning = turn.get("reasoning") or turn.get("thinking") or None

        if role == "assistant":
            # Also check for embedded <think> tags as fallback
            if not reasoning and "<think>" in content:
                if "</think>" not in content:
                    return None  # Truncated
                extracted_thinking, extracted_answer = extract_thinking(content)
                reasoning = extracted_thinking
                content = extracted_answer or content
            normalized.append({
                "role": "assistant",
                "content": content.strip() if content else "",
                "thinking": reasoning.strip() if reasoning else None,
            })
        else:
            if content or role in ("user", "system"):
                normalized.append({"role": role, "content": content})

    roles = {m["role"] for m in normalized}
    if "user" not in roles or "assistant" not in roles:
        return None

    return {
        "messages": normalized,
        "source": source,
        "model": model,
        "domain": "general",
    }


def normalize_conversations(sample: dict, source: str, model: str) -> Optional[dict]:
    """
    Normalize a sample with a `conversations` column.
    Handles role aliases: human/gpt → user/assistant.
    Extracts <think> from the LAST assistant turn only (for embedded-tag format).
    """
    convs = sample.get("conversations", [])
    if not convs:
        return None

    ROLE_MAP = {
        "human": "user",
        "user": "user",
        "system": "system",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
    }

    messages = []
    for turn in convs:
        role_raw = turn.get("role") or turn.get("from") or ""
        role = ROLE_MAP.get(role_raw.lower(), role_raw.lower())
        content = turn.get("content") or turn.get("value") or ""
        messages.append({"role": role, "content": content})

    # Extract <think> from LAST assistant turn only
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            thinking, answer = extract_thinking(messages[i]["content"])
            if thinking is None and answer is None:
                return None  # Truncated sample — drop
            messages[i] = {
                "role": "assistant",
                "content": answer or "",
                "thinking": thinking,
            }
            break

    # Validate: must have at least one user and one assistant turn
    roles = {m["role"] for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return None

    return {
        "messages": messages,
        "source": source,
        "model": model,
        "domain": "general",  # tagged later
    }


def normalize_flexible(sample: dict, source: str, model: str) -> Optional[dict]:
    """
    Flexible normalizer for TeichAI-style datasets with unknown column names.
    Tries multiple column name variants defensively.
    Falls back to conversations normalizer if `conversations` column present.
    """
    # If it has a conversations column, reuse conversations normalizer
    if "conversations" in sample:
        return normalize_conversations(sample, source, model)
    if "messages" in sample:
        # Delegate to messages normalizer — handles both `thinking` field and embedded <think> tags
        return normalize_messages_with_thinking_field(sample, source, model)

    # Flat column format — try multiple name variants
    user = (
        sample.get("prompt") or sample.get("question") or
        sample.get("instruction") or sample.get("input") or
        sample.get("problem") or ""
    )
    thinking = (
        sample.get("thinking") or sample.get("reasoning_content") or
        sample.get("reasoning") or sample.get("think") or None
    )
    answer = (
        sample.get("content") or sample.get("response") or
        sample.get("output") or sample.get("answer") or
        sample.get("solution") or ""
    )
    system = sample.get("system") or sample.get("system_prompt") or None

    if not user or not answer:
        return None

    # If answer embeds <think> tags, extract them (overrides separate thinking column)
    if "<think>" in answer:
        extracted_thinking, extracted_answer = extract_thinking(answer)
        if extracted_thinking is None and extracted_answer is None:
            return None  # Truncated
        if extracted_thinking:
            thinking = extracted_thinking
        answer = extracted_answer or answer

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user.strip()})
    messages.append({
        "role": "assistant",
        "content": answer.strip(),
        "thinking": thinking.strip() if thinking else None,
    })

    return {
        "messages": messages,
        "source": source,
        "model": model,
        "domain": "general",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Load & Normalize
# ─────────────────────────────────────────────────────────────────────────────

def load_and_normalize() -> list[dict]:
    """Load all source datasets, normalize to canonical schema."""
    all_samples: list[dict] = []

    for cfg in SOURCE_CONFIGS:
        repo = cfg["repo"]
        source = cfg["source"]
        model = cfg["model"]
        schema = cfg["schema"]

        log(f"Loading {repo} ...")
        raw = None

        # Roman schema: load directly via hf_hub_download to bypass broken dataset card
        if schema == "roman":
            try:
                from huggingface_hub import hf_hub_download
                log(f"  Roman schema: downloading opus46_final.jsonl directly ...")
                jsonl_path = hf_hub_download(
                    repo_id=repo,
                    filename="opus46_final.jsonl",
                    repo_type="dataset",
                )
                raw = load_dataset("json", data_files=jsonl_path, split="train")
                log(f"  Roman direct load: {len(raw)} samples")
            except Exception as e:
                log(f"  Roman direct load failed ({type(e).__name__}): {str(e)[:120]}")
        else:
            # Try standard load first
            for attempt_kwargs in [
                {"split": "train", "trust_remote_code": True},
                {"split": "train", "trust_remote_code": True, "download_mode": "reuse_cache_if_exists"},
            ]:
                try:
                    raw = load_dataset(repo, **attempt_kwargs)
                    break
                except Exception as e:
                    log(f"  Attempt failed ({type(e).__name__}): {str(e)[:120]}")

            # Fallback: snapshot download + manual JSON/parquet load
            if raw is None:
                try:
                    log(f"  Falling back to snapshot_download for {repo} ...")
                    from huggingface_hub import snapshot_download
                    import glob
                    local_dir = f"/tmp/hf_fallback_{source}"
                    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=local_dir)
                    # Try parquet files first, then JSONL/JSON
                    parquet_files = glob.glob(f"{local_dir}/**/*.parquet", recursive=True)
                    jsonl_files = glob.glob(f"{local_dir}/**/*.jsonl", recursive=True) + \
                                  glob.glob(f"{local_dir}/**/*.json", recursive=True)
                    if parquet_files:
                        raw = load_dataset("parquet", data_files=parquet_files, split="train")
                    elif jsonl_files:
                        raw = load_dataset("json", data_files=jsonl_files, split="train")
                    else:
                        log(f"  No loadable files found in snapshot. Skipping.")
                except Exception as e2:
                    log(f"  Fallback also failed: {e2}")

        if raw is None:
            log(f"  Skipping {repo} — could not load.")
            continue

        log(f"  Loaded {len(raw)} samples. Columns: {raw.column_names}")
        # Print 1 sample for schema inspection
        if len(raw) > 0:
            sample0 = raw[0]
            preview = {k: str(v)[:120] for k, v in sample0.items()}
            log(f"  Sample[0] preview: {preview}")

        if schema == "roman":
            normalizer = normalize_roman
        elif schema == "messages":
            normalizer = normalize_messages_with_thinking_field
        elif schema == "conversations":
            normalizer = normalize_conversations
        else:
            normalizer = normalize_flexible

        before = len(all_samples)
        dropped = 0
        for sample in tqdm(raw, desc=f"  Normalizing {source}", leave=False):
            result = normalizer(sample, source, model)
            if result is not None:
                all_samples.append(result)
            else:
                dropped += 1

        added = len(all_samples) - before
        log(f"  → {added} normalized, {dropped} dropped")

    log(f"Total after normalization: {len(all_samples)}")
    return all_samples


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Quality Filter
# ─────────────────────────────────────────────────────────────────────────────

def quality_filter(samples: list[dict], min_answer_len: int) -> list[dict]:
    """Drop empty, short, or refusal answers."""
    kept = []
    dropped = 0
    for s in samples:
        answer = s["messages"][-1].get("content", "")
        if not answer:
            dropped += 1
            continue
        if len(answer) < min_answer_len:
            dropped += 1
            continue
        if REFUSAL_RE.search(answer):
            dropped += 1
            continue
        kept.append(s)
    log(f"Quality filter: kept {len(kept)}, dropped {dropped}")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def exact_dedup(samples: list[dict]) -> list[dict]:
    """SHA256 hash of stripped user content → drop exact duplicates."""
    seen: set[str] = set()
    kept = []
    dropped = 0
    for s in samples:
        user = get_user_content(s["messages"]) or ""
        h = make_hash(user)
        if h not in seen:
            seen.add(h)
            kept.append(s)
        else:
            dropped += 1
    log(f"Exact dedup: kept {len(kept)}, dropped {dropped}")
    return kept


def near_dedup(samples: list[dict], threshold: float = 0.85, num_perm: int = 128) -> list[dict]:
    """MinHash LSH near-dedup on user content. Keeps richer (longer thinking) sample."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log("datasketch not installed — skipping near-dedup. Install with: pip install datasketch")
        return samples

    MIN_WORDS = 20  # Skip MinHash for very short prompts (unreliable)

    log(f"Near-dedup: building MinHash index (threshold={threshold}, num_perm={num_perm}) ...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, MinHash] = {}

    for i, s in enumerate(tqdm(samples, desc="  Building MinHash", leave=False)):
        user = get_user_content(s["messages"]) or ""
        words = user.lower().split()
        if len(words) < MIN_WORDS:
            continue  # Too short — already exact-deduped
        m = MinHash(num_perm=num_perm)
        for j in range(max(1, len(words) - 2)):
            shingle = " ".join(words[j:j + 3])
            m.update(shingle.encode("utf-8"))
        minhashes[i] = m

    duplicates: set[int] = set()
    for i, m in tqdm(minhashes.items(), desc="  LSH query", leave=False):
        if i in duplicates:
            continue
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass
        neighbors = lsh.query(m)
        for nb in neighbors:
            j = int(nb)
            if j == i or j in duplicates:
                continue
            # Keep the sample with more thinking content (longer reasoning = more valuable)
            thinking_i = len(samples[i]["messages"][-1].get("thinking") or "")
            thinking_j = len(samples[j]["messages"][-1].get("thinking") or "")
            if thinking_j > thinking_i:
                duplicates.add(i)
            else:
                duplicates.add(j)

    kept = [s for i, s in enumerate(samples) if i not in duplicates]
    log(f"Near-dedup: kept {len(kept)}, dropped {len(duplicates)}")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Domain Tagging
# ─────────────────────────────────────────────────────────────────────────────

def add_domain_tags(samples: list[dict]) -> list[dict]:
    """Tag each sample with a domain based on user content heuristics."""
    for s in samples:
        user = get_user_content(s["messages"]) or ""
        s["domain"] = tag_domain(user)
    domain_counts = Counter(s["domain"] for s in samples)
    log(f"Domain distribution: {dict(domain_counts)}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Build Configs & Push
# ─────────────────────────────────────────────────────────────────────────────

def build_grpo_prompt(messages: list[dict]) -> list[dict]:
    """Return all non-assistant messages (system + user turns) for GRPO."""
    return [m for m in messages if m["role"] != "assistant"]


def split_dataset(samples: list[dict], test_size: float) -> tuple[Dataset, Dataset]:
    """Stratified 95/5 train/test split by domain. Falls back to random if strata too small."""
    import random

    domain_groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        d = s["domain"]
        domain_groups.setdefault(d, []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []

    # Check if stratified split is feasible (all strata need >= 2 samples)
    can_stratify = all(len(v) >= 2 for v in domain_groups.values())

    if can_stratify:
        for idxs in domain_groups.values():
            random.shuffle(idxs)
            n_test = max(1, int(len(idxs) * test_size))
            test_idx.extend(idxs[:n_test])
            train_idx.extend(idxs[n_test:])
    else:
        log("Warning: some domain strata have < 2 samples — using random split instead")
        all_idx = list(range(len(samples)))
        random.shuffle(all_idx)
        n_test = max(1, int(len(all_idx) * test_size))
        test_idx = all_idx[:n_test]
        train_idx = all_idx[n_test:]

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    log(f"Split: {len(train_samples)} train, {len(test_samples)} test")
    return Dataset.from_list(train_samples), Dataset.from_list(test_samples)


def build_grpo_dataset(ds: Dataset) -> Dataset:
    """Convert SFT dataset to GRPO format (prompt-only, no assistant turn)."""
    grpo_rows = []
    for s in ds:
        grpo_rows.append({
            "prompt": build_grpo_prompt(s["messages"]),
            "source": s["source"],
            "domain": s["domain"],
            "model": s["model"],
        })
    return Dataset.from_list(grpo_rows)


def generate_dataset_card(samples: list[dict], output_repo: str) -> str:
    total = len(samples)
    by_source = Counter(s["source"] for s in samples)
    by_domain = Counter(s["domain"] for s in samples)
    by_model = Counter(s["model"] for s in samples)
    n_thinking = sum(
        1 for s in samples
        if s["messages"][-1].get("thinking") is not None
    )
    pct_thinking = 100 * n_thinking / total if total > 0 else 0

    source_rows = "\n".join(
        f"| {k} | {v} | {100*v/total:.1f}% |" for k, v in sorted(by_source.items(), key=lambda x: -x[1])
    )
    domain_rows = "\n".join(
        f"| {k} | {v} | {100*v/total:.1f}% |" for k, v in sorted(by_domain.items(), key=lambda x: -x[1])
    )

    return f"""---
license: apache-2.0
tags:
- reasoning
- distillation
- claude
- sft
- grpo
- qwen3
- gemma4
- lfm2
size_categories:
- 10K<n<100K
task_categories:
- text-generation
configs:
- config_name: sft
  data_files:
  - split: train
    path: sft/train-*.parquet
  - split: test
    path: sft/test-*.parquet
- config_name: grpo
  data_files:
  - split: train
    path: grpo/train-*.parquet
  - split: test
    path: grpo/test-*.parquet
---

# claude-reasoning-distillation

Merged, deduplicated, model-agnostic reasoning distillation dataset built from Claude Opus 4.6, Sonnet 4.6, Opus 4.5, and Sonnet 4.5 reasoning traces.

**{total:,} total samples** | **{pct_thinking:.0f}% with thinking traces**
Built: {datetime.now().strftime("%Y-%m-%d")}

## Configs

| Config | Description | Use case |
|--------|-------------|----------|
| `sft` | Full messages with separate `thinking` field | SFT distillation |
| `grpo` | Prompt-only (system + user turns) | GRPO reinforcement training |

## Source Breakdown

| Source | Count | % |
|--------|-------|---|
{source_rows}

## Domain Distribution

| Domain | Count | % |
|--------|-------|---|
{domain_rows}

## Schema (SFT config)

```json
{{
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "...", "thinking": "..."}}
  ],
  "source": "roman-opus-4.6 | nohurry-opus-4.6 | teichai-sonnet-4.6 | ...",
  "domain": "coding | math | science | logic | humanities | general",
  "model": "claude-opus-4.6 | claude-sonnet-4.6 | ..."
}}
```

The `thinking` field is `null` when absent. See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for per-model training adapters (Qwen3.5, LFM2.5, Gemma4).

## License

Apache 2.0 — most restrictive of source licenses (MIT is compatible).
"""


def push_to_hub(
    train_ds: Dataset,
    test_ds: Dataset,
    output_repo: str,
    samples: list[dict],
) -> None:
    """Push SFT and GRPO configs to HF Hub."""
    log("Building GRPO datasets ...")
    grpo_train = build_grpo_dataset(train_ds)
    grpo_test = build_grpo_dataset(test_ds)

    sft_dd = DatasetDict({"train": train_ds, "test": test_ds})
    grpo_dd = DatasetDict({"train": grpo_train, "test": grpo_test})

    log(f"Pushing SFT config to {output_repo} ...")
    sft_dd.push_to_hub(output_repo, config_name="sft")

    log(f"Pushing GRPO config to {output_repo} ...")
    grpo_dd.push_to_hub(output_repo, config_name="grpo")

    # Upload dataset card
    from huggingface_hub import HfApi
    api = HfApi()
    card = generate_dataset_card(samples, output_repo)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=output_repo,
        repo_type="dataset",
    )
    log(f"Dataset card uploaded.")
    log(f"Done! → https://huggingface.co/datasets/{output_repo}")


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify(sft_dd: DatasetDict, grpo_dd: DatasetDict) -> None:
    """Run assertion suite on the final dataset."""
    log("Running verification ...")
    train = sft_dd["train"]

    # Schema checks
    assert "messages" in train.column_names, "Missing 'messages' column"
    assert "source" in train.column_names, "Missing 'source' column"
    assert "domain" in train.column_names, "Missing 'domain' column"
    assert "model" in train.column_names, "Missing 'model' column"

    # Sample integrity on up to 200 random samples
    n_check = min(200, len(train))
    for i in range(n_check):
        s = train[i]
        roles = [m["role"] for m in s["messages"]]
        assert "user" in roles, f"Sample {i}: missing user turn"
        assert "assistant" in roles, f"Sample {i}: missing assistant turn"
        asst = s["messages"][-1]
        assert asst["role"] == "assistant", f"Sample {i}: last message not assistant"
        assert len(asst["content"]) >= 50, f"Sample {i}: answer too short"

    # Exact dedup check
    user_prompts = [
        m["content"]
        for s in train
        for m in s["messages"]
        if m["role"] == "user"
    ]
    assert len(user_prompts) == len(set(user_prompts)), "Exact duplicates found!"

    # GRPO format check
    grpo_train = grpo_dd["train"]
    assert "prompt" in grpo_train.column_names, "GRPO: missing 'prompt' column"
    s0 = grpo_train[0]
    assert isinstance(s0["prompt"], list), "GRPO: prompt must be a list"
    assert s0["prompt"][-1]["role"] == "user", "GRPO: last prompt turn must be 'user'"
    assert all(m["role"] != "assistant" for m in s0["prompt"]), "GRPO: must have no assistant turn"

    log(f"✓ All assertions passed")
    log(f"  SFT  train: {len(sft_dd['train'])}, test: {len(sft_dd['test'])}")
    log(f"  GRPO train: {len(grpo_dd['train'])}, test: {len(grpo_dd['test'])}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build claude-reasoning-distillation dataset")
    p.add_argument("--output_repo", default="ermiaazarkhalili/claude-reasoning-distillation")
    p.add_argument("--dedup_threshold", type=float, default=0.85,
                   help="MinHash LSH similarity threshold for near-dedup")
    p.add_argument("--min_answer_len", type=int, default=50,
                   help="Minimum assistant answer length in characters")
    p.add_argument("--test_size", type=float, default=0.05,
                   help="Fraction of data for test split")
    p.add_argument("--skip_near_dedup", action="store_true",
                   help="Skip MinHash near-dedup (faster, less thorough)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log("=" * 60)
    log("Claude Reasoning Distillation Dataset Builder")
    log("=" * 60)
    log(f"Output repo: {args.output_repo}")
    log(f"Dedup threshold: {args.dedup_threshold}")
    log(f"Min answer len: {args.min_answer_len}")
    log(f"Test size: {args.test_size}")

    # Phase 1: Load & normalize
    log("\n[Phase 1] Load & Normalize")
    samples = load_and_normalize()

    # Phase 2: Quality filter
    log("\n[Phase 2] Quality Filter")
    samples = quality_filter(samples, args.min_answer_len)

    # Phase 3: Deduplicate
    log("\n[Phase 3] Deduplication")
    samples = exact_dedup(samples)
    if not args.skip_near_dedup:
        samples = near_dedup(samples, threshold=args.dedup_threshold)
    else:
        log("Near-dedup skipped (--skip_near_dedup)")

    # Phase 4: Domain tagging
    log("\n[Phase 4] Domain Tagging")
    samples = add_domain_tags(samples)

    # Phase 5: Split & push
    log("\n[Phase 5] Split & Push")
    train_ds, test_ds = split_dataset(samples, args.test_size)
    sft_dd = DatasetDict({"train": train_ds, "test": test_ds})

    # Verify before pushing
    grpo_train = build_grpo_dataset(train_ds)
    grpo_test = build_grpo_dataset(test_ds)
    grpo_dd = DatasetDict({"train": grpo_train, "test": grpo_test})
    verify(sft_dd, grpo_dd)

    # Push to Hub
    push_to_hub(train_ds, test_ds, args.output_repo, samples)

    log("\n" + "=" * 60)
    log("Pipeline complete!")
    log("=" * 60)
    log(f"Total samples: {len(samples)}")
    log(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")
    pct = 100 * sum(1 for s in samples if s["messages"][-1].get("thinking")) / len(samples)
    log(f"  % with thinking: {pct:.1f}%")
    log(f"  Source breakdown: {dict(Counter(s['source'] for s in samples))}")
    log(f"  Domain breakdown: {dict(Counter(s['domain'] for s in samples))}")


if __name__ == "__main__":
    main()
