#!/usr/bin/env python3
"""
preprocess_distillation.py — Per-model adapter for claude-reasoning-distillation SFT training.

Loads the SFT config from HF Hub, applies the model-specific thinking-field adapter,
runs apply_chat_template() to produce a `text` column, and saves to disk.

The output directory can be passed directly to train_sft_new.py as --dataset_name.

Usage:
    python preprocess_distillation.py \
        --model_family qwen3.5 \
        --model_name_or_path Qwen/Qwen3.5-0.8B \
        --output_dir /scratch/ermia/outputs/preprocessed-qwen3.5-0.8b

Model families and their adapter strategies:
    qwen3.5: rename thinking → reasoning_content, apply_chat_template(enable_thinking=True)
    lfm2:    embed <think>{thinking}</think> into assistant content, apply_chat_template()
    gemma4:  add <|think|> system prompt, drop thinking field, apply_chat_template()
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model-specific adapters
# ─────────────────────────────────────────────────────────────────────────────

def adapt_qwen35(sample: dict, tokenizer) -> dict:
    """Qwen3.5: rename thinking → reasoning_content, enable_thinking=True."""
    messages = []
    for msg in sample["messages"]:
        m = dict(msg)
        if m["role"] == "assistant":
            thinking = m.pop("thinking", None)
            if thinking:
                m["reasoning_content"] = thinking
            else:
                m.pop("thinking", None)
        else:
            m.pop("thinking", None)
        messages.append(m)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return {"text": text}


def adapt_lfm2(sample: dict, tokenizer) -> dict:
    """LFM2.5: embed <think>{thinking}</think> into assistant content."""
    messages = []
    for msg in sample["messages"]:
        m = dict(msg)
        if m["role"] == "assistant":
            thinking = m.pop("thinking", None)
            if thinking:
                m["content"] = f"<think>{thinking}</think>\n{m['content']}"
            else:
                m.pop("thinking", None)
        else:
            m.pop("thinking", None)
        messages.append(m)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def adapt_gemma4(sample: dict, tokenizer) -> dict:
    """Gemma4: add <|think|> system prompt, drop thinking field."""
    messages = list(sample["messages"])

    THINK_TRIGGER = "<|think|>"

    # Ensure system prompt with think trigger
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": THINK_TRIGGER}] + messages
    elif THINK_TRIGGER not in (messages[0].get("content") or ""):
        messages[0] = dict(messages[0])
        messages[0]["content"] = THINK_TRIGGER + "\n" + (messages[0]["content"] or "")

    # Remove thinking field from all messages
    clean_messages = []
    for msg in messages:
        m = dict(msg)
        m.pop("thinking", None)
        clean_messages.append(m)

    text = tokenizer.apply_chat_template(
        clean_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


ADAPTER_MAP = {
    "qwen3.5": adapt_qwen35,
    "qwen3": adapt_qwen35,  # alias — same adapter
    "lfm2": adapt_lfm2,
    "lfm2.5": adapt_lfm2,  # alias
    "gemma4": adapt_gemma4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess claude-reasoning-distillation for SFT")
    p.add_argument(
        "--model_family", required=True,
        choices=list(ADAPTER_MAP.keys()),
        help="Model family determines the adapter strategy",
    )
    p.add_argument(
        "--model_name_or_path", required=True,
        help="HF model ID or local path (for tokenizer)",
    )
    p.add_argument(
        "--dataset_repo", default="ermiaazarkhalili/claude-reasoning-distillation",
        help="HF dataset repo",
    )
    p.add_argument(
        "--dataset_config", default="sft",
        help="Dataset config name",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples to process (None = all)",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to save preprocessed dataset",
    )
    p.add_argument(
        "--num_proc", type=int, default=None,
        help="Number of parallel workers for dataset.map(). Auto-detects from SLURM.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log("=" * 60)
    log("Preprocess Distillation Dataset")
    log("=" * 60)
    log(f"  Model family:  {args.model_family}")
    log(f"  Model:         {args.model_name_or_path}")
    log(f"  Dataset:       {args.dataset_repo} ({args.dataset_config})")
    log(f"  Max samples:   {args.max_samples or 'all'}")
    log(f"  Output:        {args.output_dir}")

    # Load dataset
    from datasets import load_dataset
    log(f"\nLoading dataset ...")
    ds = load_dataset(args.dataset_repo, args.dataset_config, split="train")
    log(f"  Loaded {len(ds)} samples")

    if args.max_samples is not None and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
        log(f"  Truncated to {len(ds)} samples")

    # Load tokenizer
    from transformers import AutoTokenizer, AutoProcessor
    log(f"\nLoading tokenizer for {args.model_name_or_path} ...")

    # Gemma4 uses AutoProcessor (multimodal), others use AutoTokenizer
    # Fallback to AutoTokenizer if AutoProcessor fails (e.g., older transformers)
    if args.model_family == "gemma4":
        try:
            tokenizer = AutoProcessor.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
            log(f"  Loaded AutoProcessor for Gemma4")
        except (ValueError, OSError) as e:
            log(f"  AutoProcessor failed ({e}), falling back to AutoTokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log(f"  Set pad_token = eos_token ({tokenizer.eos_token})")

    log(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'N/A'}")

    # Apply adapter
    adapter_fn = ADAPTER_MAP[args.model_family]
    log(f"\nApplying adapter: {adapter_fn.__name__} ...")

    num_proc = args.num_proc or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    # For small datasets or tokenizers that aren't picklable, use single process
    if len(ds) < 500 or args.model_family == "gemma4":
        num_proc = 1

    def apply_adapter(sample):
        return adapter_fn(sample, tokenizer)

    ds_processed = ds.map(
        apply_adapter,
        num_proc=num_proc,
        remove_columns=ds.column_names,  # keep only `text`
        desc=f"Applying {args.model_family} adapter",
    )

    # Stats
    log(f"\nProcessed {len(ds_processed)} samples")
    text_lengths = [len(t) for t in ds_processed["text"]]
    token_lengths = []
    for t in ds_processed.select(range(min(100, len(ds_processed))))["text"]:
        if hasattr(tokenizer, "tokenizer"):
            # AutoProcessor wraps tokenizer
            toks = tokenizer.tokenizer(t, add_special_tokens=False)["input_ids"]
        else:
            toks = tokenizer(t, add_special_tokens=False)["input_ids"]
        token_lengths.append(len(toks))

    log(f"  Text char lengths: min={min(text_lengths)}, avg={sum(text_lengths)//len(text_lengths)}, max={max(text_lengths)}")
    log(f"  Token lengths (first {len(token_lengths)}): min={min(token_lengths)}, avg={sum(token_lengths)//len(token_lengths)}, max={max(token_lengths)}")

    # Warn about potential truncation
    max_tok = max(token_lengths)
    if max_tok > 2048:
        log(f"  ⚠ {sum(1 for t in token_lengths if t > 2048)}/{len(token_lengths)} samples exceed 2048 tokens (max={max_tok})")
        log(f"  ⚠ Consider using --max_length 4096 in training")
    if max_tok > 4096:
        log(f"  ⚠ {sum(1 for t in token_lengths if t > 4096)}/{len(token_lengths)} samples exceed 4096 tokens")

    # Preview
    log(f"\nSample text preview (first 300 chars):")
    log(f"  {ds_processed[0]['text'][:300]}...")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ds_processed.save_to_disk(args.output_dir)
    log(f"\nSaved to {args.output_dir}")

    log("\n" + "=" * 60)
    log("Preprocessing complete!")
    log("=" * 60)


if __name__ == "__main__":
    main()
