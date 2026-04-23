#!/usr/bin/env python3
"""
Duplicate Hugging Face datasets as private mirrors.

Loads all configs of a source dataset and pushes to a target namespace
as a private dataset, preserving config layout and row content.
"""

from __future__ import annotations

import os
import sys
from datasets import get_dataset_config_names, load_dataset


DATASET_PAIRS = [
    ("Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned",
     "ermiaazarkhalili/Kimi-K2.5-Reasoning-1M-Cleaned"),
    ("Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
     "ermiaazarkhalili/GLM-5.1-Reasoning-1M-Cleaned"),
]


def duplicate(src: str, dst: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"SOURCE: {src}")
    print(f"TARGET: {dst} (private)")
    print(f"{'=' * 60}")

    configs = get_dataset_config_names(src)
    print(f"[INFO] Discovered {len(configs)} configs: {configs}")

    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Loading config: {cfg}")
        ds = load_dataset(src, cfg)
        row_count = sum(len(ds[split]) for split in ds.keys())
        print(f"[{i}/{len(configs)}] Splits: {list(ds.keys())}, total rows: {row_count:,}")

        print(f"[{i}/{len(configs)}] Pushing to {dst} (config={cfg}, private=True)")
        ds.push_to_hub(
            dst,
            config_name=cfg,
            private=True,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"[{i}/{len(configs)}] Config '{cfg}' pushed successfully")

    print(f"\n[OK] Completed duplication: {src} -> {dst}")


def main() -> int:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN not set in environment", file=sys.stderr)
        return 1

    failures: list[tuple[str, str]] = []
    for src, dst in DATASET_PAIRS:
        try:
            duplicate(src, dst)
        except Exception as exc:
            print(f"[FAIL] {src} -> {dst}: {exc}", file=sys.stderr)
            failures.append((src, dst))

    if failures:
        print(f"\n[SUMMARY] {len(failures)} failure(s):", file=sys.stderr)
        for src, dst in failures:
            print(f"  - {src} -> {dst}", file=sys.stderr)
        return 1

    print(f"\n[SUMMARY] All {len(DATASET_PAIRS)} datasets duplicated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
