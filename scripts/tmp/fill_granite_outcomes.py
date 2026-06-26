#!/usr/bin/env python3
"""Fill training_outcome=None -> real metrics for the 4 Granite 4.1 registry entries.

Metrics from train logs (2026-05-03 runs), each notebook's 'Loss:'/'Runtime:'/'Peak VRAM:' block.
"""
from __future__ import annotations
from pathlib import Path

CARD = Path("/project/6014832/ermia/HF-TRL/scripts/generate_unsloth_model_card.py")
GPU = "H100 80GB HBM3 (MIG 3g.40gb)"

OUTCOMES = {
    "ermiaazarkhalili/Granite-4.1-3B-SFT-Claude-Opus-Reasoning-Unsloth": dict(
        job_id="38330896", runtime_sec=1767, train_loss=0.8932, peak_vram_gb=10.18),
    "ermiaazarkhalili/Granite-4.1-8B-SFT-Claude-Opus-Reasoning-Unsloth": dict(
        job_id="38330897", runtime_sec=1969, train_loss=0.7895, peak_vram_gb=9.26),
    "ermiaazarkhalili/Granite-4.1-3B-Function-Calling-xLAM-Unsloth": dict(
        job_id="38330898", runtime_sec=10049, train_loss=0.2242, peak_vram_gb=10.09),
    "ermiaazarkhalili/Granite-4.1-8B-Function-Calling-xLAM-Unsloth": dict(
        job_id="38330899", runtime_sec=10937, train_loss=0.2092, peak_vram_gb=8.40),
}


def outcome_block(o: dict) -> str:
    return (
        '"training_outcome": {\n'
        f'            "job_id": "{o["job_id"]}",\n'
        f'            "runtime_sec": {o["runtime_sec"]},\n'
        f'            "train_loss": {o["train_loss"]},\n'
        f'            "peak_vram_gb": {o["peak_vram_gb"]},\n'
        f'            "gpu": "{GPU}",\n'
        '        },'
    )


def main():
    text = CARD.read_text()
    patched = 0
    for repo, o in OUTCOMES.items():
        key = f'"{repo}": {{'
        idx = text.find(key)
        if idx < 0:
            raise SystemExit(f"[FAIL] registry key not found: {repo}")
        none_marker = '"training_outcome": None,'
        rel = text.find(none_marker, idx)
        if rel < 0:
            raise SystemExit(f"[FAIL] no 'training_outcome: None' after {repo}")
        text = text[:rel] + outcome_block(o) + text[rel + len(none_marker):]
        patched += 1
        print(f"[OK] filled {repo.split('/')[1]}")
    CARD.write_text(text)
    print(f"\n[DONE] patched {patched}/4 Granite entries")


if __name__ == "__main__":
    main()
