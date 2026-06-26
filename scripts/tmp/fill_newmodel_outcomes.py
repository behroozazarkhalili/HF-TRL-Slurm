#!/usr/bin/env python3
"""Fill training_outcome=None -> real metrics for the 6 new-model registry entries.

Metrics extracted from the papermill train output notebooks (authoritative
'Loss:' / 'Runtime:' / 'Peak VRAM:' summary block each notebook prints):
  train_loss  -> trainer-internal final loss
  runtime_sec -> trainer Runtime (not job-wall; matches prior cards' convention)
  peak_vram   -> Peak VRAM GB
"""
from __future__ import annotations
import re
from pathlib import Path

CARD = Path("/project/6014832/ermia/HF-TRL/scripts/generate_unsloth_model_card.py")
GPU = "H100 80GB HBM3 (MIG 3g.40gb)"

# repo_id -> outcome
OUTCOMES = {
    "ermiaazarkhalili/VibeThinker-3B-SFT-Claude-Opus-Reasoning-Unsloth": dict(
        job_id="45169145", runtime_sec=1429, train_loss=1.3670, peak_vram_gb=12.66),
    "ermiaazarkhalili/VibeThinker-3B-Function-Calling-xLAM-Unsloth": dict(
        job_id="45169146", runtime_sec=8435, train_loss=0.3090, peak_vram_gb=13.93),
    "ermiaazarkhalili/FastContext-4B-SFT_base-SFT-Claude-Opus-Reasoning-Unsloth": dict(
        job_id="45169147", runtime_sec=1291, train_loss=0.8678, peak_vram_gb=13.30),
    "ermiaazarkhalili/FastContext-4B-SFT_base-Function-Calling-xLAM-Unsloth": dict(
        job_id="45169148", runtime_sec=7375, train_loss=0.2301, peak_vram_gb=14.52),
    "ermiaazarkhalili/FastContext-4B-RL_base-SFT-Claude-Opus-Reasoning-Unsloth": dict(
        job_id="45169149", runtime_sec=1345, train_loss=0.8673, peak_vram_gb=13.30),
    "ermiaazarkhalili/FastContext-4B-RL_base-Function-Calling-xLAM-Unsloth": dict(
        job_id="45169150", runtime_sec=7751, train_loss=0.2303, peak_vram_gb=14.52),
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
        # Find the entry's key line, then replace the FIRST 'training_outcome': None,
        # that appears after it (each entry has exactly one).
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
    print(f"\n[DONE] patched {patched}/6 entries")
    # Sanity: no stray None left among the 6 new repos
    for repo in OUTCOMES:
        k = text.find(f'"{repo}": {{')
        seg = text[k:k+1200]
        assert '"training_outcome": None' not in seg, f"still None: {repo}"
    print("[OK] verified no remaining None in the 6 new entries")


if __name__ == "__main__":
    main()
