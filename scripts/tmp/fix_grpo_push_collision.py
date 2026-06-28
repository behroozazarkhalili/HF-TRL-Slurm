#!/usr/bin/env python3
"""Fix #212: GRPO push_to_hub skipped because output_dir == trackio_space_id.

Root cause (verified): GRPO_OUT is used as BOTH `output_dir` and `trackio_space_id`.
With no explicit `hub_model_id`, push_to_hub() targets a repo named after output_dir's
basename — the SAME id Trackio already created — so the model push finds "no files
modified" and skips. The trained weights are never freshly committed.

Fix: give the model push a DISTINCT target via `hub_model_id` (a "-model" suffix repo),
decoupled from the Trackio Space id. Applied to the GRPOConfig cell of both 01 notebooks
AND the SFT cell (same collision risk: SFT_OUT). Minimal; no eval/param change.

Run:  python3 scripts/tmp/fix_grpo_push_collision.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path("/project/6014832/ermia/HF-TRL/notebooks/openenv")
TARGETS = [
    "01_reasoning_gym_sft_then_grpo.ipynb",
    "01_reasoning_gym_sft_then_grpo_DRAC_local.ipynb",
    "02_wordle_agentic_grpo.ipynb",
]


def _src(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def main():
    for fn in TARGETS:
        p = NB_DIR / fn
        nb = json.loads(p.read_text())
        changed = []
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = _src(c)
            new = s
            # GRPO: add hub_model_id distinct from trackio_space_id (=GRPO_OUT).
            if "output_dir=GRPO_OUT," in new and "hub_model_id" not in new:
                new = new.replace(
                    "    output_dir=GRPO_OUT,\n",
                    "    output_dir=GRPO_OUT,\n"
                    "    # Push weights to a repo DISTINCT from the Trackio Space id (=GRPO_OUT),\n"
                    "    # otherwise push_to_hub sees the Trackio-owned repo and skips as 'no files\n"
                    "    # modified'. A separate -model repo gets the actual checkpoint commit.\n"
                    '    hub_model_id=f"{HF_USERNAME}/{GRPO_OUT}-model" if HF_USERNAME else f"{GRPO_OUT}-model",\n',
                    1,
                )
                changed.append(f"GRPO@{i}")
            # SFT: same collision (SFT_OUT used as output_dir + would be the push target).
            if "output_dir=SFT_OUT," in new and "hub_model_id" not in new:
                new = new.replace(
                    "        output_dir=SFT_OUT,\n",
                    "        output_dir=SFT_OUT,\n"
                    '        hub_model_id=f"{HF_USERNAME}/{SFT_OUT}-model" if HF_USERNAME else f"{SFT_OUT}-model",\n',
                    1,
                )
                changed.append(f"SFT@{i}")
            if new != s:
                c["source"] = new.splitlines(keepends=True)
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"[OK] {fn}: patched {changed}")


if __name__ == "__main__":
    main()
