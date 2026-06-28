#!/usr/bin/env python3
"""Aggregate fable held-out eval JSONs into a ranked Markdown report.

Reads $SCRATCH/outputs/fable-eval/*.json (one per model, written by the eval notebook),
builds a table sorted by top-1 delta (trained - base), and writes docs/fable-eval-results.md.
Compares against the Qwable reference (top-1 token-acc 0.791 / val 0.71).

Run AFTER the eval jobs finish:  python3 scripts/tmp/aggregate_fable_eval.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
EVAL_DIR = Path(os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER','ermia')}")) / "outputs" / "fable-eval"
OUT_MD = ROOT / "docs" / "fable-eval-results.md"
QWABLE_REF = 0.791  # reference top-1 token-accuracy


def load_results():
    rows = []
    for fp in sorted(glob.glob(str(EVAL_DIR / "*.json"))):
        if os.path.basename(fp).startswith("SMOKE_"):
            continue  # skip canary smoke (partial split)
        try:
            r = json.load(open(fp))
            if "top1_trained" in r:
                rows.append(r)
        except Exception as e:
            print(f"  [warn] skip {fp}: {e}")
    return rows


def short(repo: str) -> str:
    return repo.split("/")[-1].replace("-SFT-Fable5-Glint", "")


def main():
    rows = load_results()
    if not rows:
        raise SystemExit(f"No eval JSONs found in {EVAL_DIR}")
    rows.sort(key=lambda r: r["top1_delta"], reverse=True)

    lines = [
        "# FABLE-5 (Glint) held-out eval — trained vs base",
        "",
        f"Models evaluated: **{len(rows)}** B-leg fable models on their deterministic ~5% "
        "held-out split (leakage-checked). Metric: teacher-forced token-accuracy "
        "(assistant-masked, each model's own chat template), top-1 (exact) and top-5.",
        f"Reference: Qwable top-1 token-acc ≈ **{QWABLE_REF:.3f}**.",
        "",
        "| Model | Base | top1 base | top1 trained | top1 Δ | top5 trained | top5 Δ | n_eval |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    sum_d1 = 0.0
    for r in rows:
        d1 = r["top1_delta"]
        sum_d1 += d1
        flag = "" if d1 >= 0 else " ⚠"
        lines.append(
            f"| {short(r['model'])} | {r['base'].split('/')[-1]} "
            f"| {r['top1_base']:.4f} | {r['top1_trained']:.4f} | {d1:+.4f}{flag} "
            f"| {r['top5_trained']:.4f} | {r['top5_delta']:+.4f} | {r['n_eval']} |"
        )
    mean_d1 = sum_d1 / len(rows)
    n_improved = sum(1 for r in rows if r["top1_delta"] > 0)
    best = rows[0]
    lines += [
        "",
        "## Summary",
        "",
        f"- **Improved (top-1 Δ > 0): {n_improved}/{len(rows)}**",
        f"- Mean top-1 Δ: **{mean_d1:+.4f}**",
        f"- Best top-1 Δ: **{short(best['model'])}** ({best['top1_delta']:+.4f}, "
        f"base {best['top1_base']:.4f} → trained {best['top1_trained']:.4f})",
        f"- Highest absolute top-1: **{short(max(rows, key=lambda r: r['top1_trained'])['model'])}** "
        f"({max(r['top1_trained'] for r in rows):.4f})",
        "",
        "Notes: token-accuracy is the trustworthy headline (teacher-forced, masked to the gold "
        "assistant completion). ROUGE-L on free-form generation is a weak secondary signal and is "
        "omitted from this table (per-model values are in the JSONs). B-leg = small Glint set "
        "(4,350 rows × 3 epochs); larger gains expected on the D-leg (397K rows) when it finishes.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[OK] wrote {OUT_MD}  ({len(rows)} models, {n_improved} improved, mean top1 Δ {mean_d1:+.4f})")
    # echo the table to stdout too
    print("\n".join(lines))


if __name__ == "__main__":
    main()
