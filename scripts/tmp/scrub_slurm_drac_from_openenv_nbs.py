#!/usr/bin/env python3
"""Make the off-cluster OpenEnv notebooks standalone for HF Jobs / Colab.

Removes SLURM/DRAC references (these notebooks run on HF Jobs or Colab, not on the
cluster). Targeted exact-string replacements only — no logic/parameter changes, no eval
changes. The DRAC-local notebook (01_..._DRAC_local.ipynb) is intentionally NOT touched:
it is the in-cluster variant and is excluded from the off-cluster set.

Run:  python3 scripts/tmp/scrub_slurm_drac_from_openenv_nbs.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path("/project/6014832/ermia/HF-TRL/notebooks/openenv")
TARGETS = [
    "00_openenv_quickstart.ipynb",
    "01_reasoning_gym_sft_then_grpo.ipynb",
    "02_wordle_agentic_grpo.ipynb",
]

# Exact-string replacements (old -> new). Shared phrasing across the notebooks.
REPLACEMENTS = [
    # cell[1] runtime-targets blurb
    (
        "> **Runtime targets — this notebook runs anywhere.** Colab (A100/L4/T4), a fresh\n"
        "> local GPU box, or an HPC cluster via the SLURM runner in `jobs/`. It self-installs",
        "> **Runtime targets — this standalone notebook runs on HF Jobs or Colab.** It self-installs",
    ),
    # cell[2] recipe header "off-DRAC"
    ("## ▶ How to run this notebook off-DRAC (HF Jobs · Google Colab)",
     "## ▶ How to run this notebook (HF Jobs · Google Colab)"),
    ("This notebook is portable — **no DRAC required**. Pick a lane.",
     "This notebook is standalone. Pick a lane."),
    (" so this works even from networks that block the\n"
     "hosted-env WebSocket (e.g. DRAC compute). Requires pre-paid credits on your HF account.",
     " Requires pre-paid credits on your HF account."),
    # cell[6] auth comment
    ("# Prefers an already-set HF_TOKEN (CI / SLURM / Colab secret); falls back to the",
     "# Prefers an already-set HF_TOKEN (HF Jobs / Colab secret); falls back to the"),
    # GRPOConfig vLLM trailing comment
    ("    # vLLM left OFF in-notebook: its init breaks under IPython. Enable in the SLURM/script\n"
     "    # path with use_vllm=True, vllm_mode=\"colocate\".",
     "    # vLLM left OFF in-notebook: its init breaks under IPython. The optional cell below\n"
     "    # enables it (use_vllm=True, vllm_mode=\"colocate\") for an HF Job / script run."),
    # nb02 GRPOConfig vLLM comment variant
    ("    # vLLM OFF in-notebook (IPython init). SLURM path: use_vllm=True, vllm_mode=\"colocate\".",
     "    # vLLM OFF in-notebook (IPython init). The optional cell below enables it for an HF Job."),
    # vllm-optional injected cell
    ("# Enable it (USE_VLLM=1) ONLY in a non-IPython context: an HF Job, a\n"
     "# SLURM/script run, or a fresh Colab runtime executed as a script. Colocate mode shares",
     "# Enable it (USE_VLLM=1) ONLY in a non-IPython context: an HF Job or a fresh\n"
     "# Colab runtime executed as a script. Colocate mode shares"),
    # nb01 recap bullet
    ("- For a real run: set `SMOKE=False`, deploy your own env Space, and enable vLLM in the\n"
     "  SLURM path (`jobs/openenv-grpo-reasoning-gym.sh`).",
     "- For a real run: set `SMOKE=False`, optionally deploy your own env Space, and enable\n"
     "  vLLM via `USE_VLLM=1` on an HF Job (see the run-recipe cell near the top)."),
    # nb02 'in the SLURM path.' fragment (cell[3])
    ("in the SLURM path.", "via `USE_VLLM=1` on an HF Job."),
]


def _src(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def process(path: Path) -> None:
    nb = json.loads(path.read_text())
    n_changed = 0
    for c in nb["cells"]:
        s = _src(c)
        new = s
        for old, repl in REPLACEMENTS:
            if old in new:
                new = new.replace(old, repl)
        if new != s:
            c["source"] = new.splitlines(keepends=True)
            n_changed += 1
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

    # report any residual SLURM/DRAC tokens
    residual = []
    for i, c in enumerate(nb["cells"]):
        for ln in _src(c).splitlines():
            low = ln.lower()
            if any(k in low for k in ["slurm", "drac", "sbatch", "jobs/openenv", "loopback", "localhost:8000"]):
                residual.append(f"cell[{i}]: {ln.strip()[:80]}")
    status = "CLEAN" if not residual else f"RESIDUAL x{len(residual)}"
    print(f"[OK] {path.name}: {n_changed} cells changed -> {status}")
    for r in residual:
        print(f"      ! {r}")


def main() -> None:
    for fn in TARGETS:
        p = NB_DIR / fn
        if not p.exists():
            print(f"[skip] {fn} not found"); continue
        process(p)


if __name__ == "__main__":
    main()
