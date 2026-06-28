#!/usr/bin/env python3
"""Add optional vLLM cell + HF-Jobs/Colab run-recipe cells to the OpenEnv notebooks.

Idempotent: each inserted cell carries a sentinel in its first line; re-running replaces
the marked cell in place rather than inserting a duplicate. No cell IDs required (the
notebooks predate nbformat ids), so we locate insertion points structurally by content.

Targets:
  00_openenv_quickstart.ipynb        -> run-recipe markdown only (no GRPO training here)
  01_reasoning_gym_sft_then_grpo.ipynb        -> recipe + vLLM cell
  01_..._DRAC_local.ipynb                      -> recipe + vLLM cell
  02_wordle_agentic_grpo.ipynb                 -> recipe + vLLM cell

Run on the login node:  python3 scripts/tmp/add_openenv_run_recipes.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path("/project/6014832/ermia/HF-TRL/notebooks/openenv")

VLLM_SENTINEL = "# @inject:vllm-optional"
RECIPE_SENTINEL = "<!-- @inject:run-recipes -->"


def _src(cell) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# ---- the optional vLLM cell (mutates the already-built grpo_config in place) ----
VLLM_CELL = f'''{VLLM_SENTINEL}
# --- OPTIONAL: vLLM-accelerated GRPO rollouts (5-10x faster generation) ---------
# OFF by default: vLLM's init breaks under IPython/Jupyter, so leave USE_VLLM=0 for an
# in-notebook run. Enable it (USE_VLLM=1) ONLY in a non-IPython context: an HF Job, a
# SLURM/script run, or a fresh Colab runtime executed as a script. Colocate mode shares
# the single training GPU (right for Colab / one-GPU HF-Job flavors).
#   Verified TRL v1.7.0 params: use_vllm, vllm_mode="colocate"|"server",
#   vllm_gpu_memory_utilization. (server mode is multi-GPU; not used here.)
import os

USE_VLLM = os.environ.get("USE_VLLM", "0") not in ("0", "false", "False")
if USE_VLLM:
    grpo_config.use_vllm = True
    grpo_config.vllm_mode = "colocate"
    # Leave room for the training copy of the model in colocate mode; tune per GPU/model.
    grpo_config.vllm_gpu_memory_utilization = float(
        os.environ.get("VLLM_GPU_MEM_UTIL", "0.3")
    )
    print(
        f"[vLLM] enabled: mode=colocate gpu_mem_util={{grpo_config.vllm_gpu_memory_utilization}} "
        "(requires a non-IPython runtime + `pip install vllm`)"
    )
else:
    print("[vLLM] disabled (USE_VLLM=0). In-notebook generation uses HF generate().")
'''


def extract_pip_lines(cells: list) -> str:
    """Pull the notebook's own `%pip install` lines, turned into plain `pip install`
    (chained with &&) so the HF-Jobs container installs exactly what the notebook needs —
    no hard-coded dependency list. papermill is prepended (needed to execute the .ipynb)."""
    pip_cmds = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        for ln in _src(c).splitlines():
            s = ln.strip()
            if s.startswith("%pip install") or s.startswith("!pip install"):
                pip_cmds.append("pip install " + s.split("install", 1)[1].strip())
    # papermill is required to run the notebook headlessly inside the job
    cmds = ["pip install -q papermill"] + pip_cmds
    return " && \\\n           ".join(cmds)


def extract_env_url(cells: list) -> str:
    """The notebook's own default ENV_BASE_URL (os.environ.get fallback). No guess."""
    import re as _re2
    for c in cells:
        if c["cell_type"] != "code":
            continue
        m = _re2.search(r'ENV_BASE_URL\s*=\s*os\.environ\.get\(\s*"ENV_BASE_URL"\s*,\s*"([^"]+)"', _src(c))
        if m:
            return m.group(1)
    return ""


def recipe_md(notebook_filename: str, *, has_vllm: bool, env_url: str, pip_block: str, trains: bool) -> str:
    """Run-recipe markdown tailored per notebook from FACTS extracted from that notebook
    (env_url, pip_block, whether it trains). Commands verified against the `hf jobs` CLI."""
    env_line = f"  -e ENV_BASE_URL={env_url} \\\n" if env_url else ""
    train_env = "-e SMOKE=0 -e REPORT_TO=trackio " if trains else ""
    flavor = "a10g-small" if trains else "cpu-basic"
    timeout = "10800" if trains else "1800"
    secrets = "  --secrets HF_TOKEN \\\n" if trains else ""

    vllm_jobs_block = ""
    if has_vllm:
        vllm_pip = pip_block.replace("pip install -q papermill", "pip install -q papermill vllm", 1)
        vllm_jobs_block = f"""
### B · HF Jobs — full run with vLLM (faster rollouts)

Same as A but on a bigger GPU with vLLM enabled (`USE_VLLM=1`). vLLM works here because an
HF Job is **not** an IPython runtime (the reason it stays OFF in-notebook).

```bash
hf jobs run \\
  --flavor a100-large \\
  --timeout 21600 \\
  --secrets HF_TOKEN \\
  -e SMOKE=0 -e USE_VLLM=1 -e REPORT_TO=trackio \\
{env_line}  python:3.12 \\
  bash -c "{vllm_pip} && \\
           papermill {notebook_filename} out.ipynb"
```
"""

    colab_extra = (
        '\nos.environ["REPORT_TO"] = "trackio"       # live charts; or "none" for headless'
        if trains else ""
    )
    colab_vllm_note = (
        '\n\nFor a longer run on an A100 Colab, set `os.environ["USE_VLLM"] = "1"` *before* '
        "the GRPO config cell — Colab's kernel tolerates vLLM colocate on a single A100."
        if has_vllm else ""
    )
    runtime_hint = "GPU" if trains else "CPU (no GPU needed)"

    return f"""{RECIPE_SENTINEL}
## ▶ How to run this notebook off-DRAC (HF Jobs · Google Colab)

This notebook is portable — **no DRAC required**. Pick a lane. (Dependencies and the env
URL below are taken from this notebook's own cells.)

### A · HF Jobs — run the notebook on Hugging Face infra

`hf jobs run` is *docker-run for HF infra*: the compute happens on HF's cloud, you only
submit. The control plane is plain HTTPS, so this works even from networks that block the
hosted-env WebSocket (e.g. DRAC compute). Requires pre-paid credits on your HF account.

```bash
# one-time: hf auth login   (or rely on a local HF_TOKEN)
hf jobs run \\
  --flavor {flavor} \\
  --timeout {timeout} \\
{secrets}  {train_env}\\
{env_line}  python:3.12 \\
  bash -c "{pip_block} && \\
           papermill {notebook_filename} out.ipynb"
```

Notes: `--secrets HF_TOKEN` forwards your local token (needed if the notebook pushes to the
Hub). `--flavor` options: `cpu-basic`, `t4-small`, `a10g-small`, `a100-large`, … (`hf jobs`
docs). Add `--detach` to background it; watch with `hf jobs logs <id>`.
{vllm_jobs_block}
### C · Google Colab — independent run

Open Colab → set runtime to **{runtime_hint}** → in the first cell:

```python
# Colab cell 1 — auth + knobs, then run the rest of the notebook top-to-bottom.
import os
os.environ["SMOKE"] = "0"                 # full run (omit for a quick smoke){colab_extra}
os.environ["ENV_BASE_URL"] = "{env_url}"
from huggingface_hub import notebook_login
notebook_login()                          # paste a token with write access
```

Then **Runtime → Run all**. Colab has open egress, so the hosted env Space works directly
(no in-job uvicorn needed). The install cell at the top of the notebook handles
dependencies.{colab_vllm_note}

---
"""


def upsert_cell(cells: list, sentinel: str, new_cell: dict, *, after_pred) -> str:
    """Replace the cell whose first line contains `sentinel`, else insert `new_cell`
    immediately after the first cell matching `after_pred`. Returns 'replaced'|'inserted'."""
    for idx, c in enumerate(cells):
        first = _src(c).splitlines()[0] if _src(c).strip() else ""
        if sentinel in first:
            cells[idx] = new_cell
            return "replaced"
    for idx, c in enumerate(cells):
        if after_pred(c):
            cells.insert(idx + 1, new_cell)
            return "inserted"
    raise RuntimeError(f"no anchor found for sentinel {sentinel!r}")


def process(path: Path, *, has_vllm: bool) -> None:
    nb = json.loads(path.read_text())
    cells = nb["cells"]

    # 1) run-recipe markdown after the "Runtime targets" cell (fallback: after title cell 0)
    def is_runtime_targets(c):
        return c["cell_type"] == "markdown" and "Runtime targets" in _src(c)

    def is_title(c):
        return c["cell_type"] == "markdown" and _src(c).lstrip().startswith("# ")

    anchor = is_runtime_targets if any(is_runtime_targets(c) for c in cells) else is_title
    env_url = extract_env_url(cells)
    pip_block = extract_pip_lines(cells)
    r1 = upsert_cell(
        cells, RECIPE_SENTINEL,
        make_md_cell(recipe_md(
            path.name, has_vllm=has_vllm, env_url=env_url, pip_block=pip_block, trains=has_vllm,
        )),
        after_pred=anchor,
    )

    # 2) optional vLLM code cell right after the GRPOConfig builder cell
    r2 = "n/a"
    if has_vllm:
        def builds_grpo_config(c):
            return c["cell_type"] == "code" and "grpo_config = GRPOConfig(" in _src(c)

        if not any(builds_grpo_config(c) for c in cells):
            raise RuntimeError(f"{path.name}: no 'grpo_config = GRPOConfig(' cell found")
        r2 = upsert_cell(cells, VLLM_SENTINEL, make_code_cell(VLLM_CELL), after_pred=builds_grpo_config)

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"[OK] {path.name}: recipe={r1} vllm={r2}  (cells now {len(cells)})")


def main() -> None:
    targets = [
        ("00_openenv_quickstart.ipynb", False),
        ("01_reasoning_gym_sft_then_grpo.ipynb", True),
        ("01_reasoning_gym_sft_then_grpo_DRAC_local.ipynb", True),
        ("02_wordle_agentic_grpo.ipynb", True),
    ]
    for fname, has_vllm in targets:
        p = NB_DIR / fname
        if not p.exists():
            print(f"[skip] {fname} not found")
            continue
        process(p, has_vllm=has_vllm)


if __name__ == "__main__":
    main()
