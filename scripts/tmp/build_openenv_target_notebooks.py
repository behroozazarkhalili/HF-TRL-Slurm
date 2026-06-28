#!/usr/bin/env python3
"""Generate dedicated, educational HF-Job + Colab OpenEnv notebooks.

From each cleaned source notebook (01_reasoning_gym_sft_then_grpo.ipynb,
02_wordle_agentic_grpo.ipynb) produce two STANDALONE variants per run target:
  <stem>_colab.ipynb   — Colab-baked: setup cell (env knobs + notebook_login) + badge.
  <stem>_hfjob.ipynb   — HF-Jobs-baked: a markdown with the exact `hf jobs run` command.

The training + reference-strategy eval body is copied VERBATIM (no eval/param edits).
The shared "pick a lane" recipe cell (<!-- @inject:run-recipes -->) is dropped; each
variant bakes ONE target. Section markdown is enriched HF-Cookbook style (what/why),
WITHOUT interpreting any results.

Run:  python3 scripts/tmp/build_openenv_target_notebooks.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NB_DIR = Path("/project/6014832/ermia/HF-TRL/notebooks/openenv")

# source -> (out_stem, env pip-install client line is read from the notebook itself)
SOURCES = {
    "01_reasoning_gym_sft_then_grpo.ipynb": {
        "stem": "01_reasoning_gym_grpo",
        "title": "Reasoning-Gym · SFT warm-start → GRPO → evaluate",
        "task": "chain-sum arithmetic",
    },
    "02_wordle_agentic_grpo.ipynb": {
        "stem": "02_wordle_grpo",
        "title": "Wordle · multi-turn agentic GRPO → play-the-game eval",
        "task": "multi-turn Wordle",
    },
}

RECIPE_SENTINEL = "@inject:run-recipes"


# ---- helpers ------------------------------------------------------------------
def _src(c) -> str:
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def extract_env_url(cells) -> str:
    for c in cells:
        m = re.search(r'ENV_BASE_URL\s*=\s*os\.environ\.get\(\s*"ENV_BASE_URL"\s*,\s*"([^"]+)"', _src(c))
        if m:
            return m.group(1)
    return ""


def extract_pip_lines(cells) -> list[str]:
    """Pull the notebook's own pip-install lines as plain `pip install`.

    The notebook source quotes version specs with DOUBLE quotes (e.g.
    `"transformers>=5.3.0"`). These lines get embedded inside a double-quoted
    `bash -c "..."`, so the inner double-quotes would terminate the outer string.
    Convert any double-quoted token to SINGLE quotes to keep the bash -c valid.
    """
    out = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        for ln in _src(c).splitlines():
            s = ln.strip()
            if s.startswith("%pip install") or s.startswith("!pip install"):
                cmd = "pip install " + s.split("install", 1)[1].strip()
                cmd = cmd.replace('"', "'")  # double -> single quotes (bash -c safe)
                out.append(cmd)
    return out


# ---- educational intro (per target) ------------------------------------------
def colab_intro(meta, has_sft) -> str:
    sft_bullet = (
        "- Why a short **SFT warm-start** on successful rollouts makes GRPO converge from a\n"
        "  sane starting policy instead of cold-starting from random tool-calling.\n"
        if has_sft else
        "- How GRPO can train an agent **directly from the environment reward** (no SFT\n"
        "  warm-start) on a multi-turn task.\n"
    )
    return f"""# {meta['title']} — Colab

> **An OpenEnv agent-training walkthrough, Colab edition.** You will train a small
> language model to act as an *agent* inside a live environment using **GRPO**
> (Group Relative Policy Optimization) from 🤗 TRL, then measure it. Everything runs on
> a single Colab GPU — no cluster, no local setup.

**What you'll learn**
- How an *environment* (not a static dataset) supplies the training signal: the agent
  acts, the env returns a scalar **reward**, and GRPO ranks rollouts against each other.
{sft_bullet}- How to evaluate the trained agent honestly ({meta['task']}).

**Before you start:** set the runtime to a GPU (**Runtime → Change runtime type → GPU**),
then run the setup cell below and **Runtime → Run all**."""


def hfjob_intro(meta, run_cmd, vllm_cmd, has_sft) -> str:
    pipeline_desc = (
        "with a short **SFT warm-start**, then a held-out evaluation"
        if has_sft else
        "trained directly with GRPO (no SFT warm-start), then a play-the-game evaluation"
    )
    return f"""# {meta['title']} — Hugging Face Jobs

> **An OpenEnv agent-training walkthrough, HF Jobs edition.** This notebook is meant to
> be executed **non-interactively on Hugging Face infrastructure** with `papermill`. The
> GPU work runs on HF's cloud; you only submit.

**What this trains:** a small LM to act as an agent inside a live OpenEnv environment via
**GRPO** ({meta['task']}), {pipeline_desc}.

## ▶ Submit this notebook as an HF Job

Requires pre-paid credits on your HF account. The control plane is plain HTTPS. Replace
`<YOUR_REPO_URL>` with the public Git URL hosting this notebook.

```bash
{run_cmd}
```

`--secrets HF_TOKEN` forwards your token so the trained model can push to the Hub. Set
`SMOKE=0` for a real run (150 GRPO steps) or `SMOKE=1` for a quick smoke. Tune `--flavor`
(`t4-small`, `a10g-small`, `a100-large`, …) and `--timeout` (seconds) to your run.

### Faster rollouts with vLLM

An HF Job is a non-interactive runtime, so vLLM-accelerated generation works here (it is
left off in interactive notebooks because its init conflicts with IPython). Use a bigger
GPU and install vLLM in the same command:

```bash
{vllm_cmd}
```"""


def hfjob_run_cmd(out_name, env_url, pip_lines, with_vllm=False) -> str:
    flavor = "a100-large" if with_vllm else "a10g-small"
    timeout = "21600" if with_vllm else "10800"
    vllm_env = "-e USE_VLLM=1 " if with_vllm else ""
    # python:3.12 ships without git, but `git clone` + `pip install git+...` both need it.
    apt = "apt-get update -q && apt-get install -y -q git"
    # vLLM run needs the vllm package; install it alongside papermill.
    pip = "pip install -q papermill" + (" vllm" if with_vllm else "") + " && \\\n           " + \
          " && \\\n           ".join(pip_lines)
    envline = f"  -e ENV_BASE_URL={env_url} \\\n" if env_url else ""
    return (
        "hf jobs run \\\n"
        f"  --flavor {flavor} \\\n"
        f"  --timeout {timeout} \\\n"
        "  --secrets HF_TOKEN \\\n"
        f"  -e SMOKE=0 {vllm_env}-e REPORT_TO=trackio \\\n"
        f"{envline}  python:3.12 \\\n"
        f'  bash -c "{apt} && \\\n'
        f"           {pip} && \\\n"
        "           git clone --depth 1 <YOUR_REPO_URL> /repo && cd /repo/notebooks/openenv && \\\n"
        f'           papermill {out_name} out.ipynb"'
    )


# ---- Colab setup cell ---------------------------------------------------------
def colab_setup_cell(env_url) -> str:
    return f'''# --- Colab setup: knobs + Hub auth (run me first) ------------------------------
# These environment variables drive the rest of the notebook — no cell edits needed.
import os

os.environ["SMOKE"] = "0"                  # "0" = full run (150 GRPO steps); "1" = quick smoke
os.environ["REPORT_TO"] = "trackio"        # live charts on a Trackio Space; "none" for headless
os.environ["ENV_BASE_URL"] = "{env_url}"   # the hosted environment Space
os.environ["USE_VLLM"] = "0"               # keep OFF on Colab unless on an A100 (see vLLM cell)

# Authenticate so the trained model can push to your Hub account (needed when SMOKE=0).
from huggingface_hub import notebook_login

notebook_login()
'''


# ---- per-section educational headers (augment existing ## headers) ------------
# Maps a phrase found in an existing markdown header -> extra "why" paragraph appended.
SECTION_NOTES = {
    "Install": "We install TRL (the trainer), OpenEnv (the env client), and the environment's "
               "own package. `transformers>=5.3.0` is required because GRPO's `environment_factory` "
               "path (a TRL feature) depends on tool-calling / chat-template behavior introduced in "
               "that Transformers release.",
    "collect teacher rollouts": "OpenEnv's training data is **generated**, not downloaded: a "
               "*teacher* model acts in the env and we keep the episodes it got right. These "
               "successful rollouts become the SFT warm-start set. (If no teacher key is set, we "
               "skip straight to GRPO — the notebook still runs end-to-end.)",
    "SFT warm-start": "A few epochs of supervised fine-tuning on the *correct* rollouts teach the "
               "model the env's tool-call format **before** RL, so GRPO begins from a policy that "
               "already calls the tool. The intent is a more stable start than cold-starting from "
               "the base model.",
    "Phase B — GRPO": "GRPO is value-free: for each prompt it samples a group of rollouts, scores "
               "each by the env's reward, and pushes the policy toward the better ones within the "
               "group. The environment's reward — not a static label — is the training signal. More "
               "`num_generations` gives a richer in-group ranking but costs more generation.",
    "Training signal": "We compare mean reward over the first few logged steps vs. the last few — "
               "a check on whether reward moved during training (not a held-out quality claim).",
    "held-out": "The question we want to answer is whether the agent solves problems it never "
               "trained on. We evaluate on a different seed (intended to be a disjoint held-out "
               "set) and compare the base model against the trained one with the reference eval "
               "harness, using identical decoding settings for both so the comparison is fair.",
    "play full games": "For a multi-turn env, the faithful eval is to **play complete games**: the "
               "agent acts turn by turn against live env feedback until the episode ends, and we "
               "measure win rate — the agentic analogue of held-out accuracy.",
}


def enrich_headers(cells):
    """Append a 'why' paragraph under numbered ## section headers (educational tone).

    Only matches a header by its FIRST line and only numbered sections (e.g. '## 8 · ...'),
    so the Recap and other prose cells are never enriched (avoids leaking nb01-specific
    notes like the held-out-seed text into nb02's recap)."""
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        s = _src(c)
        first = s.lstrip().splitlines()[0] if s.strip() else ""
        if not re.match(r"^##\s+\d", first):  # numbered section headers only
            continue
        for phrase, note in SECTION_NOTES.items():
            if phrase in first and note not in s:
                c["source"] = (s.rstrip() + "\n\n" + note + "\n").splitlines(keepends=True)
                break


# ---- per-code-cell teaching cells (HF-Cookbook depth) -------------------------
# Keyed by a substring that uniquely identifies a code cell. Each value is a deep
# teaching markdown (concept + why-this-choice + a 💡 callout + a doc link) inserted
# BEFORE that code cell — but only when the cell isn't already preceded by a markdown
# header (avoids doubling up). Method-only; never interprets results.
TEACHING_CELLS = {
    "import nest_asyncio": (
        "### Why `nest_asyncio`?\n\n"
        "The OpenEnv client is **async** (`await env.reset()`, `await env.step()`), but a "
        "Jupyter/Colab kernel already runs its own event loop. `nest_asyncio.apply()` lets us "
        "`await` inside notebook cells without `RuntimeError: event loop already running`.\n\n"
        "> 💡 In a plain `.py` script you'd use `asyncio.run(...)` instead — this shim is "
        "specifically for notebook runtimes."
    ),
    "from huggingface_hub import whoami": (
        "### Resolve your Hub username\n\n"
        "Downstream repo names (rollouts dataset, trained model) are built from your username, "
        "so we resolve it once via `whoami()` rather than hard-coding it. This keeps the notebook "
        "portable across accounts.\n\n"
        "> 💡 `whoami()` reads the token you authenticated with above — no extra prompt."
    ),
    "import torch\n": (
        "### Auto-detect compute, pick a model that fits\n\n"
        "Rather than hard-code a model, we read the GPU's VRAM and choose a size that fits "
        "(`Qwen3-1.7B` when there's headroom, else `Qwen3-0.6B`). Override with `MODEL_NAME`. "
        "This is what lets the same notebook run on a T4, an L4, or an A100 unchanged.\n\n"
        "> 💡 GRPO holds the policy model **and** generates rollouts, so VRAM headroom matters "
        "more than for plain inference. [TRL GRPO docs]"
        "(https://huggingface.co/docs/trl/en/grpo_trainer)."
    ),
    'SMOKE = os.environ.get("SMOKE"': (
        "### Run knobs: smoke vs. full\n\n"
        "Every expensive quantity (rollout count, GRPO steps, eval size) is gated by `SMOKE` so "
        "you can prove the whole pipeline end-to-end in minutes (`SMOKE=1`) before committing to a "
        "real run (`SMOKE=0`). The values come from environment variables, so you never edit cells "
        "to change a run.\n\n"
        "> 💡 This is the single switch that turns a 5-minute demo into a real training run."
    ),
    'TEACHER_PROVIDER = os.environ.get': (
        "### The teacher (optional)\n\n"
        "GRPO trains best from a warm-started policy. To create that warm-start we let a stronger "
        "**teacher** model play the env and keep its *correct* episodes for SFT. The teacher is a "
        "parameter: an OpenAI model, any OpenAI-compatible endpoint, or — if none is set — we skip "
        "collection and let GRPO cold-start, so the notebook always runs.\n\n"
        "> 💡 No API key? Leave it unset; you'll still get a complete GRPO run, just without the "
        "SFT warm-start."
    ),
    "from trl import GRPOConfig": (
        "### Configure GRPO\n\n"
        "`GRPOConfig` holds the RL hyperparameters. The agent-specific ones: `num_generations` "
        "(rollouts sampled per prompt — GRPO ranks them against each other), "
        "`max_completion_length` (room for the tool call), and `report_to=\"trackio\"` for live "
        "charts. `save_strategy`/`push_to_hub` control persistence.\n\n"
        "> 💡 GRPO is *value-free*: it needs no separate reward model — the **environment's** "
        "scalar reward is the only signal. [GRPO paper](https://huggingface.co/papers/2402.03300)."
    ),
    "from trl import GRPOTrainer": (
        "### Train the agent\n\n"
        "`environment_factory=<EnvClass>` is the key line: for each rollout the trainer creates an "
        "env (TRL may reuse env instances across a batch), generates the model's response, parses "
        "its tool call, steps the env, and reads the reward — the agent loop, automated.\n\n"
        "{sft_init_note}"
        "> 💡 This is what makes it *agent* training rather than text fine-tuning: the data is "
        "generated by the policy acting in the env, not read from a file."
    ),
    "base_metrics = await evaluate_model": (
        "### Base vs. trained, on held-out problems\n\n"
        "We run the **same** seeded held-out set through the base model and the trained one and "
        "compare. Using a different seed than training guarantees the problems are unseen; using "
        "the same seed for both models makes the comparison apples-to-apples.\n\n"
        "> 💡 Training reward rising is necessary but not sufficient — this held-out check is the "
        "honest measure of whether the agent generalized."
    ),
    "fine_tuned = AutoModelForCausalLM.from_pretrained(GRPO_OUT": (
        "### Play full games with the trained agent\n\n"
        "Wordle is multi-turn, so the faithful evaluation is to **play complete games**: the agent "
        "guesses, reads the env's letter feedback, and guesses again until it wins or runs out of "
        "turns. Win rate over several games is the agentic analogue of held-out accuracy.\n\n"
        "> 💡 A single-shot accuracy number can't capture multi-turn behavior — you have to let the "
        "agent actually play."
    ),
}


def inject_teaching_cells(cells, has_sft):
    """Insert a deep teaching markdown before each keyed code cell that isn't already
    preceded by a markdown cell. SFT-specific teaching cells are skipped when the
    notebook has no SFT step (e.g. Wordle). Returns a new cell list."""
    # SFT-only teaching keys: skip entirely when has_sft is False.
    sft_only = {"TEACHER_PROVIDER = os.environ.get"}
    sft_init_note = (
        "We warm-start from this run's SFT checkpoint when one exists, else from the base "
        "model.\n\n" if has_sft else
        "Here we train **directly from the base model** — no SFT warm-start — so GRPO learns "
        "purely from the environment reward.\n\n"
    )
    out = []
    for c in cells:
        prev_is_md = bool(out) and out[-1]["cell_type"] == "markdown"
        if c["cell_type"] == "code" and not prev_is_md:
            s = _src(c)
            for key, teach in TEACHING_CELLS.items():
                if key in s:
                    if key in sft_only and not has_sft:
                        break  # skip SFT-only teaching in non-SFT notebooks
                    out.append(md(teach.replace("{sft_init_note}", sft_init_note)))
                    break
        out.append(c)
    return out


# ---- build one target variant -------------------------------------------------
def build_variant(src_cells, meta, target, env_url, pip_lines, out_name):
    cells = [json.loads(json.dumps(c)) for c in src_cells]  # deep copy

    # Detect a REAL SFT step from the source body (the SFTTrainer code), not from prose.
    has_sft = any("SFTTrainer" in _src(c) for c in cells)

    # 1) drop the shared pick-a-lane recipe cell + the old title + old runtime-targets blurb
    kept = []
    for c in cells:
        s = _src(c)
        if RECIPE_SENTINEL in s:
            continue  # drop pick-a-lane
        if c["cell_type"] == "markdown" and s.lstrip().startswith("# 0") and "·" in s.split("\n")[0]:
            continue  # drop old title (we prepend our own)
        if c["cell_type"] == "markdown" and "Runtime targets" in s:
            continue  # drop old runtime-targets blurb
        kept.append(c)

    # 2) enrich section headers (educational)
    enrich_headers(kept)

    # 2b) inject deep per-code-cell teaching markdown (HF-Cookbook depth)
    kept = inject_teaching_cells(kept, has_sft)

    # 2c) normalize the Recap: fix SMOKE=False -> SMOKE=0/1 wording; drop any nb01-specific
    #     held-out-seed line leaked into a non-SFT (Wordle) recap.
    for c in kept:
        if c["cell_type"] == "markdown" and _src(c).lstrip().startswith("## Recap"):
            s = _src(c)
            if not has_sft:
                # remove the appended held-out-seed paragraph (Wordle is evaluated by play)
                s = re.sub(r"\n+The honest question is whether the agent solves problems.*$", "\n",
                           s, flags=re.DOTALL)
            c["source"] = s.splitlines(keepends=True)

    # 2d) global wording normalization across ALL copied cells (inherited from source):
    #     - SMOKE=False -> SMOKE=0 (consistent with the SMOKE=0/1 convention)
    #     - soften result-claim phrasing flagged by review (method, not conclusion)
    TEXT_FIXES = [
        ("SMOKE=False", "SMOKE=0"),
        ("a far better GRPO starting point",
         "a more prepared GRPO starting point"),
        ("measured the only way that counts —",
         "measured on"),
    ]
    for c in kept:
        s = _src(c)
        new = s
        for old, rep in TEXT_FIXES:
            new = new.replace(old, rep)
        if new != s:
            c["source"] = new.splitlines(keepends=True)

    # 3) prepend target-specific intro (+ setup cell for colab)
    head = []
    if target == "colab":
        head.append(md(colab_intro(meta, has_sft)))
        head.append(code(colab_setup_cell(env_url)))
    else:  # hfjob
        run_cmd = hfjob_run_cmd(out_name, env_url, pip_lines, with_vllm=False)
        vllm_cmd = hfjob_run_cmd(out_name, env_url, pip_lines, with_vllm=True)
        head.append(md(hfjob_intro(meta, run_cmd, vllm_cmd, has_sft)))

    nb_out = {
        "cells": head + kept,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb_out


def main():
    for src_name, meta in SOURCES.items():
        src = json.loads((NB_DIR / src_name).read_text())
        cells = src["cells"]
        env_url = extract_env_url(cells)
        pip_lines = extract_pip_lines(cells)
        for target in ("colab", "hfjob"):
            out_name = f"{meta['stem']}_{target}.ipynb"
            nb_out = build_variant(cells, meta, target, env_url, pip_lines, out_name)
            (NB_DIR / out_name).write_text(json.dumps(nb_out, indent=1, ensure_ascii=False) + "\n")
            print(f"[OK] wrote {out_name}  ({len(nb_out['cells'])} cells, env={env_url})")


if __name__ == "__main__":
    main()
