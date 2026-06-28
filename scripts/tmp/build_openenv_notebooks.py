#!/usr/bin/env python3
"""Author the OpenEnv agent-training notebooks under notebooks/openenv/.

Every code cell here is faithful to the LITERAL HF tutorial source (curl'd raw
JSON, verified 2026-06-26), made PORTABLE: no DRAC paths, no `module load`, no
`hf_unsloth`, self-installing deps, auth via notebook_login()/HF_TOKEN, env via a
BASE_URL variable, GPU auto-detect with a model-size fallback. A separate SLURM
wrapper (jobs/openenv-*.sh) runs them headless on DRAC in a fresh venv.

Notebooks produced:
  00_openenv_quickstart.ipynb           CPU-only: concepts + connect + manual steps
  01_reasoning_gym_sft_then_grpo.ipynb  full SFT-warmup -> GRPO -> agent eval
  02_wordle_agentic_grpo.ipynb          multi-turn agentic GRPO + play-the-game eval

Run:  python3 scripts/tmp/build_openenv_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/project/6014832/ermia/HF-TRL")
OUT = ROOT / "notebooks" / "openenv"
OUT.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def write_nb(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT / name
    path.write_text(json.dumps(nb, indent=1))
    # AST-validate every code cell so we never ship a syntax-broken notebook.
    import ast

    bad = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # Strip Jupyter line/cell magics + shell escapes — valid in notebooks, not in
        # bare-Python AST. We validate the surrounding Python, not the magics.
        py = "\n".join(
            "" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.splitlines()
        )
        try:
            ast.parse(py)
        except SyntaxError as e:
            bad.append((i, e))
    if bad:
        for i, e in bad:
            print(f"  [SYNTAX] cell {i}: {e}")
        raise SystemExit(f"[FAIL] {name}: {len(bad)} cell(s) failed AST parse")
    print(f"[OK] {name}: {len(cells)} cells, all code cells AST-clean")


# Shared portable preamble cells -------------------------------------------------

PORTABILITY_HEADER = """
> **Runtime targets — this notebook runs anywhere.** Colab (A100/L4/T4), a fresh
> local GPU box, or an HPC cluster via the SLURM runner in `jobs/`. It self-installs
> its dependencies, authenticates with `notebook_login()` (or `HF_TOKEN`), connects
> to an environment by **URL** (a variable you can repoint to your own Space), and
> auto-detects the GPU with a model-size fallback. There are **no hard-coded paths**.
"""

AUTH_CELL = """
# --- Authenticate with the Hugging Face Hub (portable) -------------------------
# Prefers an already-set HF_TOKEN (CI / SLURM / Colab secret); falls back to the
# interactive widget. Never hard-codes a token.
import os

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])
    print("Authenticated via HF_TOKEN.")
else:
    try:
        from huggingface_hub import notebook_login

        notebook_login()
    except Exception as exc:  # non-interactive without a token
        print(f"notebook_login unavailable ({exc}); set HF_TOKEN before running.")
"""

WHOAMI_CELL = """
# Resolve your Hub username automatically — no hard-coded usernames downstream.
from huggingface_hub import whoami

try:
    HF_USERNAME = whoami()["name"]
    print(f"Hub user: {HF_USERNAME}")
except Exception as exc:
    HF_USERNAME = os.environ.get("HF_USERNAME", "")
    print(f"Could not resolve whoami() ({exc}); using HF_USERNAME={HF_USERNAME!r}.")
"""

GPU_DETECT_CELL = """
# --- Auto-detect compute + pick a model that fits ------------------------------
# Larger model only when there is VRAM headroom; otherwise fall back. You can
# override by setting MODEL_NAME in the environment before launching.
import torch

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    DEVICE = "cuda"
else:
    vram_gb = 0.0
    DEVICE = "cpu"

DEFAULT_MODEL = "Qwen/Qwen3-1.7B" if vram_gb >= 24 else "Qwen/Qwen3-0.6B"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
print(f"device={DEVICE}  vram={vram_gb:.1f} GB  ->  MODEL_NAME={MODEL_NAME}")
"""


# ==============================================================================
# Notebook 00 — quickstart (CPU-only, connect + manual steps)
# ==============================================================================
def build_00() -> None:
    cells = [
        md("""
# 00 · OpenEnv Quickstart — connect, step, read reward (CPU-only)

**What is OpenEnv?** A standard for *execution environments* an LLM agent acts in.
Each environment exposes a Gymnasium-style surface — `reset()` to start an episode,
`step(action)` to act, and a `state` — served over **HTTP** and packaged so it can run
as a **Hugging Face Space**. The agent never holds the env in-process; it talks to a
server. That decoupling is what lets the same env drive training (TRL) and evaluation.

This first notebook uses **no GPU and trains nothing**. You will connect to a hosted
environment, drive it by hand with a few `step()` calls, and read the reward it returns.
That is the whole contract the trainer automates later (notebooks 01 and 02).
"""),
        md(PORTABILITY_HEADER),
        md("## 1 · Install\n\nJust the client libraries and the environment package. The env package is a thin\nclient pulled directly from its Space; `--no-deps` keeps it from dragging in a heavy\nserver-side stack you do not need on the client."),
        code("""
# reasoning_gym is our example env; its client ships from the Space.
%pip install -q openenv "transformers>=5.3.0"
%pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym
"""),
        code("""
# notebooks drive async env clients through a sync wrapper; nest_asyncio lets the
# event loop run inside IPython.
import nest_asyncio

nest_asyncio.apply()
"""),
        md("## 2 · Connect to a hosted environment\n\nWe point the client at a **URL**. The default is the tutorial's hosted Space; set\n`ENV_BASE_URL` in your environment to use your own duplicated Space instead. The\n`.sync()` wrapper turns the async client into blocking calls so we can step it inline."),
        code("""
import os

# Repoint this to your own Space (duplicate the hosted one) for real workloads —
# hosted tutorial Spaces are low-concurrency.
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://sergiopaniego-reasoning-gym.hf.space")
print(f"Connecting to: {ENV_BASE_URL}")

from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv

env = ReasoningGymEnv(base_url=ENV_BASE_URL).sync()
"""),
        md("## 3 · `reset()` — start an episode\n\n`reset()` configures the procedural task (here `chain_sum`: a chain of integer\nadditions) and returns the first observation. *Procedural* means the env generates a\nfresh problem each episode, so a trained model must **generalize** over the family\nrather than memorize fixed items."),
        code("""
DATASET_CONFIG = {"min_terms": 2, "max_terms": 3, "min_digits": 2, "max_digits": 2}

result = env.reset(
    dataset_name="chain_sum",
    dataset_config=DATASET_CONFIG,
    seed=0,
    size=10,
)
print("Question:", result.observation.question)
"""),
        md("## 4 · `step(action)` — act and read the reward\n\nThe env exposes an `answer` tool. We submit an answer as a `ReasoningGymAction` and\nthe env returns the score and the correct answer. This scalar reward is the *only*\nsignal GRPO needs in notebook 01."),
        code("""
# Solve it correctly to see reward = 1.0. (We cheat here by evaluating the printed
# arithmetic; a real agent would reason it out.)
import re

nums = [int(n) for n in re.findall(r"-?\\d+", result.observation.question)]
my_answer = str(sum(nums)) if nums else "0"

step = env.step(ReasoningGymAction(answer=my_answer))
print(f"submitted={my_answer}")
print(f"score={step.observation.score}  correct={step.observation.correct_answer}")
"""),
        md("## 5 · A few episodes by hand\n\nAfter the first configured `reset()`, calling `reset()` with no args advances to the\nnext question in the same dataset iterator. This loop is exactly what the trainer runs\nper rollout — just automated and batched."),
        code("""
score_sum, n = 0.0, 5
for i in range(n):
    r = env.reset() if i else result  # reuse the first question we already pulled
    q = r.observation.question
    nums = [int(x) for x in re.findall(r"-?\\d+", q)]
    ans = str(sum(nums)) if nums else "0"
    s = env.step(ReasoningGymAction(answer=ans))
    print(f"[{i}] {q!r:55s} -> ans={ans:>5s}  score={s.observation.score}")
    score_sum += float(s.observation.score or 0.0)

print(f"\\nmean score over {n} episodes: {score_sum / n:.2f}")
"""),
        md("""
## What you just learned

- An OpenEnv environment is an **HTTP server** with a `reset` / `step` contract; the
  client is a thin wrapper you connect to by URL.
- The env returns a **scalar reward** per step. That is the entire interface RL needs.
- Everything here was **manual**. In notebook 01, TRL's `GRPOTrainer` does this loop
  for you via the `environment_factory` pattern — creating an env per rollout, parsing
  the model's tool call, stepping, and collecting the reward.

**Next:** `01_reasoning_gym_sft_then_grpo.ipynb` — collect teacher rollouts, SFT
warm-start, then GRPO, then test the trained agent against held-out problems.
"""),
    ]
    write_nb("00_openenv_quickstart.ipynb", cells)


# ==============================================================================
# Notebook 01 — reasoning_gym: SFT warm-start -> GRPO -> agent eval
# ==============================================================================
def build_01() -> None:
    cells = [
        md("""
# 01 · Reasoning-Gym — SFT warm-start → GRPO → evaluate the agent

This is the **full, explained pipeline** for training an agent with OpenEnv + the HF
ecosystem, faithful to the canonical HF tutorials (Sergio Paniego / Ben Burtenshaw):

1. **Collect** reward-labeled rollouts from a *teacher* model running in the env.
2. **Filter** to the successful ones (`reward == 1.0`) and **SFT** a student on them —
   a warm-start so GRPO does not begin from random tool-calling behavior.
3. **GRPO** (Group Relative Policy Optimization): the env returns a scalar reward per
   rollout, and GRPO ranks rollouts of the same prompt against each other. Value-free,
   so the env's reward is the *only* signal it needs.
4. **Evaluate** the trained agent on a **held-out seed** and report accuracy +
   format-compliance, base vs. trained.

> **Where does a "trace dataset" (e.g. Fable) fit?** It does **not** replace the env.
> OpenEnv's training data is **reward-labeled rollouts the environment generates**, tied
> to *this env's* tool surface (`answer`). Generic imitation traces are not graded by the
> env and do not exercise its tools, so they are not a substitute. A trace corpus could
> only serve as an optional SFT warm-start *if* its tool surface matched the env's —
> which Fable's (Claude-Code-style) does not. We therefore collect our own rollouts below.
"""),
        md(PORTABILITY_HEADER),
        md("""## 0 · Install

`environment_factory` lives in current TRL and needs `transformers>=5.3.0`. The env
client ships from its Space. `trackio` gives live training charts; `openai` is only
needed if you choose the OpenAI teacher in the collect step (optional)."""),
        code("""
%pip install -q trl openenv "transformers>=5.3.0" trackio openai jmespath nest_asyncio datasets
%pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym
"""),
        code("""
import nest_asyncio

nest_asyncio.apply()
"""),
        code(AUTH_CELL),
        code(WHOAMI_CELL),
        code(GPU_DETECT_CELL),
        code("""
# --- Run knobs (smoke vs. full) ------------------------------------------------
# SMOKE keeps everything tiny so the whole notebook runs in minutes on one GPU to
# prove the pipeline end-to-end. Set SMOKE=False (or env SMOKE=0) for a real run.
SMOKE = os.environ.get("SMOKE", "1") not in ("0", "false", "False")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://sergiopaniego-reasoning-gym.hf.space")
DATASET_NAME = "chain_sum"
DATASET_CONFIG = {"min_terms": 2, "max_terms": 3, "min_digits": 2, "max_digits": 2}

N_EPISODES_COLLECT = 30 if SMOKE else 300     # teacher rollouts for SFT
GRPO_MAX_STEPS = 5 if SMOKE else 150          # GRPO optimisation steps
N_EVAL = 10 if SMOKE else 50                  # held-out eval problems
print(f"SMOKE={SMOKE}  collect={N_EPISODES_COLLECT}  grpo_steps={GRPO_MAX_STEPS}  eval={N_EVAL}")
print(f"env={ENV_BASE_URL}")
"""),
        md("""## 1 · (Optional) Deploy your own environment Space

Hosted tutorial Spaces are **low-concurrency** — fine for this notebook, a bottleneck
for a real run where the trainer spins up many parallel envs. For production, duplicate
the Space to your own account (`openenv push`, or the Hub "Duplicate this Space" button)
and set `ENV_BASE_URL` to it. Left as a documented step so the notebook stays runnable
out-of-the-box against the hosted default."""),
        code("""
# Uncomment to deploy your own copy (requires Docker locally OR use the Hub UI button):
#   !openenv push sergiopaniego/reasoning_gym --to {HF_USERNAME}/reasoning_gym
# then:  ENV_BASE_URL = f"https://{HF_USERNAME.replace('_','-')}-reasoning-gym.hf.space"
print("Using ENV_BASE_URL =", ENV_BASE_URL)
"""),
        md("""## 2 · Phase A — collect teacher rollouts (the "dataset")

We run a **teacher** model through the env with the OpenEnv *harness*, which records
each episode's messages and the env's reward. This is the real OpenEnv data-generation
step — `CollectRunner` drives the teacher, `RolloutSerializer` writes episodes to disk,
`push_to_hf_hub` publishes them as a dataset.

The teacher is a **parameter**. The tutorial uses OpenAI `gpt-5-mini`; for a fully-open,
no-API-key path you can point `create_llm_client` at any OpenAI-compatible endpoint
(e.g. a TGI/vLLM server). Set `TEACHER_PROVIDER` / `TEACHER_MODEL` to choose."""),
        code("""
SYSTEM_PROMPT = \"\"\"You are a careful arithmetic assistant.

You will be given a chain of integer additions. Compute the result and submit it as a single number.

Rules:
1. Read the question carefully.
2. Use the tool `answer` exactly once with your final number.
3. The answer must be a single integer with no units or explanation.
\"\"\"
"""),
        code("""
# Teacher is configurable. Tutorial-faithful default is OpenAI gpt-5-mini, BUT a teacher
# requires a reachable endpoint: an OpenAI key, OR a self-hosted OpenAI-compatible URL.
# If neither is available we cannot collect — fail LOUD and early with a clear message
# rather than deep inside CollectRunner. (Set TEACHER_BASE_URL to a TGI/vLLM server for a
# fully-open, no-key path.)
TEACHER_PROVIDER = os.environ.get("TEACHER_PROVIDER", "openai")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "gpt-5-mini")
TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL")  # set for a self-hosted endpoint

_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
TEACHER_AVAILABLE = bool(TEACHER_BASE_URL) or (TEACHER_PROVIDER == "openai" and _has_openai_key)
if not TEACHER_AVAILABLE:
    print(
        "[skip-collect] No teacher endpoint available "
        "(set OPENAI_API_KEY, or TEACHER_BASE_URL for an OpenAI-compatible server).\\n"
        "Phase A (collect+SFT) will be SKIPPED; GRPO will cold-start from the base model.\\n"
        "This keeps the notebook runnable end-to-end with zero external keys."
    )
"""),
        code("""
# Collect only if a teacher is reachable. Otherwise we skip straight to GRPO (cold-start),
# so the notebook always runs top-to-bottom.
RUN_SFT = TEACHER_AVAILABLE
correct = []  # populated by Phase A when it runs

if TEACHER_AVAILABLE:
    from openenv.core.harness import HarnessRunLimits, MCPHarnessAdapter
    from openenv.core.harness.collect import (
        CollectRunner,
        RolloutSerializer,
        build_model_step,
        push_to_hf_hub,
    )
    from openenv.core.llm_client import create_llm_client
    from reasoning_gym_env.client import ReasoningGymEnv
    from reasoning_gym_env.harness import ReasoningGymSessionFactory

    _client_kwargs = {"provider": TEACHER_PROVIDER, "model": TEACHER_MODEL, "max_tokens": 1024}
    if TEACHER_PROVIDER == "openai" and _has_openai_key:
        _client_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    if TEACHER_BASE_URL:
        _client_kwargs["base_url"] = TEACHER_BASE_URL

    llm_client = create_llm_client(**_client_kwargs)
    model_step = build_model_step(llm_client, system_prompt=SYSTEM_PROMPT)

    factory = ReasoningGymSessionFactory(
        lambda: ReasoningGymEnv(base_url=ENV_BASE_URL),
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
    )
    serializer = RolloutSerializer("./rollouts")
    runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=serializer,
        limits=HarnessRunLimits(max_turns=9),
    )
    result = runner.run(model_step=model_step, num_episodes=N_EPISODES_COLLECT)
    print(
        f"collected={result.num_collected} dropped={result.num_dropped} "
        f"avg_reward={result.avg_reward:.3f} success_rate={result.success_rate:.0%}"
    )
"""),
        code("""
# Publish the rollouts to the Hub only when authenticated (optional, for sharing/reuse).
# The SFT step reads from the LOCAL ./rollouts dir, so a Hub push is never required to run.
if TEACHER_AVAILABLE and HF_USERNAME:
    try:
        url = push_to_hf_hub(output_dir="./rollouts", repo_id=f"{HF_USERNAME}/chain-sum-rollouts")
        print(f"Rollouts dataset: {url}")
    except Exception as exc:
        print(f"[warn] Hub push skipped ({exc}); continuing from local ./rollouts.")
"""),
        md("## 3 · Filter + format the rollouts for SFT\n\nKeep only episodes where the teacher was **correct** (`reward == 1.0`), and rewrite the\nassistant tool call into the `<tool_call>{json}</tool_call>` text form the student learns\nto emit. We strip the env's `tool` responses — SFT only supervises the assistant turn.\nWe read rollouts from the **local** `./rollouts` directory (no Hub round-trip needed)."),
        code("""
import json

from datasets import load_dataset

if RUN_SFT:
    # Load from local disk — portable, no Hub auth required.
    ds = load_dataset("json", data_files="./rollouts/*.jsonl", split="train")
    raw_rollouts = list(ds)
    print(f"loaded {len(raw_rollouts)} episodes from ./rollouts")
else:
    raw_rollouts = []
    print("Phase A skipped (no teacher) — no rollouts to format.")


def to_chat_messages(record):
    converted = []
    for msg in record["messages"]:
        if msg["role"] == "tool":
            continue  # SFT supervises only the assistant turn
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            tool_call_text = (
                "<tool_call>\\n"
                + json.dumps({"name": "answer", "arguments": {"answer": args.get("answer", "")}})
                + "\\n</tool_call>"
            )
            converted.append({"role": "assistant", "content": tool_call_text})
        else:
            converted.append(msg)
    return {"messages": converted, "reward": record["reward"]}


import re as _re


def _slug(s: str) -> str:
    # HF/trackio Space ids prefer hyphen-lowercase; underscores get normalised in the
    # Space subdomain, so build a clean slug up front (matches the canonical naming).
    return _re.sub(r"-+", "-", _re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")


_RUN_TAG = _slug(f"reasoning-gym-{DATASET_NAME}-{MODEL_NAME.split('/')[-1]}")
SFT_OUT = f"{_RUN_TAG}-sft"

rollouts = [to_chat_messages(r) for r in raw_rollouts]
correct = [r for r in rollouts if r["reward"] == 1.0]
if RUN_SFT:
    print(f"correct: {len(correct)} / {len(rollouts)} ({len(correct) / max(len(rollouts),1):.1%})")
    if not correct:
        # No usable teacher rollouts -> skip SFT rather than abort; GRPO cold-starts.
        RUN_SFT = False
        print("[skip-sft] 0 correct rollouts; GRPO will cold-start from the base model.")
"""),
        md("## 4 · Size the sequence length from the data\n\nWhen SFT runs, rather than guess `max_length` we measure it: tokenize the kept rollouts\nand set the cap at the 99th percentile + a small margin. Data-driven, model-agnostic."),
        code("""
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_SEQ_LEN = 1024  # default if SFT is skipped
if RUN_SFT and correct:
    lengths = []
    for row in correct:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer.encode(text)))
    lengths = np.array(lengths)
    MAX_SEQ_LEN = int(np.percentile(lengths, 99)) + 16
    print(f"p50={np.percentile(lengths,50):.0f} p95={np.percentile(lengths,95):.0f} "
          f"p99={np.percentile(lengths,99):.0f} max={lengths.max()}  -> MAX_SEQ_LEN={MAX_SEQ_LEN}")
else:
    print(f"SFT skipped; MAX_SEQ_LEN default = {MAX_SEQ_LEN}")
"""),
        md("## 5 · Phase A — SFT warm-start\n\n`assistant_only_loss=True` masks the loss to the assistant tokens, so the student learns\n*to produce the tool call*, not to parrot the prompt. The result is a checkpoint that\nalready calls the `answer` tool in the right format — a far better GRPO starting point\nthan the base model. **Runs only when Phase A collected usable rollouts**; otherwise GRPO\ncold-starts from the base model and the notebook continues."),
        code("""
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

if RUN_SFT and correct:
    sft_dataset = Dataset.from_list([{"messages": r["messages"]} for r in correct])
    sft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    sft_config = SFTConfig(
        output_dir=SFT_OUT,
        max_length=MAX_SEQ_LEN,
        num_train_epochs=1 if SMOKE else 3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        assistant_only_loss=True,
        push_to_hub=not SMOKE,
    )
    sft_trainer = SFTTrainer(
        model=sft_model,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )
    sft_trainer.train()
    sft_trainer.save_model(SFT_OUT)
    if not SMOKE:
        sft_trainer.push_to_hub(commit_message="SFT warm-up on reasoning_gym chain_sum")
    print(f"SFT checkpoint -> {SFT_OUT}")
else:
    print("SFT skipped — GRPO will cold-start from", MODEL_NAME)
"""),
        md("""## 6 · Phase B — GRPO

Now the RL phase. The pieces, all verified from the canonical walkthrough:

- **An env wrapper class.** Its public, documented methods become the agent's *tools*.
  Here `answer(...)` is the single tool; `reset()` pulls the next question. The class
  tracks `self.reward` / `self.done` so the trainer can read the outcome.
- **A reward function** that just reads `env.reward` off each rollout's env instance.
- **A dummy prompt dataset** whose only job is to set the episode count — the *real*
  prompts come from `env.reset()` inside the factory.
- **`environment_factory=<class>`** on `GRPOTrainer`: the trainer creates one env per
  rollout, generates the completion, parses the tool call, steps the env, and collects
  the reward — the entire loop from notebook 00, automated."""),
        code("""
import random

from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv


class ReasoningGymTrainEnv:
    \"\"\"One rollout = one question -> one `answer` tool call -> done.\"\"\"

    DATASET_SIZE = 1000

    def __init__(self):
        self.client = ReasoningGymEnv(base_url=ENV_BASE_URL).sync()
        self._dataset_seed = random.randint(0, 2**31 - 1)  # per-instance: parallel envs diverge
        self._initialized = False
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str:
        if not self._initialized:
            result = self.client.reset(
                dataset_name=DATASET_NAME,
                dataset_config=DATASET_CONFIG,
                seed=self._dataset_seed,
                size=self.DATASET_SIZE,
            )
            self._initialized = True
        else:
            result = self.client.reset()  # next question; re-sending config would rewind to 0
        self.reward = 0.0
        self.done = False
        return result.observation.question

    def answer(self, answer: str) -> str:
        \"\"\"Submit the final answer for the current question.

        Args:
            answer: The agent's answer (parsed as a number server-side).

        Returns:
            Feedback string with the score and the correct answer.
        \"\"\"
        if self.done:
            raise ValueError("Episode is already finished.")
        result = self.client.step(ReasoningGymAction(answer=str(answer)))
        self.reward = float(result.observation.score or 0.0)
        self.done = True
        return f"score={self.reward} correct={result.observation.correct_answer}"


def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
"""),
        code("""
from datasets import Dataset

# Dummy prompts: count == number of rollout episodes. Real prompts come from reset().
N_PROMPTS = 50 if SMOKE else 1000
grpo_dataset = Dataset.from_dict(
    {"prompt": [[{"role": "user", "content": SYSTEM_PROMPT}] for _ in range(N_PROMPTS)]}
)
"""),
        code("""
from trl import GRPOConfig

GRPO_OUT = f"{_RUN_TAG}-grpo"

# Logging backend is a knob: "trackio" (default, live charts) or "none" for a fully
# offline/headless run with no Space setup. trackio_space_id is only set when using trackio.
REPORT_TO = os.environ.get("REPORT_TO", "trackio")
_report_kwargs = {"report_to": REPORT_TO}
if REPORT_TO == "trackio":
    _report_kwargs["trackio_space_id"] = GRPO_OUT

grpo_config = GRPOConfig(
    num_train_epochs=1,
    max_steps=GRPO_MAX_STEPS,
    learning_rate=1e-6,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=1,
    warmup_steps=min(10, GRPO_MAX_STEPS),
    optim="adamw_torch",
    max_grad_norm=1.0,
    num_generations=2,
    max_completion_length=256,
    log_completions=True,
    num_completions_to_print=2,
    chat_template_kwargs={"enable_thinking": False},
    output_dir=GRPO_OUT,
    logging_steps=1 if SMOKE else 10,
    gradient_checkpointing=True,
    save_strategy="no",
    push_to_hub=not SMOKE,
    **_report_kwargs,
    # vLLM left OFF in-notebook: its init breaks under IPython. Enable in the SLURM/script
    # path with use_vllm=True, vllm_mode="colocate".
)
"""),
        code("""
from trl import GRPOTrainer

# Warm-start GRPO from the SFT checkpoint when available; else from base.
import os as _os

# Warm-start from THIS run's SFT checkpoint only. Guard on RUN_SFT so a stale SFT dir
# left in the cwd by a PRIOR run does not silently warm-start a cold-start run.
GRPO_INIT = SFT_OUT if (RUN_SFT and _os.path.isdir(SFT_OUT)) else MODEL_NAME
print(f"GRPO initialising from: {GRPO_INIT}")

grpo_trainer = GRPOTrainer(
    model=GRPO_INIT,
    reward_funcs=reward_func,
    train_dataset=grpo_dataset,
    args=grpo_config,
    environment_factory=ReasoningGymTrainEnv,
)
grpo_trainer.train()
grpo_trainer.save_model(GRPO_OUT)
if not SMOKE:
    grpo_trainer.push_to_hub(commit_message="GRPO fine-tune on reasoning_gym chain_sum")
"""),
        md("## 7 · Training signal — reward delta\n\nThe quickest sanity check: did mean reward rise over training? We read it from the\ntrainer's log history (first-5 vs last-5 logged steps)."),
        code("""
import statistics

rewards = [log["reward"] for log in grpo_trainer.state.log_history if "reward" in log]
if len(rewards) < 5:
    print(f"Only {len(rewards)} reward logs — increase max_steps / lower logging_steps for a clean delta.")
else:
    initial, final = statistics.mean(rewards[:5]), statistics.mean(rewards[-5:])
    print(f"initial reward (first5): {initial:.2%}")
    print(f"final   reward (last5):  {final:.2%}")
    print(f"delta:                   {(final - initial) * 100:+.2f} pp")
"""),
        md("""## 8 · Test the trained agent against held-out problems

Training reward can rise for the wrong reasons (reward hacking, format luck). The real
question is: **does the trained agent solve held-out problems it never saw?** We evaluate
on a fresh `seed` and report two numbers, **base vs. trained**:

- **accuracy** — fraction of held-out problems answered correctly (the env's score);
- **format-compliance** — fraction where the model emitted a parseable `answer`.

This is the canonical `evaluate_model` harness from the HF SFT-warmup tutorial."""),
        code("""
import re

from transformers import pipeline
from reasoning_gym_env.client import ReasoningGymEnv
from reasoning_gym_env.models import ReasoningGymAction


async def evaluate_model(model_name_or_path, n_eval=N_EVAL, seed=999):
    \"\"\"Held-out accuracy + format-compliance for one model. Higher seed => unseen problems.\"\"\"
    gen = pipeline(
        "text-generation",
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        device_map="auto",
        dtype="auto",
    )
    gen.model.generation_config.max_length = None
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    eval_env = ReasoningGymEnv(base_url=ENV_BASE_URL)

    obs = await eval_env.reset(
        dataset_name=DATASET_NAME, dataset_config=DATASET_CONFIG, seed=seed, size=n_eval
    )

    rewards, format_hits = [], 0
    for i in range(n_eval):
        if i:
            obs = await eval_env.reset()
        question = obs.observation.question
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion = gen(prompt, max_new_tokens=64)[0]["generated_text"][len(prompt):]

        # Accept signed integers — chain_sum results can be negative.
        m = re.search(r'"answer"\\s*:\\s*"?(-?\\d+)"?', completion)
        if m:
            format_hits += 1
            answer = m.group(1)
        else:
            nums = re.findall(r"(-?\\d+)", completion)
            answer = nums[-1] if nums else "0"

        res = await eval_env.step(ReasoningGymAction(answer=answer))
        rewards.append(float(res.observation.score or 0.0))

    await eval_env.close()
    del gen
    return {"accuracy": sum(rewards) / len(rewards), "format_compliance": format_hits / n_eval}
"""),
        code("""
# Compare base vs. the GRPO-trained checkpoint on held-out problems.
base_metrics = await evaluate_model(MODEL_NAME)
trained_metrics = await evaluate_model(GRPO_OUT)

print(f"\\n{'Metric':<22}{'Base':>10}{'Trained':>10}{'Delta':>10}")
print("-" * 52)
for key, label in [("format_compliance", "Format compliance"), ("accuracy", "Accuracy")]:
    b, t = base_metrics[key], trained_metrics[key]
    print(f"{label:<22}{b:>10.1%}{t:>10.1%}{(t - b) * 100:>+9.1f}pp")
"""),
        md("""
## Recap

You ran the complete OpenEnv agent-training loop: **collect → filter → SFT warm-start →
GRPO → held-out evaluation**, all against a live environment, all portable. The trained
agent's quality is measured the only way that counts — accuracy on problems it never saw.

- For a real run: set `SMOKE=False`, deploy your own env Space, and enable vLLM in the
  SLURM path (`jobs/openenv-grpo-reasoning-gym.sh`).
- **Next:** `02_wordle_agentic_grpo.ipynb` applies the same shape to a genuinely
  *multi-turn* agentic environment (6 guesses per episode with stateful feedback).
"""),
    ]
    write_nb("01_reasoning_gym_sft_then_grpo.ipynb", cells)


# ==============================================================================
# Notebook 02 — Wordle: multi-turn agentic GRPO + play-the-game eval
# ==============================================================================
def build_02() -> None:
    cells = [
        md("""
# 02 · Wordle — multi-turn **agentic** GRPO + play-the-game evaluation

Notebook 01's env was single-turn (one question → one answer). **Wordle is genuinely
agentic**: up to **6 guesses** per episode, and each guess returns colour-coded feedback
(GREEN/YELLOW/GRAY) the model must *reason over* to choose its next guess. The episode is
**stateful** and **multi-turn** — the closest OpenEnv shape to a real tool-using agent.

The training mechanics are identical to notebook 01 (`environment_factory` + GRPO), which
is the point: once you have the pattern, swapping the environment is the only change. We
reuse the same SFT→GRPO→evaluate spine and highlight what differs for a multi-turn env.

> **Why Wordle and not a "coding env"?** OpenEnv's coding agent (`opencode_env`) is real
> but needs an **E2B sandbox** *and* an OpenAI-compatible LLM endpoint — two external
> dependencies that break "runs anywhere." Wordle (`textarena`) is multi-turn, agentic,
> needs **no API key**, and is the canonical agentic tutorial. The production coding path
> is documented in the README for when those dependencies are available.
"""),
        md(PORTABILITY_HEADER),
        md("## 0 · Install\n\nThe Wordle env client ships from the `openenv/wordle` Space. `trl[vllm]` is the\ntutorial default; in-notebook we keep vLLM off (IPython init issue) and enable it only\nin the SLURM path."),
        code("""
%pip install -q trl openenv "transformers>=5.3.0" trackio jmespath nest_asyncio datasets
%pip install -q --no-deps git+https://huggingface.co/spaces/openenv/wordle
"""),
        code("""
import nest_asyncio

nest_asyncio.apply()
"""),
        code(AUTH_CELL),
        code(WHOAMI_CELL),
        code(GPU_DETECT_CELL),
        code("""
import os

SMOKE = os.environ.get("SMOKE", "1") not in ("0", "false", "False")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://openenv-wordle.hf.space")
GRPO_MAX_STEPS = 5 if SMOKE else 150
N_EVAL_GAMES = 3 if SMOKE else 20
print(f"SMOKE={SMOKE}  grpo_steps={GRPO_MAX_STEPS}  eval_games={N_EVAL_GAMES}  env={ENV_BASE_URL}")
"""),
        md("## 1 · System prompt — teach the rules and the tool\n\nThe prompt states the Wordle rules, the feedback colour code, and — critically — that\nthe model must call the `guess` tool. The `environment_factory` loop drives tool calls,\nso the model has to know the tool exists."),
        code("""
WORDLE_PROMPT = \"\"\"You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
\"\"\"
"""),
        md("""## 2 · The multi-turn environment class

Two things differ from notebook 01's single-turn env:

- **`reset()` returns the running feedback transcript**, and the env appends new feedback
  each turn. We keep `self._last_full_feedback` and slice out only the *newly appended*
  part so the model sees just the latest result.
- **`done` is set by the env**, not by us — the game ends on a win or after 6 guesses, so
  a single rollout spans multiple `guess` tool calls. The trainer keeps stepping until
  `done=True`.

`guess` is the one tool (public + docstring → auto-discovered). It penalises invalid
moves (reward 0) and otherwise records the env's reward."""),
        code("""
from textarena_env import TextArenaAction, TextArenaEnv


class WordleEnv:
    \"\"\"Multi-turn Wordle rollout: up to 6 `guess` tool calls until done.\"\"\"

    def __init__(self):
        self.client = TextArenaEnv(base_url=ENV_BASE_URL)

    def reset(self, **kwargs) -> None | str:
        result = self.client.reset()
        # Env returns cumulative feedback; store the full text so we can diff each turn.
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        \"\"\"Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'.

        Returns:
            The feedback message from the environment.
        \"\"\"
        if self.done:
            raise ValueError("Game over.")
        result = self.client.step(TextArenaAction(message=guess))
        full = result.observation.messages[0].content
        feedback = full[len(self._last_full_feedback):]  # only the newly appended part
        self._last_full_feedback = full
        self.reward = 0.0 if "You attempted an invalid move" in feedback else result.reward
        self.done = result.done
        return feedback


def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
"""),
        md("## 3 · Dataset + GRPO config\n\nSame dummy-prompt trick (count = episodes). The notable config difference vs. notebook\n01 is a larger `max_completion_length` (multi-turn games are longer) and a bigger\n`gradient_accumulation_steps` per the Wordle tutorial."),
        code("""
from datasets import Dataset

N_PROMPTS = 30 if SMOKE else 3000
dataset = Dataset.from_dict(
    {"prompt": [[{"role": "user", "content": WORDLE_PROMPT}] for _ in range(N_PROMPTS)]}
)
"""),
        code("""
import re as _re

from trl import GRPOConfig

# Hyphen-lowercase slug for clean HF/trackio Space ids.
GRPO_OUT = _re.sub(r"-+", "-", _re.sub(r"[^a-z0-9]+", "-", f"wordle-grpo-{MODEL_NAME.split('/')[-1]}".lower())).strip("-")

# Logging backend knob: "trackio" (default) or "none" for a fully offline/headless run.
REPORT_TO = os.environ.get("REPORT_TO", "trackio")
_report_kwargs = {"report_to": REPORT_TO}
if REPORT_TO == "trackio":
    _report_kwargs["trackio_space_id"] = GRPO_OUT

grpo_config = GRPOConfig(
    num_train_epochs=1,
    max_steps=GRPO_MAX_STEPS,
    learning_rate=1e-6,
    gradient_accumulation_steps=8 if SMOKE else 64,
    per_device_train_batch_size=1,
    warmup_steps=min(10, GRPO_MAX_STEPS),
    optim="adamw_torch",
    max_grad_norm=1.0,
    num_generations=2,
    max_completion_length=1024,           # multi-turn games are longer than chain_sum
    log_completions=True,
    num_completions_to_print=2,
    chat_template_kwargs={"enable_thinking": False},
    output_dir=GRPO_OUT,
    logging_steps=1 if SMOKE else 10,
    save_strategy="no",
    gradient_checkpointing=True,
    **_report_kwargs,
    push_to_hub=not SMOKE,
    # vLLM OFF in-notebook (IPython init). SLURM path: use_vllm=True, vllm_mode="colocate".
)
"""),
        code("""
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=MODEL_NAME,
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=grpo_config,
    environment_factory=WordleEnv,
)
trainer.train()
trainer.save_model(GRPO_OUT)
if not SMOKE:
    trainer.push_to_hub(commit_message="GRPO fine-tune on Wordle (textarena)")
"""),
        md("## 4 · Training signal — reward delta\n\nSame quick check as notebook 01: mean reward, first-5 vs last-5 logged steps."),
        code("""
import statistics

rewards = [log["reward"] for log in trainer.state.log_history if "reward" in log]
if len(rewards) < 5:
    print(f"Only {len(rewards)} reward logs — increase max_steps for a clean delta.")
else:
    initial, final = statistics.mean(rewards[:5]), statistics.mean(rewards[-5:])
    print(f"initial={initial:.2%}  final={final:.2%}  delta={(final - initial) * 100:+.2f}pp")
"""),
        md("""## 5 · Test the trained agent — play full games

For a multi-turn agent, the meaningful test is **playing complete games**: load the
trained model, let it guess turn-by-turn against live feedback, and measure the win rate
over `N_EVAL_GAMES`. This is the canonical `play_wordle` harness, wrapped to score a win
(`env.done and env.reward > 0`) and report aggregate performance — the agentic analogue
of notebook 01's accuracy."""),
        code("""
import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer


def play_one_game(model, tokenizer, verbose=False):
    \"\"\"Play a single Wordle game with the model; return (won: bool, final_reward: float).\"\"\"
    env = WordleEnv()
    obs = env.reset()
    messages = [{"role": "user", "content": WORDLE_PROMPT}]
    if obs:
        messages.append({"role": "user", "content": obs})

    for _turn in range(6):
        if env.done:
            break
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=512)
        text = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        if verbose:
            print(f"  model: {text[:120]}")
        try:
            if "guess" in text and "{" in text:
                args = json.loads(text[text.index("{"):text.rindex("}") + 1])
                args = args.get("arguments", args)
                word = args.get("guess", "")
            else:
                m = re.search(r"\\[([a-zA-Z]{5})\\]", text)
                word = m.group(1) if m else text.strip()[:5]
            feedback = env.guess(f"[{word}]")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": feedback})
        except Exception as exc:  # malformed output ends the game
            if verbose:
                print(f"  parse error: {exc}")
            break

    return (env.done and env.reward > 0), float(env.reward)
"""),
        code("""
# Win-rate of the trained agent over several games.
fine_tuned = AutoModelForCausalLM.from_pretrained(GRPO_OUT, torch_dtype="auto", device_map="auto")
ft_tokenizer = AutoTokenizer.from_pretrained(GRPO_OUT)

wins, total_reward = 0, 0.0
for g in range(N_EVAL_GAMES):
    won, r = play_one_game(fine_tuned, ft_tokenizer, verbose=(g == 0))
    wins += int(won)
    total_reward += r
    print(f"game {g + 1}/{N_EVAL_GAMES}: {'WON' if won else 'lost'}  reward={r:.2f}")

print(f"\\nwin rate: {wins}/{N_EVAL_GAMES} = {wins / N_EVAL_GAMES:.0%}   mean reward={total_reward / N_EVAL_GAMES:.2f}")
"""),
        md("""
## Recap

Same `environment_factory` + GRPO spine as notebook 01, applied to a **multi-turn agentic**
environment. The only real changes were the env class (stateful feedback, env-driven
`done`) and a longer completion budget. The trained agent is judged by **playing full
games** and measuring win rate — the agentic counterpart to held-out accuracy.

This is the template for any OpenEnv environment: define the env wrapper (tools = public
documented methods), a reward function that reads `env.reward`, a dummy prompt dataset for
the episode count, and hand the class to `GRPOTrainer(environment_factory=...)`. Swap the
env, keep the spine.
"""),
    ]
    write_nb("02_wordle_agentic_grpo.ipynb", cells)


if __name__ == "__main__":
    build_00()
    build_01()
    build_02()
    print(f"\n[DONE] notebooks under {OUT}")
