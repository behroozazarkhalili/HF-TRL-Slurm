# OpenEnv agent-training notebooks — run guide (Colab · HF Jobs)

How to run the OpenEnv agent-training notebooks on **Google Colab** or **Hugging Face Jobs**.
Every command below is taken from the notebooks' own cells (env URLs, dependencies, GPU
flavors); nothing is invented.

The notebooks live in [`notebooks/openenv/`](../notebooks/openenv/).

---

## 1 · The notebooks

| Notebook | What it does | Env | Trains? | Min runtime |
|---|---|---|---|---|
| `00_openenv_quickstart.ipynb` | connect → `step()` → read reward | reasoning_gym | No | CPU |
| `01_reasoning_gym_sft_then_grpo.ipynb` | SFT warm-start → GRPO → held-out eval | reasoning_gym | **Yes** | GPU |
| `02_wordle_agentic_grpo.ipynb` | multi-turn agentic GRPO + play-the-game eval | Wordle (textarena) | **Yes** | GPU |

**Which to pick**
- **Learning the API / a laptop or CPU runtime:** `00`.
- **A real training run:** `01` (reasoning_gym) or `02` (Wordle). Colab and HF Jobs both have
  open network egress, so the hosted environment Space works directly — no extra setup.

---

## 2 · The knobs (set as environment variables)

Every notebook reads these from the environment, so you control a run without editing cells:

| Var | Default | Meaning |
|---|---|---|
| `SMOKE` | `1` | `1` = tiny end-to-end proof (minutes). `0` = real run (150 GRPO steps, pushes to Hub). |
| `MODEL_NAME` | auto (`Qwen/Qwen3-1.7B` if VRAM ≥ 24 GB else `Qwen/Qwen3-0.6B`) | base model to train. |
| `ENV_BASE_URL` | per-notebook (see table below) | the environment Space URL. |
| `REPORT_TO` | `trackio` | live charts on a Trackio Space; set `none` for headless. |
| `USE_VLLM` | `0` | `1` = vLLM-accelerated GRPO rollouts (5–10× faster). **Only** in a non-IPython runtime (HF Job / script / fresh Colab) — vLLM init breaks under Jupyter. |
| `HF_TOKEN` | — | write token; needed when `SMOKE=0` pushes the trained model to the Hub. |

Per-notebook `ENV_BASE_URL` defaults:

| Notebook | `ENV_BASE_URL` default |
|---|---|
| `00`, `01` | `https://sergiopaniego-reasoning-gym.hf.space` |
| `02` | `https://openenv-wordle.hf.space` |

`SMOKE=0` scales (from the notebooks): `N_EPISODES_COLLECT 30→300`, `GRPO_MAX_STEPS 5→150`,
`N_EVAL 10→50` (nb01), `N_EVAL_GAMES 3→20` (nb02).

---

## 3 · Google Colab

Open the notebook in Colab → **Runtime → Change runtime type → GPU** (T4/L4/A100; CPU is fine for `00`).

**First cell** — set knobs + authenticate, then **Runtime → Run all**:

```python
# Colab cell 1 — knobs + auth, then Run all.
import os
os.environ["SMOKE"] = "0"                  # full run; omit/"1" for a quick smoke
os.environ["REPORT_TO"] = "trackio"        # live charts; or "none" for headless
# Env URL: use the notebook's default, or point at your own duplicated Space.
os.environ["ENV_BASE_URL"] = "https://sergiopaniego-reasoning-gym.hf.space"  # nb00/nb01
# nb02 (Wordle):  os.environ["ENV_BASE_URL"] = "https://openenv-wordle.hf.space"

from huggingface_hub import notebook_login
notebook_login()                           # paste a token with WRITE access
```

Colab has open egress, so the hosted env Space works directly. The notebook's install cell
handles dependencies. For a longer run on an **A100** Colab you can also enable vLLM *before*
the GRPO config cell:

```python
os.environ["USE_VLLM"] = "1"               # A100 Colab tolerates vLLM colocate
```

---

## 4 · Hugging Face Jobs

`hf jobs run` is *docker-run for HF infra*: the GPU work happens on HF's cloud; you only submit.
**Requires pre-paid credits** on your HF account.

### Setup (once)

```bash
pip install -U huggingface_hub          # provides the `hf` CLI (v1.21+)
hf auth login                           # or rely on a local HF_TOKEN
hf jobs ps -a                           # sanity check: lists your jobs (empty is fine)
```

### A · Run a training notebook on a GPU (nb01, reasoning_gym)

```bash
hf jobs run \
  --flavor a10g-small \
  --timeout 10800 \
  --secrets HF_TOKEN \
  -e SMOKE=0 -e REPORT_TO=trackio \
  -e ENV_BASE_URL=https://sergiopaniego-reasoning-gym.hf.space \
  python:3.12 \
  bash -c "pip install -q papermill && \
           pip install -q trl openenv 'transformers>=5.3.0' trackio openai jmespath nest_asyncio datasets && \
           pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym && \
           papermill 01_reasoning_gym_sft_then_grpo.ipynb out.ipynb"
```

> `papermill` needs the `.ipynb` present in the job's working dir. Easiest options: run the
> command from a clone of `notebooks/openenv/` (the file is uploaded with the job), or `git clone`
> inside the `bash -c` before `papermill`. `--secrets HF_TOKEN` forwards your local token so the
> trained model can push to the Hub.

### B · Full run with vLLM (faster rollouts)

vLLM works on HF Jobs because a Job is **not** an IPython runtime:

```bash
hf jobs run \
  --flavor a100-large \
  --timeout 21600 \
  --secrets HF_TOKEN \
  -e SMOKE=0 -e USE_VLLM=1 -e REPORT_TO=trackio \
  -e ENV_BASE_URL=https://sergiopaniego-reasoning-gym.hf.space \
  python:3.12 \
  bash -c "pip install -q papermill vllm && \
           pip install -q trl openenv 'transformers>=5.3.0' trackio openai jmespath nest_asyncio datasets && \
           pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym && \
           papermill 01_reasoning_gym_sft_then_grpo.ipynb out.ipynb"
```

### C · Wordle notebook (nb02 — different env + client)

```bash
hf jobs run \
  --flavor a10g-small \
  --timeout 10800 \
  --secrets HF_TOKEN \
  -e SMOKE=0 -e REPORT_TO=trackio \
  -e ENV_BASE_URL=https://openenv-wordle.hf.space \
  python:3.12 \
  bash -c "pip install -q papermill && \
           pip install -q trl openenv 'transformers>=5.3.0' trackio jmespath nest_asyncio datasets && \
           pip install -q --no-deps git+https://huggingface.co/spaces/openenv/wordle && \
           papermill 02_wordle_agentic_grpo.ipynb out.ipynb"
```

### D · Quickstart (nb00 — CPU, no GPU)

```bash
hf jobs run \
  --flavor cpu-basic \
  --timeout 1800 \
  -e ENV_BASE_URL=https://sergiopaniego-reasoning-gym.hf.space \
  python:3.12 \
  bash -c "pip install -q papermill && \
           pip install -q openenv 'transformers>=5.3.0' && \
           pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym && \
           papermill 00_openenv_quickstart.ipynb out.ipynb"
```

### Managing jobs

```bash
hf jobs run --detach ...        # background; prints the Job ID only
hf jobs ps                      # running jobs    (ps -a = all)
hf jobs logs   <job_id>         # stream logs
hf jobs inspect <job_id>        # full metadata
hf jobs cancel  <job_id>        # stop a job
hf jobs wait    <job_id>        # block until done (non-zero exit on failure)
```

GPU flavors: `cpu-basic`, `t4-small`, `a10g-small`, `a10g-large`, `a100-large`, … (`hf jobs` docs).
Default timeout is 30 min — always pass `--timeout <seconds>` for training.

---

## 5 · Reading the results

When `SMOKE=0`, a training run produces three artifacts:

1. **Trained model** — pushed to `<your-username>/reasoning-gym-chain-sum-<model>-grpo`
   (public model repo with `model.safetensors`, config, tokenizer, and the logged rollout parquets).
2. **Trackio dashboard** — a Space of the same name (live reward/loss charts), created when
   `REPORT_TO=trackio`.
3. **Held-out eval table** — printed at the end, base vs trained:

   ```
   Metric                          Base   Trained     Delta
   Accuracy (tool-call)            ...       ...       ...
   Accuracy (robust)               ...       ...       ...
   Tool-call format rate           ...       ...       ...
   ```

   - **Accuracy (tool-call)** — env-scored from the model's `<tool_call>` wrapper (the channel it
     was trained on). The faithful metric.
   - **Accuracy (robust)** — last integer after stripping `<think>` spans, vs the env's gold
     answer. Format-agnostic capability probe.
   - **Tool-call format rate** — how often the model emitted a parseable tool call. The gap
     between the two accuracies is the format story.

   The **training reward delta** (first-5 vs last-5 logged steps) is also printed — the primary
   "did GRPO help?" signal, independent of output formatting.

---

## 6 · Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `vLLM` crashes under Jupyter/Colab kernel | vLLM init breaks under IPython → keep `USE_VLLM=0` in-notebook; enable it only in HF Jobs / a script / a fresh A100 Colab runtime. |
| `papermill: file not found` on HF Jobs | the `.ipynb` isn't in the job's working dir → run from a repo clone or `git clone` inside the `bash -c`. |
| Model didn't push to Hub | `SMOKE=1` never pushes (by design); for a real push use `SMOKE=0` **and** `--secrets HF_TOKEN` (HF Jobs) or a logged-in token (Colab). |
| `hf jobs` says no credits | HF Jobs requires pre-paid credits on the account. |
| Eval shows 0% format rate | the base model doesn't emit the `<tool_call>` wrapper without an SFT warm-up; `acc_robust` still shows true capability. Expected on short cold-start runs. |
