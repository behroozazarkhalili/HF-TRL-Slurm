#!/usr/bin/env python3
"""Revert nb01 eval to the canonical HF reference STRATEGY, keeping OUR parameters.

Reference (huggingface/OpenEnv examples/sft_warmup.ipynb, verified from source):
  single-turn pipeline() -> apply_chat_template([system,user]) -> one generate ->
  regex on `"answer": <int>` (sets format_hits) -> env.step(ReasoningGymAction) ->
  accuracy from env score. Returns {accuracy, format_compliance}.

We keep OUR parameters (per user): max_new_tokens=256 (ref uses 64), signed-int
`-?\\d+` (ref uses unsigned \\d+ — chain_sum can be negative). No <tool_call> parser,
no dual acc_toolcall/acc_robust split (those changed the STRATEGY), no interpretive
prose in the notebook.

Edits cells in BOTH notebooks/openenv/01_reasoning_gym_sft_then_grpo.ipynb and the
_DRAC_local copy: the eval-def cell and the eval-table cell (located structurally).

Run:  python3 scripts/tmp/fix_openenv_eval_to_reference.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path("/project/6014832/ermia/HF-TRL/notebooks/openenv")
TARGETS = [
    "01_reasoning_gym_sft_then_grpo.ipynb",
    "01_reasoning_gym_sft_then_grpo_DRAC_local.ipynb",
]

# ---- eval-def cell: reference strategy + our params ----------------------------
EVAL_DEF = r'''import re

from transformers import pipeline
from reasoning_gym_env.client import ReasoningGymEnv
from reasoning_gym_env.models import ReasoningGymAction


async def evaluate_model(model_name_or_path, n_eval=N_EVAL, seed=999):
    """Held-out accuracy + format-compliance for one model (reference strategy).

    Single-turn: render [system, user], generate once, extract the answer, step the env
    once, read the env score. `format_compliance` = fraction emitting a parseable
    `"answer": <int>` (the trained tool-call surface). Higher seed => unseen problems.
    """
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
        completion = gen(prompt, max_new_tokens=256)[0]["generated_text"][len(prompt):]

        # Reference extraction: `"answer": <int>` sets format compliance; else last integer.
        # Signed ints kept (our param): chain_sum results can be negative.
        m = re.search(r'"answer"\s*:\s*"?(-?\d+)"?', completion)
        if m:
            format_hits += 1
            answer = m.group(1)
        else:
            nums = re.findall(r"(-?\d+)", completion)
            answer = nums[-1] if nums else "0"

        res = await eval_env.step(ReasoningGymAction(answer=answer))
        rewards.append(float(res.observation.score or 0.0))

    await eval_env.close()
    del gen
    return {"accuracy": sum(rewards) / len(rewards), "format_compliance": format_hits / n_eval}
'''

# ---- eval-table cell: reference two-metric table, no interpretation ------------
EVAL_TABLE = r'''# Compare base vs. the GRPO-trained checkpoint on held-out problems.
base_metrics = await evaluate_model(MODEL_NAME)
trained_metrics = await evaluate_model(GRPO_OUT)

print(f"\n{'Metric':<22}{'Base':>10}{'Trained':>10}{'Delta':>10}")
print("-" * 52)
for key, label in [("format_compliance", "Format compliance"), ("accuracy", "Accuracy")]:
    b, t = base_metrics[key], trained_metrics[key]
    print(f"{label:<22}{b:>10.1%}{t:>10.1%}{(t - b) * 100:>+9.1f}pp")
'''

# ---- sec-8 markdown: reference framing, no result interpretation ---------------
SEC8_MD = r'''## 8 · Test the trained agent against held-out problems

Run both the base model and the GRPO checkpoint on a held-out `seed` (problems not seen
in training) and report two numbers, base vs. trained:

- **accuracy** — fraction of held-out problems the env scores correct;
- **format compliance** — fraction where the model emitted a parseable `answer`.

This is the reference `evaluate_model` harness from the HF SFT-warmup tutorial.'''


def _src(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def process(path: Path):
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    did = []
    for i, c in enumerate(cells):
        s = _src(c)
        if c["cell_type"] == "markdown" and "Test the trained agent against held-out" in s:
            cells[i] = md_cell(SEC8_MD); did.append(f"md@{i}")
        elif c["cell_type"] == "code" and "def evaluate_model" in s:
            cells[i] = code_cell(EVAL_DEF); did.append(f"def@{i}")
        elif c["cell_type"] == "code" and "base_metrics = await evaluate_model" in s:
            cells[i] = code_cell(EVAL_TABLE); did.append(f"table@{i}")
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"[OK] {path.name}: replaced {did}")


def main():
    for fn in TARGETS:
        p = NB_DIR / fn
        if not p.exists():
            print(f"[skip] {fn} not found"); continue
        process(p)


if __name__ == "__main__":
    main()
