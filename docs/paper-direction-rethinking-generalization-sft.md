# Paper Direction: Extending "Rethinking Generalization in Reasoning SFT"

**Date:** 2026-04-22
**Status:** Scoped, parked. Ready to pick up later.
**Source paper:** [arXiv:2604.06628](https://huggingface.co/papers/2604.06628) — Ren et al., "Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability" (Apr 8, 2026)

---

## 1. Source paper summary

**Central claim:** Cross-domain generalization in reasoning SFT is **conditional** — jointly shaped by:

1. **Optimization dynamics** (dip-and-recovery pattern during training)
2. **Data quality / structure** (verified long-CoT traces crucial)
3. **Base-model capability** (stronger models internalize transferable procedural patterns, weaker ones imitate surface verbosity)

**Key empirical findings:**

- **Dip-and-recovery:** Cross-domain performance *initially degrades* before recovering and improving with extended training. Short checkpoint windows underestimate generalization.
- **Data quality matters:** Low-quality solutions broadly hurt generalization; verified long-CoT traces yield consistent cross-domain gains.
- **Model capability is critical:** Stronger models internalize transferable procedural patterns (e.g., backtracking); weaker models imitate surface verbosity.
- **Asymmetric generalization:** Reasoning improves cross-domain while safety degrades.
- **SFT vs RL response-length dynamics:** SFT sharp-increase-then-gradual-decrease; RL continuous-increase.

**Datasets and models used by the paper:**

- Math-CoT-20k, Math-NoCoT-20k, Countdown-CoT-20k, Math-CoT-44k-Qwen3-32b (44k queries × 32 responses, with token-level probabilities)
- Teacher: Qwen3-32B
- Various base models across a capability range

**Methodology:** Multi-checkpoint evaluation across training steps to observe dip-and-recovery; controlled comparison across data quality, data structure, and base model capability.

---

## 2. Four candidate directions from our artifacts

### Direction 1 — "Small models dip harder" (capability axis extension)
**Claim:** Dip-and-recovery is *deeper and later* for small models. Our pipeline spans 350M → 9B across 4 model families (LFM2.5, Qwen3.5, Gemma4, Carnice) — finer capability axis than the paper used.
**Scope:** Full paper.
**Compute:** ~2 days of lm-eval compute across ~6 models × 3-4 benchmarks × multiple checkpoints.
**Status:** User's preferred direction. Blocked on checkpoint retention (see §4).

### Direction 2 — "Teacher asymmetry in distillation" (data axis extension)
**Claim:** Different distillation teachers (Claude-Opus, Kimi-K2.5, GLM-5.1) produce systematically different cross-domain transfer profiles. Paper treats teacher choice as fixed; we would isolate teacher effect under controlled base-model and compute.
**Scope:** Full paper.
**Compute:** Train one base model (e.g., Qwen3-4B) on each of 3 teacher datasets at matched compute (~3-6h each on b2). Plus eval harness.
**Status:** Feasible. Kimi-K2.5 and GLM-5.1 already mirrored as private datasets on user's HF Hub (see §3). Claude-Opus also already in pipeline.

### Direction 3 — "xLAM vs reasoning SFT — task specificity of dip-and-recovery"
**Claim:** Function-calling SFT (xLAM) shows *different* dip-and-recovery than reasoning SFT. Tests whether the paper's findings generalize beyond reasoning to non-reasoning SFT.
**Scope:** Full paper or strong workshop.
**Compute:** xLAM logs already exist. Needs BFCL/ToolBench + general-reasoning eval suite at 5-8 checkpoints per model.
**Status:** Feasible if we re-train with checkpoint retention.

### Direction 4 — "350M reproducibility" (workshop paper)
**Claim:** LFM2.5-350M runs (finish in ~10 min each) demonstrate the paper's findings hold at the smallest-open-science scale.
**Scope:** Workshop / short arxiv.
**Compute:** Minimal — 5 checkpoint evaluations per 350M model.
**Status:** Lowest effort. Good for NeurIPS/ICML workshops. Still requires checkpoint retention.

---

## 3. Artifacts we already have (inventory as of 2026-04-22)

### Trained merged models on HF Hub (no intermediate checkpoints)
All public, all merged-final-only (no `checkpoint-*/` in the repos):

- `ermiaazarkhalili/LFM2.5-1.2B-SFT-Claude-Opus-Reasoning-Unsloth`
- `ermiaazarkhalili/LFM2.5-350M-SFT-Claude-Opus-Reasoning-Unsloth`
- `ermiaazarkhalili/Qwen3.5-0.8B-SFT-Claude-Opus-Reasoning-Unsloth`
- `ermiaazarkhalili/Gemma4-E2B-SFT-Claude-Opus-Reasoning-Unsloth`
- `ermiaazarkhalili/LFM2.5-1.2B-Function-Calling-xLAM-Unsloth`
- `ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth`
- `ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth`
- `ermiaazarkhalili/Gemma4-E2B-Function-Calling-xLAM-Unsloth`
- `ermiaazarkhalili/Gemma4-E4B-Function-Calling-xLAM-Unsloth`

### Training datasets (private mirrors on user's HF Hub)
Both confirmed private, full-scale, 2026-04-22:

| Dataset | Teacher | Configs | Rows |
|---|---|---|---|
| `ermiaazarkhalili/Kimi-K2.5-Reasoning-1M-Cleaned` | KIMI-K2.5 | General-Distillation, PHD-Science, General-Math, MultilingualSTEM | 844,388 |
| `ermiaazarkhalili/GLM-5.1-Reasoning-1M-Cleaned` | GLM-5.1 | main, PHD-Science, Multilingual-STEM, Math | 746,321 |

Plus the existing `claude-reasoning-distillation` dataset (already in pipeline, Claude-Opus teacher).

This gives us a **3-teacher axis** for Direction 2 with minimal additional prep.

### GRPO-run checkpoints on `/scratch/ermia/outputs/` (research-grade density)
GRPO runs retained intermediate checkpoints (unlike SFT Unsloth runs). Present:

- `granite-4.0-micro-grpo-numinamath-cot-33490409` — checkpoints at 4000, 4500, 5000
- `granite-4.0-micro-grpo-numinamath-cot-32957809` — checkpoints at 1500, 2000, 2500
- `smollm3-3b-grpo-numinamath-cot-33362919`
- `llama-3.2-3b-instruct-grpo-numinamath-cot-33388484`
- `deepseek-r1-distill-qwen-1.5b-grpo-dapo-math-17k-processed-32925729`
- `lfm2.5-1.2b-instruct-grpo-numinamath-cot-32957930` and `-32925728`
- `qwen3-0.6b-grpo-numinamath-cot-32336832`, `qwen3-1.7b-grpo-numinamath-cot-32281529`, `qwen3-1.7b-grpo-numinamath-cot-32418124`
- `openmath-nemotron-1.5b-grpo-numinamath-cot-32768548`

Each has multiple `checkpoint-<step>/` directories with `trainer_state.json`.

### Unsloth SFT checkpoints
Only smoke-test `checkpoint-5/` exists for SFT runs. Full training runs went merge → push without retaining `save_steps=500` intermediates.

---

## 4. Critical blocker: intermediate checkpoint retention for SFT

**The paper's dip-and-recovery finding requires multi-checkpoint evaluation at 5-10 points per training run.** Our Unsloth SFT pipeline does not retain those.

**Current behavior** (in every `sft_distillation_*` and `xlam_function_calling_*` Unsloth notebook):
- `save_strategy="steps"`, `save_steps=500` — sets TRL trainer to save intermediates
- But the notebook ends with `model.save_pretrained_merged(...)` + `push_to_hub_merged(...)` which only uploads the merged final weights
- Intermediate `checkpoint-500/`, `checkpoint-1000/` etc. are written to scratch but eventually cleaned or overwritten by subsequent runs

**Required changes for a paper-quality SFT run:**

```python
# In the training args cell:
save_strategy="steps",
save_steps=100,           # was 500 — need finer granularity for dip-and-recovery
save_total_limit=-1,      # NEW: retain ALL checkpoints (not just the last 3)
output_dir="/scratch/<user>/paper-artifacts/<model>-<dataset>/",  # stable path
```

**Disk-space math:**
- 1.2B bf16 ≈ 2.4 GB per checkpoint × 12 checkpoints × 6 models ≈ 170 GB
- 4-8B ≈ 8-16 GB per checkpoint × 12 × 2 models ≈ 200-400 GB
- Total budget for Direction 1: ~500 GB on `/scratch/ermia/` — fits comfortably

**Alternative pivot that avoids re-running SFT:**
- Shift the paper angle to **SFT vs GRPO dip-and-recovery comparison** — our GRPO runs *already* have checkpoint density. The paper's Sec 5 calls out SFT/RL asymmetry but doesn't do a head-to-head same-base-model comparison. That's a direct novel contribution.

---

## 5. Recommended sequence when we resume

1. **Decide paper angle:**
   - Direction 1 (capability axis, re-run SFT with checkpoint retention), or
   - Pivot to SFT-vs-GRPO dip-and-recovery (use existing GRPO checkpoints + new SFT runs)
2. **Patch training notebooks** for checkpoint retention (`save_total_limit=-1`, finer `save_steps`, stable `output_dir`)
3. **Build eval harness** using lm-eval-harness or similar:
   - In-domain: GSM8K, Hendrycks MATH
   - Out-of-domain: MMLU, BBH, HumanEval
   - (xLAM angle: add BFCL, ToolBench)
4. **Pilot with 350M-1.2B** models first (cheap, fast feedback)
5. **Scale up** to 4B/8B once pilot confirms signal
6. **Analysis + writeup**: dip-and-recovery curves per model, cross-domain matrix, statistical tests

---

## 6. References

- Paper: <https://huggingface.co/papers/2604.06628>
- arXiv: 2604.06628 (April 8, 2026)
- Title: *Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability*
- Authors: Qihan Ren, Peng Wang, Ruikun Cai, Shuai Shao, Dadi Guo, Yuejin Xie, Yafu Li, Quanshi Zhang, Xia Hu, Jing Shao, Dongrui Liu
