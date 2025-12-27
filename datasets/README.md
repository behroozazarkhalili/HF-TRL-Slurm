# Training Datasets Reference

This folder contains comprehensive documentation for datasets used in SFT and GRPO training.

## Quick Reference

| Type | File | Datasets |
|------|------|----------|
| **GRPO** | [GRPO_DATASETS.md](./GRPO_DATASETS.md) | 7 math reasoning datasets |
| **SFT** | [SFT_DATASETS.md](./SFT_DATASETS.md) | 4 instruction-tuning datasets |

---

## Summary Statistics

### GRPO Datasets (Math Reasoning)

| Dataset | Samples | Best For |
|---------|---------|----------|
| OpenMathInstruct-2 | 14M | Large-scale training |
| NuminaMath-1.5 | 896K | Comprehensive coverage |
| NuminaMath-CoT | 860K | Standard CoT training |
| OpenR1-Math-220k | 450K | Long reasoning chains |
| orca-math-200k | 200K | Quick baseline |
| NuminaMath-TIR | 73K | Tool-integrated reasoning |
| olympiads-ref | 13K | High-quality proofs |

### SFT Datasets (Instruction Following)

| Dataset | Samples | Best For |
|---------|---------|----------|
| OpenOrca | 2.94M | Large-scale training |
| OpenHermes-2.5 | 1M | GPT-4 quality |
| ultrachat_200k | 515K | Multi-turn conversations |
| FineTome-100k | 100K | Quick experiments |

---

## Recommended Combinations

### For Math Models (GRPO)
```
Stage 1: NuminaMath-CoT (baseline)
Stage 2: OpenMathInstruct-2 (scale up)
Stage 3: olympiads-ref (refinement)
```

### For Chat Models (SFT)
```
Stage 1: FineTome-100k (quick test)
Stage 2: ultrachat_200k (production)
```

---

## Files in This Directory

```
datasets/
├── README.md              # This file (index)
├── GRPO_DATASETS.md       # GRPO dataset details with tables
└── SFT_DATASETS.md        # SFT dataset details with tables
```

---

## Last Updated
2025-12-25
