# Base Models Reference

This folder contains comprehensive documentation for base models used in SFT and GRPO training.

## Quick Reference

| Type | File | Models |
|------|------|--------|
| **SFT** | [SFT_MODELS.md](./SFT_MODELS.md) | 8 pretrained models |
| **GRPO** | [GRPO_MODELS.md](./GRPO_MODELS.md) | 13 instruction-tuned models |

---

## Summary Statistics

### SFT Models (Pretrained Only)

Use pretrained models that have NOT been instruction-tuned for SFT training.

| Model Family | Count | Size Range |
|--------------|-------|------------|
| Qwen2.5 | 3 | 0.5B - 3B |
| Qwen3 | 3 | 0.6B - 4B |
| NVIDIA Nemotron-Flash | 2 | 1B - 3B |
| LiquidAI | 0 | - |
| **Total** | **8** | 0.5B - 4B |

### GRPO Models (Instruction-Tuned)

Use instruction-tuned models for GRPO training.

| Model Family | Count | Size Range |
|--------------|-------|------------|
| Qwen2.5 | 3 | 0.5B - 3B |
| Qwen3 | 5 | 0.6B - 4B |
| NVIDIA Nemotron-Flash | 1 | 3B |
| LiquidAI LFM2 | 4 | 350M - 2.6B |
| **Total** | **13** | 350M - 4B |

---

## Important: Naming Conventions

**Qwen3 uses the OPPOSITE naming convention from Llama/other models:**

| Model Family | Pretrained (for SFT) | Instruction-Tuned (for GRPO) |
|-------------|---------------------|------------------------------|
| **Qwen2.5** | No suffix (e.g., `Qwen2.5-1.5B`) | `-Instruct` suffix |
| **Qwen3** | `-Base` suffix (e.g., `Qwen3-1.7B-Base`) | No suffix (e.g., `Qwen3-1.7B`) |
| **Nemotron-Flash** | No suffix | `-Instruct` suffix |
| **LiquidAI LFM2** | None available | All models (no suffix) |

> **Source**: [Fine-Tuning Qwen3: Base vs. Reasoning Models](https://kaitchup.substack.com/p/fine-tuning-qwen3-or-qwen3-base)

---

## Recommended Training Pipeline

### For Chat/Instruction Models
```
Stage 1: SFT on pretrained model (use SFT_MODELS.md)
         Dataset: ultrachat_200k or FineTome-100k

Stage 2: GRPO on SFT-tuned model (optional)
         Dataset: math reasoning datasets
```

### For Math Reasoning Models
```
Stage 1: Start with instruction-tuned model (use GRPO_MODELS.md)
Stage 2: GRPO training with math datasets
         Dataset: NuminaMath-CoT or OpenMathInstruct-2
```

---

## Files in This Directory

```
models/
├── README.md              # This file (index)
├── SFT_MODELS.md          # Pretrained models for SFT
└── GRPO_MODELS.md         # Instruction-tuned models for GRPO
```

---

## Last Updated
2025-12-25
