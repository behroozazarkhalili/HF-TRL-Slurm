---
license: cc-by-nc-4.0
tags:
- gguf
- granite-4.0-micro
- grpo
- math-reasoning
- quantized
- ollama
- llama-cpp
base_model: ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-10K
language:
- en
pipeline_tag: text-generation
---

# Granite 4.0 Micro — GRPO NuminaMath 10K (GGUF)

GGUF quantized versions of [granite-4.0-micro-GRPO-NuminaMath-10K](https://huggingface.co/ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-10K) for local inference with Ollama, llama.cpp, and LM Studio.

## Available Quantizations

| File | Quant | Size | Quality | Use Case |
|------|-------|------|---------|----------|
| `model-Q4_K_M.gguf` | Q4_K_M | ~2 GB | Good | **Recommended** — best balance |
| `model-Q5_K_M.gguf` | Q5_K_M | ~2.3 GB | Better | Higher quality, slightly larger |
| `model-Q8_0.gguf` | Q8_0 | ~3.5 GB | Best | Maximum quality |

## Quick Start

### Ollama
```bash
ollama pull hf.co/ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-10K-GGUF:Q4_K_M
ollama run granite-4.0-micro-GRPO-NuminaMath-10K-GGUF "Solve: If 3x + 7 = 22, what is x?"
```

### llama.cpp
```bash
./llama-cli -m model-Q4_K_M.gguf -p "Solve: What is the sum of all integers from 1 to 100?" -n 512
```

### LM Studio
Download any `.gguf` file and load it in LM Studio.

## Model Details

- **Base model**: [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro)
- **Training method**: GRPO (Group Relative Policy Optimization) with LoRA
- **Dataset**: [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) (10K samples)
- **Architecture**: Granite MoE Hybrid (SSM + Attention, ~3.4B params, ~1B active)
- **Training loss**: -0.016

See the [full model card](https://huggingface.co/ermiaazarkhalili/granite-4.0-micro-GRPO-NuminaMath-10K) for training details and metrics.

## Credits

- **IBM Research** for Granite 4.0
- **AI-MO/Numina** for NuminaMath-CoT
- **Hugging Face** for TRL
- **Digital Research Alliance of Canada** for compute
