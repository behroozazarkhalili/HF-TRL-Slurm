#!/usr/bin/env python3
"""Upload GGUF files to HuggingFace Hub."""

from huggingface_hub import HfApi, create_repo
import os
from datetime import datetime

api = HfApi()

# Models to upload (with sample size in naming)
MODELS = [
    {
        "source_model": "ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-10K",
        "base_model": "LiquidAI/LFM2-350M",
        "gguf_repo": "ermiaazarkhalili/LFM2-350M-GRPO-NuminaMath-10K-GGUF",
        "gguf_dir": "/scratch/ermia/outputs/lfm2-350m-grpo-numina-16912214/gguf/gguf_output",
        "model_name": "lfm2-350m-grpo-numina-10k",
        "params": "350M",
        "train_samples": "10K",
    },
    {
        "source_model": "ermiaazarkhalili/LFM2-700M-GRPO-NuminaMath-10K",
        "base_model": "LiquidAI/LFM2-700M",
        "gguf_repo": "ermiaazarkhalili/LFM2-700M-GRPO-NuminaMath-10K-GGUF",
        "gguf_dir": "/scratch/ermia/outputs/lfm2-700m-grpo-numina-16912215/gguf/gguf_output",
        "model_name": "lfm2-700m-grpo-numina-10k",
        "params": "700M",
        "train_samples": "10K",
    },
    {
        "source_model": "ermiaazarkhalili/LFM2-1.2B-GRPO-NuminaMath-10K",
        "base_model": "LiquidAI/LFM2-1.2B",
        "gguf_repo": "ermiaazarkhalili/LFM2-1.2B-GRPO-NuminaMath-10K-GGUF",
        "gguf_dir": "/scratch/ermia/outputs/lfm2-1.2b-grpo-numina-16912216/gguf/gguf_output",
        "model_name": "lfm2-1.2b-grpo-numina-10k",
        "params": "1.2B",
        "train_samples": "10K",
    },
]


def create_gguf_readme(source_model: str, model_name: str, gguf_repo_id: str, base_model: str, params: str) -> str:
    """Generate comprehensive README for GGUF repository matching SFT GGUF format."""
    # Extract display name from model_name
    display_name = model_name.replace("-", " ").title().replace(" ", "-")
    short_name = source_model.split("/")[-1]

    return f"""---
license: cc-by-nc-4.0
language:
  - en
tags:
  - gguf
  - llama.cpp
  - ollama
  - lm-studio
  - quantized
  - lfm2
  - liquid
  - grpo
  - math
  - reasoning
  - numina
base_model: {source_model}
datasets:
  - AI-MO/NuminaMath-CoT
pipeline_tag: text-generation
---

# {short_name}-GGUF

GGUF quantized versions of [{short_name}](https://huggingface.co/{source_model}) for efficient CPU and mixed CPU/GPU inference.

## Model Overview

This is a quantized version of **{short_name}**, a {params} parameter model fine-tuned using **Group Relative Policy Optimization (GRPO)** on the [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset for mathematical reasoning tasks.

### Key Features

- **Mathematical Reasoning**: Optimized for step-by-step math problem solving
- **GRPO Training**: Uses reinforcement learning with verifiable rewards
- **Efficient Inference**: Quantized for fast CPU/GPU inference
- **Wide Compatibility**: Works with Ollama, llama.cpp, LM Studio, and more

## Available Quantizations

| Quantization | File | Size | Description |
|--------------|------|------|-------------|
| **Q4_K_M** | `{model_name}-q4_k_m.gguf` | ~40% of original | Best balance of quality and size |

## Quick Start

### Using Ollama

```bash
# Pull and run directly from HuggingFace
ollama pull hf.co/{gguf_repo_id}:Q4_K_M
ollama run hf.co/{gguf_repo_id}:Q4_K_M "Solve step by step: What is 15% of 80?"
```

### Alternative: Create Custom Modelfile

```bash
# Download the GGUF file first
huggingface-cli download {gguf_repo_id} \\
    {model_name}-q4_k_m.gguf --local-dir ./models

# Create Modelfile with custom system prompt
cat > Modelfile << 'EOF'
FROM ./models/{model_name}-q4_k_m.gguf

SYSTEM "You are a helpful math tutor. When given a math problem, solve it step by step, showing your reasoning clearly. Always verify your final answer."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create and run the model
ollama create {model_name} -f Modelfile
ollama run {model_name}
```

### Using llama.cpp

```bash
# Download the GGUF file
huggingface-cli download {gguf_repo_id} \\
    {model_name}-q4_k_m.gguf --local-dir ./models

# Run inference
./llama-cli -m ./models/{model_name}-q4_k_m.gguf \\
    -p "Solve step by step: If a train travels at 60 mph for 2.5 hours, how far does it travel?" \\
    -n 256

# Or start a server
./llama-server -m ./models/{model_name}-q4_k_m.gguf \\
    --host 0.0.0.0 --port 8080
```

### Using llama-cpp-python

```python
from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="./models/{model_name}-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=-1  # Use all GPU layers if available
)

# Generate response
prompt = '''Solve step by step:
A store has a 25% off sale. If an item originally costs $80, what is the sale price?

Solution:'''

output = llm(
    prompt,
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    echo=False
)

print(output['choices'][0]['text'])
```

### Using LM Studio

1. Download the GGUF file from this repository
2. Open LM Studio and navigate to the Models tab
3. Click "Import Model" and select the downloaded GGUF file
4. Load the model and start chatting about math problems!

## Example Prompts

Here are some example prompts that work well with this model:

```
Solve step by step: What is 23 × 17?

Solve step by step: A rectangle has a length of 12 cm and a width of 8 cm. What is its area and perimeter?

Solve step by step: If 3x + 7 = 22, what is the value of x?

Solve step by step: A car travels 150 miles in 2.5 hours. What is its average speed in miles per hour?
```

## Source Model

This is a quantized version of [{short_name}](https://huggingface.co/{source_model}).

### Training Details

| Property | Value |
|----------|-------|
| Base Model | [{base_model}](https://huggingface.co/{base_model}) |
| Training Method | GRPO (Group Relative Policy Optimization) |
| Dataset | [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| Training Samples | 10,000 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |

See the [source model card](https://huggingface.co/{source_model}) for full training details and usage examples with Transformers.

## Hardware Requirements

| Quantization | RAM Required | GPU VRAM (optional) |
|--------------|--------------|---------------------|
| Q4_K_M | ~1-2 GB | ~1-2 GB |

## Conversion Details

| Property | Value |
|----------|-------|
| Source Model | [{source_model}](https://huggingface.co/{source_model}) |
| Conversion Date | {datetime.now().strftime("%Y-%m-%d")} |
| Quantization | Q4_K_M |
| Converter | [llama.cpp](https://github.com/ggerganov/llama.cpp) |

## License

CC-BY-NC-4.0 (same as source model)

## Acknowledgments

- [Liquid AI](https://www.liquid.ai/) for the LFM2 base model
- [AI-MO](https://huggingface.co/AI-MO) for the NuminaMath-CoT dataset
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for quantization tools
- [ermiaazarkhalili](https://huggingface.co/ermiaazarkhalili) for training and quantization

---
*Quantized using the HF-TRL GGUF conversion pipeline on Compute Canada infrastructure*
"""


def main():
    """Upload GGUF files to HuggingFace Hub."""
    for model in MODELS:
        repo_id = model["gguf_repo"]
        source_model = model["source_model"]
        base_model = model["base_model"]
        gguf_dir = model["gguf_dir"]
        model_name = model["model_name"]
        params = model["params"]

        print(f"\n=== Processing {repo_id} ===")

        # Create repo
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print("  Created/verified repo")
        except Exception as e:
            print(f"  Error creating repo: {e}")
            continue

        # Upload Q4_K_M GGUF file
        gguf_file = os.path.join(gguf_dir, "model-Q4_K_M.gguf")
        if os.path.exists(gguf_file):
            new_name = f"{model_name}-q4_k_m.gguf"
            size_mb = os.path.getsize(gguf_file) / 1024**2
            print(f"  Uploading {new_name} ({size_mb:.1f} MB)...")
            try:
                api.upload_file(
                    path_or_fileobj=gguf_file,
                    path_in_repo=new_name,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  Uploaded: {new_name}")
            except Exception as e:
                print(f"  Error uploading GGUF: {e}")
        else:
            print(f"  GGUF file not found: {gguf_file}")

        # Upload comprehensive README
        readme = create_gguf_readme(source_model, model_name, repo_id, base_model, params)
        try:
            api.upload_file(
                path_or_fileobj=readme.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            print("  Uploaded README.md")
        except Exception as e:
            print(f"  Error uploading README: {e}")

    print("\n=== GGUF Upload Complete ===")


if __name__ == "__main__":
    main()
