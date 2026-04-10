#!/usr/bin/env python3
"""
GGUF Conversion Script
======================
Convert trained models to GGUF format for use with:
- Ollama
- LM Studio
- llama.cpp
- GPT4All

Features:
- Merge LoRA adapters with base model
- Convert to GGUF format
- Create multiple quantized versions
- Upload to HF Hub

Usage:
    python convert_gguf.py \
        --model username/my-model \
        --output_repo username/my-model-gguf \
        --quantizations Q4_K_M,Q5_K_M,Q8_0
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")

    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Model to convert (HF model ID or local path)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model (if --model is a LoRA adapter)")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model revision")

    # Output arguments
    parser.add_argument("--output_repo", type=str, required=True,
                        help="Output HF repo for GGUF files")
    parser.add_argument("--output_dir", type=str, default="/tmp/gguf_conversion",
                        help="Local output directory")

    # Quantization arguments
    parser.add_argument("--quantizations", type=str, default="Q4_K_M,Q5_K_M,Q8_0",
                        help="Comma-separated quantization types")
    parser.add_argument("--include_fp16", action="store_true",
                        help="Include FP16 GGUF (larger file)")

    # Hub arguments
    parser.add_argument("--private", action="store_true",
                        help="Create private repo")

    return parser.parse_args()


def run_command(cmd: List[str], cwd: Optional[str] = None) -> bool:
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n=== Installing Dependencies ===")

    # Install build tools
    deps = ["cmake", "build-essential"]
    print("Note: Ensure build-essential and cmake are available")

    # Install Python dependencies
    run_command([sys.executable, "-m", "pip", "install", "-q",
                 "transformers", "peft", "torch", "accelerate",
                 "huggingface_hub", "sentencepiece", "protobuf", "gguf"])


def clone_llama_cpp(work_dir: str) -> str:
    """Clone llama.cpp repository."""
    llama_cpp_dir = os.path.join(work_dir, "llama.cpp")

    if os.path.exists(llama_cpp_dir):
        print("llama.cpp already exists, updating...")
        run_command(["git", "pull"], cwd=llama_cpp_dir)
    else:
        print("\n=== Cloning llama.cpp ===")
        run_command([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            llama_cpp_dir
        ])

    return llama_cpp_dir


def build_llama_cpp(llama_cpp_dir: str) -> str:
    """Build llama.cpp quantization tools."""
    print("\n=== Building llama.cpp ===")

    build_dir = os.path.join(llama_cpp_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Configure with CMake (CPU only for conversion)
    run_command([
        "cmake", "..",
        "-DGGML_CUDA=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
    ], cwd=build_dir)

    # Build quantize tool
    run_command([
        "cmake", "--build", ".",
        "--target", "llama-quantize",
        "-j", str(os.cpu_count() or 4),
    ], cwd=build_dir)

    quantize_path = os.path.join(build_dir, "bin", "llama-quantize")
    if not os.path.exists(quantize_path):
        # Try alternative path
        quantize_path = os.path.join(build_dir, "llama-quantize")

    if os.path.exists(quantize_path):
        print(f"Quantize tool built: {quantize_path}")
        return quantize_path
    else:
        raise RuntimeError("Failed to build llama-quantize")


def merge_lora_adapter(model_path: str, base_model: str, output_dir: str) -> str:
    """Merge LoRA adapter with base model."""
    print("\n=== Merging LoRA Adapter ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merged_dir = os.path.join(output_dir, "merged_model")

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    return merged_dir


def convert_to_gguf(model_dir: str, llama_cpp_dir: str, output_dir: str) -> str:
    """Convert model to GGUF format."""
    print("\n=== Converting to GGUF ===")

    gguf_fp16_path = os.path.join(output_dir, "model-fp16.gguf")

    # Use llama.cpp convert script
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")

    success = run_command([
        sys.executable, convert_script,
        model_dir,
        "--outfile", gguf_fp16_path,
        "--outtype", "f16",
    ])

    if not success:
        raise RuntimeError("GGUF conversion failed")

    print(f"FP16 GGUF created: {gguf_fp16_path}")
    return gguf_fp16_path


def quantize_gguf(
    fp16_path: str,
    quantize_tool: str,
    output_dir: str,
    quantizations: List[str]
) -> List[str]:
    """Create quantized versions of GGUF."""
    print("\n=== Quantizing GGUF ===")

    quantized_files = []

    for quant_type in quantizations:
        output_path = os.path.join(output_dir, f"model-{quant_type}.gguf")
        print(f"Creating {quant_type} quantization...")

        success = run_command([
            quantize_tool,
            fp16_path,
            output_path,
            quant_type,
        ])

        if success and os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Created: {output_path} ({size_mb:.1f} MB)")
            quantized_files.append(output_path)
        else:
            print(f"  Warning: Failed to create {quant_type}")

    return quantized_files


def create_readme(output_dir: str, model_name: str, quantizations: List[str]) -> str:
    """Create README for the GGUF repo."""

    # Extract model short name for file naming
    model_short = model_name.split('/')[-1].lower()
    repo_name = f"{model_name}-GGUF"

    readme_content = f"""---
tags:
- gguf
- llama.cpp
- ollama
- lm-studio
- quantized
base_model: {model_name}
license: cc-by-nc-4.0
---

# {model_name.split('/')[-1]}-GGUF

GGUF quantized versions of [{model_name}](https://huggingface.co/{model_name}) for use with llama.cpp, Ollama, LM Studio, and other GGUF-compatible tools.

## Available Quantizations

| File | Quantization | Quality | Use Case |
|------|--------------|---------|----------|
| `{model_short}-q4_k_m.gguf` | Q4_K_M | Good | **Recommended** - Best balance of quality and size |
| `{model_short}-q5_k_m.gguf` | Q5_K_M | Better | Higher quality, moderate size increase |
| `{model_short}-q8_0.gguf` | Q8_0 | Best | Highest quality quantization |

## Download Specific Quantization

### Using huggingface-cli

```bash
# Download Q4_K_M (recommended)
huggingface-cli download {repo_name} {model_short}-q4_k_m.gguf --local-dir ./models

# Download Q5_K_M (higher quality)
huggingface-cli download {repo_name} {model_short}-q5_k_m.gguf --local-dir ./models

# Download Q8_0 (best quality)
huggingface-cli download {repo_name} {model_short}-q8_0.gguf --local-dir ./models

# Download all quantizations
huggingface-cli download {repo_name} --local-dir ./models
```

### Using wget

```bash
# Q4_K_M
wget https://huggingface.co/{repo_name}/resolve/main/{model_short}-q4_k_m.gguf

# Q5_K_M
wget https://huggingface.co/{repo_name}/resolve/main/{model_short}-q5_k_m.gguf

# Q8_0
wget https://huggingface.co/{repo_name}/resolve/main/{model_short}-q8_0.gguf
```

## Usage

### Ollama

```bash
# Pull specific quantization
ollama pull hf.co/{repo_name}:Q4_K_M

# Or create from local file
cat > Modelfile << EOF
FROM ./{model_short}-q4_k_m.gguf
EOF

ollama create {model_short} -f Modelfile
ollama run {model_short}
```

### llama.cpp

```bash
# Run with llama-cli
./llama-cli -m {model_short}-q4_k_m.gguf -p "Your prompt here" -n 256

# Run as server
./llama-server -m {model_short}-q4_k_m.gguf --host 0.0.0.0 --port 8080
```

### llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="{model_short}-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=-1  # Use all GPU layers
)

output = llm(
    "What is machine learning?",
    max_tokens=256,
    temperature=0.7,
)
print(output['choices'][0]['text'])
```

### LM Studio

1. Download the desired GGUF file from this repository
2. Open LM Studio and navigate to the Models tab
3. Click "Add Model" and select the downloaded GGUF file
4. Load the model and start chatting

### GPT4All

1. Download the Q4_K_M GGUF file
2. Open GPT4All and go to Settings > Models
3. Add the GGUF file path
4. Select the model and start using

## Original Model

This is a quantized version of [{model_name}](https://huggingface.co/{model_name}).
See the original model card for:
- Training details and methodology
- Dataset information
- Performance metrics
- Full usage examples with Transformers

## Conversion Details

| Property | Value |
|----------|-------|
| Source Model | [{model_name}](https://huggingface.co/{model_name}) |
| Conversion Date | {datetime.now().strftime("%Y-%m-%d")} |
| Quantizations | {', '.join(quantizations)} |
| Converter | llama.cpp |

## License

Same license as the original model. See [{model_name}](https://huggingface.co/{model_name}) for details.

---
*Converted using the Slurm Model Trainer skill*
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    return readme_path


def upload_to_hub(output_dir: str, repo_id: str, private: bool = False):
    """Upload GGUF files to HF Hub."""
    print("\n=== Uploading to Hub ===")

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Upload all files
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            print(f"Uploading: {file}")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="model",
            )

    print(f"\nUploaded to: https://huggingface.co/{repo_id}")


def main():
    args = parse_args()

    print("=" * 60)
    print("GGUF Conversion")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_repo}")
    print(f"Quantizations: {args.quantizations}")

    # Create work directory
    work_dir = args.output_dir
    os.makedirs(work_dir, exist_ok=True)

    # Parse quantizations
    quantizations = [q.strip() for q in args.quantizations.split(",")]

    try:
        # Install dependencies
        install_dependencies()

        # Clone and build llama.cpp
        llama_cpp_dir = clone_llama_cpp(work_dir)
        quantize_tool = build_llama_cpp(llama_cpp_dir)

        # Detect model type: full model vs LoRA adapter
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

        model_dir = args.model
        is_adapter = False

        try:
            files = list_repo_files(args.model)
            is_adapter = "adapter_config.json" in files
        except OSError:
            pass  # Local path or private repo — will detect below

        if is_adapter:
            # LoRA adapter — merge with base model
            base_model = args.base_model
            if not base_model:
                # Auto-detect base model from adapter_config.json
                adapter_config_path = hf_hub_download(args.model, "adapter_config.json")
                with open(adapter_config_path) as f:
                    config = json.load(f)
                base_model = config.get("base_model_name_or_path")
            if not base_model:
                raise ValueError(f"Adapter detected but no base model found. Pass --base_model explicitly.")
            print(f"LoRA adapter detected. Base model: {base_model}")
            model_dir = merge_lora_adapter(args.model, base_model, work_dir)
        else:
            # Full model — download to local dir for llama.cpp conversion
            print(f"Full model detected (no adapter_config.json)")
            local_dir = os.path.join(work_dir, "full_model")
            print(f"Downloading model to: {local_dir}")
            model_dir = snapshot_download(
                args.model,
                local_dir=local_dir,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            )
            print(f"Model downloaded: {model_dir}")

        # Convert to GGUF
        gguf_dir = os.path.join(work_dir, "gguf_output")
        os.makedirs(gguf_dir, exist_ok=True)

        fp16_path = convert_to_gguf(model_dir, llama_cpp_dir, gguf_dir)

        # Create quantized versions
        quantized_files = quantize_gguf(fp16_path, quantize_tool, gguf_dir, quantizations)

        # Remove FP16 if not requested (it's large)
        if not args.include_fp16 and os.path.exists(fp16_path):
            os.remove(fp16_path)

        # Create README
        create_readme(gguf_dir, args.model, quantizations)

        # Upload to Hub
        upload_to_hub(gguf_dir, args.output_repo, args.private)

        print("\n" + "=" * 60)
        print("GGUF Conversion Complete!")
        print("=" * 60)
        print(f"Repository: https://huggingface.co/{args.output_repo}")
        print(f"\nTo use with Ollama:")
        print(f"  ollama pull hf.co/{args.output_repo}:Q4_K_M")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
