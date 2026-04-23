#!/usr/bin/env python3
"""
Direct GGUF conversion — reuses cached llama.cpp toolchain.

Replaces convert_gguf.py's every-run rebuild. Expects llama.cpp already
built at LLAMA_CPP_DIR (see --llama_cpp_dir), with:
  - build/bin/llama-quantize (compiled)
  - convert_hf_to_gguf.py (Python script)

Stages:
  1. Download HF model to local path (skipped if --model is a path).
  2. Convert HF → FP16 GGUF via convert_hf_to_gguf.py.
  3. Quantize FP16 → each requested format via llama-quantize.
  4. Upload each quant to --output_repo on HF Hub.

Exits 0 on full success, nonzero on any stage failure.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: str | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_tools(llama_cpp_dir: Path) -> tuple[Path, Path]:
    quantize = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    convert = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not quantize.is_file():
        sys.exit(f"[FATAL] llama-quantize missing at {quantize}. Rebuild cache.")
    if not convert.is_file():
        sys.exit(f"[FATAL] convert_hf_to_gguf.py missing at {convert}")
    return quantize, convert


def materialize_model(model_id: str, work_dir: Path, base_model: str | None) -> Path:
    """Return a local directory containing a merged model ready for GGUF conversion."""
    if Path(model_id).is_dir():
        return Path(model_id)

    from huggingface_hub import snapshot_download

    target = work_dir / "hf_download"
    target.mkdir(parents=True, exist_ok=True)

    print(f"[....] Downloading {model_id} → {target}", flush=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json", "*.txt", "*.model",
            "*.safetensors", "*.bin",
            "tokenizer*", "special_tokens*",
        ],
    )

    adapter_cfg = target / "adapter_config.json"
    if adapter_cfg.is_file():
        if not base_model:
            sys.exit(f"[FATAL] {model_id} is a LoRA adapter but --base_model not given")
        return merge_lora(target, base_model, work_dir)

    return target


def merge_lora(adapter_dir: Path, base_model: str, work_dir: Path) -> Path:
    print(f"[....] Merging LoRA {adapter_dir} onto base {base_model}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merged = work_dir / "merged"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="cpu", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged))
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(str(merged))
    return merged


def convert_to_fp16(model_dir: Path, convert_script: Path, out_path: Path) -> None:
    run([sys.executable, str(convert_script), str(model_dir),
         "--outfile", str(out_path), "--outtype", "f16"])


def quantize(fp16: Path, quantize_bin: Path, out: Path, quant_type: str) -> None:
    run([str(quantize_bin), str(fp16), str(out), quant_type])


def upload_file(local: Path, repo: str, path_in_repo: str, private: bool = False) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    try:
        create_repo(repo, repo_type="model", private=private, exist_ok=True)
    except Exception as e:
        print(f"[warn] create_repo: {e}", flush=True)
    print(f"[....] Uploading {local.name} → {repo}:{path_in_repo}", flush=True)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo ID or local path")
    ap.add_argument("--base_model", default=None, help="Base model if --model is a LoRA adapter")
    ap.add_argument("--output_repo", required=True, help="HF repo to upload GGUFs to")
    ap.add_argument("--quantizations", default="Q4_K_M,Q5_K_M,Q8_0")
    ap.add_argument("--llama_cpp_dir", default="/scratch/ermia/tools/llama.cpp")
    ap.add_argument("--work_dir", default=None, help="Scratch dir (default: /tmp/gguf-<pid>)")
    ap.add_argument("--keep_fp16", action="store_true", help="Upload FP16 GGUF too")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir) if args.work_dir else Path(f"/tmp/gguf-{os.getpid()}")
    work.mkdir(parents=True, exist_ok=True)

    quantize_bin, convert_script = ensure_tools(Path(args.llama_cpp_dir))
    model_dir = materialize_model(args.model, work, args.base_model)

    short = args.output_repo.split("/")[-1].replace("-GGUF", "").lower()
    fp16 = work / f"{short}-f16.gguf"
    convert_to_fp16(model_dir, convert_script, fp16)

    if args.keep_fp16:
        upload_file(fp16, args.output_repo, f"{short}.f16.gguf", private=args.private)

    failed: list[str] = []
    for qt in [q.strip() for q in args.quantizations.split(",") if q.strip()]:
        out = work / f"{short}-{qt.lower()}.gguf"
        try:
            quantize(fp16, quantize_bin, out, qt)
            upload_file(out, args.output_repo, f"{short}.{qt.lower()}.gguf", private=args.private)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {qt}: exit {e.returncode}", flush=True)
            failed.append(qt)

    if failed:
        print(f"[SUMMARY] {len(failed)} quant(s) failed: {failed}", flush=True)
        return 1
    print("[SUMMARY] all quants converted + uploaded", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
