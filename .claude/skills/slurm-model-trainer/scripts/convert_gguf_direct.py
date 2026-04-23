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


def _probe_tokenizer(model_dir: Path) -> tuple[bool, str]:
    """Try AutoTokenizer.from_pretrained(model_dir). Return (ok, error_msg)."""
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _infer_fallback_class(model_dir: Path) -> str | None:
    """Pick a substitute tokenizer_class based on which tokenizer files exist.

    transformers ships three standard base classes that cover 99% of
    HF Hub models without vendor-specific code:
      - PreTrainedTokenizerFast : any `tokenizer.json` (BPE, Unigram, WordPiece)
      - LlamaTokenizer          : `tokenizer.model` + SentencePiece
      - GPT2Tokenizer           : `vocab.json` + `merges.txt`
    Returns None if nothing fits (caller should bail).
    """
    if (model_dir / "tokenizer.json").is_file():
        return "PreTrainedTokenizerFast"
    if (model_dir / "tokenizer.model").is_file():
        return "LlamaTokenizer"
    if (model_dir / "vocab.json").is_file() and (model_dir / "merges.txt").is_file():
        return "GPT2Tokenizer"
    return None


def patch_tokenizer_class(model_dir: Path) -> None:
    """Ensure the model dir's tokenizer loads via AutoTokenizer in the current env.

    Capability-probe, not allowlist: we actually attempt to load the tokenizer,
    and only rewrite `tokenizer_class` if the probe fails. Generalizes to ANY
    vendor whose tokenizer_config.json names a class that:
      (a) is not in transformers, AND
      (b) has no remote code (auto_map) — or remote code fails to load.

    Substitute class is inferred from the files present (not hard-coded).
    Original config is backed up to tokenizer_config.json.orig for auditability.
    """
    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        return

    ok, err = _probe_tokenizer(model_dir)
    if ok:
        return  # Tokenizer already loads — no patch needed.

    import json
    cfg = json.loads(cfg_path.read_text())
    original_cls = cfg.get("tokenizer_class", "<unset>")

    fallback = _infer_fallback_class(model_dir)
    if fallback is None:
        sys.exit(
            f"[FATAL] Tokenizer at {model_dir} does not load and no fallback "
            f"class fits the files present. Error: {err}"
        )

    backup = cfg_path.with_suffix(".json.orig")
    if not backup.is_file():
        backup.write_text(cfg_path.read_text())

    print(f"[patch] tokenizer_class: {original_cls!r} → {fallback!r} "
          f"(probe failed: {err.splitlines()[0][:140]})", flush=True)
    cfg["tokenizer_class"] = fallback
    # Strip auto_map pointers to custom classes — they'll be tried again
    # by AutoTokenizer and would re-trigger the same failure.
    cfg.pop("auto_map", None)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    ok2, err2 = _probe_tokenizer(model_dir)
    if not ok2:
        # Restore backup so we don't leave the model in a broken intermediate state.
        cfg_path.write_text(backup.read_text())
        sys.exit(f"[FATAL] Patched tokenizer_class={fallback!r} still fails: {err2}")


def materialize_model(model_id: str, work_dir: Path, base_model: str | None) -> Path:
    """Return a local directory containing a merged model ready for GGUF conversion."""
    if Path(model_id).is_dir():
        patch_tokenizer_class(Path(model_id))
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
    patch_tokenizer_class(target)

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
    patch_tokenizer_class(merged)
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
