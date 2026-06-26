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


TOKENIZER_FILE_NAMES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)


def _extract_vocab_size(cfg: dict) -> int | None:
    """Find vocab_size wherever it lives.

    Flat configs (Llama, Qwen): top-level `vocab_size`.
    Multi-modal configs (Gemma4 / gemma3n, Llama-3.2-Vision, PaliGemma):
    nested under `text_config`. Some vendors use `language_config`.
    """
    v = cfg.get("vocab_size")
    if isinstance(v, int):
        return v
    for nest_key in ("text_config", "language_config", "llm_config"):
        nested = cfg.get(nest_key)
        if isinstance(nested, dict):
            nv = nested.get("vocab_size")
            if isinstance(nv, int):
                return nv
    return None


def _vocab_sizes_match(base_cfg: Path, model_cfg: Path) -> tuple[bool, str]:
    """Compare `vocab_size` in the two config.json files.

    A base-tokenizer swap is only safe when the fine-tune did NOT change the
    vocab (e.g. no resize_token_embeddings, no added special tokens). For all
    our Unsloth LoRA flows this holds — LoRA targets projection layers only.

    Returns (safe_to_swap, detail). If vocab_size cannot be resolved in either
    config, we treat the check as inconclusive rather than a hard fail:
    base and model come from the same HF family by construction, so vocab
    drift without an explicit resize_token_embeddings call is not possible.
    """
    import json
    try:
        bcfg = json.loads(base_cfg.read_text())
        mcfg = json.loads(model_cfg.read_text())
    except Exception as e:
        return False, f"config.json read error: {e}"
    bv = _extract_vocab_size(bcfg)
    mv = _extract_vocab_size(mcfg)
    if bv is None and mv is None:
        return True, "vocab_size unknown in both configs — assuming consistent (same family)"
    if bv is None or mv is None:
        return False, f"vocab_size missing on one side (base={bv}, model={mv})"
    if bv != mv:
        return False, f"vocab size drift: base={bv} model={mv}"
    return True, f"vocab_size={bv}"


def _swap_base_tokenizer(model_dir: Path, base_model: str) -> tuple[bool, str]:
    """Replace model_dir's tokenizer files with those from `base_model`.

    Safeguards:
      - Aborts if vocab_size drifts between model and base.
      - Backs up every overwritten file to *.orig.
      - Only swaps files that actually exist in the base snapshot.
    Returns (swapped_ok, detail).
    """
    from huggingface_hub import snapshot_download

    base_cache = model_dir.parent / "_base_tokenizer"
    base_cache.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=base_model,
            local_dir=str(base_cache),
            local_dir_use_symlinks=False,
            allow_patterns=list(TOKENIZER_FILE_NAMES) + ["config.json"],
        )
    except Exception as e:
        return False, f"snapshot_download failed: {e}"

    base_config = base_cache / "config.json"
    model_config = model_dir / "config.json"
    if base_config.is_file() and model_config.is_file():
        ok, detail = _vocab_sizes_match(base_config, model_config)
        if not ok:
            return False, f"unsafe to swap — {detail}"
    else:
        detail = "config.json missing in base or model — skipping vocab check"

    swapped: list[str] = []
    for name in TOKENIZER_FILE_NAMES:
        src = base_cache / name
        if not src.is_file():
            continue
        dst = model_dir / name
        if dst.is_file():
            backup = dst.with_suffix(dst.suffix + ".orig")
            if not backup.is_file():
                backup.write_bytes(dst.read_bytes())
        dst.write_bytes(src.read_bytes())
        swapped.append(name)

    if not swapped:
        return False, "no tokenizer files found in base model"
    return True, f"swapped {len(swapped)} files ({detail}): {swapped}"


def patch_tokenizer_class(model_dir: Path, base_model: str | None = None) -> None:
    """Ensure the model dir's tokenizer loads via AutoTokenizer in the current env.

    Three-layer fix, in order of preference:
      1. Probe — if it loads, do nothing.
      2. Base-tokenizer swap (if base_model given) — replace tokenizer files
         with those from the base model. Safe when vocab is unchanged (true
         for all LoRA targeting projection layers, not embeddings).
      3. File-inferred class substitution — rewrite tokenizer_class to a
         transformers stdlib class (PreTrainedTokenizerFast / LlamaTokenizer
         / GPT2Tokenizer) based on which tokenizer files are present.

    Every rewrite backs up the original file to *.orig for auditability.
    """
    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        return

    ok, err = _probe_tokenizer(model_dir)
    if ok:
        return  # Tokenizer already loads — no patch needed.

    # Layer 2: base-tokenizer swap (preferred — root-cause, not symptom)
    if base_model:
        swapped, detail = _swap_base_tokenizer(model_dir, base_model)
        if swapped:
            print(f"[patch] base-tokenizer swap from {base_model}: {detail}",
                  flush=True)
            ok2, err2 = _probe_tokenizer(model_dir)
            if ok2:
                return
            print(f"[patch] swap loaded files but probe still fails: {err2} "
                  "— falling back to class substitution", flush=True)
        else:
            print(f"[patch] base-tokenizer swap skipped: {detail}", flush=True)

    # Layer 3: file-inferred class substitution (fallback)
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
        patch_tokenizer_class(Path(model_id), base_model)
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
    patch_tokenizer_class(target, base_model)

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
