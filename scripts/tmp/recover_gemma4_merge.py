#!/usr/bin/env python3
"""Recover the Gemma4-E4B-SFT merge that Unsloth's in-place addmm_ crashed on.

Root cause: Unsloth 2026.4.6 save_pretrained_merged folds LoRA via an IN-PLACE
W.addmm_(lora_B, lora_A). Unsloth's own regex auto-targeted Gemma4's special
per-layer projections (per_layer_input_gate / per_layer_projection /
embedding_projection), whose widths differ from the standard MLP — so the
in-place target W [2560,6144] couldn't hold the [2560,10240] product. Training
itself succeeded; the adapter is intact.

Fix (route around the broken in-place merge): attach the saved adapter to the
FP16 base via stock PEFT and merge_and_unload() (out-of-place, shape-correct),
then push the merged fp16 model to the Hub.
"""
from __future__ import annotations
import os, sys, traceback

ADAPTER_DIR = "/project/6014832/ermia/HF-TRL/notebooks/gemma4_e2b_distill_lora"
FP16_BASE = "unsloth/gemma-4-E4B-it"   # NOT the bnb-4bit base — merge into fp16
HUB_ID = "ermiaazarkhalili/Gemma4-E4B-SFT-Claude-Opus-Reasoning-Unsloth"

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from peft import PeftModel

    try:
        from dotenv import load_dotenv; load_dotenv("/project/6014832/ermia/HF-TRL/.env")
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN")

    print(f"[recover] base : {FP16_BASE}", flush=True)
    print(f"[recover] adapter: {ADAPTER_DIR}", flush=True)
    print(f"[recover] target: {HUB_ID}", flush=True)

    print("[recover] loading fp16 base (text model)...", flush=True)
    # Gemma4 is multimodal; load the text causal LM. trust_remote_code for arch.
    model = AutoModelForCausalLM.from_pretrained(
        FP16_BASE, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    print(f"[recover][OK] base loaded: {type(model).__name__}", flush=True)

    print("[recover] attaching adapter via PEFT...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    print("[recover][OK] adapter attached", flush=True)

    print("[recover] merge_and_unload() (out-of-place, shape-correct)...", flush=True)
    model = model.merge_and_unload()
    print(f"[recover][OK] merged: {type(model).__name__}", flush=True)

    # Tokenizer/processor: prefer the adapter dir (it saved tokenizer + chat_template)
    try:
        tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
        print("[recover][OK] tokenizer from adapter dir", flush=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(FP16_BASE, trust_remote_code=True)
        print("[recover][OK] tokenizer from base", flush=True)

    print(f"[recover] pushing merged model -> {HUB_ID} ...", flush=True)
    # transformers 5.5.0 dropped the safe_serialization kwarg (safetensors is default).
    model.push_to_hub(HUB_ID, token=token)
    tok.push_to_hub(HUB_ID, token=token)
    print(f"[recover][PASS] pushed merged fp16 model to {HUB_ID}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[recover][FAIL] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
