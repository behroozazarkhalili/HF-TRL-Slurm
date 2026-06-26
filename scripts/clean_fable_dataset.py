#!/usr/bin/env python3
"""Clean a Fable-5 / Mythos trace dataset → standalone PRIVATE clean dataset.

Per docs/recipes/fable-mythos-distillation.md. Produces ONE JSONL supporting 3
training shapes (full-SFT / reasoning-split / instruction-only).

Dataset B (Glint pi_agent) path: the parquet already holds teich-converted
`messages[]` (Pi-agent format). So we extract messages, clean+anonymize, reshape
to the 3-mode schema, dedup, and push private. No teich CLI re-conversion needed
for B (the trace→messages step was done upstream by Glint).

Usage:
  python clean_fable_dataset.py --source B --out_repo ermiaazarkhalili/Fable-5-Glint-Clean-Private [--push]
"""
from __future__ import annotations
import argparse, json, re, hashlib, sys
from collections import Counter

# ── source registry ──────────────────────────────────────────────────────────
SOURCES = {
    "B": dict(
        repo="Glint-Research/Fable-5-traces", config="pi_agent", split="train",
        parquet="https://huggingface.co/datasets/Glint-Research/Fable-5-traces/resolve/refs%2Fconvert%2Fparquet/pi_agent/train/0000.parquet",
        license="agpl-3.0",
    ),
    # C = the RAW upstream of B (armand0e: "Glint-Research/Fable-5-traces was created
    # from formatting/splitting THIS same data"). Identical 11-col schema; 63 full-session
    # rows (vs B's 4,665 split shards). Same cleaner applies. Don't mix B+C in one tune.
    "C": dict(
        repo="armand0e/claude-fable-5-claude-code", config="default", split="train",
        parquet="https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        license="apache-2.0",
    ),
}

# ── cleaning helpers (replicate teich anonymize + clean_fable5 normalization) ──
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
HOME = re.compile(r"/home/[^/\s]+/")
WINHOME = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
APIKEY = re.compile(r"\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")
# command-injection / harness noise blocks
CMD_CAVEAT = re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.S)
CMD_NAME = re.compile(r"<command-(name|message|args)>.*?</command-\1>", re.S)
CMD_STDOUT = re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.S)
MODEL_SET = re.compile(r"USER:\s*Set model to.*?(?:\n|$)")

COT_REASONING_THRESHOLD = 450  # chars; > ⇒ task_type "reasoning"


def scrub(text: str) -> str:
    if not text:
        return text or ""
    text = ANSI.sub("", text)
    text = CMD_CAVEAT.sub("", text)
    text = CMD_NAME.sub("", text)
    text = CMD_STDOUT.sub("", text)
    text = MODEL_SET.sub("", text)
    text = HOME.sub("/home/user/", text)
    text = WINHOME.sub(r"C:\\Users\\user\\", text)
    text = EMAIL.sub("user@example.com", text)
    text = APIKEY.sub("<REDACTED_KEY>", text)
    return text.strip()


def _as_dict(m):
    """messages elements arrive as JSON strings (parquet List[Json]); parse them."""
    if isinstance(m, str):
        try:
            return json.loads(m)
        except Exception:
            return {"role": "unknown", "content": m}
    return m


def scrub_deep(obj):
    """Recursively scrub string values in nested structures (e.g. tool_calls args
    that embed real file paths like /home/<user>/...)."""
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, dict):
        return {k: scrub_deep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_deep(v) for v in obj]
    return obj


def norm_messages(raw_msgs):
    """Clean an OpenAI-style messages list; return (messages, thinking, response, output_type)."""
    out = []
    thinking = ""
    response = ""
    output_type = "text"
    for m in raw_msgs:
        m = _as_dict(m)
        role = m.get("role")
        content = m.get("content", "") or ""
        content = scrub(content) if isinstance(content, str) else content
        msg = {"role": role, "content": content}
        # carry tool_calls (structured) if present — scrub nested paths/keys/emails
        tc = m.get("tool_calls")
        if tc:
            msg["tool_calls"] = scrub_deep(tc)
            output_type = "tool_use"
        if role == "assistant":
            # reasoning_content is the genuine CoT in Glint pi-agent format
            rc = m.get("reasoning_content")
            if rc:
                thinking = scrub(rc)
            if isinstance(content, str) and content:
                response = content
        out.append(msg)
    return out, thinking, response, output_type


def build_completion(thinking: str, response: str) -> str:
    if thinking:
        return f"<think>\n{thinking}\n</think>\n{response}"
    return response


def clean_rows(df):
    """Yield cleaned 3-mode-schema rows from the source dataframe (pandas)."""
    seen = set()
    kept = 0
    dropped_empty = 0
    dropped_dup = 0
    for _, row in df.iterrows():
        raw_msgs = row.get("messages")
        if raw_msgs is None or len(raw_msgs) == 0:
            dropped_empty += 1
            continue
        # pandas may give numpy array of dicts
        raw_msgs = list(raw_msgs)
        messages, thinking, response, output_type = norm_messages(raw_msgs)
        # require a non-empty assistant response OR a tool call
        has_asst = any(m["role"] == "assistant" and (m.get("content") or m.get("tool_calls")) for m in messages)
        if not has_asst:
            dropped_empty += 1
            continue
        # context = all-but-last-assistant rendered; here keep messages as the source of truth
        context = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
            if isinstance(m.get("content"), str) and m["content"]
        )
        completion = build_completion(thinking, response)
        cot_len = len(thinking)
        task_type = "reasoning" if cot_len > COT_REASONING_THRESHOLD else "agentic"
        if output_type == "tool_use":
            task_type = "agentic"
        # dedup by normalized content signature
        sig = hashlib.md5((context[:1000] + "||" + response[:500]).encode()).hexdigest()
        if sig in seen:
            dropped_dup += 1
            continue
        seen.add(sig)
        kept += 1
        yield {
            "model": "claude-fable-5",
            "origin": "hf",
            "task_type": task_type,
            "output_type": output_type,
            "context_truncated": bool(row.get("context_truncated", False)),
            "messages": messages,
            "context": context,
            "thinking": thinking,
            "response": response,
            "completion": completion,
            "cot_length": cot_len,
            "context_length": len(context),
            "response_length": len(response),
            "session_id": row.get("session_id"),
        }
    return kept, dropped_empty, dropped_dup


# ── Dataset D (2M meta-aggregation) — quality-routed extraction ───────────────
# KEEP genuine fable-trace sources; DROP synthetic/off-domain. Verified by content
# sampling 2026-06-25: Poumrm (776K real convo), 1EYE4ALL raw sessions, PawanKrd,
# TheFusionCube, HelioAI = keep. attentionAllYouNeed (1.1M templated alpaca),
# BerkayBB (Turkish news), ansulev (mythos LARP) = drop.
# Evidence-derived keep-set (genuineness audit + investigate-pro root-cause + pro-review,
# 2026-06-25). D = full labeled SUPERSET (user choice): includes armand0e (=Dataset C)
# and 1EYE4ALL (raw form of B) — dedup against B/C at train time via source_dataset col.
D_KEEP_SOURCES = {
    "Poumrm/Mythos-5-and-Fabel-5-Class-Model-Outputs",   # 776K real multi-turn (inline [Tool:])
    "1EYE4ALL/Fable-5-traces",                            # 23K raw Claude sessions (=B upstream)
    "armand0e/claude-fable-5-claude-code",               # 14K (=Dataset C)
    "Swarm-AI-Research/fable5-traces-sft",               # 4.6K genuine (extractor-fixed)
    "lordx64/agentic-distill-fable-5-sft",              # 4.5K agentic SFT
    "victor/fable-5-boeing-747-trace",                  # 1K real session
    "Glint-Research/mythosmini",                         # 951 Glint
    "PawanKrd/claude-fable-5-code",                      # 603 prompt/response
    "TheFusionCube/Fable-5-CoT-Traces",                 # 353 genuine (row-filter category!=decoy)
}
# DROPPED (verified): attentionAllYouNeed (templated alpaca), BerkayBB (Turkish news),
# ansulev (mythos LARP), desiree (synthetic curriculum), umm-maybe (Cthulhu fiction),
# ox-ox (roleplay), ronniealfaro (Spanish), HelioAI (Russian).


def clean_dataset_D(out_jsonl, out_repo, push, limit=0):
    """Quality-routed clean of the 2M meta-aggregation via polars streaming."""
    import polars as pl
    url = ("https://huggingface.co/datasets/Crownelius/Complete-FABLE.5-traces-2M/"
           "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    print(f"[clean-D] lazy scan {url}", flush=True)
    lf = pl.scan_parquet(url)
    # keep only messages[]-format genuine sources (covers Poumrm = the bulk)
    lf = lf.filter(pl.col("first_source_dataset").is_in(list(D_KEEP_SOURCES)))
    if limit:
        lf = lf.head(limit)
    print("[clean-D] collecting filtered rows (streaming)...", flush=True)
    df = lf.select(["first_source_dataset", "row_json"]).collect(engine="streaming")
    print(f"[clean-D] {len(df)} rows from kept sources", flush=True)

    seen = set()
    rows = []
    kept = dropped_empty = dropped_dup = dropped_noparse = dropped_decoy = 0
    for src, rj in zip(df["first_source_dataset"], df["row_json"]):
        try:
            d = json.loads(rj)
        except Exception:
            dropped_noparse += 1
            continue
        if not isinstance(d, dict):
            dropped_empty += 1
            continue

        # TheFusionCube: drop deliberate decoy rows (README: "filter out the decoys")
        if src == "TheFusionCube/Fable-5-CoT-Traces" and d.get("category") == "decoy":
            dropped_decoy += 1
            continue

        # ── multi-format extraction → messages[], thinking, response, output_type ──
        messages = []
        thinking = ""
        response = ""
        output_type = "text"
        raw = d.get("messages")
        if isinstance(raw, list) and raw:
            messages, thinking, response, output_type = norm_messages(raw)
        elif "thinking" in d or "query" in d:  # HelioAI-style reasoning {query, thinking}
            q = scrub(str(d.get("query", "")))
            thinking = scrub(str(d.get("thinking", "")))
            response = scrub(str(d.get("response", d.get("solution", ""))))
            messages = [{"role": "user", "content": q},
                        {"role": "assistant", "content": response or thinking}]
        else:  # prompt/response or instruction/output (PawanKrd, TheFusionCube)
            user = scrub(str(d.get("prompt", d.get("instruction", d.get("user_prompt", "")))))
            response = scrub(str(d.get("response", d.get("output", d.get("solution", d.get("completion", ""))))))
            if not user and not response:
                dropped_empty += 1
                continue
            messages = [{"role": "user", "content": user},
                        {"role": "assistant", "content": response}]

        has_asst = any(m["role"] == "assistant" and (m.get("content") or m.get("tool_calls")) for m in messages)
        if not has_asst:
            dropped_empty += 1
            continue
        # inline-[Tool:] detection for output_type
        if output_type != "tool_use" and any("[Tool:" in str(m.get("content", "")) for m in messages):
            output_type = "tool_use"

        context = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
            if isinstance(m.get("content"), str) and m["content"])
        completion = build_completion(thinking, response)
        cot_len = len(thinking)
        task_type = "reasoning" if cot_len > COT_REASONING_THRESHOLD and output_type != "tool_use" else "agentic"
        sig = hashlib.md5((context[:1000] + "||" + response[:500]).encode()).hexdigest()
        if sig in seen:
            dropped_dup += 1
            continue
        seen.add(sig)
        kept += 1
        rows.append({
            "model": "claude-fable-5", "origin": "hf",
            "source_dataset": src,
            "task_type": task_type, "output_type": output_type,
            "context_truncated": False,
            "messages": messages, "context": context,
            "thinking": thinking, "response": response, "completion": completion,
            "cot_length": cot_len, "context_length": len(context), "response_length": len(response),
        })

    print(f"\n[clean-D] kept={kept} dropped_empty={dropped_empty} dropped_dup={dropped_dup} noparse={dropped_noparse}", flush=True)
    if rows:
        print(f"[clean-D] by source: {dict(Counter(r['source_dataset'] for r in rows))}")
        print(f"[clean-D] task_type: {dict(Counter(r['task_type'] for r in rows))}")
        print(f"[clean-D] output_type: {dict(Counter(r['output_type'] for r in rows))}")

    import os
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[clean-D] wrote {len(rows)} rows -> {out_jsonl}", flush=True)

    if push and out_repo:
        from datasets import Dataset
        try:
            from dotenv import load_dotenv; load_dotenv("/project/6014832/ermia/HF-TRL/.env")
        except Exception:
            pass
        push_rows = [{**r, "messages": json.dumps(r["messages"], ensure_ascii=False)} for r in rows]
        Dataset.from_list(push_rows).push_to_hub(out_repo, private=True)
        print(f"[clean-D][PASS] pushed {len(push_rows)} rows -> {out_repo} (private)", flush=True)


# ── Dataset A (mythos-25K synthetic) — boilerplate strip + quality filter ─────
# Full canned opener = "Drawing from ... Claude Mythos (distilled ...), I approach this
# with multi-layered analysis, ... and ethical guardrails." Strip the WHOLE sentence.
MYTHOS_PREAMBLE_RE = re.compile(
    r"Drawing from the autonomous, frontier-level reasoning characteristic of Claude Mythos.*?"
    r"(?:ethical guardrails\.|\bguardrails\.)",
    re.S)
MYTHOS_PREAMBLE = "Drawing from the autonomous, frontier-level reasoning characteristic of Claude Mythos"
MYTHOS_CLOSING = re.compile(r"\*?This response was generated to exemplify.*?purposes\.\*?", re.S)
# stub / fake-code / fabricated markers (drop these rows entirely)
MYTHOS_STUBS = ["Full implementation would be", "200+ LOC", "would be 200",
                "// Full implementation", "23.4M ops/sec"]


def clean_dataset_A(out_jsonl, out_repo, push, limit=0):
    """Aggressive clean of the synthetic mythos-25K: strip canned preamble/closing,
    drop stub/fake-code rows, dedup. Honest survivor count reported."""
    import pandas as pd
    url = ("https://huggingface.co/datasets/WithinUsAI/claude_mythos_distilled_25k/"
           "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    print(f"[clean-A] loading {url}", flush=True)
    df = pd.read_parquet(url)
    if limit:
        df = df.head(limit)
    print(f"[clean-A] loaded {len(df)} rows", flush=True)

    seen = set()
    rows = []
    kept = dropped_stub = dropped_dup = dropped_empty = 0
    for _, row in df.iterrows():
        raw = row.get("messages")
        if raw is None or len(raw) == 0:
            dropped_empty += 1
            continue
        msgs = [_as_dict(m) for m in list(raw)]
        user = next((scrub(str(m.get("content", ""))) for m in msgs if m.get("role") == "user"), "")
        asst = next((str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"), "")
        if not user or not asst:
            dropped_empty += 1
            continue
        # DROP rows with stub/fake code
        if any(s in asst for s in MYTHOS_STUBS):
            dropped_stub += 1
            continue
        # strip the FULL canned opener sentence + closing tag
        clean_asst = MYTHOS_PREAMBLE_RE.sub("", asst).strip()
        if clean_asst == asst.strip():  # regex missed → fall back to clause strip
            clean_asst = asst.replace(MYTHOS_PREAMBLE, "").strip()
        clean_asst = MYTHOS_CLOSING.sub("", clean_asst).strip()
        clean_asst = scrub(clean_asst)
        if len(clean_asst) < 40:  # nothing left after stripping boilerplate
            dropped_empty += 1
            continue
        # dedup on stripped response signature
        sig = hashlib.md5((user[:300] + "||" + clean_asst[:400]).encode()).hexdigest()
        if sig in seen:
            dropped_dup += 1
            continue
        seen.add(sig)
        kept += 1
        messages = [{"role": "user", "content": user},
                    {"role": "assistant", "content": clean_asst}]
        rows.append({
            "model": "claude-mythos", "origin": "hf",
            "category": row.get("category"),
            "task_type": "reasoning", "output_type": "text",
            "context_truncated": False,
            "messages": messages, "context": f"USER: {user}",
            "thinking": "", "response": clean_asst, "completion": clean_asst,
            "cot_length": 0, "context_length": len(user), "response_length": len(clean_asst),
        })

    print(f"\n[clean-A] kept={kept} dropped_stub={dropped_stub} dropped_dup={dropped_dup} dropped_empty={dropped_empty}", flush=True)
    print(f"[clean-A] SURVIVORS: {kept}/{len(df)} ({100*kept/len(df):.1f}%)")
    if rows:
        print(f"[clean-A] category dist: {dict(Counter(r['category'] for r in rows))}")
        import statistics as st
        print(f"[clean-A] response len: median={int(st.median(r['response_length'] for r in rows))}")

    import os
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[clean-A] wrote {len(rows)} rows -> {out_jsonl}", flush=True)

    if push and out_repo:
        from datasets import Dataset
        try:
            from dotenv import load_dotenv; load_dotenv("/project/6014832/ermia/HF-TRL/.env")
        except Exception:
            pass
        push_rows = [{**r, "messages": json.dumps(r["messages"], ensure_ascii=False)} for r in rows]
        Dataset.from_list(push_rows).push_to_hub(out_repo, private=True)
        print(f"[clean-A][PASS] pushed {len(push_rows)} rows -> {out_repo} (private)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES) + ["A", "D"])
    ap.add_argument("--out_jsonl", default=None, help="local JSONL output path")
    ap.add_argument("--out_repo", default=None, help="HF dataset repo to push (PRIVATE)")
    ap.add_argument("--push", action="store_true", help="push to Hub (private)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows")
    args = ap.parse_args()

    if args.source == "D":
        out_jsonl = args.out_jsonl or "/scratch/ermia/fable-clean/source_D_clean.jsonl"
        clean_dataset_D(out_jsonl, args.out_repo, args.push, args.limit)
        return
    if args.source == "A":
        out_jsonl = args.out_jsonl or "/scratch/ermia/fable-clean/source_A_clean.jsonl"
        clean_dataset_A(out_jsonl, args.out_repo, args.push, args.limit)
        return

    src = SOURCES[args.source]
    import pandas as pd
    print(f"[clean] loading {src['repo']} ({src['config']}/{src['split']}) parquet...", flush=True)
    df = pd.read_parquet(src["parquet"])
    if args.limit:
        df = df.head(args.limit)
    print(f"[clean] loaded {len(df)} rows; columns: {list(df.columns)}", flush=True)

    rows = []
    gen = clean_rows(df)
    try:
        while True:
            rows.append(next(gen))
    except StopIteration as e:
        kept, dropped_empty, dropped_dup = e.value

    print(f"\n[clean] kept={kept}  dropped_empty={dropped_empty}  dropped_dup={dropped_dup}", flush=True)
    if rows:
        tt = Counter(r["task_type"] for r in rows)
        ot = Counter(r["output_type"] for r in rows)
        with_cot = sum(1 for r in rows if r["cot_length"] > 0)
        print(f"[clean] task_type: {dict(tt)}")
        print(f"[clean] output_type: {dict(ot)}")
        print(f"[clean] rows with thinking/CoT: {with_cot} ({100*with_cot/len(rows):.1f}%)")
        import statistics as st
        print(f"[clean] completion len: median={int(st.median(len(r['completion']) for r in rows))} max={max(len(r['completion']) for r in rows)}")

    out_jsonl = args.out_jsonl or f"/scratch/ermia/fable-clean/source_{args.source}_clean.jsonl"
    import os
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[clean] wrote {len(rows)} rows -> {out_jsonl}", flush=True)

    if args.push and args.out_repo:
        print(f"[clean] pushing PRIVATE dataset -> {args.out_repo} ...", flush=True)
        from datasets import Dataset
        try:
            from dotenv import load_dotenv; load_dotenv("/project/6014832/ermia/HF-TRL/.env")
        except Exception:
            pass
        # Serialize the nested `messages` (variable tool_calls structure) as a JSON
        # string column to avoid Arrow "cannot mix struct and non-struct" — the
        # standard pattern for agent-trace datasets. Trainer does json.loads(messages).
        push_rows = []
        for r in rows:
            r2 = dict(r)
            r2["messages"] = json.dumps(r["messages"], ensure_ascii=False)
            push_rows.append(r2)
        ds = Dataset.from_list(push_rows)
        ds.push_to_hub(args.out_repo, private=True)
        print(f"[clean][PASS] pushed {len(push_rows)} rows to {args.out_repo} (private)", flush=True)
        print("[clean] NOTE: `messages` is a JSON string column — json.loads() it at train time.", flush=True)
    elif args.push:
        print("[clean][WARN] --push set but no --out_repo; skipped push", file=sys.stderr)


if __name__ == "__main__":
    main()
