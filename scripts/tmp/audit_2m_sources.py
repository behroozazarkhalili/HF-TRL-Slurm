#!/usr/bin/env python3
"""Genuineness audit of the 17 sources in Complete-FABLE.5-traces-2M.
Per-source population-level signals → falsifiable KEEP/DROP gate.
Run before the full clean. Streams; never loads 2M in memory."""
from __future__ import annotations
import json, re, hashlib
from collections import Counter
import polars as pl

URL=("https://huggingface.co/datasets/Crownelius/Complete-FABLE.5-traces-2M/"
     "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
SAMPLE=2000
LARP="Drawing from the autonomous, frontier-level reasoning characteristic of Claude Mythos"
# Root-caused 2026-06-25: added inline `[Tool: ` (Poumrm format) + Cyrillic-safe
# markers. Tool-call detection must cover BOTH structured tool_calls AND inline
# "[Tool: Name]" rendering.
AGENTIC_MARKERS=("tool_calls",'"Bash"','"Read"','"Edit"','"Write"',"sessionId",
                 "<think>","reasoning_content","[Tool:","[tool:")

def _longest_assistant(msgs):
    """Fix for Swarm-AI false-DROP: pick the LONGEST assistant content, not the
    last (some sources have empty trailing assistant turns)."""
    best=""
    for m in msgs:
        mm=json.loads(m) if isinstance(m,str) else m
        if isinstance(mm,dict) and mm.get("role")=="assistant":
            c=str(mm.get("content","") or "")
            if len(c)>len(best): best=c
    return best

def extract_response(rj:str):
    """Multi-format response extractor. Returns (text, has_agentic_marker, fmt)."""
    try: d=json.loads(rj)
    except Exception: return rj[:4000], any(m in rj for m in AGENTIC_MARKERS), "raw"
    marker=any(m in rj for m in AGENTIC_MARKERS)
    if isinstance(d,dict):
        msgs=d.get("messages")
        if isinstance(msgs,list) and msgs:
            return _longest_assistant(msgs) or str(msgs[-1]), marker, "messages"
        # HelioAI reasoning format {query, thinking}
        if "thinking" in d: return str(d.get("thinking","")), True, "thinking"
        for k in ("output","response","completion","solution","content","text"):
            if d.get(k): return str(d[k]), marker, k
        if "message" in d: return json.dumps(d.get("message"))[:4000], marker, "session"
    return rj[:4000], marker, "other"

def eng_ratio(s:str)->float:
    s=s[:400]
    if not s: return 0.0
    ascii_latin=sum(1 for c in s if ord(c)<128)
    return ascii_latin/len(s)

def main():
    lf=pl.scan_parquet(URL)
    dist=(lf.group_by("first_source_dataset").agg(pl.len().alias("n"))
            .sort("n",descending=True).collect(engine="streaming"))
    print(f"{'source':<52} {'n':>8} {'eng':>5} {'pfx':>5} {'uniq':>5} {'agen':>5} {'mlen':>6} {'larp':>5}  VERDICT")
    print("-"*120)
    keep=[]
    for row in dist.iter_rows(named=True):
        src=row["first_source_dataset"]; n=row["n"]
        samp=(lf.filter(pl.col("first_source_dataset")==src).select("row_json")
                .head(SAMPLE).collect(engine="streaming"))
        texts=[]; markers=0; fmts=Counter()
        for rj in samp["row_json"]:
            t,mk,fmt=extract_response(rj); texts.append(t); markers+=int(mk); fmts[fmt]+=1
        m=len(texts) or 1
        engs=sum(1 for t in texts if eng_ratio(t)>=0.9)/m
        prefixes=Counter(hashlib.md5(t[:120].encode()).hexdigest() for t in texts if t)
        pfx_top=(max(prefixes.values())/m) if prefixes else 1.0
        uniq=len(set(texts))/m
        agen=markers/m
        lens=sorted(len(t) for t in texts)
        mlen=lens[len(lens)//2] if lens else 0
        larp=sum(1 for t in texts if LARP in t)/m
        # gate
        ok=(engs>=0.8 and pfx_top<=0.15 and uniq>=0.6 and larp<=0.02 and (agen>=0.3 or mlen>=800))
        # agentic-floor patch: reject English-but-non-agentic long fiction
        if ok and agen<0.05 and mlen>=800 and "messages" not in fmts:
            ok=False; verdict="DROP(non-agentic)"
        else:
            verdict="KEEP" if ok else "DROP"
        if ok: keep.append(src)
        print(f"{src[:52]:<52} {n:>8} {engs:>5.2f} {pfx_top:>5.2f} {uniq:>5.2f} {agen:>5.2f} {mlen:>6} {larp:>5.2f}  {verdict}")
    print("\n=== EVIDENCE-DERIVED KEEP SET ===")
    for s in keep: print(f"  KEEP  {s}")
    print(f"\n[audit] {len(keep)}/17 sources pass the genuineness gate")

if __name__=="__main__":
    main()
