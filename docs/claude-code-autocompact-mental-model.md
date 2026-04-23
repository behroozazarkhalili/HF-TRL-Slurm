# Claude Code Auto-Compact — Mental Model (Teaching Version)

The simple-to-understand explanation. For the authoritative, binary-extracted reference see `claude-code-autocompact.md`.

---

## The knobs

Auto-compact has just **TWO knobs**, even though they appear under three names:

1. **WINDOW** — how much context to treat as your budget
2. **PERCENTAGE** — what fraction of the WINDOW fires compaction

### Names

| Knob | Name 1 (settings.json) | Name 2 (env var) |
|---|---|---|
| WINDOW | `autoCompactWindow` | `CLAUDE_CODE_AUTO_COMPACT_WINDOW` |
| PERCENTAGE | *(none)* | `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` |

Two names for WINDOW, one name for PERCENTAGE. Same knob regardless of which name you use for WINDOW.

---

## The (simple) trigger formula

```
TRIGGER_POINT  ≈  WINDOW × PERCENTAGE
```

Auto-compact fires when `used_tokens` crosses `TRIGGER_POINT`.

## The (real) trigger formula

```
effectiveWindow = min(modelNative, WINDOW) − 20000   // tool-result budget always reserved
threshold       = effectiveWindow − 13000            // blocking buffer always reserved
// if PCT is set, pick whichever fires EARLIER:
if PCT:
  threshold = min(floor(effectiveWindow × PCT/100), threshold)
```

So the real formula has a **fixed 33k safety buffer** the simple formula hides. Always round down from your WINDOW by 33k.

---

## Rules of thumb

1. **Set WINDOW** to treat the model as smaller than it really is. Useful for 1M-context models when you want earlier compaction.
2. **Set PERCENTAGE** to fire earlier within the window. PERCENTAGE can only make compaction fire **earlier**, never later.
3. Pick ONE way to express WINDOW — either settings.json or env var. Env var silently wins if both are set.
4. Skip PERCENTAGE entirely and rely on the 33k default buffer — simpler.

---

## Examples (on opus[1m] — 1M model)

| Config | Effective Window | Trigger |
|---|---|---|
| Nothing set (default) | 1,000,000 | **~967k** (1M − 33k) |
| `autoCompactWindow: 500000` | 500,000 | **~467k** (500k − 33k) |
| `autoCompactWindow: 500000` + `PCT=80` | 500,000 | **~384k** (min(80% of 480k, 467k) = 384k) |
| `PCT=50` only | 1,000,000 | **~490k** (50% of 980k) |

---

## Gotchas

- Setting PERCENTAGE=100 does nothing — the `min()` clamp pins you to the 33k-default anyway
- Setting WINDOW above the model's native cap has no effect (cannot exceed model capacity)
- There's no reliable way to fully disable auto-compact in 2.1.116
