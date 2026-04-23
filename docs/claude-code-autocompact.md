# Claude Code Auto-Compact — Definitive Reference

**Source:** Decompiled from Claude Code 2.1.116 binary (`/home/ermia/.local/share/claude/versions/2.1.116`, ELF x86-64, 237 MB, not stripped) + cross-verified against [Claude Code official docs](http://code.claude.com/docs/en/env-vars) and real-world GitHub issues.

Last verified: 2026-04-21 against Claude Code 2.1.116.

---

## The three mechanisms

| Identifier | Where set | Type | Controls |
|---|---|---|---|
| `autoCompactWindow` | `~/.claude/settings.json` top-level | integer 100000–1000000 | Effective context window size (tokens) |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | env var | integer tokens | Same as above; env-var form, **higher precedence** |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | env var | integer 1–100 | % of effective window at which to fire |

There is no `autoCompactThreshold` or `autoCompactEnabled` field that actually disables compaction — both community-claimed keys (`autoCompactEnabled`, `DISABLE_AUTO_COMPACT`) have been reported unreliable (issue #42394).

---

## The actual logic (deminified from the binary)

```js
// Constants baked into the binary
MIN_WINDOW         = 100_000     // $f6 = 1e5 — schema floor
MAX_WINDOW         = 1_000_000   // yP7 = 1e6 — schema ceiling
BLOCKING_BUFFER    = 13_000      // _f6 — reserved for compaction summary
TOOL_RESULT_BUDGET = 20_000      // bY1 — reserved for tool-result cap

// Source resolution — env var WINS over settings.json WINS over experiment WINS over model default
function an(model, settingsWindow) {
  let modelNative = nativeWindowForModel(model);   // e.g. 1,000,000 for opus[1m]

  // 1. Env var — highest precedence
  if (process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW) {
    let parsed = validate(CLAUDE_CODE_AUTO_COMPACT_WINDOW, MIN_WINDOW, MAX_WINDOW);
    if (parsed.status !== "invalid") {
      let configured = Math.max(MIN_WINDOW, parsed.effective);
      return { window: Math.min(modelNative, configured), configured, source: "env" };
    }
  }

  // 2. settings.json autoCompactWindow
  if (settingsWindow !== undefined) {
    return { window: Math.min(modelNative, settingsWindow), configured: settingsWindow, source: "settings" };
  }

  // 3. Experimental A/B test flag
  if (experimentFlagActive) { ... source: "experiment" ... }

  // 4. Fallback: model's native window
  return { window: modelNative, configured: modelNative, source: "model" };
}

// Threshold calculator — returns the token count at which compaction fires
function l9$(model, settingsWindow) {
  let effectiveWindow  = an(model, settingsWindow).window - TOOL_RESULT_BUDGET; // minus 20k
  let defaultThreshold = effectiveWindow - BLOCKING_BUFFER;                      // minus another 13k

  // If PCT override is set, compute alternate threshold and pick EARLIER one
  let pct = process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;
  if (pct) {
    let p = parseFloat(pct);
    if (!isNaN(p) && p > 0 && p <= 100) {
      let pctThreshold = Math.floor(effectiveWindow * (p / 100));
      return Math.min(pctThreshold, defaultThreshold);  // earliest fire wins
    }
  }

  return defaultThreshold;
}
```

---

## Key behavioral facts from the binary

1. **A fixed 33,000-token buffer is ALWAYS reserved** (TOOL_RESULT_BUDGET 20k + BLOCKING_BUFFER 13k). You cannot make compaction fire later than `effectiveWindow - 33000`, even with `PCT_OVERRIDE=100`.

2. **PCT_OVERRIDE can only make compaction fire EARLIER, never later.** The `Math.min(pctThreshold, defaultThreshold)` clamp guarantees this.

3. **Safety clamps:**
   - Cannot configure below `MIN_WINDOW` (100,000)
   - Cannot exceed the model's native capacity — `Math.min(modelNative, configured)` clips it
   - Setting `CLAUDE_CODE_AUTO_COMPACT_WINDOW=2000000` on a 1M model acts as if you set 1M

4. **Env var > settings.json.** If both are set, env var wins. Use one or the other.

5. **Telemetry confirms the source.** The Claude Code status bar (when debug logging active) emits:
   ```
   autocompact: tokens=<used> threshold=<fire-at> effectiveWindow=<after 20k budget>
   ```

---

## Example calculations

### Your current setup — `autoCompactWindow: 500000` on opus[1m]

```
modelNative       = 1,000,000
configured (settings) = 500,000
effectiveWindow   = min(1_000_000, 500_000) - 20,000 = 480,000
defaultThreshold  = 480,000 - 13,000 = 467,000
```

**Compaction fires at ~467,000 tokens.**

### Default (no config) on opus[1m]

```
modelNative       = 1,000,000
effectiveWindow   = 1,000,000 - 20,000 = 980,000
defaultThreshold  = 980,000 - 13,000 = 967,000
```

**Compaction fires at ~967,000 tokens** — right before the hard ceiling.

### With `PCT_OVERRIDE=80` on top of your 500k window

```
effectiveWindow   = 480,000 (as above)
defaultThreshold  = 467,000
pctThreshold      = floor(480,000 × 0.80) = 384,000
threshold         = min(384,000, 467,000) = 384,000
```

**Compaction fires at ~384,000 tokens** (80% of effective window, since that's earlier than 33k-buffer default).

### With `PCT_OVERRIDE=100` — why it can't push beyond the buffer

```
effectiveWindow   = 480,000
defaultThreshold  = 467,000
pctThreshold      = floor(480,000 × 1.00) = 480,000
threshold         = min(480,000, 467,000) = 467,000  ← falls back to default
```

Setting 100% doesn't help because `min()` picks the earlier trigger.

---

## Precedence (empirically extracted from binary)

| Priority | Source | Name used in binary's `source:` field |
|---|---|---|
| 1 (highest) | `CLAUDE_CODE_AUTO_COMPACT_WINDOW` env var | `"env"` |
| 2 | `autoCompactWindow` in settings.json | `"settings"` |
| 3 | A/B experiment flag | `"experiment"` |
| 4 (lowest) | Model's native window | `"model"` |

---

## Tips + gotchas

- **Debug your setting:** set `"env": {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "500000"}` and also enable debug logging to see the `autocompact: threshold=X` line.
- **Don't set both env-var AND settings.json.** Pick one. Env var silently wins and you'll be confused.
- **Opus 1M-context models:** with default config, compaction fires at ~967k — very late. Set `autoCompactWindow` lower (e.g. 500k) to compact at ~467k, or add `PCT_OVERRIDE=60` to compact at 60% of 1M = 600k.
- **VS Code extension:** same binary, same logic. Setting applies globally.
- **Cannot fully disable.** Neither `autoCompactEnabled: false` nor `DISABLE_AUTO_COMPACT=1` reliably stop compaction (community-reported, issues #38483, #42394). If you absolutely need to avoid compaction, raise the window and pray.

---

## Sources

- Binary: `/home/ermia/.local/share/claude/versions/2.1.116`
- [Claude Code docs — Environment variables](http://code.claude.com/docs/en/env-vars)
- [Issue #40757: Auto-compact triggers at ~420K on Opus 1M](https://github.com/anthropics/claude-code/issues/40757)
- [Issue #24079: Auto-Compact triggers prematurely with 51k free — confirms 33k buffer](https://github.com/anthropics/claude-code/issues/24079)
- [Issue #42394: Auto-compact fires despite DISABLE_AUTO_COMPACT=1](https://github.com/anthropics/claude-code/issues/42394)
- [Issue #42149: Add autoCompact: false setting](https://github.com/anthropics/claude-code/issues/42149)
- [BSWEN: How to Configure Claude Code Auto-Compact Settings](https://docs.bswen.com/blog/2026-03-21-claude-code-auto-compact-settings/)
