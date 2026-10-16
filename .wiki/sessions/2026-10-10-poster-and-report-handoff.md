# 2026-10-10 — Poster delivered; report handoff

Sessions 2026-10-06 → 10-10. Poster planned, verified, edited in Figma and exported. Report planned,
not started. **Full handoff: `report/HANDOFF.md`.**

## What was done

- **Report kit** unzipped (`report/kit/`); `report/REPORT_PLAN.md` written: research questions,
  results-by-RQ structure, argument map C1–C30, consistency risks, figure plan.
- **Poster context pack** (`poster/POSTER_CONTEXT.md`) for Johnson's design agent: message, copy,
  A0/A1 spec, figure slots, glossary, do-not-claim list.
- **Independent verification** of every poster number (`analysis/verify_poster_numbers.py`): 58/61
  exact, 3 fourth-decimal drift, 0 fail. It caught shortcut SVM strength 8 resting on n = 4.
- **Two new analyses** (`analysis/extra_evidence.py`):
  - the protocol correction made the geometry signal *clearer* (wine hole-vs-random d −0.18 → +0.80),
    not just accuracy honest;
  - a generalisation scorecard: on 4-D iris, spread separates the hole from random deletion
    reliably only for tree + white-box; it is partial on SVM + black-box and never separates on
    tree + black-box.
- **Poster** edited in Figma via the MCP server (client `poster/tools/figma_mcp.mjs`). Final:
  Johnson's layout + agreed wording. Exported `poster/Poster_124.pdf` (A1 + 6 mm bleed, verified).

## Corrections discovered

- **Aidan (not Aiden) Jared's code used the correct protocol** (split, then delete from training).
  The before-split error was ours, from the July rebuild. `02-background-aiden.md` now carries a
  correction notice.
- "Only if it bends the boundary globally" → "**reliably** … only when" (the local outlier was caught
  white-box).
- Aidan's setting is the *most reliable*, not the *strongest*.

## Framing settled (for the poster, carried to the report)

Visibility rule + **no universal metric**; "most favourable setting", not "lucky"; keep two claims
apart: harder conditions weaken spread vs. a different defect breaks it. Aolin's analogy: an audit
needs a **panel of tests**, like a medical check-up.

## Next

Report conversation. Start from `report/HANDOFF.md` §9 (open questions: deadline, page limit,
MiKTeX, Overleaf, AI policy, contribution boundary, rerun scope).
