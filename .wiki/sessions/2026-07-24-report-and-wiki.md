# Session — 2026-07-24

Long working session covering understanding → report → presentation → wiki.

## What happened (chronological)

1. **Understood the `hsj_svm` experiment** — confirmed it studies the
   relationship between SVM and coverage-gap bias; the SVM is RBF-kernel; it uses
   one-vs-one for multi-class.
2. **Explained the ML mechanics** — how sklearn SVC does multi-class (OvO), how
   HopSkipJump (decision-based, label-only) attacks it without seeing the OvO
   structure, why density decreases with bias.
3. **Committed + documented the repo** — wrote the top-level README, organised
   the three experiment grids, computed headline Cohen's d values.
4. **Verified all findings against data** — corrected several of the agent's own
   overstatements (see below).
5. **Built the mid-year report** (`midyear_report.docx`) — first generic, then
   reformatted to the UoA P4P template (no title/abstract/refs, 4–6 pages).
6. **Fact-checked the conference deck** — all numbers correct; flagged the
   "spread not density" framing and the missing compression-ratio slide; fixed
   both in `presentation/deck.html` (kept original HTML style).
7. **Regenerated figures** — went from cluttered dual-axis plots to clean
   Tufte-style (p1–p6), then z-scored the main finding plot like Aiden's.
8. **Investigated the Tree+HSJ bias-0.9 drop** — found it's real: the depleted
   class's decision region collapses, attack reverts to intact-class boundary.
9. **Built this wiki** and ingested the session.

## Key corrections made this session (things that were initially wrong)

- **"Aiden's density increased with bias"** → WRONG. His original decreased
  (0.0309→0.0298). The agent's `dtree_attack` reconstruction increased only
  because it used `n_points/(mean+1)` instead of Aiden's `n_pairs/sum`. See
  [../06-lessons-gotchas.md](../06-lessons-gotchas.md).
- **"mean_dist is a better metric than density"** → WRONG. They're equivalent
  (spread ≈ 1/density). Reframed the whole report/deck narrative: we report
  spread for interpretability, not because it's superior. This was the user's
  catch — "were you just hallucinating?"
- **"Tree+HSJ weak because no gradient to follow"** → WRONG. The boundary has a
  well-defined normal; the real issue is HSJ's *sampling-based* direction
  estimate is noisy on flat axis-aligned facets → non-convergence. User caught
  this too.
- **Metric decomposition "our contribution"** → softened. Decomposition clarifies
  what moves (size vs spread) but isn't a novel metric.

## User's confirmed mental models (correct)

- Label noise adds boundaries → adversarial points barely move → measured
  distribution ≈ original test points → signal useless. Verified: compression
  ratio → 1, perturbation ↓.
- Coverage gap at extreme bias (0.9) collapses the depleted class's region → most
  attacks happen at the healthy intact-class boundary → spread reverts down.
  Verified: class-0 test predictions drop from 9 to <1.
- Unified picture: **adversarial spread tracks where the model's usable
  boundaries are.** More boundaries (label noise) → scatter; collapsed region
  (extreme gap) → revert to healthy geometry.

## Artifacts produced

- `midyear_report.docx` + `generate_report.py`
- `presentation/deck.html` (10 slides, fixed)
- `figures/report/p1–p6*.png` (+ Altair variants)
- `.wiki/` (this wiki)
- Commits `b8a821c` → `52d3251` on origin/master

## Deliverable status

- **Mid-year report** — drafted, needs user review + supervisor consult
- **Conference deck** — fixed, needs the two narrative tweaks reviewed
- **Report due** 2026-07-24 23:59 (Turnitin file upload)

## Loose ends

- `_probe_cg.py` compression-ratio result is hardcoded into the discriminant
  figure from a completed run — reproducible but not saved to parquet.
- Coverage-gap test-set confound still not fixed (train-fold-only injection).
- Report/deck should be read end-to-end by the user before submission.
