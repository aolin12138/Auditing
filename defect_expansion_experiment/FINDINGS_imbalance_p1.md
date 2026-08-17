# Class Imbalance — Phase 1 Findings (2026-08-16)

Extends Phase 0 (`FINDINGS_imbalance.md`, tree + white-box DTA) to a change of **model** and
**attack** and a **class-asymmetry** test. All arms delete a matched COUNT from the train fold
only (clean test), `feat=2` (petal length) for the spatial arm.

- Runner: `run_p1.py` (hang-safe subprocess-per-row driver, resumable).
- Data: `results_p1.parquet` (svm 210 rows, tree 210 rows; 10 seeds), `results_p1_rf.parquet`
  (rf 39 rows; 3 seeds — slow ~73 s/cell, 8 cells hung → NaN), `results_confound.parquet`
  (tree+DTA, 30 seeds, 660 rows).
- Figs: `plots/imbalance/p1_models.png`, `p1_asymmetry.png`, `p1_confound.png`.
- All spreads are normalised to each model's own clean (frac=0) baseline; 95% CI = 1.96·σ/√n.

Attacks/models: rbf-SVM + HSJ, single overfit tree + HSJ (black-box control), RandomForest
(60 trees) + HSJ. tree+HSJ hangs on the flat-facet pathology (50/210 cells → NaN); its cells
are **survivorship-biased** (4–10 valid seeds/cell), stated on every tree claim below.

---

## F1 — The minority-recall discriminator is MODEL-AGNOSTIC (the robust Phase-1 result)

At matched count removed from class 2 (virginica), **minority-class recall on the clean test
separates spatial (coverage gap) from random (imbalance) on every model**, with tight CIs, even
where scalar spread does not:

| model (attack) | frac | random recall | spatial recall | separation |
|----------------|------|---------------|----------------|------------|
| SVM + HSJ      | 0.85 | 0.75 ± 0.03 | **0.26 ± 0.01** | decisive |
| SVM + HSJ      | 0.95 | 0.45 ± 0.05 | 0.35 ± 0.01 | narrows (class nearly gone) |
| tree + HSJ†    | 0.85 | 0.74 ± 0.02 | **0.24 ± 0.02** | decisive |
| tree + HSJ†    | 0.95 | 0.64 ± 0.12 | **0.14 ± 0.00** | decisive |
| RF + HSJ‡      | 0.85 | 0.77 ± 0.06 | **0.35 ± 0.01** | decisive |
| RF + HSJ‡      | 0.95 | 0.55 ± 0.10 | **0.15 ± 0.01** | decisive |

† survivorship-biased (some seeds hung). ‡ n = 2–3 seeds.

Reproduces the Phase-0 mechanism across models: spatial deletion removes a **contiguous
petal-length band** → test virginica in that band are systematically misclassified → recall
craters. Random deletion thins the class **uniformly** → the full range stays covered → recall
degrades gently. **This is the clean, model-agnostic separator.**

## F2 — The scalar-spread gap survives PARTIALLY, and is model-dependent

Whether *spatial spread > random spread* survives a change of attack/model is weaker and
model-specific (contrast: Phase-0 white-box DTA separated CIs at frac ≥ 0.5):

- **RF + HSJ — clearest direction (but underpowered).** spatial 1.50 / 1.48× vs random 1.09 /
  1.26× at frac 0.85 / 0.95. Direction matches the model-family result *coverage gap survives
  bagging*; **n = 3 seeds** (some hung) → wide CIs, treat as directional not conclusive.
- **SVM + HSJ — separates only at the extreme.** frac 0.95: spatial 1.46 ± 0.11 [1.35, 1.57]
  vs random 1.23 ± 0.09 [1.14, 1.32] → CIs separate. frac 0.85: spatial 1.50 ± 0.20 vs random
  1.14 ± 0.16 → CIs touch (~1.30), **borderline**.
- **tree + HSJ — NO spread gap.** spatial 1.12 ± 0.07 vs random 1.14 ± 0.06 (frac 0.85); CIs
  overlap. The known weak black-box-on-tree combo (also 24 % of cells hung). Spread carries no
  spatial signal here — only recall (F1) does.

**Takeaway:** under black-box HSJ, the *spatial-hole spread signal* is real on
global-boundary / ensemble models (SVM at high severity, RF directionally) but does **not**
generalise the way the white-box DTA spread did. The robust cross-model separator is
**minority recall (F1)**, not scalar spread.

## F3 — Class ASYMMETRY: deleting from a SEPARABLE class is invisible (H-asym CONFIRMED)

Same deletion applied to `tc=0` (setosa, linearly separable) vs `tc=2` (virginica, contested),
rbf-SVM + HSJ:

- **tc=0 setosa:** normalised spread stays flat ~1.0× across all severities; minority recall
  stays **1.00** up to frac 0.85 (a tiny wobble only at 0.95 where the class is nearly deleted).
- **tc=2 virginica:** spread rises to 1.50×; recall craters to 0.26.

A defect only registers when it distorts a **contested** boundary. Deleting a band from a class
that no boundary passes near costs neither geometry nor accuracy — the strongest confirmation of
the organizing principle (`PLAN.md §0`: *visible iff it distorts a boundary separable from
accuracy*). Confirmed on SVM; RF shows the same tc=0 null (recall 1.00 all frac).

## F4 — The coverage-gap ACCURACY CONFOUND, quantified (H-confound CONFIRMED, 30 seeds)

Same overfit tree, same DTA, same class, same spatial deletion — only **where** the deletion
happens changes (`run_confound.py`). For `tc=2` (contested), at frac 0.95:

| protocol | test acc | minority recall |
|----------|----------|-----------------|
| **before-split** (deletes the band from train AND test — the flagship protocol) | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **train-only** (clean test — the correct protocol) | **0.714 ± 0.001** | **0.143 ± 0.003** |

before-split injection makes accuracy **rise to a perfect 1.000** because deleting the hard
low-petal-length virginica band from the *test* set removes exactly the points the model would
miss. train-only keeps them → accuracy and recall collapse. The Δ (0.286 test-acc, 0.857 recall)
is a **pure protocol artifact** with tight CIs. For `tc=0` (separable) both protocols stay ~flat
(recall 1.00) — nothing to remove near a boundary.

This numerically confirms the Phase-0 nuance and the `.wiki/06` lesson: the flagship
coverage-gap "accuracy stays flat" headline was **partly an artifact of before-split injection**.
Under a clean test, coverage gap is visible in *both* accuracy (esp. minority recall) and
geometry.

---

## Hypotheses scorecard
- **H1-survive** (spatial > random spread survives new model/attack): **PARTIAL** — RF
  directional (n=3), SVM at frac 0.95 only, absent on tree+HSJ. Not the clean cross-model result.
- **H2-recall** (minority recall the clean discriminator): **CONFIRMED, model-agnostic** — the
  robust Phase-1 result (F1).
- **H-asym** (separable class = null): **CONFIRMED** (F3).
- **H-confound** (before-split vs train-only accuracy artifact): **CONFIRMED, decisive** (F4).

## Caveats
- RF+HSJ underpowered (3 seeds, 8/39 cells hung) — F2 RF claim is directional. Firming to 10
  seeds would need ~1 h detached; deferred.
- tree+HSJ survivorship-biased (50/210 hung) — its spread panel is not reliable; recall (F1) is.
- iris only; single feature (petal length) for the spatial arm. Class-2 depletion at frac 0.95
  leaves only ~2 training virginica → high-severity cells are near-degenerate by construction.

## Next gates (PLAN §6)
- Optional: firm RF to 10 seeds (detached) to make F2-RF conclusive.
- Move to **Phase 0 shortcut / spurious feature** (PLAN §3) — the strongest "geometry catches
  what accuracy misses" candidate, and the natural next defect now that imbalance is characterised.
