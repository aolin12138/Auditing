# 08 — Open Questions & Future Work

## Urgent / next up

### Fix the coverage-gap test-set confound
Bias is injected before the CV split, so the test set changes with bias level.
The compression ratio partially addresses this, but the clean fix is to **inject
bias into the training fold only and keep a fixed clean test set**. This isolates
"does the model's geometry change" from "do we have different test points."
Cheapest high-value experiment. See [06-lessons-gotchas.md](06-lessons-gotchas.md).

### Higher-dimensional validation — ✅ CLOSED (2026-08-17): spread is dimension-limited
Re-ran the imbalance contrast on **wine (13-D)** + **svm/HSJ** (`FINDINGS_variance.md`, Finding 6):
the adversarial-spread signal collapses in higher dimensions (iris spatial 1.26× → wine ~1.04×)
and under black-box HSJ — measured cause = distance concentration (std/mean 0.54 iris vs 0.29 wine).
**§8 then tested four dimension-robust candidates** (kNN-ratio, LOF, PCA-then-spread, kNN-local) on
the same clouds: **none recovers a meaningful wine signal** (`FINDINGS_robust_metric.md`). Best case
(kNN-local) is a +4% ripple on a clean baseline already as large as the defect arms. **Conclusion:
scalar spread is genuinely dimension-limited; per-class recall is the recommended black-box
diagnostic.** A >50-D set would only confirm this.

### Signal-directed dimension reduction before the metric (proposed 2026-08-17, OPEN)
The dimensionality problem that sank the spread signal invites an obvious fix: **reduce dimensions
before computing the metric.** This was *partly* tested — **M3 (PCA-then-spread)** in §8 projected the
adversarial cloud onto its top-3 PCs, then measured spread. It **fixed the concentration**
(std/mean 0.56→0.82) but did **not enlarge the effect**, because on wine the defect barely
perturbs the geometry at all (the clean cloud is already as dispersed as the defect clouds).
**Lesson: dimension reduction fixes the *measurement*, not the *existence* of a signal.**

The sharper, untested version: reduce to the **signal-carrying** subspace, not the top-variance
subspace. Unsupervised PCA finds max-variance directions, which need not align with the directions
the defect distorts. Candidate: find the directions where the *defect* adversarial cloud differs
most from a *clean-baseline* cloud (a supervised / discriminative reduction), then measure spread
only there. **Evidence this direction is right:** the shortcut result (Finding 7) is exactly this
idea in its *known-axis* form — we measure displacement on the one axis carrying the signal, and it
works and survives to wine + black-box. **Open question:** can the signal-carrying subspace be
*discovered* from the adversarial cloud when we don't know it a priori (coverage-gap / imbalance)?
**Honest caveat:** if a defect genuinely doesn't distort the boundary in high-D (the wine finding),
no subspace selection can manufacture a signal.

**→ CONCRETE PLAN (2026-08-17): `defect_expansion_experiment/PLAN.md §9`** — a controlled study on
the verified defects (coverage gap + imbalance): pad iris with `m` noise dimensions (signal stays in
the original 4), sweep `m`, and test whether (a) raw spread decays with `m`, (b) dimension reduction
(unsupervised PCA vs signal-directed) recovers the KNOWN signal buried in noise, (c) feature
normalization (whitening vs z-score) and metric normalization (Cohen's d, ÷expected-distance) keep
the signal consistent across dimensions. Disambiguates the §8 negative: dimensionality-per-se vs
defect-not-distorting (does the same reduction that rescues padded-iris also rescue wine?).

### Shortcut / spurious feature — ✅ DONE (2026-08-17)
The "defect that hurts data quality without collapsing accuracy" premise, tested (Finding 7).
A per-axis metric detects it, survives wine + black-box HSJ — but the signal co-moves with accuracy
(never zero-cost). The pure accuracy-*inflating* case is next:

### Train–test leakage / duplication — ⭐ NEXT (`PLAN.md §4`)
Leakage inflates accuracy (the model looks *great*) while creating memorized, over-confident local
geometry. Tests whether the probe flags a defect that accuracy actively *hides*. Metric:
memorization locality. This is the inverse of every defect so far — the only one where accuracy
moves the *wrong* way.

### A defect that hurts data quality without collapsing accuracy
The strongest test of the whole premise. Label noise fails partly because it
collapses accuracy. Design a defect that degrades data quality while accuracy
stays high (e.g. systematic mislabeling of one sub-region, feature corruption in
a subpopulation). If the geometry fires when accuracy is blind, that's the
headline result.

## Medium term

- **More model families:** neural networks with gradient-based attacks (PGD).
  Does the signal generalise beyond tree/SVM? **→ PLANNED:** RandomForest +
  XGBoost via HSJ, plus a 2-feature viz setup and stretch MLP+PGD. See
  [09-planned-experiments.md](09-planned-experiments.md) /
  `model_family_experiment/PLAN.md`.
- **More defect types:** feature noise, class imbalance, outliers, systematic
  mislabeling. Map where the signal exists and where it doesn't — a
  characterisation study. **→ PLANNED (outliers):** correctly-labeled anomalies
  with controlled distance/count/direction/target-class, train-only injection,
  per-class asymmetry test. See [09-planned-experiments.md](09-planned-experiments.md)
  / `outlier_experiment/PLAN.md`.
- **Tree strategy under coverage gap:** the overfit-vs-pruned flip was only
  tested for label noise. Does pruning change the coverage-gap signal? And
  record spread separately this time (the v2 grid only stored density).
- **Statistical rigour:** n=36 per cell, only 3 seeds. A mixed-effects model over
  the fold structure would strengthen the effect-size claims. Weak effects
  (Tree+HSJ d=+0.38) are not significant at this n.

## Known unknowns / caveats to keep in view

- **Tree+HSJ is fragile** — weak signal, non-convergence, non-monotonic at
  extreme bias. Don't lean on it. SVM+HSJ and Tree+DTA are the trustworthy
  combinations.
- **The spread metric itself is dimension- and attack-fragile** (2026-08-16, Finding 6): trust it
  only for iris + white-box DTA; on wine / under HSJ it goes flat. Prefer **recall** cross-dataset.
  → **RESOLVED AS NEGATIVE (2026-08-17, §8):** four dimension-robust candidates tested on the
  same clouds — **none recovers a meaningful wine signal.** Best case (kNN-local spread) is a
  +4% ripple with clean CIs on a clean baseline already as large as the defect arms; ratio
  metrics provably null (uniform stretch); PCA fixes concentration (0.56→0.82) but not the
  effect. Conclusion: spread is **genuinely dimension-limited**; **recall is the recommended
  diagnostic**. Spread usable only for white-box iris-like cases.
- **before-split is the wrong protocol** for both accuracy and the geometry metric — always
  train-only with a clean test (`06-lessons-gotchas.md`).
- **Benchmark against existing bias detection.** Aiden's notes mention Katerina
  Dost's bias-detection work. Should we compare our signal against an existing
  method rather than treating it as purely exploratory?
- **The coverage-gap mechanism is a hypothesis.** "Boundary in extrapolated
  space → points scatter" is consistent with the data (compression ratio,
  perturbation) but not directly proven. Visualising the actual boundary and
  adversarial landing points in 2D would strengthen it.

## The direction decision (for the supervisor)

- **Option A (recommended): deepen.** Make coverage-gap bulletproof — fix the
  confound, prove it scales to higher dimensions, find one defect where geometry
  beats accuracy. Narrow but defensible.
- **Option B: broaden.** Map the signal across more defects/models/datasets — a
  characterisation study rather than one strong claim.
- **Option C: pivot** if the supervisor thinks the accuracy-confound weakens the
  premise.
