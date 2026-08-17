# 08 — Open Questions & Future Work

## Urgent / next up

### Fix the coverage-gap test-set confound
Bias is injected before the CV split, so the test set changes with bias level.
The compression ratio partially addresses this, but the clean fix is to **inject
bias into the training fold only and keep a fixed clean test set**. This isolates
"does the model's geometry change" from "do we have different test points."
Cheapest high-value experiment. See [06-lessons-gotchas.md](06-lessons-gotchas.md).

### Higher-dimensional validation — ✅ INVESTIGATED (2026-08-16): spread does NOT survive
Re-ran the imbalance contrast on **wine (13-D)** + **svm/HSJ** (`FINDINGS_variance.md`, Finding 6).
**The adversarial-spread signal collapses in higher dimensions** (iris spatial 1.26× → wine
~1.04×) and under black-box HSJ (~1.11×) — measured cause = distance concentration (std/mean 0.54
iris vs 0.29 wine). Per-class **recall** stayed robust. **→ next: a dimension-robust spread
metric** (kNN-ratio, LOF-style density ratio, PCA-then-spread, kNN-graph local spread) — test
whether any recovers the wine signal raw spread lost. Full spec in
`defect_expansion_experiment/PLAN.md §8`. A >50-D set (digits) would stress it further.
**Interim conclusion:** if no robust metric is found, recall is the recommended black-box diagnostic.

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
  → **PARTIALLY RESOLVED (2026-08-17, §8):** use **kNN-local spread (M4)** instead of raw OPTICS
  spread — it recovers wine (1.055 vs 1.012 @0.8) and keeps iris; PCA-then-spread (M3) also works
  @0.9. Black-box HSJ recovery is partial (M3 wine @0.9; M4 direction-flipped). Recall still king.
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
