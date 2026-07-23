# 08 — Open Questions & Future Work

## Urgent / next up

### Fix the coverage-gap test-set confound
Bias is injected before the CV split, so the test set changes with bias level.
The compression ratio partially addresses this, but the clean fix is to **inject
bias into the training fold only and keep a fixed clean test set**. This isolates
"does the model's geometry change" from "do we have different test points."
Cheapest high-value experiment. See [06-lessons-gotchas.md](06-lessons-gotchas.md).

### Higher-dimensional validation
Everything is on iris (4 features). OPTICS is density-based and degrades in high
dimensions (reachability distances concentrate). **Does the spread signal
survive?** Add 1–2 higher-dim datasets (digits, or synthetic 10–20D). If OPTICS
fails, may need dimension-robust measures (relative density vs. original data
density, Hopkins statistic, kNN/rank-based).

### A defect that hurts data quality without collapsing accuracy
The strongest test of the whole premise. Label noise fails partly because it
collapses accuracy. Design a defect that degrades data quality while accuracy
stays high (e.g. systematic mislabeling of one sub-region, feature corruption in
a subpopulation). If the geometry fires when accuracy is blind, that's the
headline result.

## Medium term

- **More model families:** neural networks with gradient-based attacks (PGD).
  Does the signal generalise beyond tree/SVM?
- **More defect types:** feature noise, class imbalance, outliers, systematic
  mislabeling. Map where the signal exists and where it doesn't — a
  characterisation study.
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
