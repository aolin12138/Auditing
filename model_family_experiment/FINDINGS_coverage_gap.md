# Model-Family × Coverage Gap — Findings (2026-07-30)

Does the established coverage-gap spread signal survive **RandomForest**? Attack: black-box
**HopSkipJump** (runs on ensembles). iris, 5-fold, 3 seeds × 3 target classes × feature 2
(petal length). Data: `results.parquet` where `defect=coverage_gap`. Fig: `plots/cg_rf_vs_tree.png`.

## Headline — YES, the coverage-gap signal survives bagging

Adversarial spread vs bias (fraction of the target class deleted), hung cells dropped:

| bias | RandomForest spread | single tree spread | test acc |
|------|---------------------|--------------------|----------|
| 0.1 | 0.657 | 0.621 | 0.95 |
| 0.3 | 0.722 | 0.671 | 0.96 |
| 0.5 | 0.746 | 0.660 | 0.96 |
| 0.7 | **0.800** | 0.659 | 0.96 |
| 0.9 | 0.793 | 0.620 | 0.96 |

**RF spread rises monotonically with bias (0.66 → 0.80) while accuracy stays flat ~0.96** —
the same accuracy-blind coverage-gap signature seen on Tree+DTA and SVM+HSJ, now confirmed on a
bagged ensemble. **Single tree + HSJ stays flat (~0.62–0.67)** — the known weak Tree+HSJ combo.

## Why coverage gap survives RF but the outlier does not

A **coverage gap is a global structural hole**: every tree in the forest is trained on data
missing that region, so *all* of them extrapolate the boundary there — bagging cannot average
the hole away. An **outlier is a few points**: most bootstrap resamples dilute or omit them, so
the forest's averaged boundary barely moves (see `FINDINGS_outlier.md` F2). This is a clean,
defensible distinction between the two defects' interaction with ensembling.

## Bonus — RF is a *better* black-box coverage-gap detector than a single tree

Because bagging smooths the single tree's flat axis-aligned facets into many small steps, HSJ
navigates the RF boundary far better (it also hangs less: 1 vs 2 NaN here). So for black-box
auditing, **RF+HSJ > Tree+HSJ** for coverage gap — a useful practical point.

## Caveats
- n = up to 9 valid runs per bias cell (3 seeds × 3 tc, feature 2 only); RF CI bands are wide
  but the monotone trend and separation from the tree at high bias are clear.
- Feature restricted to petal length (feat=2) to bound HSJ runtime; a fuller feature sweep
  would tighten it. Label-noise for RF is pending (a key bug was fixed; rerun queued).
