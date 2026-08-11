# Model-Family × Outlier — Findings (2026-07-30)

RandomForest (60 trees) and a single overfit tree, both attacked with the **black-box
HopSkipJump** (the only attack that runs on ensembles). Outlier defect, tc=2, injected into
the training fold only, iris 5-fold. Data: `results.parquet` (rows where `defect=outlier`).
Runner: `run_experiment.py --defect outlier`. **UPDATED to 10 seeds (2026-07-30) — the earlier
n=3 "flip" was noise and is retracted (see F1).**

Strong cell (k=8, n_out=10), normalised to each model's clean baseline, 10 seeds, 95% CI:

| model (HSJ black-box) | toward | outward | random | test acc |
|-----------------------|--------|---------|--------|----------|
| single tree (overfit) | 0.94× [0.87,1.01] | 1.02× [0.95,1.09] | 0.97× [0.88,1.05] | ~0.95 (flat) |
| RandomForest (bagging)| 0.92× [0.86,0.98] | 1.04× [0.98,1.10] | **1.13× [1.02,1.24]** | ~0.95 (flat) |

## F1 — The outlier signal needs the WHITE-BOX attack; black-box HSJ mostly can't see it

At 10 seeds, black-box **HSJ on a single tree shows NO reliable outlier signal** in any
direction (toward 0.94×, outward 1.02×, random 0.97× — all CIs straddle the baseline). This
**retracts** the earlier n=3 claim that HSJ "sees outward" (1.22×) — that was sampling noise.
Contrast the **white-box DecisionTreeAttack** on the same overfit tree, which *does* detect it
robustly at 10 seeds: `toward` **1.25–1.36×** (`../outlier_experiment/FINDINGS_phase05.md`).

**So the outlier is an attack-dependent, fragile signal:** the deterministic white-box attack
that jumps to the nearest leaf feels the `toward` tendril; the stochastic black-box attack
averages it away. (Lesson: n=3 seeds are unreliable — firming to 10 flipped the conclusion.)

## F2 — RandomForest largely suppresses it too; only a weak `random` blip

RF at 10 seeds is ~1.0 for toward (0.92×, actually slight compression) and outward (1.04×);
the one point that clears baseline is `random` at **1.13× [1.02,1.24]** — a weak signal.
Bagging averages the per-tree tendrils away, so RF behaves like the robust rbf-SVM, not like a
single white-box-attacked tree. **The outlier signal does not meaningfully survive ensembling.**

(Contrast coverage gap — see `FINDINGS_coverage_gap.md` — which DOES survive RF, because a
coverage gap is a *global structural hole* every tree shares, whereas an outlier is a few
points most bootstrap samples dilute.)

## F3 — Pushing outliers to the class-size ceiling (n_out = 40 = 100% of class)

All models taken to the ceiling on the `toward` axis (k=8, 10 seeds). The 40-per-class train
fold means n_out=40 doubles the class. Three qualitatively different behaviours:

| model (attack)        | low dose | plateau (25-75%) | ceiling (100%) | accuracy at ceiling |
|-----------------------|----------|------------------|----------------|---------------------|
| tree d10 + DTA (WB)   | rises    | ~1.25×          | **COLLAPSES 1.04×** | train 1.0, test ~0.95 (blind) |
| tree d3 + DTA (WB)    | rises    | ~1.36×          | partial 1.20×   | train ~0.97, test ~0.93 (blind) |
| SVM + HSJ             | ~1.0     | slowly rises     | **RISES 1.09×** (1.17× at k=12) | train/test **DEGRADE** 0.97→0.89 |
| RandomForest + HSJ    | ~1.0     | flat null        | flat 0.99×      | train 1.0, test ~0.95 (blind) |

**Interpretation — the ceiling separates three mechanisms:**
- **Tree (white-box): the signal is non-monotone and COLLAPSES at the ceiling.** Up to ~75% the
  outliers are a minority the overfit tree isolates with long tendrils (big spread). At 100% they
  are no longer anomalies — they are half the class, form their own compact blob, and the tree
  stops stratifying them → spread falls back to baseline. So there is an *optimal detection dose*
  (~25-75%); too many outliers hide themselves.
- **SVM: the signal RISES monotonically and, uniquely, test accuracy DEGRADES.** rbf-SVM ignores a
  few outliers (locality) but 40 same-labeled points finally move the support vectors, widening
  the margin region (spread up) and mislabelling real test points (accuracy down 0.97→0.89).
- **RF: flat null at every dose** — bagging averages the outliers away throughout; nothing to
  collapse or grow. Accuracy stays blind (train 1.0, test ~0.95).

**Headline:** accuracy never flags the outlier for the tree-based models even when injected
outliers equal 100% of the class — only the adversarial-geometry spread reacts, and only via the
white-box attack, and only below the ceiling. Figures: `../outlier_experiment/plots/`
`toward_all_models.png`, `model_performance.png`, `random_all_models.png`.

**Runner note (bug fixed):** the `--ceiling` grid extension was initially invisible to the worker
subprocess because `filter_args()` did not forward the boolean flag → the driver looped ~45 min
with zero progress (1397 log lines stuck at `[130/160]`). Fixed by appending `--ceiling` in
`filter_args()`; a working RF-ceiling batch is ~35 min (23/30 valid, 7 HSJ hangs at high dose).

## F3 — Accuracy blind across all models/attacks

Test accuracy ~0.95 and train ~1.0 for every model/direction — the outlier defect never shows
up in accuracy, only (sometimes) in geometry.

## Caveats / next
- **n=3 seeds** — the white-box/black-box flip (F1) is the headline but needs 10 seeds to firm up.
- Only the strong cell (k=8, n_out=10) summarised; full grid is k∈{4,8}×n_out∈{5,10}.
- XGBoost deferred: XGB+HSJ hangs pervasively (flat-facet pathology; see journal / chat).
- HSJ hung on only 2/40 RF cells (caught by the 90s subprocess timeout); 0/39 tree cells.
