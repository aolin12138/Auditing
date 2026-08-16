# Phase 1 unified sweep + second-dataset variance check (2026-08-16)

One (model, attack) — **overfit tree + white-box DecisionTreeAttack** (deterministic, no hangs,
30 seeds) — swept across defect severity, factored by defect type (**spatial = coverage gap**
vs **random = imbalance**) and injection **protocol** (**train-only** clean test vs
**before-split** deletes the band from test too). Run on **two datasets** to test whether the
findings are iris-specific.

- Runner: `run_variance.py` · data: `results_variance.parquet` (1260 rows) · figs:
  `plots/variance_iris.png`, `plots/variance_wine.png`, `plots/variance_spread_summary.png`.
- Features **standardized** (required for wine: proline otherwise dominates all distances).
- Per dataset, target class + spatial feature chosen by measurement (most contested class =
  lowest baseline recall; spatial feature = most class-discriminative):
  **iris** tc=2 virginica, feat=petal width (f3) · **wine** tc=0, feat=proline (f12).
- 13-D check: OPTICS still forms adversarial clusters on wine (nclust≈4), so the spread metric
  is *defined* there — the question is whether it still *carries signal*.

Each figure: rows = [test accuracy, minority recall, normalised spread], cols = [train-only,
before-split], lines = spatial (red) vs random (green), 95% CI.

---

## F1 — Accuracy + minority-recall discriminator REPLICATES on both datasets ✅ (robust)

At frac=0.9, train-only (matched count removed from the contested class):

| dataset | minority recall — random | minority recall — spatial | gap |
|---------|--------------------------|---------------------------|-----|
| iris (4-D)  | 0.702 ± 0.025 | 0.575 ± 0.017 | 0.13 |
| wine (13-D) | 0.705 ± 0.026 | 0.519 ± 0.021 | 0.19 |

Spatial deletion (a contiguous band) craters minority recall harder than uniform random
deletion on **both** datasets, CIs separate. Overall accuracy shows the same ordering (spatial
lower) but blunted (~⅓ the effect, since only 1 class of 3 is depleted). **The per-class-recall
signal is dataset-robust.**

## F2 — The adversarial-SPREAD signal is DATASET-FRAGILE ⚠️ (the key variance result)

Same cell, normalised spread (× clean baseline), frac=0.9, train-only:

| dataset | spread — random | spread — spatial | spatial − random |
|---------|-----------------|------------------|------------------|
| iris (4-D)  | 1.12 ± 0.04 | **1.26 ± 0.04** | +0.14 (CIs separate) |
| wine (13-D) | 1.00 ± 0.02 | **1.04 ± 0.02** | +0.04 (marginal) |

On iris the spatial hole clearly inflates adversarial spread (the flagship geometry signal); on
**wine (13-D) the spread barely moves** (1.04× spatial vs 1.00× random — CIs nearly touch).
`variance_spread_summary.png` shows it at a glance: iris-spatial climbs to ~1.25×, wine-spatial
crawls to ~1.04×. This is direct evidence for the long-standing caveat (`MEETING_NOTES.md`,
`.wiki/06`): **the OPTICS-spread metric degrades as dimensionality rises** — it is *not* a
reliable cross-dataset diagnostic. The robust cross-dataset separator is **recall (F1)**, not
spread.

## F3 — The before-split ACCURACY CONFOUND replicates on both datasets ✅

Spatial deletion, class-contested, frac=0.9 — test accuracy by protocol:

| dataset | train-only (clean test) | before-split (deletes test band) | lift |
|---------|-------------------------|----------------------------------|------|
| iris | 0.857 ± 0.005 | 0.982 ± 0.001 | +0.125 |
| wine | 0.796 ± 0.008 | 0.924 ± 0.007 | +0.128 |

before-split injection *lifts* accuracy by ~0.13 on both — deleting the hard band from the test
set removes exactly the points the model would miss. The minority-recall panels show the same
masking (before-split keeps recall high until the class is nearly gone). **The confound is a
protocol artifact that generalises across datasets**, not an iris quirk.

---

## Scorecard
- **Recall/accuracy discriminator (coverage gap vs imbalance): robust** — replicates iris → wine.
- **Adversarial-spread geometry signal: fragile** — strong on iris (4-D), near-null on wine
  (13-D). Dimensionality is the likely cause (OPTICS density estimates degrade).
- **before-split vs train-only accuracy confound: robust** — replicates on both.

## Honest takeaway
The project's headline "adversarial geometry as a black-box diagnostic" rests on the **spread**
metric, which this variance check shows does **not** survive a move from 4-D iris to 13-D wine.
What *does* survive is the cheaper, model-agnostic **per-class recall** signal and the
**protocol-confound correction**. Next step to rescue the geometry angle would be a
dimension-robust spread measure (kNN-distance ratio, relative density vs clean data, or PCA-then-
OPTICS) — see PLAN / open questions.

## Caveats
- One model+attack (tree+DTA) so far; SVM+HSJ / RF+HSJ cross-dataset not yet run (cheap for SVM,
  slow for RF). Global StandardScaler (mild leakage, acceptable for this exploratory check).
- Two datasets only — wine is still low-dimensional by real-world standards; a >50-D set
  (digits, or a real tabular dataset) would test the spread collapse harder.
