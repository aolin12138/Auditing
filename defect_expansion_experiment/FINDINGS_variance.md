# Phase 1 unified sweep + second-dataset variance check (2026-08-16)

One (model, attack) — **overfit tree + white-box DecisionTreeAttack** (deterministic, no hangs,
30 seeds) — swept across defect severity, factored by defect type (**spatial = coverage gap**
vs **random = imbalance**) and injection **protocol** (**train-only** clean test vs
**before-split** deletes the band from test too). Run on **two datasets** to test whether the
findings are iris-specific.

- Runner: `run_variance.py` · data: `results_variance.parquet` (1260 rows) · figs:
  `plots/variance/tree_dta_iris.png`, `plots/variance/tree_dta_wine.png`, `plots/variance/spread_fragility_map.png`.
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
`spread_fragility_map.png` shows it at a glance: iris-spatial climbs to ~1.25×, wine-spatial
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

## F4 — SVM + HSJ (black-box) replicates the mechanisms AND weakens the spread signal further

Ran **svm + HopSkipJump** across both datasets (2×2 structure×protocol × frac × 15 seeds,
`results_variance_svm.parquet`; `plots/variance/svm_hsj_{iris,wine}.png`,
`plots/variance/spread_fragility_map.png` fragility map).

- **Recall discriminator replicates and is strong** (train-only, frac 0.9): spatial vs random
  minority recall — iris 0.44 vs 0.67, **wine 0.48 vs 0.90** (gap 0.42). Robust across dataset
  **and** model/attack.
- **Spread signal weakens *further* under HSJ:** iris spatial only ~1.11× (vs 1.26× under white-box
  DTA); wine flat ~1.0×. So the geometry-spread signal is fragile to **both** dimensionality
  (dataset) **and** attack — black-box HSJ lands on the smooth SVM boundary surface and is less
  geometry-sensitive than white-box DTA, which lands on discrete tree leaves. The fragility map
  shows it at a glance: strong *only* on iris+tree/DTA (~1.25×), everything else ≈ 1.0×.
- **Before-split composition artifact + accuracy confound both replicate** under SVM+HSJ.
- **Conclusion.** Recall is the robust **cross-dataset AND cross-model** signal; the
  spread/geometry signal clearly fires only for iris + white-box tree/DTA.

---

# Mechanisms (observation → hypothesis → evidence → conclusion)

Each mechanism was measured, not reasoned by analogy (probe scripts run ad-hoc; numbers below
reproducible from the same seeds range(300,320), overfit tree + DTA, standardized features).

## M1 — Why random imbalance moves the spread far less than coverage gap (train-only)

- **Observation.** Under train-only injection, spatial (coverage gap) raises adversarial spread
  much more than random (imbalance): iris 1.26× vs 1.12×; wine 1.04× vs 1.00× (frac 0.9).
- **Hypothesis.** Random deletion keeps the class's spatial extent (only thins density) → the
  decision boundary stays put → adversarial points land in the same leaves → spread unchanged.
  Spatial deletion removes a contiguous band → the boundary must extrapolate across the gap →
  adversarial points relocate to farther leaves → spread rises.
- **Evidence FOR.** At frac 0.9 on iris the tree loses the *same* number of leaves for both arms
  (−4.6) yet only spatial moves the spread → it is *where* the boundary shifts (directional), not
  how much. The deleted class's own adversarial spread rises more under spatial than random.
- **Evidence AGAINST "pure density".** At extreme severity (90%) random *also* coarsens the
  boundary somewhat: the tc-class adversarial spread rises +25% (iris) / +7% (wine) and leaves
  drop −4.2 (iris). So random is "mostly density thinning **plus** mild boundary coarsening at the
  extreme," not strictly density-only.
- **Conclusion.** Coverage gap = a *directional, structured* boundary distortion (strong signal);
  random imbalance = density thinning that only coarsens the boundary mildly at high severity.
  Confirmed — with the refinement that at 90% deletion random is not perfectly geometry-preserving.

## M2 — Why before-split makes BOTH arms move the spread identically (a composition artifact)

- **Observation.** Under before-split injection, random and coverage-gap give the *same* spread
  (wine ~2.9 both; iris both rise together), unlike train-only where they separate.
- **Hypothesis A (composition).** Before-split removes the target class from the *test* set
  equally for both arms → tc adversarial examples drop out of the cloud → the surviving c1/c2 cloud
  is identical → same spread; the rise vs baseline is because the dropped tc cluster was tight.
- **Hypothesis B (boundary).** With tc nearly gone from *training*, the surviving c1/c2 points
  land differently → they themselves spread out.
- **Evidence FOR A, AGAINST B** (wine, frac 0.9, decomposition): tc points in the test fold drop
  11.8 → 1.2 (identical both arms); tc adversarial examples drop 8 → 0.8; c1/c2 counts are
  identical across arms (13.2 / 8.7). Baseline **c1+c2-only** spread = 2.899 ≈ before-split full
  spread 2.917. The **c1 and c2 individual spreads stay flat** across every condition
  (c1 3.48→3.50, c2 3.12→3.19) → the surviving boundary did **not** change. So it is pure
  composition (A); B is refuted.
- **Conclusion.** The before-split spread change is a **test-set composition artifact**: deleting
  the tight tc cluster from the test cloud raises the mean, identically for both arms (same tc
  count removed), and reflects nothing about the defect's boundary effect. It is the
  geometry-sibling of the accuracy confound. **Train-only is the correct protocol.**

## M3 — Why the spread signal is dataset-fragile (iris strong, wine flat): dimensionality

- **Observation.** Spatial spread reaches 1.26× on iris but only 1.04× on wine; random rises to
  1.12× on iris but stays flat (1.00×) on wine.
- **Hypothesis.** Dimensionality (13-D wine) weakens the geometry signal, via (a) a boundary more
  robust to deletion and (b) distance concentration numbing the mean-pairwise-distance metric.
  Rival hypothesis: it is class *overlap*, not dimensionality.
- **Evidence FOR dimensionality.** Adversarial-distance concentration (std/mean) = 0.54 (iris) vs
  0.29 (wine) — high-D concentrates distances. Boundary coarsening under random deletion: −4.2
  leaves (iris) vs −2.0 (wine). tc-class spread rise: +25% (iris) vs +7% (wine). Full-cloud
  dilution: wine's +7% tc rise shows as only +2% in the full cloud (numb metric). DTA displacement
  carried by ~2.4/4 axes on iris (60% of the space) vs 2.8/13 on wine (22%).
- **Evidence on overlap (secondary).** Both target classes are contested (iris virginica baseline
  recall 0.92; wine class-0 0.88) and a *separable* class (iris setosa) gives a null signal — so
  overlap *enables* any signal but is equal-ish across the two target classes, so it does not
  explain the iris→wine difference.
- **Conclusion.** The spread signal's fragility is **primarily dimensionality** (robust boundary +
  numb metric), not overlap. Overlap is necessary for any signal but not the cause of wine's
  flatness. Implication: the geometry/spread diagnostic needs a **dimension-robust** metric to
  generalize; the recall discriminator (distance-free) is the robust cross-dataset signal.

## M4 — How minority recall is computed, and why (method note)

- **Definition.** `min_recall` = recall of the depleted **target class** `tc` on the *clean
  held-out test fold* = (#test points truly `tc` predicted `tc`) / (#test points truly `tc`) =
  TP / (TP + FN). Code (`run_variance.py:90`, `run_imbalance.py:79`):
  `mmask = yv == tc; min_recall = (pv[mmask] == tc).mean()`. Computed per fold, averaged over 5
  folds then over seeds; NaN if a fold has no `tc` test points.
- **Naming caveat.** `min_recall` means **minority-class recall** (recall of `tc`), **not** the
  minimum recall across classes. `run_p1.py` additionally stores `recall_c0/c1/c2`.
- **Why recall, why per-class.** (1) It **isolates the damaged class** — overall accuracy averages
  all 3 classes so depleting one barely moves it (the other ⅔ of the test set is fine), hiding the
  harm. (2) **Recall, not precision** — the defect makes the model *under-predict* `tc`, so its
  true instances become false negatives (recall↓); precision is far less affected. (3) It is
  **distance-free**, so unlike spread it does not degrade with dimensionality — which is why it
  stayed the robust discriminator from iris → wine.
- **Caveat.** Under **before-split**, recall is measured on only ~1.2 tc test points (M2), so those
  panels are noisy/near-meaningless — another reason train-only is the correct protocol.

## Caveats
- Model coverage: cross-dataset variance now covers **tree + white-box DTA (30 seeds)** and
  **SVM + black-box HSJ (15 seeds)** on iris + wine, both protocols (F4). RF + HSJ remains
  impractical cross-dataset (~73 s/cell + hangs) — not run. Global StandardScaler (mild leakage,
  acceptable for this exploratory check).
- Two datasets only — wine is still low-dimensional by real-world standards; a >50-D set
  (digits, or a real tabular dataset) would test the spread collapse harder.
