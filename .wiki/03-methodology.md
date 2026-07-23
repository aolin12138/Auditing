# 03 — Methodology

## The pipeline (5 stages)

```
1. Inject defect      →  coverage gap OR label noise into iris
2. 5-fold stratified CV  →  train model on each training fold
3. Attack             →  DecisionTreeAttack or HopSkipJump on
                          correctly-classified test points; keep
                          only successful (label-flipping) perturbations
4. Cluster            →  OPTICS on the adversarial points
5. Measure            →  per cluster: n_points, mean pairwise distance
                          (spread), density; average over clusters, then folds
```

## Dataset

**Iris** — 150 samples, 4 continuous features, 3 balanced classes. Chosen for
continuity with Aiden and because it's small/well-understood. **Limitation:**
low-dimensional; OPTICS degrades in high dimensions (see [08-open-questions.md](08-open-questions.md)).

Label noise also tested on **wine** (well-separated continuous), **Car
Evaluation** (categorical), and **3 synthetic 3D datasets** varying only in
class separability (well-separated / overlapping / categorical).

## Defects

- **Coverage gap:** sort a target class along one feature, delete the bottom
  `bias` fraction (0.1–0.9). Simulates an under-sampled subpopulation. Leaves a
  clean contiguous hole.
- **Label noise:** flip a `noise` fraction (0.1–0.5) of *training* labels to a
  random other class; test labels stay clean. Simulates annotation errors.

## Models

- **DecisionTreeClassifier(max_depth=3)** — continuity with Aiden.
- **SVC(kernel='rbf', probability=True)** — different hypothesis class (smooth
  boundary). Handles multi-class via **one-vs-one** (3 pairwise SVMs for iris).

## Attacks

- **DecisionTreeAttack** — white-box, tree-specific, exact, fast. Baseline.
- **HopSkipJump (HSJ)** — decision-based **black-box**: needs only predicted
  labels. Binary-searches the label-flip boundary, refines via a sampled
  boundary-direction estimate. L2, max_iter=10, max_eval=200, init_eval=50,
  untargeted. **Because HSJ needs no internals, the same attack runs on both
  tree and SVM** — this is what makes cross-model comparison valid.

## Why OPTICS (not DBSCAN)

DBSCAN needs a global density threshold ε — all clusters must share one density,
so it merges dense clusters or discards sparse ones as noise. **We are measuring
changes in adversarial density/spread, so the clustering must be
density-adaptive.** OPTICS orders points by reachability and extracts clusters
at varying density via ξ. Params: min_samples=3, ξ=0.05, min_cluster_size=3.

## The metric

Per cluster:
```
spread   = mean pairwise Euclidean distance   (scipy pdist)
density  = n_pairs / (sum_of_pairwise_distances + 1)  ≈  1 / spread
```

**Density and spread carry the same signal — one is the inverse of the other.**
We report **spread** because a distance is more directly interpretable. We do
NOT claim spread is a "better metric" — see [06-lessons-gotchas.md](06-lessons-gotchas.md)
for the formula subtlety that makes this important.

## Experiment grids (360 runs each)

| Grid | Models | Attack | Defects |
|------|--------|--------|---------|
| 1 (`dtree_attack_experiment/`) | Tree | DecisionTreeAttack | Coverage gap + Label noise |
| 2 (`hsj_svm_experiment/`) | Tree + SVM | HopSkipJump | Coverage gap |
| 3 (`hsj_label_noise_experiment/`) | Tree + SVM | HopSkipJump | Label noise |

Each: 5-fold CV, 3 seeds, all class×feature combinations. HSJ runs each row in a
subprocess with a hard timeout because HSJ frequently hangs on trees (12–27
rows/grid terminated as NaN).
