# CONTEXT.md — Domain Glossary

> Terms that have specific meanings in this project, or differ from standard ML usage.

## Core metrics

**spread** — Mean pairwise Euclidean distance of adversarial points within one OPTICS cluster.
Higher = points are farther apart ("scattering"). The primary reported metric.
Equivalent to the inverse of Aiden's density.

**density** — `n_pairs / (sum_pairwise_dist + 1)` per cluster. Approximately `1/spread`.
Do NOT confuse with probability density. In this project, density and spread carry
the same signal; spread is reported because a distance is more interpretable.

**compression ratio** — `spread(adv cloud) / spread(original test points)`. Ratio < 1
means the adversarial cloud is tighter than the test points — genuine attack effect.
Ratio ≈ 1 means points barely moved — no attack signal.

**perturbation** — `‖adv_point − original_test_point‖` (L2). How far a single point
traveled. Not the same as spread (which measures distance *between* points).

## Defects

**coverage gap** — Delete the bottom `bias` fraction of one class, sorted along one
feature. Simulates an under-sampled subpopulation. Creates a clean contiguous hole.
Grid: `bias` ∈ [0.1, 0.3, 0.5, 0.7, 0.9].

**label noise** — Randomly flip `noise` fraction of training labels to a different
class. Test labels stay clean. Simulates annotation errors.
Grid: `noise` ∈ [0.1, ..., 0.9].

**bias** — In this project: the fraction deleted in a coverage gap (0.1–0.9).
NOT model bias (statistical bias-variance).

## Data columns (results.parquet)

| Column | Meaning |
|--------|---------|
| `tc` | Target class — which class is depleted (0, 1, or 2). −1 for label noise. |
| `feat` | Feature index (0–3) along which the coverage gap deletion happens. −1 for label noise. |
| `bias` / `level` / `noise` | Defect severity fraction (0.1–0.9). Named differently across experiment grids. |
| `tacc` / `vacc` | Training accuracy / validation (test) accuracy. |
| `asucc` | Attack success rate — fraction of correctly-classified points that were successfully flipped. |
| `nadv` | Number of successful adversarial examples produced. |
| `nclust` | Number of OPTICS clusters found in the adversarial set. |
| `clust_size` | Mean number of points per cluster. |
| `mean_dist` | Mean pairwise distance per cluster (= spread). |
| `aiden_density` | Aiden's original density formula (uses `np.linalg.norm((p1,p2))` — Frobenius norm bug). |
| `seed` | StratifiedKFold split seed. |
| `noise_seed` | Per-run random seed for label-noise injection. |

## Attacks

**HSJ (HopSkipJump)** — Decision-based black-box attack from ART. Untargeted, L2.
Config: `max_iter=10, max_eval=200, init_eval=50`. Needs only predicted labels.
Non-deterministic: random init + Monte-Carlo boundary-direction estimation.
Hangs frequently on decision trees (flat axis-aligned facets) — rows terminated
via subprocess timeout (15 s) and recorded as NaN.

**DTA (DecisionTreeAttack)** — White-box attack from ART. Reads tree structure
directly, finds exact minimal perturbation to reach the nearest different-class
leaf. Deterministic. Tree-only.

## Models

**DecisionTree** — `DecisionTreeClassifier(max_depth=3, random_state=42)`.
**SVM** — `SVC(kernel='rbf', probability=True, random_state=42)`. Multi-class via one-vs-one.

## Clustering

**OPTICS** — Density-adaptive clustering (`min_samples=3, xi=0.05, min_cluster_size=3`).
Used instead of DBSCAN because DBSCAN requires a fixed global density threshold ε,
which would pre-filter the density variation being measured.

## Experiment grids

| Grid | File | Rows | Defects | Model | Attack |
|------|------|------|---------|-------|--------|
| 1 | `dtree_attack_experiment/results.parquet` | 504 | CG + LN | Tree | DTA |
| 2 | `hsj_svm_experiment/results.parquet` | 360 | CG only | Tree + SVM | HSJ |
| 3 | `hsj_label_noise_experiment/results.parquet` | 648 | LN only | Tree + SVM | HSJ |

All: iris dataset, 5-fold stratified CV, 3 random seeds, all (class, feature) combos for CG.
