# Auditing — Detecting Dataset Bias via Adversarial-Example Geometry

Can you tell that a model was trained on **biased data** just by attacking it?

We train classifiers on deliberately defective versions of a dataset, generate
adversarial examples against them, cluster those adversarial points, and ask
whether the **geometry of the adversarial clusters** (density / spread) tracks
the severity of the training-data defect. If it does, adversarial geometry is a
black-box **audit signal** for dataset bias — no access to the training data
needed.

## TL;DR — Key findings

1. **Coverage-gap bias is detectable.** Mean pairwise distance of adversarial
   points within clusters ("clean spread") increases monotonically with bias on
   **every** model + attack combination we tried — Tree + DecisionTreeAttack
   (Cohen's d = +2.06), Tree + HopSkipJump (d = +0.38), SVM + HopSkipJump
   (d = +0.75) — while test accuracy stays flat (~0.96). The model looks fine
   on accuracy; the adversarial geometry gives the bias away.
2. **The metric matters — a lot.** The original "density" metric had a norm bug
   (`np.linalg.norm((p1, p2))` computes the Frobenius norm of two *stacked*
   points, not the distance `‖p1 − p2‖`) and, even after fixing it, the
   `n / (dist + 1)` form confounds cluster **size** with cluster **geometry**.
   The buggy metric inverted the true signal on trees: it reported adversarial
   points *densifying* with bias when they are actually **spreading apart**.
3. **Label noise is not cleanly detectable.** Spread also grows with label
   noise, but only alongside an accuracy collapse (SVM: 0.96 → 0.78 at 50%
   noise), so plain accuracy already flags the problem — the geometric signal
   adds nothing model-agnostic there.
4. **The signal transfers across model families.** The same untargeted
   HopSkipJump (decision-based, label-only) recovers the coverage-gap signal on
   both a depth-3 decision tree and an RBF-kernel SVC, so the diagnostic is not
   an artifact of one hypothesis class.

## The three experiments

All experiments use **iris** (150 samples, 4 features, 3 classes) and a
**5-fold stratified CV** harness. Each folder is self-contained: a runner, a
`results.parquet` (complete, 360 rows), a `progress.txt`, and `plots/`.

| Folder | Model(s) | Attack | Defect | Grid (360 runs each) |
|---|---|---|---|---|
| `dtree_attack_experiment/` | DecisionTree (depth 3) | DecisionTreeAttack (white-box, structural) | coverage gap **and** label noise | 5 levels × 3 seeds × (3 classes × 4 features \| 12 noise-seeds) |
| `hsj_svm_experiment/` | DecisionTree **and** SVC(rbf) | HopSkipJump (black-box, L2, untargeted) | coverage gap | 5 bias × 3 seeds × 3 classes × 4 features × 2 models |
| `hsj_label_noise_experiment/` | DecisionTree **and** SVC(rbf) | HopSkipJump (black-box, L2, untargeted) | label noise | 5 noise × 3 split-seeds × 12 noise-seeds × 2 models |

**The two defects:**

- **Coverage gap** (`inject_bias`): for a target class, sort by one feature and
  delete the bottom `bias` fraction (0.1 … 0.9) of that class from the data —
  simulating a systematically under-sampled subpopulation.
- **Label noise** (`flip_labels`): flip a fraction (0.1 … 0.5) of *training*
  labels to a random other class; test labels stay clean.

## Pipeline (identical in every run)

1. Inject the defect into iris at the given level.
2. 5-fold stratified CV; train the model on each training fold.
3. Attack only the **correctly-classified** test-fold points; keep only
   **successful** adversarials (predicted label actually changed).
4. Cluster the adversarial points with **OPTICS**
   (`min_samples=3, xi=0.05, min_cluster_size=3`).
5. Per cluster, compute **three metrics on the same points** (the metric
   decomposition — see below); average over clusters, then over folds.

HopSkipJump settings everywhere: `norm=2, max_iter=10, max_eval=200,
init_eval=50`, untargeted. The HSJ runners execute each row in a fresh
subprocess with a hard timeout, because HSJ occasionally fails to converge on
trees (piecewise-constant surface) — hung rows are recorded as NaN and skipped
(12 rows in `hsj_svm`, 27 in `hsj_label_noise`).

## The metric decomposition (why three metrics)

| Column | Formula | Problem |
|---|---|---|
| `aiden_density` | `n / (Frobenius_norm_of_stacked_points + 1)` | the original **buggy** metric — norm of stacked points, not pairwise distance; sensitive to point magnitudes, inverted the true signal |
| `density` | `n / (mean_pairwise_dist + 1)` | norm fixed, but still **confounds** cluster size `n` with geometry |
| `mean_dist` | `mean pairwise distance` (scipy `pdist`) | **clean spread** — geometry only, the metric we trust |

Every `results.parquet` carries all three, so the whole story is reproducible
from one file per experiment.

## Results at a glance (clean spread, low → high defect, Cohen's d)

| Combination | Coverage gap (0.1 → 0.9) | Label noise (0.1 → 0.5) |
|---|---|---|
| Tree + DecisionTreeAttack | 0.44 → 0.56 (**d = +2.06**) | 0.46 → 0.62 (d = +2.02) |
| Tree + HopSkipJump | 0.62 → 0.65 (d = +0.38) | 0.62 → 0.75 (d = +1.57) |
| SVM + HopSkipJump | 0.53 → 0.59 (**d = +0.75**) | 0.53 → 0.66 (d = +1.08) |
| Test accuracy | **flat** (~0.96 at every bias) | **collapses** (0.96 → 0.78) |

The coverage-gap column is the interesting one: the geometry moves while
accuracy doesn't. Intuition: a large coverage gap makes the remaining class
easy to separate — the boundary sits in an extrapolated, data-free region, so
successful adversarials scatter instead of concentrating at natural
weak spots. The label-noise column is confounded — the spread grows because
the model itself is degrading, which accuracy already shows.

## Repository map

```
├── README.md                     ← you are here
├── master_analysis.ipynb         ← THE analysis notebook: reads the three
│                                    results.parquet files, decomposes the metrics,
│                                    produces all comparison plots & Cohen's d tables
├── dtree_attack_experiment/      ← Tree + DecisionTreeAttack, both defects
│   ├── run_experiment.py         ← resumable runner (360 rows)
│   ├── results.parquet           ← complete results (committed)
│   └── plots/
├── hsj_svm_experiment/           ← Tree & SVM + HopSkipJump, coverage gap
│   ├── hsj_bias_experiment.ipynb ← resumable runner notebook (360 rows)
│   ├── run_experiment.py         ← script version of the same grid
│   ├── results.parquet
│   └── plots/
├── hsj_label_noise_experiment/   ← Tree & SVM + HopSkipJump, label noise
│   ├── run_experiment.py         ← subprocess-per-row runner with hang timeout
│   ├── results.parquet
│   └── plots/
├── experiments/                  ← exploratory phase: earlier metric-bug
│                                    investigation scripts, density variants,
│                                    label-noise EDA (eda_v2.ipynb), plot scripts
├── eda.ipynb / eda_bias.ipynb    ← early EDA of the bias-injection idea
├── bias.py / util.py / main.py   ← original tree-bias audit harness (pre-dates
│                                    the three experiment folders)
└── *.png                         ← investigation-phase figures (metric-fix
                                     validation, label-noise comparisons)
```

`data/` is gitignored (large logs and intermediate parquets); the three
experiment folders carry their own committed `results.parquet`, so nothing in
`data/` is needed to reproduce the analysis.

## How to run

Environment (Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync                                  # creates .venv from uv.lock
.venv/Scripts/activate                   # Windows (source .venv/bin/activate on Unix)
```

**Just the analysis** (fast, no experiments re-run):

> Open `master_analysis.ipynb`, select the `.venv` Jupyter kernel
> ("Auditing (.venv)"), Run All. It only reads the three committed
> `results.parquet` files.

**Re-running an experiment** (each ~15–30 min, all resumable — completed rows
are skipped via the existing `results.parquet`; delete it for a fresh run):

```bash
python dtree_attack_experiment/run_experiment.py
python hsj_svm_experiment/run_experiment.py
python hsj_label_noise_experiment/run_experiment.py
```

`progress.txt` in each folder shows live progress / ETA.

## Attack primer (for context)

- **HopSkipJump** is *decision-based*: it only sees predicted labels, never
  gradients or internals. Untargeted: start from a correctly classified point,
  jump to any point of another class, binary-search the label flip to locate
  the boundary, then iteratively refine with estimated gradients. For the
  multi-class SVC (sklearn one-vs-one under the hood) the attack is blind to
  the pairwise structure — any label change counts.
- **DecisionTreeAttack** is *white-box* and tree-specific: it walks the tree
  structure to find the minimal path change, so it is exact and fast (no
  convergence issues) — used as the structural baseline.
