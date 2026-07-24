# Project Handoff — Adversarial-Geometry Dataset Auditing

> Drop this into Claude / ChatGPT / any LLM session. It contains everything needed
> to answer questions about the project. Self-contained, no exploration required.

---

## 1. What this project is

**One sentence:** We train models on deliberately biased data, attack them from the outside (black-box), and check whether the **spread of adversarial examples** reveals the bias — even when test accuracy says everything is fine.

**Answer:** For coverage-gap bias, **yes**. Spread rises while accuracy stays flat (~96%). For label noise, **no** — accuracy collapses in lockstep, so the defect was never hidden.

University of Auckland, Part IV (Honours) Engineering. Repo: `github.com/aolin12138/Auditing`.

---

## 2. Repo structure (where everything lives)

```
E:/OneDrive - The University of Auckland/Desktop/Model Auditing/Auditing/
├── dtree_attack_experiment/        Grid 1: Tree + DecisionTreeAttack (both defects)
│   ├── run_experiment.py           Runner (resumable, appends to parquet)
│   ├── results.parquet             504 rows x 15 cols (defect, level, seed, ..., mean_dist)
│   └── progress.txt
├── hsj_svm_experiment/             Grid 2: Tree+SVM + HopSkipJump (coverage gap only)
│   ├── run_experiment.py
│   ├── results.parquet             360 rows
│   └── plots/
├── hsj_label_noise_experiment/     Grid 3: Tree+SVM + HopSkipJump (label noise, full 0.1-0.9)
│   ├── run_experiment.py
│   ├── results.parquet             648 rows (89 hung/skipped)
│   └── plots/
├── experiments/exploratory/        Early-stage scripts (synth3d, v2 tree strategies)
├── experiments/probes/             Diagnostic probes
│   ├── _probe_move.py              Label noise compression-ratio verification
│   ├── _probe_cg.py                Coverage gap compression-ratio verification
│   └── _probe_review.py            Per-point perturbation + source-class decomposition
├── figures/report/                 Publication-quality PNGs
│   ├── p1_coverage_gap.png         Main finding (z-scored spread, 3 combos + accuracy)
│   ├── p2_label_noise.png          Full range 0.1-0.9, 3 combos, shaded >0.5 regime
│   ├── p3_metrics.png              Metric decomposition
│   ├── p4_discriminant.png         Compression ratio + perturbation
│   ├── p5_strategy_iris.png        Overfit vs pruned trees
│   └── p6_aiden_original.png       Aiden's original signal
├── presentation/
│   ├── deck.html                   11-slide conference deck (HTML, Space Grotesk + IBM Plex)
│   ├── deck-stage.js / support.js  Deck runtime
│   ├── p1_coverage_gap.png         Copy of main figure
│   ├── p2_label_noise.png          Copy of label-noise figure
│   ├── p4_discriminant.png         Copy of verification figure
│   └── SPEAKER_SCRIPT.md           Aolin's script for slides 07-10
├── .wiki/                          Project knowledge base
│   ├── README.md                   Index + headline numbers + status
│   ├── 01-research-question.md
│   ├── 02-background-aiden.md
│   ├── 03-methodology.md
│   ├── 04-findings.md
│   ├── 05-key-decisions.md
│   ├── 06-lessons-gotchas.md
│   ├── 07-repo-map.md
│   ├── 08-open-questions.md
│   └── sessions/2026-07-24-report-and-wiki.md
├── aiden_original/                 Prior work (preserved as-is)
├── generate_report.py              Builds midyear_report.docx from parquet data
├── generate_figures_v2.py          Clean Tufte-style report figures (preferred)
├── midyear_report.docx             Mid-year technical report (4-6 pages)
├── master_analysis.ipynb           Aggregates all 3 experiments
├── pyproject.toml                  Python 3.14, uv-managed
└── .venv/                          Virtual environment

# Key outputs NOT in repo (gitignored):
├── data/data_bias.parquet          Aiden's original bias output
├── data/data_v2.parquet            Tree strategy experiment
└── data/data_label_noise*.parquet  Early label-noise runs
```

---

## 3. Exact pipeline (5 stages)

```
1. Inject defect → iris dataset, 3 classes × 4 features
2. 5-fold StratifiedKFold → train DecisionTree(max_depth=3) or SVC(rbf, probability=True)
3. Attack → DecisionTreeAttack (white-box) or HopSkipJump (black-box, L2,
              max_iter=10, max_eval=200, init_eval=50, untargeted)
              Only correctly-classified test points; keep only successful (label-flipping) adv
4. Cluster → OPTICS(min_samples=3, xi=0.05, min_cluster_size=3) on adversarial points
5. Measure → per cluster: mean pairwise Euclidean distance (spread);
              also: density, cluster_size, aiden_density
```

## 4. Two defects

### Coverage gap (THE result)
Sort one class's points along one feature, delete the bottom `bias` fraction
(0.1–0.9). Contiguous hole. Simulates an under-sampled subpopulation.
Grid: 5 bias × 3 seeds × 3 classes × 4 features = 180 rows per model/attack.

### Label noise (negative result)
Randomly flip `noise` fraction (0.1–0.9) of training labels to a different class.
Test labels stay clean. Simulates annotation errors.
Grid: 9 noise × 3 seeds × 12 noise-seeds = 324 rows per model (HSJ: ×2 models = 648).

---

## 5. Key findings — exact numbers

### Coverage gap: spread rises, accuracy flat
(Target class bleeds from 50→4 points as bias 0.1→0.9.)

| Combination | spread 0.1→0.9 | Cohen's d | Accuracy |
|-------------|----------------|-----------|----------|
| Tree + DTA | 0.44 → 0.56 | **+2.06** | flat ~0.95 |
| SVM + HSJ | 0.53 → 0.59 | **+0.75** | flat ~0.96 |
| Tree + HSJ | 0.62 → 0.65 | +0.38 (weak) | flat |

**Mechanism:** deleting a contiguous class region forces the boundary into an
extrapolated, data-free zone. Adversarial points crossing it scatter — no
training-data density to constrain them. Surviving data still classifies
correctly → accuracy flat.

**Tree+HSJ is the weak combination:** HSJ estimates boundary direction by
sampling perturbations; on a tree's flat axis-aligned facets most samples
stay inside the same leaf, so the estimate is noisy → frequent non-convergence
(12-27 rows terminated as NaN per grid).

**Bias 0.7→0.9 drop (Tree+HSJ only):** at bias 0.9 only 4/50 target-class
points survive. The depleted class runs out of attackable test points → HSJ
reverts to probing the intact-class boundary in dense space → spread drops.
**Tree+DTA does NOT drop** — DTA is deterministic and still finds the far
stretched boundary. It's an attack-mechanism effect (stochastic HSJ vs
deterministic DTA), not a model flaw. See per-class data:

| tc=0 spread | bias 0.7 | bias 0.9 |
|-------------|----------|----------|
| Tree+HSJ | 0.806 | 0.627 ↓ |
| SVM+HSJ | 0.616 | 0.533 ↓ |
| Tree+DTA | 0.574 | 0.610 ↑ |

### Label noise: spread rises BUT confounded, then destabilises
(Full 0.1–0.9 range, grids extended. All three combos.)

| combo | spread 0.1→0.5 | Cohen's d | accuracy 0.1→0.5 |
|-------|----------------|-----------|-------------------|
| Tree+DTA | 0.455 → 0.616 | **+2.02** | 0.927 → 0.641 |
| Tree+HSJ | 0.620 → 0.745 | **+1.57** | 0.927 → 0.641 |
| SVM+HSJ | 0.526 → 0.663 | **+1.08** | 0.956 → 0.776 |

Spread rises — the geometry moves. But **accuracy falls in lockstep** →
defect is NOT hidden → geometry adds nothing.

**Above noise 0.5:** variance explodes (std 0.05→0.6-0.9, ~10×), valid runs
collapse (Tree+DTA: 36→12 at 0.9), model accuracy falls below the 1/3 chance
line → metric thrashes, no usable trend. Trees explode (spread→1.8), SVM stays
bounded (~0.8) — both are downstream of a dead model.

**Direction depends on class separability:** spread increases on iris/wine
(well-separated), reverses on Car Evaluation (categorical/overlapping).
Confirmed on synthetic 3D datasets.

### Verification — signal is real (compression ratio)

| Defect | Compression ratio | Perturbation | What it means |
|--------|-------------------|--------------|---------------|
| Label noise | 0.77 → **0.98** | 0.81 → 0.64 (↓) | Points barely move, cloud = copy of test data → artifact |
| Coverage gap | **~0.70** (flat) | 1.38 → 1.70 (↑) | Cloud genuinely compressed, points travel farther → real signal |

Compression ratio = adv_spread / original_test_spread.
Perturbation = ‖adv - original_test_point‖ (measured per-point, not aggregate).

---

## 6. HSJ attack mechanics (verified from ART source)

`art/attacks/evasion/hop_skip_jump.py`, untargeted, for one test point `x`:

1. **Init:** draw uniform random noise; keep trying (up to 100) until one is
   misclassified differently from `x` → this random point is the seed.
2. **Binary search:** move the random point toward `x`, stopping just on the
   misclassified side of the nearest boundary. Endpoint ≈ `x + δ` where δ is
   the minimal nudge to cross → sits right at the boundary.
3. **Boundary walk (10 iterations):** estimate boundary normal by sampling
   ~50-200 random perturbations (Monte-Carlo, non-deterministic); step toward
   `x` along the boundary; line-search to maintain misclassification.

**Result:** one adversarial point per correctly-classified test point (minus
failures). The endpoint is always across the boundary (misclassified).
Perturbation = ‖adv − x‖ ≈ distance from `x` to its nearest boundary.

**Why it's non-deterministic:** random init seed + Monte-Carlo direction
estimates → different runs produce different adversarial points. This is the
fundamental difference from DecisionTreeAttack (white-box, deterministic,
exact).

---

## 7. How to run things

```bash
# Activate environment
cd "E:/OneDrive - The University of Auckland/Desktop/Model Auditing/Auditing"
.venv/Scripts/activate

# Analysis only (fast, reads committed parquets):
#   Open master_analysis.ipynb, select .venv kernel, Run All

# Regenerate figures:
python generate_figures_v2.py

# Regenerate report:
python generate_report.py

# Re-run an experiment (resumable, appends to parquet, skips done rows):
python dtree_attack_experiment/run_experiment.py
python hsj_label_noise_experiment/run_experiment.py  # slow, subprocess per row

# Data queries (Polars, fast):
python -c "
import polars as pl
df = pl.read_parquet('hsj_svm_experiment/results.parquet')
print(df.filter(pl.col('model')=='svm').group_by('bias').agg(pl.col('mean_dist').mean()))
"
```

---

## 8. Presentation deck (11 slides)

| # | Slide | Presenter | Content |
|---|-------|-----------|---------|
| 01 | Title | Partner | "Auditing models by attacking them" |
| 02 | Motivation | Partner | 96% accuracy, still biased |
| 03 | Our question | Partner | "Can we find the blind spot by attacking?" → handoff |
| 04 | Attacks 101 | Aolin | Adv examples, HopSkipJump, black-box |
| 05 | Pipeline | Aolin | Inject→Train→Attack→Cluster→Measure |
| 06 | Metric | Aolin | Spread = inverse of density |
| **07** | **Finding** | **Aolin** | **Coverage gap — spread rises, accuracy flat. 3 combos.** |
| **08** | **Label noise** | **Aolin** | **Spread rises but confounded, >0.5 destabilises** |
| **09** | **Verification** | **Aolin** | **Compression ratio: 0.70 vs 0.98** |
| **10** | **Conclusion** | **Aolin** | **One metric can't catch everything → set of metrics** |
| 11 | Takeaway | Partner | "The geometry of failure can." |

Aolin's script: `presentation/SPEAKER_SCRIPT.md`

---

## 9. Common Q&A (prepared answers)

**Q: Why iris? That's tiny.**
A: Correct — it's a proof of concept. We wanted to establish whether the signal
exists at all before scaling. The low dimensionality is a known limitation;
higher-dim datasets are the next phase.

**Q: Doesn't the coverage gap also change the test set?**
A: Yes — that's a real confound. We partially address it with the compression
ratio (slide 09), which shows the signal survives even when we control for the
changing test set. The clean fix — biasing only the training fold — is on our
list.

**Q: If label noise also moves the geometry, isn't that still a signal?**
A: It moves, but accuracy moves with it. The point of our metric is to detect
defects that accuracy *misses*. If accuracy already collapses, you don't need
the geometry — you can see the problem in the first number you check. Label
noise is a negative result, not a failure: it tells us the geometry is
sensitive to *structural* holes specifically.

**Q: Why is Tree+HSJ so weak?**
A: HSJ estimates the boundary direction by sampling perturbations. On a tree's
flat, axis-aligned facets, most random samples stay inside the same leaf
without crossing the boundary → noisy estimate → frequent non-convergence.
It's a geometry mismatch between the attack method and the model surface.

**Q: Why does the blue line dip at bias 0.9?**
A: At bias 0.9 only 4 of 50 points survive in the target class. Almost no
test points in that class remain to be attacked. HSJ (non-deterministic) can't
reliably reach the far stretched boundary of the depleted class, so it defaults
to attacking the intact classes' boundary in dense space → spread drops.
DecisionTreeAttack (deterministic, white-box) still finds the far boundary, so
the green line doesn't drop. It's an attack-mechanism effect.

**Q: How does HopSkipJump actually work?**
A: For each test point: (1) draw random noise until misclassified, (2) binary
search toward the test point to sit just across the nearest boundary,
(3) iteratively walk the boundary toward the test point by estimating the
boundary normal via Monte-Carlo sampling. Endpoint = test point nudged just
across its nearest boundary. Non-deterministic because of random init + sampling.

**Q: What's the "compression ratio"?**
A: adversarial spread ÷ original test-point spread. ~0.70 for coverage gap
means the adversarial cloud is 30% tighter than the test cloud — genuine attack
effect. ~0.98 for label noise means the points barely moved — no attack effect.

**Q: What's next?**
A: Higher-dimensional datasets, more defect types (outliers, feature corruption),
more models (random forest, XGBoost), and a **suite of metrics** — because
different defects need different detectors, and a single metric can't catch
everything.

**Q: Where's the data?**
A: The three `results.parquet` files in each experiment folder. Committed to
GitHub. Each is ~360-648 rows with 15 columns. Readable with `polars.read_parquet()`.

---

## 10. Critical gotchas (don't repeat these)

1. **Aiden's density used `np.linalg.norm((p1,p2))`** — Frobenius norm, not
   pairwise distance. A bug, but his pair-count-invariant form still trended
   correctly. Our spread ≈ 1/Aiden's density — they carry the same signal.
2. **"Density" and "spread" are equivalent** — spread = 1/density ≈ mean
   pairwise distance. Don't claim one is "better." We report spread because
   a distance is more interpretable.
3. **Tree+HSJ failure is NOT "no gradient"** — the boundary has a well-defined
   normal on each flat facet. The real issue: HSJ's *sampling-based* direction
   estimate is noisy on flat surfaces → frequent non-convergence.
4. **Coverage gap deletes ONE class along ONE feature** — not 90% of ALL data.
   At bias 0.9 there are still 104 of 150 points (only setosa collapses to 4).
5. **Label noise spread is NOT flat** — it rises (d=+1.08 to +2.02 up to 0.5).
   The point is that accuracy falls in lockstep, not that the geometry doesn't
   move.
6. **Polars gotchas:** `is_in` not `isin`; NaN propagates through `.mean()`;
   duplicate aggregation names need `.alias()`.
7. **HSJ on trees hangs** — use subprocess timeouts (15s in our runners).
