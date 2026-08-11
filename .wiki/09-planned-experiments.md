# 09 — Planned Experiments (Outliers + Model Families + Defect Expansion)

Three experiment threads. A & B are **complete** (2026-08); C is **planned** (2026-08-11).
Detailed contracts live in the experiment folders; this page is the index and rationale.

## Thread A — Outlier defect  → `outlier_experiment/PLAN.md`

A **third defect geometry**: correctly-labeled points placed *far* from their class cloud.
The strongest candidate for the project's headline goal — **a defect that hurts data quality
without collapsing accuracy** (coverage gap is a hole; label noise only fires by wrecking
accuracy; an outlier is a mislocated-but-correct point an over-fit model chases).

Controlled knobs: distance-to-centroid `k`, count `n_out`, target class (one vs all —
the **asymmetry test**), direction (outward/toward), reference (class/global centroid),
kind (correct/mislabeled), axis. **How the outlier is injected is itself a controlled
variable** — Phase 2 sweeps it, because the injection method plausibly changes the result.

Two design choices baked in:
1. **Inject into the training fold only, fixed clean test** — fixes the coverage-gap
   test-set confound ([06-lessons-gotchas.md](06-lessons-gotchas.md)) from the start.
2. **Measure spread per class** — clean classes are the in-run baseline for the asymmetry test.

**Tree depth is now a controlled variable, not `max_depth=3`.** Primary = an *overfitting*
tree (fits training ~100%) so the outlier actually deforms the boundary; a pruned depth-3
arm is kept to prove capacity mediates the signal (ties to [04-findings.md](04-findings.md)
Finding 4, the overfit-vs-pruned flip).

## Thread B — Model families → `model_family_experiment/PLAN.md`

Does the signal generalise beyond Tree/SVM? Add **RandomForest** (bagging → expected to
*damp* the signal) and **XGBoost** (boosting → may *sharpen* it), attacked with **HSJ**
(the only attack that runs on every model — `DecisionTreeAttack` can't read an ensemble).
Plus a **2-feature iris setup** to *visualise* the boundary and adversarial landing points,
closing the "prove the mechanism in 2D" open question. Stretch: small MLP + PGD gradient attack.

## Thread C — Defect expansion → `defect_expansion_experiment/PLAN.md`

Three **new defects**, chosen by the principle learned from A & B: *a defect is visible to the
adversarial-geometry probe iff it imposes a **global, structured** boundary distortion that is
**separable from accuracy***. Priority order:

1. **Class imbalance** ★ — the **count-controlled sibling** of coverage gap: uniform random
   deletion vs spatial (coverage-gap) deletion at matched count. Answers whether the
   coverage-gap signal is the **spatial hole** or just **fewer samples**. Sharpens the
   clearest result. Report **per-class recall** (imbalance dents minority recall).
2. **Shortcut / spurious feature (Clever Hans)** — a 5th feature label-correlated in train but
   not test; the boundary leans on a fake axis while IID accuracy can look fine. Strongest
   "geometry catches what accuracy misses" candidate. Metric: **per-axis** adversarial
   displacement.
3. **Train–test leakage / duplication** — leakage *inflates* accuracy while creating
   memorized, over-confident local geometry. Tests whether the probe flags a defect accuracy
   actively hides. Metric: **memorization locality**.

## Shared infrastructure

- One set of model adapters (RF, XGBoost) feeds **both** threads (outlier H4 = robustness×model).
- Reuse `cluster_stats`, `attack_adv`, OPTICS params, and the **subprocess-timeout HSJ runner**
  (ensembles hang more than single trees).
- New dependency: `xgboost` (add to `pyproject.toml`); `torch` only if pursuing PGD.

## Status

- 2026-07-30: threads A & B plans written, folders scaffolded.
- 2026-08: **Thread A (outlier) COMPLETE** — white-box DTA detects `toward` outliers
  (1.25–1.36×, accuracy-blind), non-monotone, **collapses at the class-size ceiling**;
  `outward`/`random` weak-null; fragile/white-box-only under HSJ. See
  `outlier_experiment/FINDINGS_*` + `plots/{toward,random,model_performance}_all*`.
- 2026-08: **Thread B (model families) COMPLETE** — **coverage gap SURVIVES RandomForest**
  (clearest signal; RF+HSJ > tree+HSJ); outlier does **not** survive bagging; label noise
  **intractable** via HSJ (boundary fragmentation → hangs). XGBoost deferred (hangs). See
  `model_family_experiment/FINDINGS_*`.
- 2026-08-11: **Thread C (defect expansion) planned** — `defect_expansion_experiment/PLAN.md`.
  Next gate: **Phase 0 class imbalance** (tree+DTA, spatial-vs-random deletion control) to
  test the spatial-hole hypothesis before scaling.
- Deferred/open: XGBoost per-point-timeout rework; 2-feature boundary visualisation of the
  `toward` tendril; structured (boundary-localized) label noise.
