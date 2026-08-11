# 09 — Planned Experiments (Outliers + Model Families)

Two new experiment threads, both drawn from [08-open-questions.md](08-open-questions.md)
("more model families", "outliers"). Detailed contracts live in the experiment folders;
this page is the index and rationale.

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

## Shared infrastructure

- One set of model adapters (RF, XGBoost) feeds **both** threads (outlier H4 = robustness×model).
- Reuse `cluster_stats`, `attack_adv`, OPTICS params, and the **subprocess-timeout HSJ runner**
  (ensembles hang more than single trees).
- New dependency: `xgboost` (add to `pyproject.toml`); `torch` only if pursuing PGD.

## Status

- 2026-07-30: plans written, folders scaffolded. No code run yet. Next gate: Phase 0
  prototype (Tree-overfit + DTA, `tc=0`, sweep `k`,`n_out`) to confirm the outlier signal
  exists before scaling the grid.
