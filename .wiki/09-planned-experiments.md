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
- 2026-08: **Thread C Phase 0 class imbalance DONE** (tree+DTA, 30 seeds) — spatial hole adds
  spread on top of the count effect (H1, CIs separate at frac≥0.5); minority recall the clean
  discriminator. `FINDINGS_imbalance.md`, `plots/imbalance/phase0_spread_vs_random.png`.
- 2026-08-16: **Thread C Phase 1 class imbalance DONE** (`FINDINGS_imbalance_p1.md`) —
  across SVM/tree/RF + black-box HSJ, **minority recall is the robust model-agnostic separator**
  (spatial craters it to 0.14–0.35 vs random 0.55–0.77, tight CIs); the scalar-spread gap
  survives only partially (RF directional n=3, SVM at frac 0.95, absent on tree+HSJ). **Class
  asymmetry confirmed** (deleting from separable setosa = null geometry + recall). **Accuracy
  confound quantified** (before-split acc→1.000 vs train-only→0.714, 30 seeds). Figs
  `plots/imbalance/p1_{models,asymmetry,confound}.png`.
- 2026-08-17: **§8 dimension-robust spread metric DONE** (`FINDINGS_robust_metric.md`; same-clouds
  guarantee verified — tree re-run 1260/1260 cells identical, svm re-run reproduces documented
  numbers to 3 decimals) — **substantive verdict = honest-negative: the spread signal is
  genuinely dimension-limited; recall is the recommended diagnostic.** Best candidate (kNN-local
  spread) = +4% ripple on wine with clean CIs (baseline already as large as defect arms); ratio
  metrics provably null (uniform stretch); PCA equalises concentration (0.56→0.82) without
  enlarging the effect; raw LOF numerically unstable. Next gate: **Phase 0 shortcut / spurious
  feature** (PLAN §3).
- 2026-08-17: **Thread C Phase 0 shortcut DONE** (`FINDINGS_shortcut.md`, `plots/shortcut/phase0.png`) —
  **GO**: H1 CONFIRMED on iris+wine (spurious-axis fraction of adv displacement rises
  monotonically, 0.11→0.50 iris / 0.00→1.00 wine, tight CIs — the first geometry signal to fire
  on wine at all); first spurious split rises to the root (reliance manipulation check);
  cheap-escape prediction REFUTED (adv_l2 grows with corr — thresholds recede); geometry
  separates before the accuracy collapse completes but never at zero accuracy cost.
- 2026-08-17: **Thread C Phase 1 shortcut svm+HSJ DONE** (`results_shortcut_svm.parquet`,
  `plots/shortcut/svm_hsj.png`) — **per-axis signal SURVIVES the black-box attack**: monotone,
  separated from control by corr=2, peak corr=4 (iris 0.354 / wine 0.455 — wine now leads,
  model-type flip vs the tree), survivorship dip at corr=8; cheap escape partially reappears on
  wine (adv_l2 dips at moderate corr); scalar spread still a late byproduct (fires only after
  acc collapse). RF+HSJ deferred (hang-prone).
- Deferred/open: XGBoost per-point-timeout rework; 2-feature boundary visualisation of the
  `toward` tendril; structured (boundary-localized) label noise.
