# Session — 2026-08-15 (spans the 2026-08 research run)

Post-presentation research run: completed the **model-family** study (RF/tree under
black-box HSJ on all three defects), built the **outlier** defect out fully (ceiling
push, all-models toward/random/distance plots, model performance), **resolved the
outlier mechanism from first principles**, discovered the **coverage-gap accuracy
confound**, and opened the **defect-expansion** thread (comprehensive plan + Phase 0
class imbalance). All work committed and pushed to `origin/master`.

## Topics resolved (evidence-driven)

### Model families — coverage gap SURVIVES RandomForest (headline)
`model_family_experiment/` (RF 60 trees + single overfit tree, black-box HSJ, hang-safe
subprocess-per-row driver). Under HSJ:
- **Coverage gap survives bagging:** RF spread rises 0.66→0.80 with bias, accuracy flat
  ~0.96. RF+HSJ is a *better* black-box CG detector than tree+HSJ (which stays flat ~0.62,
  the known weak combo). Mechanism: a coverage gap is a *global structural hole* every tree
  shares → bagging cannot average it away. → `FINDINGS_coverage_gap.md`.
- **Outlier does NOT survive bagging** and is **white-box-only**: the earlier n=3 "flip"
  (tree+HSJ sees outward) was retracted at 10 seeds — HSJ shows ~1.0× in every direction;
  RF only a weak `random` 1.13×. → `FINDINGS_outlier.md`.
- **Label noise intractable via HSJ:** 53/90 cells hung (boundary fragmentation → HSJ
  loops). Survivors confirm label noise is accuracy-confounded, not independent geometry.
  → `FINDINGS_label_noise.md`. XGBoost deferred (hangs pervasively).

### Outlier ceiling push (n_out to class size = 40) — three model behaviours
`outlier_experiment/` (toward, k=8, 10 seeds). Trees COLLAPSE at the ceiling (d10
1.25×→1.04× at 100%; non-monotone in *count*, optimal detection ~25–75%); SVM RISES
(0.98×→1.09×) *with* test-accuracy degradation 0.97→0.89; RF stays FLAT (bagging).
Accuracy is blind to the outlier for tree/RF even at 100% injection. →
`toward_all_models.png`, `model_performance.png`, `random_all_models.png`,
`distance_all_models.png`, `interaction_all_models.png`.

### Outlier MECHANISM — fully resolved (`outlier_experiment/MECHANISM.md`)
Driven by a chain of user mechanism questions, each answered by direct measurement:
- **DTA's "nearest" = tree-GRAPH distance**, not spatial (verified in ART source: walk up
  ancestors, DFS sibling subtrees, `threshold ± offset` landing coordinates).
- **The outlier's coordinates ARE split candidates** (thresholds = midpoints of consecutive
  point values). The k=8 outlier at plen 2.40 *created* root threshold 2.15 = (1.90+2.40)/2
  → it *becomes* the setosa boundary point. Count is irrelevant; location matters.
- **A pure outlier leaf is an UNBOUNDED STRIP** (`pwid ≤ 0.74 AND plen > 2.06`): one split
  achieves purity, tree stops (optimises purity, not compactness). DTA's *one-attribute*
  perturbation leaves the other axis at the original test value (measured mean |Δpwid|=0.000)
  → adversarial points string out along the strip → spread rises.
- **The signal is NON-MONOTONE in distance (Goldilocks window):** null inside versicolor
  (k≤4, the dip); fires in the empty gap between clusters (k=5–10); **COLLAPSES back to
  baseline at k≥12** when the toward ray carries the outlier into setosa territory
  (plen 1.3 at k=12) — its leaf moves to the setosa branch, tree-far from versicolor, so DTA
  reverts to the compact boundary leaves. 50-seed sweep, tight CIs. → `kplateau.png`.

### Coverage-gap ACCURACY CONFOUND (important correction to the flagship result)
All flagship coverage-gap runs (`dtree_attack_experiment`, `hsj_svm_experiment`,
`model_family_experiment`) inject the bias **before the CV split** (`inject_bias(X,y)` then
`skf.split(Xb,yb)`) → the deleted band is absent from the **test** set too. So "accuracy
stays flat" is partly artifact: there are no test points in the hole to misclassify, and for
the contested class tc=2 test accuracy even *rises* 0.96→0.99 (deleting hard boundary cases
makes the test easier). Train-fold-only injection (clean test) *drops* accuracy 0.95→0.71.
The **spread signal remains real** (Finding 3 compression ratio); only "accuracy-blind" is
protocol-dependent. (`util.py:audit_tree_bias` biases train-only but is unused by flagship
runs.)

### Defect expansion — Phase 0 class imbalance (`defect_expansion_experiment/`)
New thread from the principle: *a defect is visible to the geometry probe iff it imposes a
global, structured boundary distortion separable from accuracy*. Priority
imbalance→shortcut→leakage (`PLAN.md`). **Phase 0 (tree+DTA, 30 seeds):** random deletion
(imbalance) vs spatial deletion (coverage gap) at matched count. H1 **confirmed** — the
spatial hole adds spread on top of the count effect (both rise; spatial > random, CIs
separate at frac≥0.5, 1.37× vs 1.24× at 95%). H2 **confirmed** — minority-class recall is
the clean discriminator (coverage gap craters it to 0.14; imbalance holds ~0.61). →
`FINDINGS_imbalance.md`, `plots/imbalance/phase0_spread_vs_random.png`.

## Artifacts produced
- `outlier_experiment/`: `MECHANISM.md`, `run_kplateau.py`+`plot_kplateau.py` (kplateau.png),
  `plot_toward_all.py`, `plot_random_all.py`, `plot_distance_all.py`, `plot_interaction_all.py`,
  `plot_performance.py`, `plot_viz2d.py`, `_diag_distance.py`; parquets `results_extended`
  (3390 rows), `results_threshold`, `results_kplateau` (800 rows); plots
  `{toward,random,distance,interaction}_all_models.png`, `model_performance.png`,
  `viz2d_outlier_mechanism.png`, `distance_geometry.png`.
- `model_family_experiment/`: `run_experiment.py` (hang-safe driver; fixed KEYCOLS(noise) +
  filter_args(--ceiling) bugs), `FINDINGS_{coverage_gap,outlier,label_noise}.md`,
  `plot_cg.py`+`cg_rf_vs_tree.png`, `results.parquet` (440 rows).
- `defect_expansion_experiment/`: `PLAN.md`, `run_imbalance.py`+`plot_imbalance.py`,
  `FINDINGS_imbalance.md`, `imbalance_phase0.png`, `results_imbalance.parquet`.

## Wiki corrections
- `.wiki/04-findings.md`: added ⚠️ before-split accuracy-confound caveat to the coverage-gap
  headline; added **Finding 5** (outlier Goldilocks/collapse, CG-survives-RF, imbalance
  Phase 0) with pointers to the experiment folders.
- `.wiki/09-planned-experiments.md`: Thread C (defect expansion) indexed; A/B marked complete.
- `.wiki/05-key-decisions.md`, `08-open-questions.md`: overfit-tree + train-only injection
  decisions; open questions marked PLANNED (earlier in the run).
- `.wiki/README.md`: threads A/B done, C in progress.

## Corrections made during this session (user caught each)
- **"n=3 flip is real" (tree+HSJ sees outward 1.22×)** → NOISE. Retracted at 10 seeds (~1.0×).
- **"tree grows a diagonal tendril to the outlier"** → wrong; trees are axis-aligned. Real
  mechanism is the unbounded-strip leaf + one-attribute perturbation.
- **"spread keeps rising with k (6→8)"** → over-read point estimates; CIs overlap, it's a
  PLATEAU. (Then the 50-seed sweep found the real structure: plateau → collapse at k≥12.)
- **"empty space past versicolor"** → wrong; the toward ray runs *into setosa's corner*.
- **"the RF ceiling run taking 30 min is normal"** → NO, it was an infinite-loop bug
  (`--ceiling` not forwarded to the worker); fixed, then it legitimately took ~35 min.
