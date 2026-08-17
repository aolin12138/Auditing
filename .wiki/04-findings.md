# 04 — Findings

## Headline

**Coverage-gap bias is detectable from adversarial geometry while accuracy stays
flat. Label noise is not.**

> ⚠️ **2026-08 caveat (read before quoting the headline):** the "accuracy stays
> flat" half is **partly a before-split-injection artifact**. All flagship coverage-gap
> runs delete the class region from the *whole dataset before* the CV split, so the
> deleted band is missing from the **test** set too → there are no test points in the
> hole to misclassify → accuracy stays flat (and for the contested class tc=2 it even
> *rises* 0.96→0.99, because deleting the hard boundary cases makes the test easier).
> When injected **train-fold-only** (clean test), the same coverage gap **drops** test
> accuracy 0.95→0.71 and minority recall to 0.14. The **spread signal is still real**
> (Finding 3, compression ratio) — but "accuracy-blind" is protocol-dependent, not an
> intrinsic property of the defect. See `defect_expansion_experiment/FINDINGS_imbalance.md`
> and Finding 5 below.

---

## Finding 1 — Coverage gap: spread rises, accuracy flat (THE result)

Geometric spread (mean pairwise distance) increases monotonically with bias on
every model+attack combination, while test accuracy stays ~0.96.

| Combination | spread 0.1→0.9 | Cohen's d | Accuracy |
|-------------|----------------|-----------|----------|
| Tree + DTA | 0.44 → 0.56 | **+2.06** | flat ~0.95 |
| SVM + HSJ | 0.53 → 0.59 | **+0.75** | flat ~0.96 |
| Tree + HSJ | 0.62 → 0.65 | +0.38 (weak) | flat |

**Mechanism:** deleting a contiguous class region forces the model's boundary
into an extrapolated, data-free zone. Adversarial points crossing it scatter
(no data density to constrain them) → spread rises. The surviving data still
classifies correctly → accuracy flat.

Figure: `figures/report/p1_coverage_gap.png` (z-scored per combination).

### Tree+HSJ is the weak/unreliable combination — two reasons
1. **HSJ can't navigate a flat tree surface.** HSJ estimates the boundary
   direction by sampling perturbations; a tree's boundary is flat axis-aligned
   facets, so most perturbations stay inside a leaf without crossing — noisy
   estimate, frequent non-convergence.
2. **Non-monotonic drop at bias 0.9 — it's an ATTACK effect (HSJ vs DTA), not
   purely a model effect.** Driven entirely by tc=0 (setosa depleted): spread
   0.806 → 0.627. At bias 0.9 only 4/50 setosa points survive, so <1 lands in
   the test fold — the attack has almost nothing to target there and reverts to
   the **intact class-1-vs-2 boundary** in dense space → spread drops. Why only
   HSJ shows it: **DecisionTreeAttack is deterministic** and always captures the
   far, stretched gap boundary, so **Tree+DTA keeps rising** (0.574 → 0.610).
   **HSJ must *reach* that far boundary by random-init + sampling**; when the
   depleted class runs out of test points it can't, so its points revert to the
   healthy boundary (and it fails more — 4 NaN at 0.9, zero for SVM). Note SVM+HSJ
   *also* drops for tc=0 (0.616 → 0.533) — it's just **masked in the aggregate by
   class averaging** over tc=1/tc=2, whereas Tree+HSJ's swing is large enough to
   survive averaging. So "SVM avoids it" is an aggregation artifact. See
   [06-lessons-gotchas.md](06-lessons-gotchas.md).

---

## Finding 2 — Label noise: no independent signal

Spread **rises** under label noise (it is NOT flat), but it is still **not a
useful diagnostic** for two reasons. Measured across the full 0.1–0.9 range for
all three combos (grids extended past 0.5 for this):

- **The rise is real but accuracy-CONFOUNDED.** Spread 0.1→0.5 rises with large
  effect: Tree+DTA d=+2.02, Tree+HSJ d=+1.57, SVM+HSJ d=+1.08. But test accuracy
  falls in lockstep (tree 0.93→0.64, SVM 0.96→0.78). The defect is already
  plainly visible in accuracy, so the geometry adds nothing. (Note: in *density*
  terms this reads as a *decrease* — density ≈ 1/spread; earlier notes quoting
  d=−0.80 / −0.09 were the size-sensitive density, not spread.)
- **Direction depends on class separability.** Spread *increases* on iris/wine
  (well-separated), little change / opposite on Car Evaluation (categorical).
  Confirmed on synthetic 3D.
- **Above noise 0.5: the metric destabilises entirely.** The model tends toward a
  random classifier (accuracy below the 1/3 chance level by 0.8); spread variance
  grows ~an order of magnitude (std 0.05 → 0.6–0.9) and the number of valid runs
  collapses (Tree+DTA n: 36 → 12 at 0.9; HSJ hangs more). No stable trend exists
  — both which points stay correct and where the stochastic HSJ lands are noise.
- **Mechanism:** well-separated data + noise → tree grows many leaves to
  memorise noise → every point sits next to a boundary → tiny perturbation → adv
  cloud ≈ original test cloud (compression ratio → 0.98). Past 0.5 this saturates.

Figure: `figures/report/p2_label_noise.png` (full range, 3 combos, shaded >0.5
randomness regime).

---

## Finding 3 — Verification: the signal is real (compression ratio)

To rule out "spread just tracks the changing test set" (coverage gap is injected
before the CV split), we measured **compression ratio = adv_spread /
original_test_point_spread**, and perturbation magnitude.

| Defect | Compression ratio | Perturbation | Interpretation |
|--------|-------------------|--------------|----------------|
| Label noise | 0.77 → **0.98** | 0.81 → 0.64 (↓) | Adv cloud mirrors original test points — **artifact** |
| Coverage gap | **~0.70** (flat) | 1.38 → 1.70 (↑) | Adv cloud genuinely compressed — **real signal** |

Under label noise the adversarial points barely move (ratio → 1): we're just
re-measuring the fixed test data. Under coverage gap the cloud stays ~30% tighter
than the test data and points travel *farther* as the gap widens — a genuine
attack effect. Figure: `figures/report/p4_discriminant.png`. Probes:
`experiments/probes/_probe_move.py` (label noise), `_probe_cg.py` (coverage gap).

---

## Finding 4 — Tree training strategy flips the signal (label noise)

Exploratory (`experiments/exploratory/collect_v2.py`, `data/data_v2.parquet`):
under label noise, **overfit** trees show density rising (leaves 4→19), **pruned**
trees show density falling (leaves 4→0.5). Same defect, opposite signal — because
the training strategy changes how many boundaries form. Further evidence that
label-noise geometry tracks model structure, not the defect. (Spread not recorded
separately in this earlier grid — a limitation.)

---

## Finding 5 — New threads (2026-08): outlier defect, model families, defect expansion

Three threads added after the report; full write-ups live in the experiment folders.

- **Outlier defect** (`outlier_experiment/`, `MECHANISM.md`, `FINDINGS_phase*.md`): a
  correctly-labelled anomaly at k·σ from the class centroid (train-fold only). White-box
  DTA detects `toward` outliers (1.25–1.36×, accuracy-blind); `outward`/`random` weak-null.
  The signal is **non-monotone in distance** — a *Goldilocks window*: null while the outlier
  is inside versicolor (k≤4), fires in the empty gap between clusters (k=5–10), then
  **collapses back to baseline at k≥12** when the outlier enters setosa territory (50-seed
  sweep, `plots/kplateau.png`). Mechanism fully resolved: DTA's "nearest" = tree-graph
  distance; the outlier's coordinates are split candidates; a pure outlier leaf is an
  *unbounded strip*, and DTA's one-attribute perturbation strings adversarial points along it.
- **Model families** (`model_family_experiment/`): coverage gap **SURVIVES RandomForest**
  (spread 0.66→0.80 with bias; RF+HSJ is a *better* black-box CG detector than tree+HSJ);
  outlier does **not** survive bagging; label noise is **intractable** via HSJ (boundary
  fragmentation → pervasive hangs). XGBoost deferred (hangs).
- **Defect expansion** (`defect_expansion_experiment/`, `PLAN.md`): principle = *a defect is
  visible iff it imposes a global, structured boundary distortion separable from accuracy*.
  **Phase 0 class imbalance DONE:** random deletion (imbalance) vs spatial deletion
  (coverage gap) at matched count — the **spatial hole adds spread on top of the count
  effect** (H1, CIs separate at frac≥0.5), and **minority recall is the clean discriminator**
  (coverage gap craters it to 0.14; imbalance holds ~0.61).
  **Phase 1 class imbalance DONE (2026-08-16, `FINDINGS_imbalance_p1.md`):** across models
  (SVM/tree/RF + black-box HSJ), **minority recall is the robust, model-agnostic separator**
  of coverage gap vs imbalance (spatial recall craters to 0.14–0.35 vs random 0.55–0.77 at
  frac 0.85–0.95, tight CIs on every model). The *scalar-spread* gap survives only partially
  — RF directional (n=3), SVM at frac 0.95 (1.46±0.11 vs 1.23±0.09), **absent on tree+HSJ**
  (the weak/hang-prone combo). **Class ASYMMETRY confirmed:** deleting from a *separable* class
  (setosa) is null in both geometry and recall — the defect must distort a *contested* boundary.
  **Coverage-gap accuracy confound quantified (30 seeds):** before-split injection makes
  test acc RISE to 1.000±0.000 (recall 1.000) while train-only DROPS to 0.714±0.001
  (recall 0.143) — a pure protocol artifact confirming the flagship “accuracy-blind” headline
  was partly a test-set-removal artifact. Next: shortcut-feature (PLAN §3), then leakage.

## Finding 6 — Cross-dataset variance: the SPREAD signal is fragile, RECALL is robust (2026-08-16)

`defect_expansion_experiment/` (`run_variance.py`, `run_confound.py`, `FINDINGS_variance.md`,
`plots/variance/tree_dta_{iris,wine}.png`, `svm_hsj_{iris,wine}.png`, `spread_fragility_map.png`).
Re-ran the imbalance-vs-coverage-gap contrast on a **second dataset (wine, 13-D)** and a **second
model+attack (svm + black-box HSJ)**, factored by injection **protocol** (train-only vs
before-split), features standardized, 30 seeds (tree+DTA) / 15 (svm+HSJ).

- **Recall discriminator is robust** across dataset AND model/attack: spatial (coverage gap)
  craters minority recall much harder than random (imbalance) — wine train-only frac 0.9:
  spatial 0.48–0.52 vs random 0.90 (tight CIs, both tree+DTA and svm+HSJ).
- **The adversarial-SPREAD signal is fragile.** Strong only on **iris + white-box tree/DTA**
  (spatial 1.26×); collapses to ~1.04× on **13-D wine** and to ~1.11× under **black-box HSJ** on
  iris — fragile to *both dimensionality and attack*. Measured cause = **distance concentration**
  (pairwise-distance std/mean 0.54 iris vs 0.29 wine): the mean pairwise distance goes numb in
  higher dimensions. → motivates a **dimension-robust spread metric** (`PLAN.md §8`).
- **before-split spread “signal” is a test-set COMPOSITION artifact** (not boundary detection):
  before-split deletes the target class from the *test* set (11.8→1.2 pts, same for both arms),
  so its tight adversarial cluster drops out and the surviving c1/c2 cloud is identical → both
  arms give the same spread; the surviving-class spreads are unchanged (c1 3.48→3.50, c2 3.12→
  3.19). It's the geometry-sibling of the accuracy confound (Finding 5 caveat). **train-only is
  the correct protocol.**
- **Class asymmetry + accuracy confound both replicate** across datasets/models (before-split
  lifts accuracy +0.13; deleting from a separable class = null geometry + recall).

**Honest bottom line:** the transferable, model-/dimension-robust signal is **per-class recall**;
the adversarial-**geometry/spread** signal only clearly fires for iris + white-box tree/DTA and
needs a dimension-robust metric to generalize.

**RESOLUTION (2026-08-17, `FINDINGS_robust_metric.md`):** §8 tested 4+1 dimension-robust
candidates on the same clouds. **Substantive verdict = the honest-negative: the spread
signal is genuinely dimension-limited; recall is the recommended diagnostic.** The best
candidate (kNN-local spread, M4) gives clean wine CIs (1.055 vs 1.012 @0.8) but only a **+4%
ripple** on a clean baseline already as large as the defect arms (iris contrast +9.8% on a
small baseline) — detectability of the ripple, not survival of a meaningful signal. Ratio
metrics (kNN-ratio, LOF) are provably null (uniform stretch → ratios invariant);
PCA-then-spread equalises concentration (0.56→0.82) without enlarging the effect; raw LOF is
numerically unstable on white-box clouds. M4 = marginal upgrade for white-box iris-like
cases only.

## Metric decomposition note

Density combines cluster size and spread. On Tree+DTA coverage gap, cluster size
*grows* with bias while spread also grows, so `n_points/(mean+1)` density can go
the "wrong" way — this is why we report spread. But Aiden's actual `n_pairs/sum`
density ≈ 1/spread and is size-invariant, so it agrees with spread. See
[06-lessons-gotchas.md](06-lessons-gotchas.md) for the full story.
