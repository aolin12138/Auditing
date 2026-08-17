# FINDINGS — Phase 0 shortcut / spurious feature (Clever Hans) (2026-08-17)

Append a 5th feature: **train** = `corr·(y−1) + N(0,1)` (label-correlated), **test** =
`N(0,1)` (pure noise). The model can cheat by leaning on the shortcut axis; an IID split
keeps the test set "clean" so the reliance is invisible to train accuracy. Sweep
`corr ∈ {0, .5, 1, 2, 4, 8}` (0 = pure-noise control axis), overfit tree + white-box DTA,
30 seeds, iris (4-D) + wine (13-D).

- Runner: `run_shortcut.py` · data: `results_shortcut.parquet` (360 cells) · fig:
  `plots/shortcut/phase0.png` · all numbers below are fold-mean ± 95% CI over 30 seeds.

## 0. What we measure

- **test accuracy `vacc`** — the Clever Hans cost: it drops once the tree leans on an axis
  that is noise at test time.
- **first-spurious-split depth `spur_depth`** (root = 0) — manipulation check: where in the
  tree the shortcut first appears. High depth = shortcut ignored; 0 = shortcut at the root.
- **`spur_frac`** — mean over successful adversarial examples of `Δx_spur² / ‖Δx‖²`: the
  fraction of the adversarial L2 displacement carried by the spurious axis (H1 metric;
  control ≈ axis-symmetry share, far below 1/d here because the noise axis is rarely useful).
- **`adv_l2`** — mean adversarial L2 displacement (PLAN's "cheap escape" check).
- `nsp_splits` (count of spurious splits), `tacc`, `nadv` recorded as supporting evidence.

## F1 — Reliance (manipulation check): the tree leans on the shortcut, dose-responsively ✅

| corr | iris vacc | iris spur_depth | wine vacc | wine spur_depth |
|---|---|---|---|---|
| 0    | 0.942±.005 | 2.99±.14 | 0.903±.006 | 3.04±.07 |
| 1    | 0.936±.006 | 2.73±.11 | 0.901±.005 | 2.82±.34 |
| 2    | 0.890±.010 | 2.21±.08 | 0.860±.010 | 1.71±.18 |
| 4    | 0.730±.017 | **1.04±.03** | 0.532±.017 | **0.00 (root)** |
| 8    | 0.667±.000 | 1.00±.00 | 0.399±.000 | 0.00 (root) |

At corr ≥ 4 the shortcut is the root split on wine and near-root on iris; test accuracy
collapses accordingly. (At corr ≤ 1 the tree barely uses the shortcut — the real features
still win: iris's classes are nearly separable, wine's proline separates class 0. The
reliance knob is `corr` *relative to* the real features' separability, not `corr` alone.)

## F2 — H1 CONFIRMED: adversarial displacement concentrates on the spurious axis ✅

`spur_frac` (spurious-axis fraction of adversarial L2 displacement):

| corr | iris | wine |
|---|---|---|
| 0 (control) | 0.108±.021 | 0.002±.002 |
| 1 | 0.148±.023 | 0.009±.006 |
| 2 | **0.262±.021** | 0.082±.023 |
| 4 | 0.321±.038 | 0.197±.050 |
| 8 | **0.500±.000** | **1.000±.000** |

Monotone rise on **both** datasets; control-vs-strongest CIs massively separated
(iris 0.11 → 0.50, wine 0.00 → 1.00). **This is the first adversarial-geometry signature in
this project that fires on wine at all** — but honestly: it saturates only at extreme
reliance (corr=8, where accuracy has collapsed to 0.40), and at intermediate corr the wine
signal is an order of magnitude below iris (0.08–0.20 vs 0.26–0.32 at corr 2–4): on wine the
13 real axes still carry most of the displacement until the shortcut is overwhelming.

## F3 — "Cheap escape" prediction REFUTED: displacement grows with corr ❌ (informative)

PLAN §3 predicted "lower total spread (cheap escape)". Measured `adv_l2` **rises**:
iris 0.78 → 0.93 → 1.10 → 2.34; wine 1.33 → 1.35 → 1.60 → 4.06 (corr 0 → 2 → 4 → 8).
Mechanism: correctly-classified test points sit in the "right" band of the spurious axis;
escaping means crossing a spurious threshold, and stronger shortcuts push the thresholds
*farther* away (train-time means at ±corr). So the escape gets *more expensive*, not cheaper.
The signature of shortcut reliance is **per-axis concentration**, not displacement size.

## F4 — H2 (re-specified): geometry fires before the accuracy collapse completes, but never at zero cost

- **iris, corr=2:** `spur_frac` 0.262±.021 separates from control 0.108±.021 (CIs: .241 vs
  .129) while `vacc` is 0.890 — an accuracy cost of only **−0.05** from 0.942. At corr=1 the
  geometry does not yet separate (0.148±.023 vs 0.108±.021, overlap). → The first separated
  dose costs ~5 points of test accuracy. Geometry leads, but it is not free.
- **wine, corr=2:** `spur_frac` 0.082±.023 separates from 0.002±.002 (CIs: .059 vs .004) at
  an accuracy cost of −0.04 (0.903→0.860) — but the absolute effect (0.08) is weak; the next
  dose (corr=4) carries 0.197 at an accuracy collapse to 0.53.

**Honest verdict:** the per-axis displacement signature is a sharper, dose-responsive readout
of shortcut reliance than the accuracy drop at matched severity, and it separates *before*
accuracy collapses. But it is **not independent of accuracy** — both are functions of the
same reliance, and no dose shows geometry firing at literally zero accuracy change.

## 5. Success criteria

1. `spur_frac` CIs separate control from strongest corr, direction = rise — **MET both
   datasets** (F2).
2. Manipulation check confirms reliance — **MET** (F1: spur_depth → root, vacc ↓).
3. Dose curves plotted + per-dataset verdict — **MET** (plots/shortcut/phase0.png; F1–F5, incl. the
   previously-unplotted scalar-spread row).
4. Plots numerically cross-checked against the parquet (all cells match); **visual check
   pending user** (no image rendering this session); commit pending.

## F5 — Scalar adversarial spread responds only LATE and only on wine ⚠️ (was unplotted)

The runner recorded the project's scalar spread (m0 raw OPTICS + m4 kNN-local, normalised ×
clean baseline) but the original figure omitted it — now row i–j of `plots/shortcut/phase0.png`:

| corr | iris m0 | iris m4 | wine m0 | wine m4 |
|---|---|---|---|---|
| 2 | 1.033±.033 | 1.022±.014 | 1.002±.016 | 1.020±.007 |
| 4 | 1.052±.038 | 0.999±.017 | **1.114±.034** | **1.056±.010** |
| 8 | 0.985±.032 | 0.978±.012 | **1.194±.040** | 0.962±.003 |

Scalar spread is **null on iris** (≤5%, within noise) and fires on **wine only at extreme
doses** (m0 +11–19% at corr≥4, where accuracy has already collapsed to 0.53/0.40). This is
exactly why the per-axis fraction (`spur_frac`) is the right metric for this defect: the
shortcut's geometry is *directional* (displacement along one axis), and the scalar spread —
which averages over all axes — misses it until the effect is overwhelming, and on iris misses
it entirely.

## F6 — Phase 1 svm + black-box HSJ: the per-axis signal SURVIVES the attack (15 seeds) ✅

`results_shortcut_svm.parquet` · `plots/shortcut/svm_hsj.png` (rows: accuracy / spur_frac / adv_l2 / spread).

| corr | iris vacc | iris spur_frac | wine vacc | wine spur_frac |
|---|---|---|---|---|
| 0 (control) | 0.952±.007 | 0.015±.002 | 0.981±.002 | 0.024±.003 |
| 2 | 0.845±.010 | **0.127±.010** | 0.920±.009 | **0.226±.010** |
| 4 | 0.578±.019 | **0.354±.019** | 0.611±.014 | **0.455±.009** |
| 8 | 0.334±.001 | 0.230±.125 ↓ | 0.400±.001 | 0.365±.031 ↓ |

- **H1 SURVIVES the black-box attack:** monotone rise, separated from control by corr=2 on
  both datasets, peak at corr=4 (iris 24× control, wine 19×), then a **survivorship dip** at
  corr=8 (accuracy collapsed → tiny attacked set → noisy). The HSJ baseline is near-zero
  (0.015–0.024) vs the tree's 0.108 — smooth-boundary escapes are real-axis-dominated at
  control, so the shortcut's rise is even sharper in relative terms.
- **Model-type flip:** wine now *leads* (0.455 vs 0.354 at corr=4), opposite of tree+DTA
  (iris 0.321 vs wine 0.197). Mechanism: the overfit tree exploits iris's nearly-separable
  real structure, so the shortcut competes; the smooth rbf-SVM leans on the shortcut's global
  axis, and wine's messier real structure makes the shortcut relatively more attractive.
- **"Cheap escape" exists here (partially):** wine adv_l2 *dips* at moderate corr
  (2.23 → 2.00 at corr=2) before exploding at extreme doses (6.17 at corr=8) — the minimal
  boundary crossing goes along the shortcut direction cheaply, on the smooth model only.
- **Scalar spread fires earlier than on the tree but still only after the collapse:** m0
  iris 1.22× at corr=4 (null on the tree arm!), wine 1.10×/1.54× at corr≥4 — but vacc is
  already 0.58/0.61 there. m4 kNN-local flattens or drops at extreme doses (clumping).
  Same conclusion as F5: scalar spread = late byproduct; per-axis fraction = early signature.

## 6. GO/NO-GO status

Phase 1 svm+HSJ arm **DONE (F6): GO confirmed** — the per-axis signature survives the
black-box attack, with a model-type flip (wine now leads) and the same accuracy co-movement.
**RF+HSJ arm remains deferred** (hang-prone ensembles, per the model-family lessons; the
svm+HSJ answer already covers the black-box question, and RF was the weakest outlier model).
Carried caveats resolved or confirmed: (i) confirmed — no zero-cost detector; (ii) partially
resolved — under HSJ wine's dilution disappears (smooth model leans on the global axis);
(iii) resolved — HSJ *does* show per-axis concentration (peak corr=4).

## 7. Limitations / debt

- The reliance knob is dataset-relative: iris and wine have different real-feature
  separability, so matched `corr` is not matched "reliance" across datasets (spur_depth is
  the honest reliance coordinate — replotting F2 against spur_depth would be fairer).
- Survivorship at high corr: the attacked set is the *correctly classified* remainder, which
  shrinks as vacc collapses (nadv drops ~30% at corr=8) — the fraction is computed on
  survivors only.
- z-encoding `{−1,0,+1}` assumes ordinal class distances; a one-hot multi-axis shortcut is a
  natural variant, untested.
- svm+HSJ at 15 seeds; the corr=8 survivorship dip has wide CIs (tiny attacked set);
  RF+HSJ not run (deferred).
