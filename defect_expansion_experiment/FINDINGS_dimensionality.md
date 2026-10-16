# FINDINGS — Dimensionality, normalization & signal-directed reduction (PLAN §9)

**Date:** 2026-09-02 · **Design:** padded-iris controlled study · **Model/attack:** overfit tree
(`max_depth=None`) + white-box DecisionTreeAttack (deterministic) · **Seeds:** 30 · **tc=2, feat=2**

**Runners:** `run_dimensionality.py` (Arm 1 + clouds) · `run_reduction.py` (Arm 3/4 offline on clouds)
· `run_fitvsmeasure.py` (the decisive fit-time vs measure-time isolation) · `run_wine_h4.py` (the wine
disambiguation via pre-fit feature selection).
**Figures:** `plots/dimensionality/{dimensionality_decay, reduction_recovery, fit_vs_measure,
wine_h4_disambiguation}.{png,pdf}`.

---

## 0. Why this study exists (the §8 question it answers)

§8 found the OPTICS spread signal (spatial coverage-gap vs random imbalance) collapses on 13-D wine
and closed the "just re-normalise the metric" avenue as a negative. It could **not** separate two causes:
1. **dimensionality per se** — distance concentration burying a real signal, vs
2. the defect **genuinely not distorting** the higher-D geometry.
Wine confounds "more dims" with "different dataset." This study removes the confound: take the
**verified** iris defects (where spread works at 4-D) and **pad with `m` pure-noise N(0,1) dims**
(`m ∈ {0,5,10,20,50,100}`). The signal lives entirely in the original 4 dims; padding is
ground-truth-uninformative. **Determinism guard:** the noise columns are drawn from a separate rng, so
the injection rng that picks deleted rows is untouched by `m` — the deleted rows and the 4 real
features are byte-identical across all `m`; only the noise columns change.

---

## Finding 1 — Raw spread's signal decays monotonically with dimensionality (H1 ✅)

- **Observation.** Normalised OPTICS spread's spatial-vs-random gap (Cohen's d, 30 seeds) falls with `m`:

  | m | d @frac0.5 | d @frac0.7 | d @frac0.85 | concentration std/mean |
  |---|---|---|---|---|
  | 0 | +1.96 ✅ | +1.68 ✅ | +1.01 ✅ | 0.536 (= iris ref 0.54) |
  | 5 | +1.00 ✅ | +1.04 ✅ | +1.85 ✅ | 0.277 (≈ wine 0.29) |
  | 10 | 0.00 | +0.10 | +1.76 ✅ | 0.212 |
  | 20 | +0.12 | +0.39 | +0.29 | 0.154 |
  | 50 | −0.15 | −0.04 | −0.01 | 0.098 |
  | 100 | +0.11 | +0.18 | +0.39 | 0.070 |

- **Hypothesis.** Distance concentration (the pairwise-distance std/mean shrinking as isotropic noise
  dims are added) numbs the *mean* pairwise distance, so the spatial-vs-random contrast washes out.
- **Evidence for.** Concentration collapses 0.54→0.07 in lockstep with the d decay; the absolute mean
  pairwise distance balloons 2.25→14.4 (√-of-added-variance growth). The m=0 value reproduces the
  Phase-0 iris signal and the FINDINGS_variance iris concentration reference (0.54).
- **Conclusion.** H1 confirmed decisively. The signal is alive at m=0–5, degraded at m=10, dead by
  m=20–50. **(frac=0.95 self-hides even at m=0 — the known ceiling effect; excluded above.)**

## Finding 2 — Per-class recall is dimensionality-INVARIANT (the robust contrast)

- **Observation.** Minority-class recall (spatial) is `0.22` at both m=0 and m=100; the geometry/spread
  signal decays but recall does not move.
- **Conclusion.** The recall discriminator is distance-free (per-class accuracy), so padding cannot
  touch it. It remains the recommended cross-dimension diagnostic — consistent with every prior thread.

## Finding 3 — Dimension reduction BEFORE the metric does NOT recover the signal; the damage is at FIT time (H2 ❌ as stated → reframed) ⭐

This is the headline. Three pipelines on the same padded data (`run_fitvsmeasure.py`):

| pipeline | what it does | d @frac0.5 (m=0 → m=100) |
|---|---|---|
| **measure_full** (raw) | fit on all 4+m dims, measure spread in full space | +1.96 → +0.11 (dies by m≈10) |
| **fit_full** (oracle) | fit on all 4+m dims, measure spread in the **known real-4 dims** | +1.96 → +0.23 (decays) |
| **fit_real** (pre-select) | **drop noise before fitting**, fit on real-4, measure there | **+1.96 → +1.96 (flat, all m)** |

- **Observation.** Even the **oracle** — spread measured in the *known* signal subspace — decays
  (fit_full: +1.96→+0.23). Yet feature-selection-**before**-fit (fit_real) is exactly flat at the m=0
  signal for every `m` (+1.96/+1.68/+1.01 at frac 0.5/0.7/0.85).
- **Hypothesis (tested, not assumed).** The signal is degraded at **fit** time, not measurement time:
  the mere *presence* of extra candidate features perturbs the overfit tree's split selection, changing
  the adversarial cloud even in the real subspace. Post-fit reduction is therefore too late.
- **Evidence for.**
  - The overfit tree splits on a noise dim **0 %** of the time (iris is separable in petal features →
    pure leaves are reached without noise) — so the effect is **not** noise entering the splits.
  - Yet the fit_full vs fit_real trees disagree on **2–4 %** of test predictions (up to 9.9 % at high
    severity) — the noise features perturb split *selection / tie-breaking* just enough.
  - In the real-4 subspace, the **spatial** cloud is essentially constant across `m` (0.78; byte-identical
    where nadv matches), but the **random**-arm cloud creeps UP (0.66→0.75) toward it → the delicate
    ~20 % spread contrast erodes. The spread gap is fragile to a 2–4 % model change.
  - fit_real proves the signal EXISTS and is FULLY recoverable — but only by selecting features
    *before* training.
- **Evidence against / nuance.** fit_full (post-fit oracle) decays *slower* than raw (measure_full):
  at frac0.7 it holds d≈0.5 through m=50 while raw is already null by m=10. So post-fit reduction is a
  **partial** measurement fix — it removes the concentration component but is **capped** by the fit-time
  component it cannot undo.
- **Conclusion.** **H2 (reduction-before-the-metric recovers the buried signal) is refuted in its
  post-fit form and reframed:** the loss is two-part — (i) measurement concentration (raw, fixable by
  reduction) + (ii) fit-time model perturbation (oracle still decays, fixable ONLY by pre-fit feature
  selection). The actionable fix is **feature selection before training**, which here recovers the full
  signal because we control the real dims; in practice it requires *discovering* the informative
  subspace pre-training — the same discovery problem, moved earlier in the pipeline.

## Finding 4 — Blind discovery methods do not beat the oracle, and produce false positives at high m

- **Observation.** Unsupervised PCA-to-4 (`pca4`) and supervised axis selection (`axis_var` = top-4
  var(defect)/var(clean); `axis_mean` = top-4 |Δmean|/pooled-std) track the oracle's decay at frac≤0.7,
  and none clears the no-signal band reliably beyond m=10. At frac=0.85 + high m, `axis_var`/`pca4`
  spike ABOVE the m=0 signal (d up to +2.4/+2.9) — but this coincides with tiny adversarial sets
  (self-hiding ceiling), i.e. the selection picks spurious high-variance noise axes.
- **Conclusion.** These spikes are **small-sample axis-selection artifacts**, not recovery — a
  cautionary result: a blind "select the most-different axes then measure spread" pipeline manufactures
  false positives on small clouds. The clean panels (frac 0.5/0.7, `reduction_recovery.png`) show all
  discovery methods decaying to the oracle's capped level.

## Finding 5 — Metric normalization (Arm 4) does not rescue raw spread

- **Observation.** `raw_divmed` (spread ÷ median pairwise) and `raw_divsqrtd` (spread ÷ √d) both decay
  with `m` just like raw; `raw_divsqrtd` is numerically identical to raw's d (the √d factor cancels in
  the spatial-vs-random ratio). Cohen's d itself (used throughout) is the dimensionless normalization
  and still decays — because the underlying effect (fit-time erosion) is real, not a scaling artifact.
- **Conclusion.** Consistent with the §8 negative: no *metric-level* normalization recovers a signal the
  *model* no longer produces. Reported for completeness.

## Finding 6 — The wine defect DOES distort geometry; the §8 null was dimensionality (H4 ANSWERED) ⭐

- **Observation.** Restrict wine to its top-`k` most class-discriminative features (ANOVA F-score) and
  run the SAME coverage-gap (spatial) vs imbalance (random) defect there. The spatial-vs-random spread
  signal appears at low `k` and vanishes at full `k` (Cohen's d, 30 seeds; effect size vs baseline):

  | wine k | spatial (×baseline) | random (×baseline) | Cohen's d @frac0.5 |
  |---|---|---|---|
  | 2 | **1.48× (+48%)** | 0.96× | **+2.04 ✅** |
  | 3 | 1.11× | 0.95× | +1.47 ✅ |
  | 4 | 1.06× | 0.94× | +1.51 ✅ |
  | 6 | 1.01× | 0.98× | +0.38 |
  | 13 (full) | 0.99× | 0.99× | −0.12 (reproduces §8 null) |

  Iris positive control (top-k) keeps its signal at all low k (d +1.4 to +2.2) — low-k selection does
  not manufacture a signal, it preserves a real one.
- **Conclusion (answers the original §8 question).** The wine coverage-gap defect genuinely DOES distort
  the boundary geometry — the spread signal is real and LARGE (+48% at k=2, cleanly CI-separated) in
  wine's low-D discriminative subspace. The §8 wine null was **dimensionality per se** (13-D distance
  concentration burying it), **NOT** defect-not-distorting. **H4 verdict: RECOVERABLE.**
- **This revises the §8 conclusion.** §8 said the spread signal is "genuinely dimension-limited; recall
  is the recommended diagnostic; a >50-D set would only confirm this." The refinement: the signal is
  dimension-*limited* but not dimension-*dead*. The distinction is the INTERVENTION — §8's **post-fit
  metric reduction** gave a +4% ripple (correctly a non-result); **pre-fit feature selection** (this
  finding, and fit_real in Finding 3) recovers a +48% effect. Consistent with the padded-iris headline:
  the fix is selecting features BEFORE fitting, never reducing dimensions after.

## Finding 7 — The signal is a property of the MODEL's boundary, not the data: you cannot keep the full-accuracy model AND see it (`run_wine_measure_vs_fit.py`) ⭐

(Prompted by the user's objection: shrinking features to see the signal either hurts performance, or —
if you keep all the good features — puts you back in high-D where the signal is buried.)

- **Observation (wine, 30 seeds, full-model clean vacc = 0.90):**

  | approach | model | Cohen's d @frac0.5 |
  |---|---|---|
  | full 13-D model, measure full space (§8) | full (acc 0.90) | ~null |
  | **full 13-D model, MEASURE-ONLY project → top-k axes** | **full (acc 0.90)** | **−0.05 / −0.06 / −0.01 (null)** |
  | fit model IN top-k subspace (H4) | reduced (acc 0.82–0.89) | **+2.04 / +1.47 / +1.51 ✅** |

- **Key point.** Projecting the FULL model's adversarial cloud onto the top-k discriminative axes — even
  when the projection **includes the exact deletion axis** — recovers NOTHING (both arms shrink together,
  d≈0). Only FITTING the tree in the subspace recovers the signal. (Matches padded-iris: the oracle =
  fit-full-measure-real-dims also failed, Finding 3.)
- **Conclusion / mechanism.** The adversarial-spread signal is a property of the fitted model's
  DECISION BOUNDARY, not of the data directly. A full-D model smears the coverage-gap distortion across
  all dims (concentration); no after-the-fact projection reconstructs a distortion that was never
  concentrated. Only a model whose boundary is *built* in the low-D subspace bends around the gap
  visibly — and that is a different, lower-accuracy model. **Both horns of the user's objection are real:**
  shrink features → visible signal but weaker model (−8% acc at k=2); keep all quality features → full
  accuracy but signal reburied (k=6: acc 0.91, d=0.38). **You cannot have both with this probe.**
- **Honest resolution (the usable takeaway).**
  1. **Probe/deployed split.** The geometry probe need not use the deployed model. Fit a cheap LOW-D
     PROBE model on a discriminative subspace purely to DETECT the data defect; keep the deployed model
     full-D. The probe's lower accuracy is irrelevant (it is a detector, not a predictor). Open caveat:
     whether a probe-detected defect transfers to the deployed model's behaviour is untested.
  2. **Per-class recall is the model-agnostic, full-D detector** — it flags spatial vs random on the
     FULL 13-D wine model without any subspace crutch (recall gap −0.3 at every k, incl k=13 where the
     spread signal is null). Consistent with every prior thread: recall = robust detector; adversarial
     spread = a mechanistic LENS that only resolves in low-D (iris-like data or a low-D probe).

## Finding 8 — Negative control: the low-D spatial>random gap is the DEFECT, not a low-D artifact ⭐

(Prompted by the user: "how do I know the metric reflects the defect, not that spread naturally grows
 in low-D? test whether the same pattern happens on a healthy dataset.")

- **Design.** Run the IDENTICAL low-D (3-D) spatial-vs-random pipeline on four conditions and compare
  Cohen's d (30 seeds). `run_healthy_control.py`.

  | condition | clean acc | spatial/random spread | Cohen's d @frac0.5 | recall Δ |
  |---|---|---|---|---|
  | **iris contested** (real coverage gap) | 0.95 | **1.48 / 1.10** | **+2.23 ✅** | −0.25 |
  | iris separated class (same data/dim/manip) | 0.95 | 1.06 / 1.07 | −0.05 (null) | 0.00 |
  | well-separated blobs (healthy synthetic) | 1.00 | 0.96 / 0.99 | −0.26 (null) | −0.02 |
  | random labels (no structure) | 0.34 | 1.09 / 1.03 | +0.29 (null) | −0.04 |

- **Tightest control (dispositive).** iris *contested* vs iris *separated* is the SAME dataset, SAME
  3-D, SAME spatial-vs-random manipulation — only the deleted class changes. The signal collapses from
  **d=+2.23 to −0.05**. So it is NOT the low dimensionality; it is whether the deletion distorts a real
  *contested* boundary. Healthy blobs (wide margins) and random labels (no boundary) are both null.
- **Conclusion.** The low-D spatial>random spread gap reflects the boundary-distorting DEFECT, not a
  generic low-D metric growth. The metric passes the negative control.
- **Honest caveat.** `rand_null` shows a faint positive lean (d≈+0.29, NOT CI-separated) — a tiny low-D
  "contiguity" effect (a contiguous deletion nudges spread up even with no structure) exists, but it is
  ~8× below the real-defect effect (2.23) and does not clear the CI bar. Keep it in view; it is a
  reason to require a real effect size, not just CI separation, at very small `k`.

## Finding 9 — What if the problem TRULY needs high-D? Then the probe has no window (and the defect is milder) ⭐

(Prompted by the user: wine kept accuracy at k=4 because its top features carry the signal — but what
if we genuinely need 7/8/13 features? Does reduction still keep accuracy, and what happens to the probe?)

- **Design.** Two synthetic 10-feature / 3-class sets, same size, differing ONLY in intrinsic dimension:
  `redundant` (n_informative=3, rest redundant — like wine) vs `all_informative` (all 10 independently
  needed). Coverage-gap vs imbalance on the contested class, sweep #features kept `k`, tree+DTA, 30 seeds.
  `run_highdim_need.py`.

  | | redundant (intrinsic ~3-D) | all_informative (true 10-D) |
  |---|---|---|
  | accuracy k=2 / k=8 / k=10 | 0.83 / 0.96 / 0.96 (plateaus early) | 0.65 / 0.82 / 0.81 (keeps climbing) |
  | spread signal (d) | separated at k=3,6,8,10 (up to +2.1) | **null at every k** |
  | k with BOTH ≥95% acc AND signal | **{6, 8, 10}** | **NONE** |
  | minority-recall gap @0.5 | strong at all k (−0.16 to −0.29) | weak at high k (−0.02 to −0.05) |

- **Conclusion (answers "what if we truly need high-D").** Three things happen, all confirmed:
  1. **Reducing costs accuracy** when features are genuinely needed (all_informative: 0.82→0.65 as k 8→2).
     Reduction keeps accuracy ONLY when the extra features are redundant (the problem is secretly low-D,
     like wine/iris). You can *diagnose which* directly from the accuracy-vs-k curve: plateaus early →
     redundant, safe to reduce; keeps climbing → truly high-D.
  2. **The geometry probe has NO usable window** for a truly-high-D problem — no `k` gives both accuracy
     and a separated spread signal (sweet-spot = NONE). It is a tool for low-intrinsic-D data only.
  3. **A single-axis coverage gap is also genuinely MILDER in high-D** — the model routes around it via
     the other informative features, so even recall's signal fades at full-D (−0.05). The defect is both
     less detectable and less harmful. (Reassuring: the high-D defects that hurt — crater recall/accuracy
     — are the ones recall still catches; the ones nothing sees are ones the model absorbs.)
- **Untested harder case.** A high-D defect spread across MANY axes (genuinely harmful in high-D) would
  crater recall while staying invisible to geometry — the clean "use recall" regime. Only the single-axis
  gap was tested here.
- **Bottom line for the method.** The adversarial-geometry probe works when the data's *defect-relevant*
  structure is low-D (redundant features / low intrinsic dimension — much real tabular data). When the
  problem truly needs high-D, there is no free low-D probe; fall back to **per-class recall on the full
  deployed model** (the project's recurring robust detector).

## Finding 10 — Absolute vs normalized gap + CIs, and the genuinely-high-D verdict (the honest scope) ⭐

(Prompted by the user: (i) is the ABSOLUTE increase similar across dims and only the NORMALIZED change
diluted — with CIs going from separated to overlapping? (ii) test on a dataset that GENUINELY needs high
dimension, because a reducible dataset should have been feature-selected before training anyway — so it
tells us nothing universal.)

### (i) Absolute vs normalized, with paired CIs
- **Padded iris (added dims are NOISE):** the *absolute* spatial−random spread gap stays roughly the
  same size (~0.13 at m=0, m=5, m=20), the *normalized* gap collapses (0.208 → 0.002 as the baseline
  balloons 0.61→14.3), AND the paired-gap CI widens ~4× (±0.03 → ±0.13). So detectability dies from
  BOTH normalization-dilution AND growing variance — the CI goes SEP → overlap. **The user's mechanism is
  correct here.**
- **Genuinely-high-D (all-informative):** the absolute gap is ~0 from the start (−0.039±0.059 at 10-D),
  not "preserved but diluted." A single-axis gap barely distorts a genuinely-high-D boundary, so there is
  no absolute signal to dilute. Looking at absolute-instead-of-normalized does NOT rescue detection here.

### (ii) Genuinely irreducible high-D + a HARMFUL defect (`run_highdim_harmful.py`)
Dataset: `make_classification`, 15 informative dims, `n_clusters_per_class=3` — accuracy climbs with
dimension (top-4 ≈ 81% of full), so it is genuinely high-D, NOT reducible. Defect: a boundary-directed
coverage gap (delete the contested class's points nearest the neighbouring class — genuinely harmful).
Three detectors, all on the FULL 15-D model:

  | detector | frac0.5 | frac0.7 |
  |---|---|---|
  | **recall** (contested class) | clean 0.575 → spatial **0.355** / random 0.407; gap **−0.052±0.015 (sep)** | 0.575 → 0.261/0.287; −0.026±0.017 (sep) |
  | spread full-15D | −0.063±0.091 (null) | +0.091±0.120 (null) |
  | spread **defect-direction** (1-D projection) | +0.001±0.005 (null) | +0.000±0.004 (null) |
  | spread top-4 prediction axes | +0.016±0.027 (null) | +0.001±0.030 (null) |

- **Conclusion (the honest scope of the whole geometry angle).** On genuinely-high-D data with a harmful
  defect, the adversarial-spread probe is BLIND — in the full space, projected onto the defect's own
  direction (the idea that would keep the model full-D), AND on the prediction axes. No measurement
  projection recovers it, and feature selection is not an option (it would cripple a model that needs the
  dims). **Only per-class recall detects the harm** (recall craters 0.575→0.355) and distinguishes the
  spatial structure (−0.052, CI-separated).
- **Therefore:** the adversarial-geometry spread signal is fundamentally a **low-intrinsic-dimension
  diagnostic** — it does NOT provide a universal high-D auditing method (the user is right: on reducible
  data the recovery is trivial; on genuinely-high-D data it fails). **Per-class recall is the robust,
  dimension-general detector.** This bounds and sharpens the project's geometry angle rather than
  extending it.
- **Caveat.** The harmful defect was boundary-directed along the class-mean-difference direction; other
  harmful high-D defect shapes are untested, but the defect-direction projection (the most favourable
  measurement) was already null.

## Finding 11 — Fitting a reduced model does NOT reveal the signal when features genuinely matter (direct test) ⭐

(Finding 10 tested only MEASUREMENT projection on the harmful high-D gap. This tests FIT-IN-SUBSPACE
— the wine-H4 recovery method — directly: `run_highdim_reduce.py`. Inject the harmful boundary-directed
gap in FULL 15-D, then select top-k F-score features, FIT the tree on them, attack, measure spread. Sweep k.)

  | k | clean accuracy | spread d@0.5 (sep) | spread d@0.7 (sep) | recall gap |
  |---|---|---|---|---|
  | 2 | 0.387 | −0.65 (n) | +0.44 (n) | +0.005 |
  | 4 | 0.435 | +0.47 (n) | −0.61 (n) | −0.005 |
  | 6 | 0.510 | +0.10 (n) | −0.03 (n) | −0.012 |
  | 10 | 0.541 | +0.27 (n) | +0.32 (n) | −0.044 |
  | 15 (full) | 0.558 | −0.23 (n) | +0.44 (n) | −0.049 |

- **Result.** At EVERY k the spread signal is null (never CI-separated) and its sign flips randomly
  between −0.69 and +0.47 — noise, not signal. Meanwhile accuracy craters (0.558 → 0.387 at k=2, a ~30%
  relative loss). So fit-in-subspace pays a large accuracy cost AND reveals nothing. Even recall degrades
  under reduction (−0.049 at full-D → ~0 at k=2).
- **Principle (why reduction ≡ reducibility).** Reduction reveals the coverage gap ONLY when the defect's
  boundary footprint is concentrated in a low-D subspace — which is the SAME condition that makes the data
  reducible in the first place (wine: gap along ~2 features → selecting them recovered +48%, Finding 6).
  When features genuinely matter, the boundary (and the gap's distortion of it) is spread across all dims,
  so no top-k subspace captures the defect and shrinking the model only destroys it. **You cannot use
  reduction to see a signal in data that genuinely needs its dimensions — the two are the same coin.**
- Directly confirms (rather than infers) the answer to "does reducing reveal the signal when features
  really matter?": **no.**

## Methodological note — `lda1` is not evaluable under baseline normalization

The 1-D Fisher (clean-vs-defect) projection is a valid per-cell reduction (5362/5400 folds give a valid
projection; 38 degenerate), but its **frac=0 baseline is defect==clean → between-class scatter ≈ 0 →
degenerate → norm = NaN**, so every normalised value is NaN. It is excluded from the recovery figure.
(Fixed a crash it caused first run: guard `z.shape[1]==0 → NaN`; added per-fold try/except and
per-variant parquet checkpointing so a late-variant crash never discards the whole run.)

---

## Answer to the §8 question (ANSWERED)

- **On padded iris:** the signal genuinely EXISTS (fit_real flat) and is buried by BOTH dimensionality
  (measurement, raw dies) AND fit-time model perturbation (oracle still decays). The fix is pre-fit
  selection, not post-fit reduction.
- **On wine (H4):** the signal genuinely EXISTS too (+48% at k=2, Finding 6) and the §8 null was 13-D
  concentration burying it. **RECOVERABLE by pre-fit feature selection** to a low-D discriminative
  subspace — NOT defect-not-distorting.
- **Unified conclusion:** on BOTH datasets the adversarial-spread geometry signal is real and
  dimension-LIMITED (not dead); it is recovered by **feature selection BEFORE fitting**, and NOT by any
  post-fit metric reduction (§8) nor post-fit oracle projection (Finding 3). BUT (Finding 7) fitting in
  the subspace = a **different, lower-accuracy model** — the signal is a property of the fitted boundary,
  so you cannot keep the full-accuracy deployed model and see it by projecting the measurement. Practical
  reframing: use adversarial spread as a **low-D probe-model diagnostic** (a probe/deployed split), and
  **per-class recall as the model-agnostic full-D detector**.

## Limitations / not done
- **Arm 2 (feature normalization: z-score vs PCA-whitening vs robust-scale, H3)** — not run.
- **Arm 5 external validity: digits 64-D** — not run (wine 13-D done as H4).
- **Tree-specificity of the fit-time effect (Finding 3)** — only overfit tree + DTA tested. Whether
  SVM+HSJ (smooth boundary, less tie-break-fragile) or a pruned tree (cannot memorize) shows the same
  fit-time fragility is untested — the natural robustness check.
- **Pre-fit selection uses a SUPERVISED criterion** (ANOVA F-score / oracle real-dims). It assumes we
  can identify the informative subspace before training; discovering it blind on an unknown defect is
  the open problem (blind post-fit discovery produced false positives, Finding 4).
- Reduction target in Arm 3 fixed at k=4 (the true dim); no k-sweep there (the wine H4 does sweep k).

## Final scope statement (§9 overall)
The adversarial-geometry spread probe is a **low-intrinsic-dimension diagnostic lens**: it detects a
coverage-gap defect when the defect-relevant structure is low-D (iris; wine *after* pre-fit selection),
via a probe model distinct from the deployed one. It does **not** generalise to genuinely high-D
irreducible problems — neither feature selection (cripples a model that needs the dims) nor any
measurement projection (full-space, defect-direction, or prediction-axis) recovers a harmful high-D
defect. **Per-class recall is the dimension-general, model-agnostic detector** and is the recommended
primary auditing signal; adversarial spread is a mechanistic explainer for the low-D regime.

## One-sentence summary
On a controlled padded-iris design the adversarial-spread signal decays with added noise dimensions
exactly as distance concentration predicts, and dimension reduction *after* fitting — even an oracle
that knows the true subspace — cannot recover it (the noise features perturb the overfit tree at fit
time, 2–4 % of predictions); **only feature selection *before* training restores the full signal, and
the same pre-fit selection recovers a large (+48 %) signal on 13-D wine that §8's post-fit reduction
could not** — so the wine §8 null was dimensionality per se (recoverable), the geometry signal is
dimension-limited but not dead, and the correct fix is "select the informative features before you
train," not "reduce dimensions after."
