# Defect Expansion Experiment — Comprehensive Plan

**Created:** 2026-08-11 · **Status:** planned (Phase 0 pending) · **Owner:** research thread 3

Extends the adversarial-geometry dataset-auditing study with three new training-data
defects, chosen by a single predictive principle learned from the first two threads
(coverage-gap, outlier, label-noise). Priority order: **(1) class imbalance, (2) shortcut /
spurious feature, (3) train–test leakage.**

---

## 0. Organizing principle (why these three)

From the completed threads:

| defect | geometry signal | why |
|--------|-----------------|-----|
| coverage gap | **clear** (survives RF, accuracy-blind) | global **spatial hole** → boundary must extrapolate through empty space |
| outlier | fragile (white-box-only, self-hides at ceiling) | few local points → averaged away / become own blob |
| label noise (random) | null | unstructured → only moves accuracy |

**Predictive rule:** a defect is visible to the adversarial-geometry probe **iff it imposes a
GLOBAL, STRUCTURED distortion of the decision boundary that is SEPARABLE from accuracy.**
The three new defects are chosen to test the *edges* of this rule:

- **Class imbalance** — the **count-controlled sibling** of coverage gap. Isolates *spatial
  structure* from *sample count*. Directly sharpens our clearest result.
- **Shortcut / spurious feature** — a *global* distortion that is maximally **accuracy-blind**
  on an IID split (the boundary leans on a fake axis). Strongest candidate for "geometry
  catches what accuracy misses."
- **Train–test leakage / duplication** — *inflates* accuracy (looks great) while creating
  memorized, over-confident geometry. Tests whether the probe can flag a defect that accuracy
  actively *hides*.

---

## 1. Shared method (all three defects)

- **Dataset:** iris, `StratifiedKFold(5, shuffle, random_state=seed)`. Train fold = 40/class.
- **Injection is train-fold-only** (except leakage, which is defined across the split — see §4).
  Keeps the test set clean so the geometry signal is not a test-set coverage artifact.
- **Models & attacks (report both attacks — attack choice is a first-class variable):**
  - overfit single tree (`max_depth=None`) + **white-box DecisionTreeAttack (DTA)**
  - RandomForest (60 trees) + **black-box HopSkipJump (HSJ)**
  - rbf-SVM + HSJ (fast; where tractable)
  - (single tree + HSJ as the black-box-on-tree control)
- **Metric:** adversarial **spread** = `cluster_stats(adv)[2]` (OPTICS within-cluster mean
  pairwise distance), reused from `outlier_experiment/run_experiment.py`.
- **Normalisation:** report **spread / each model's own clean baseline** (n=0 injection).
- **Accuracy:** log `tacc` (train) + `vacc` (test) always; **plus per-class recall** for
  imbalance (the minority class is the point). Never average accuracy over the injected factor.
- **Grids:** full factorial with all other knobs fixed; state fixed values on every plot.
  **Never couple severity and a second knob in a diagonal sweep.** ≥10 seeds for fast
  (tree/SVM) cells; ≥3 for RF+HSJ (slow, may hang).
- **Reuse (import by path, avoid `run_experiment.py` name collision):**
  `model_family_experiment/run_experiment.py` → `make_and_wrap`, `attack_and_measure`,
  `cluster_stats`, hang-safe subprocess `driver`/`worker`, `COLS`/`KEYCOLS` schema, and
  `inject_cg` (spatial deletion) as the coverage-gap anchor. New injections live in this dir.
- **Hang safety:** tree/RF/XGB + HSJ can hang on fragmented boundaries → keep the
  subprocess-per-row timeout driver (`{'rf':90,'tree':25}`), hung rows → NaN, never retried.

### Success-criteria template (verification-protocol; observable yes/no)
Every defect ships with criteria of the form: *"At severity S, is normalised spread of
condition A vs B separated with non-overlapping 95% CIs, while accuracy metric M behaves as
predicted?"* — not "does it work".

---

## 2. Defect A — Class imbalance  ★ PRIORITY 1

### Definition
Reduce one class `tc` to a fraction `keep` of its train-fold count by **uniform random
deletion** (no spatial structure). Contrast with coverage gap = **spatially-biased deletion**
of the *same count* (delete the `1-keep` fraction lowest along feature `feat`).

### The controlled comparison (the whole point)
`inject_imbalance(X, y, tc, frac_removed)` (random) vs `inject_cg(X, y, tc, feat, frac_removed)`
(spatial) at **matched `frac_removed`**. Answers: is the coverage-gap signal about the
**spatial hole** or just **fewer samples**?

### Injection knobs (first-class controlled variables)
- `frac_removed` ∈ {0, 0.25, 0.5, 0.7, 0.85, 0.95} — severity (0 = balanced baseline).
- `structure` ∈ {`random` (imbalance), `spatial` (coverage-gap control)} — **the key factor.**
- `tc` ∈ {0 setosa (separable), 2 virginica (contested)} — target class.
- `feat` ∈ {2} for the spatial arm (petal length; full = {0,1,2,3} later).
- `seed` (fold), model, attack.

### Hypotheses
- **H1 (spatial-hole):** at matched `frac_removed`, `spatial` spread > `random` spread with
  non-overlapping CIs at ≥1 severity → the spatial hole (directional extrapolation), not
  count, drives coverage gap. **Predicted: TRUE.**
- **H2 (accuracy separability):** overall `vacc` stays ~flat for `random` on a class that is a
  minority of the *test* set, but **minority-class recall drops monotonically** → imbalance is
  visible in *per-class* accuracy, unlike coverage gap. So report per-class recall.
- **H3 (dose):** `random` spread is weak/flat or mild-rising (no big empty region), separable
  from `spatial` which rises like the established coverage-gap curve.

### Phased grid
- **Phase 0 (pilot):** tree + DTA, `tc=2`, `feat=2`, both `structure`, all `frac_removed`,
  10 seeds. ~2 fast. GO/NO-GO: is `spatial` > `random` at high severity?
- **Phase 1:** add RF+HSJ and SVM+HSJ; add `tc=0`; per-class recall plots; normalised
  spread dose-response `spatial` vs `random`.

### Success criteria
1. H1 check: ∃ severity where `spatial` and `random` normalised-spread 95% CIs do **not**
   overlap (spatial higher). ✅/❌
2. H2 check: minority-class recall for `random` decreases monotonically with `frac_removed`
   while overall `vacc` moves < half as much. ✅/❌
3. Reproduces coverage-gap curve on the `spatial` arm (sanity: matches
   `model_family_experiment/FINDINGS_coverage_gap.md`). ✅/❌

---

## 3. Defect B — Shortcut / spurious feature (Clever Hans)  ★ PRIORITY 2

### Definition
Append a 5th feature that is **label-correlated in train but not in test**. In train,
`x_spurious = onehot-ish(y) + N(0, leak_noise)`; in test, `x_spurious = N(0, leak_noise)`
(pure noise). The model can "cheat" by reading the shortcut axis.

### Injection knobs
- `corr` = strength of the train-time label signal in the spurious feature ∈ {0, 0.5, 1, 2,
  4} (as a multiple of noise σ; 0 = no shortcut = control).
- `leak_noise` (fixed, e.g. 1.0).
- model, attack, seed. (Split is IID — that is the point: accuracy can look fine.)

### Hypotheses
- **H1:** as `corr` rises, the model relies on the spurious axis → adversarial examples move
  **preferentially along that axis** → measurable signature: `spread` along the spurious
  feature ≫ along real features, and/or lower total spread (cheap escape). Report **per-axis
  adversarial displacement**, not just scalar spread.
- **H2 (accuracy-blind):** train `tacc` inflates; test `vacc` may stay moderate on an IID
  split even though the boundary is wrong → geometry separates from accuracy.

### Phased grid
- **Phase 0:** tree + DTA, sweep `corr`, 10 seeds. Metric: fraction of adversarial L2
  displacement carried by the spurious axis vs baseline. GO/NO-GO: does that fraction rise
  with `corr`?
- **Phase 1:** RF+HSJ, SVM+HSJ; per-axis displacement plots; accuracy-vs-geometry separation.

### Success criteria
1. Spurious-axis displacement fraction rises monotonically with `corr` (CIs separate control
   from strongest). ✅/❌
2. There exists a `corr` where `tacc` is inflated (≥ +0.03 over control) while the geometry
   signal is already firing. ✅/❌

---

## 4. Defect C — Train–test leakage / duplication  ★ PRIORITY 3

### Definition
Copy `n_leak` train points (optionally with tiny jitter) **into the test fold** (leakage), or
duplicate `n_dup` points many times **within train** (memorization). Standard accuracy is
*inflated* by leakage.

### Injection knobs
- `mode` ∈ {`leak` (train→test copies), `dup` (in-train replication)}.
- `n_leak` / `n_dup` ∈ {0, 2, 5, 10, 20} · `jitter` ∈ {0, small}.
- `dup_factor` (for `dup`) ∈ {2, 5, 10}. model, attack, seed.

### Hypotheses
- **H1:** leaked/duplicated points are memorized → the boundary grows tight, over-confident
  pockets around them → **local spikes**: high `density`/low local spread near leaked points,
  and `asucc` (attack success) or margin anomalies. Report **memorization locality**, not just
  global spread.
- **H2 (accuracy hides it):** `vacc` *rises* with `n_leak` (leakage inflates it) while the
  geometry flags the memorization → the probe catches what accuracy actively conceals.

### Phased grid
- **Phase 0:** tree + DTA, `mode=leak`, sweep `n_leak`, 10 seeds. GO/NO-GO: `vacc` rises AND a
  local-geometry anomaly appears near leaked points.
- **Phase 1:** `mode=dup`; RF+HSJ; locality metric (per-point neighbourhood spread/margin).

### Success criteria
1. `vacc` increases monotonically with `n_leak` (confirms leakage inflates accuracy). ✅/❌
2. A local-geometry statistic near leaked points separates (CIs) from clean regions. ✅/❌

---

## 5. Schema (superset of the model-family schema)

Reuse `COLS` from `model_family_experiment` + add defect-specific columns:
`frac_removed, structure` (imbalance) · `corr, leak_noise, axis_frac_spurious` (shortcut) ·
`mode, n_leak, n_dup, jitter, dup_factor, local_spread, minority_recall`. New parquet:
`defect_expansion_experiment/results.parquet`. One row = one (defect, model, seed, knobs) cell.

---

## 6. Sequence & gates

1. **Phase 0 class imbalance** (tree+DTA, the spatial-vs-random control) → GO/NO-GO on H1.
2. If GO: Phase 1 imbalance (RF/SVM, per-class recall).
3. **Phase 0 shortcut** → GO/NO-GO on per-axis displacement.
4. **Phase 0 leakage** → GO/NO-GO on accuracy-inflation + local anomaly.
5. Each phase: findings doc `FINDINGS_<defect>.md`, plots in `plots/`, journal Attempt entry.

**Alignment gate:** confirm this plan with the user before Phase 1 of any defect (the Phase 0
pilots are cheap and safe to run first).

## 7. Risks / notes
- iris has only 4 features + 3 classes → shortcut feature is a *5th appended* axis; per-axis
  displacement is clean to measure there.
- RF/tree + HSJ hang risk persists on any defect that fragments the boundary → keep the
  timeout driver; expect NaN cells at high severity.
- Imbalance changes class priors → for HSJ, ensure enough correctly-classified test seeds of
  the minority class exist before attacking (guard against empty attack sets).
