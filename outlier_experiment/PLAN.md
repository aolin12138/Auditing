# Outlier Experiment — Plan

> **Question this thread answers:** does injecting *outliers* (correctly-labeled points
> placed far from their class cloud) into the training data produce a detectable
> adversarial-geometry signal *while test accuracy stays flat*? Outliers are the best
> candidate for the project's strongest test: **a defect that hurts data quality without
> collapsing accuracy** (see `.wiki/08-open-questions.md`). Coverage gap fires but is a
> "hole"; label noise fires only by collapsing accuracy. An outlier is a *third* geometry:
> a mislocated but correctly-labeled point that an over-fit model will chase.

Status: **planned, not yet run.** This file is the contract; build `run_experiment.py`
against it.

---

## 0. Why this is different from the two existing defects

| Defect | What it does | Accuracy impact | Signal (established) |
|--------|--------------|-----------------|----------------------|
| Coverage gap | Deletes a contiguous class region (a *hole*) | flat ~0.96 | spread ↑ (real) |
| Label noise | Flips a fraction of training labels | *collapses* | confounded / null |
| **Outlier (this)** | Adds a correctly-labeled point *far* from its class cloud | ~flat (few points) | **hypothesised spread ↑ near injected class** |

The outlier is a training point. Adversarial examples are generated from **test** points.
So the outlier influences the *model* but never enters the measured adversarial cloud —
a clean separation the other defects don't fully have.

---

## 1. The defect definition (primary)

**Correctly-labeled anomaly.** Take target class `c`. Place `n_out` new points at distance
`k · s_c` from the class centroid, in a chosen direction, and label them `c` (their *true*
class). The model must decide whether to stretch class `c`'s territory out to cover them.

Secondary contrast (Phase 2): **mislabeled far point** — same location, but labeled as a
*different* class. Lets us separate "far + correct" from "far + wrong."

## 2. The controllable knobs (the crux — user flagged injection sensitivity)

| Knob | Symbol | Grid | Parallels | Notes |
|------|--------|------|-----------|-------|
| Target class | `tc` | {0,1,2} individually, and `all` | coverage-gap `tc` | one-class runs = the asymmetry test |
| Distance | `k` | {2, 4, 6, 8} (× class std) | coverage-gap `bias` | primary severity dial; dimensionless |
| Reference point | `ref` | `class` centroid (default) / `global` centroid | — | user's "cluster or overall" |
| Direction | `dir` | `outward` (default) / `toward` nearest class | — | outward = empty space; toward = invade boundary |
| Count | `n_out` | {1, 3, 5, 10} | — | ~2.5%–25% of a ~40-pt training-fold class |
| Injection kind | `kind` | `correct` (default) / `mislabeled` | label noise | Phase 2 contrast |
| Displacement axis | `axis` | `radial` (default) / single `feat`∈{0..3} | coverage-gap `feat` | radial = true "distance from centroid"; axis-aligned = visualizable |
| Seed | `outlier_seed` | 0..N | `noise_seed` | angular jitter for n_out>1 |

**Baseline / control:** `n_out = 0` (clean). Every effect is measured against this and
against the clean classes *within the same run*.

## 3. Concrete injection function (build this)

```python
def inject_outliers(X_tr, y_tr, tc, k, n_out, direction='outward', ref='class',
                    kind='correct', axis='radial', feat=None, rng=None):
    """Add n_out outliers to class `tc` in the TRAINING data only.
    Returns (X_aug, y_aug). Distance k is in units of the class's characteristic std."""
    if n_out == 0:
        return X_tr.copy(), y_tr.copy()
    Xc = X_tr[y_tr == tc]
    mu_c   = Xc.mean(axis=0)
    sig_c  = Xc.std(axis=0)
    s      = sig_c.mean()                      # isotropic characteristic scale
    g_mu   = X_tr.mean(axis=0)                  # global centroid
    base   = mu_c if ref == 'class' else g_mu

    # direction unit vector
    if axis == 'feat':                          # axis-aligned displacement
        d = np.zeros_like(mu_c); d[feat] = 1.0
        if direction == 'toward':               # flip sign toward nearest class
            others = [m for cl, m in class_means(X_tr, y_tr) if cl != tc]
            nearest = min(others, key=lambda m: np.linalg.norm(m - mu_c))
            d[feat] = np.sign((nearest - mu_c)[feat])
    else:                                        # radial
        if direction == 'outward':
            d = mu_c - g_mu                      # away from data center
        else:                                    # toward nearest other class
            others = [(cl, m) for cl, m in class_means(X_tr, y_tr) if cl != tc]
            _, nearest = min(others, key=lambda p: np.linalg.norm(p[1] - mu_c))
            d = nearest - mu_c
        d = d / (np.linalg.norm(d) + 1e-12)

    pts = []
    for _ in range(n_out):
        jitter = rng.normal(0, 0.3, size=mu_c.shape) * sig_c   # small angular spread
        pts.append(base + k * s * d + jitter)
    Xo = np.vstack(pts)
    label = tc if kind == 'correct' else nearest_other_class(mu_c, X_tr, y_tr)
    yo = np.full(n_out, label)
    return np.vstack([X_tr, Xo]), np.hstack([y_tr, yo])
```

**Critical:** inject **after** the CV split, into `X_train` only. Test fold stays clean.
This is the fix for the coverage-gap test-set confound (`.wiki/06-lessons-gotchas.md`) —
done right from the start here.

## 4. Model + attack matrix

Same rule as the rest of the project: **HSJ is the common attack** so cross-model results
are comparable; DTA is the fast deterministic tree-only baseline.

| Model | Attack | Prediction for outliers |
|-------|--------|-------------------------|
| Tree, **overfit** (`max_depth=None`) | DTA (primary), HSJ | memorises outlier → tendril → **strong spread ↑** |
| Tree, **pruned** (`max_depth=3`) | DTA, HSJ | may ignore outlier → **weak/absent** (capacity test) |
| SVM (rbf) | HSJ | soft margin absorbs it → **damped** |
| RandomForest | HSJ | bagging averages it out → **robust, weak** |
| XGBoost | HSJ | boosting chases hard points → **possibly amplified** |

**Tree depth is a controlled variable, not a fixed choice.** Per the user: use an
overfitting tree as the primary (fits training ~100%, so the outlier actually changes the
boundary). Keep depth-3 as the pruned arm — the overfit-vs-pruned contrast *proves capacity
mediates the signal* and links to Finding 4.

## 5. Measurement (per class — enables the asymmetry test)

Reuse `cluster_stats` + `attack_adv`. Record adversarial spread/density **broken down by
the true class of the attacked point** (the existing `tc` column already supports this).
In a one-class injection run (`tc=0`), classes 1 and 2 are the **in-run clean baseline**.

Also record, per class: `asucc` (attack success), mean perturbation `‖adv − x‖`,
`nclust`, `clust_size`.

## 6. Result schema (extend the existing parquet contract)

Existing columns kept: `tacc, vacc, asucc, nadv, density, nclust, mean_dist, clust_size,
aiden_density, seed, tc, feat`. **New columns:**

| Column | Meaning |
|--------|---------|
| `defect` | `'outlier'` |
| `k` | distance multiplier (0 = clean baseline) |
| `n_out` | outliers injected into `tc` |
| `direction` | `outward` / `toward` |
| `ref` | `class` / `global` |
| `kind` | `correct` / `mislabeled` |
| `depth` | `-1` for None(overfit), else the cap |
| `model` | `tree` / `svm` / `rf` / `xgb` |
| `attack` | `dta` / `hsj` |
| `per_class_spread` | dict/JSON: spread per true class (asymmetry test) |

## 7. Hypotheses (pre-registered)

- **H1 (premise test):** overfit-tree adversarial spread for the injected class rises with
  `k` and `n_out`, while `vacc` stays within ~0.02 of the `n_out=0` baseline.
- **H2 (capacity):** the depth-3 pruned tree shows a markedly weaker/absent spread response
  than the overfit tree.
- **H3 (asymmetry):** in one-class injection, the injected class's spread exceeds the clean
  classes' spread in the same run (Cohen's d > 0.5).
- **H4 (robustness × model):** RF damps the signal (bagging); XGBoost may amplify it
  (boosting chases hard points); SVM damps (soft margin).
- **H5 (non-monotonic distance):** very large `k` may *reduce* the signal — the model prunes
  or out-votes an outlier that is too extreme (inverted-U). Worth watching, not assumed.
- **H6 (injection sensitivity — user's point):** `outward` vs `toward`, `correct` vs
  `mislabeled`, `class` vs `global` each produce a documented, reproducible change in the
  signal. Establishing *what the signal is and isn't sensitive to* is a contribution.

## 8. Confounds & controls

- **Train-only injection, fixed clean test.** (fixes the known confound)
- **Clean baseline** `n_out=0` per (model, seed, tc).
- **Verify the model actually changed:** log whether the outlier's leaf/region exists
  (overfit tree) — if the model ignored the outlier, spread won't move and that's H2, not a bug.
- **Outlier not in the adversarial cloud** (it's a train point; adv from test) — no leakage.
- **Seeds:** ≥3 split seeds × several `outlier_seed`s; report mean ± spread (n is small, as
  in the existing grids — flag weak effects).

## 9. Phased execution (de-risk before scaling)

- **Phase 0 — prototype / signal check. ✅ DONE 2026-07-30 — see `FINDINGS_phase0.md`.**
  Tree(overfit)+DTA. **Corrected design after the smoke test:** target a *contested* class
  `tc=2` (virginica), and treat **`direction` as a factor** — the smoke test showed
  `tc=0`+`outward` (my original default) is the single guaranteed-null cell (a tree is blind
  to correctly-labeled outward outliers). Sub-steps:
  - **0a — go/no-go:** `toward, k=8, n_out=5` vs clean baseline, with `outward` as the null
    contrast. **Result: GO** — spread(all) +0.161 while `vacc` exactly flat (0.953→0.953);
    `outward` +0.000.
  - **0b — factorial (`direction × k × n_out`, 3 seeds):** full factorial, not OFAT, not
    coupled — all three crossed while other knobs held fixed, so we read each main effect and
    the interactions. **Result:** dose-response confirmed (spread rises with `k` and `n_out`,
    saturating ~`n_out=5`), direction decisive (toward 0.446 vs outward 0.377 ≈ baseline).
  - **Analysis order:** main effects first (marginalise), then interactions. **Never couple
    `k` and `n_out`** (diagonal sweep) — it confounds them.
  - **Lesson folded into later phases:** `direction` promoted to a first-class factor;
    `tc` sweep (Phase 1) must compare contested classes; global-boundary models (SVM) may
    see the `outward` signal a tree cannot (sharpens Phase 3).
- **Phase 1 — asymmetry + comparable models.** Add `tc∈{0,1,2,all}`, add Tree+HSJ, SVM+HSJ.
- **Phase 2 — injection sensitivity (user priority).** Sweep `direction`, `ref`, `kind`.
- **Phase 3 — ensembles + capacity.** Add RF, XGBoost; sweep `depth∈{None,3}`.
- **Phase 4 — visualization.** 2-feature iris; plot boundary + adversarial landings around
  the injected outlier (see `../model_family_experiment/PLAN.md`).

## 10. Success criteria for THIS experiment (observable yes/no)

1. Overfit-tree injected-class spread increases with `k` (monotone or inverted-U) while
   `|vacc − vacc_clean| ≤ 0.02`. → the accuracy-blind claim.
2. One-class run: injected-class spread > clean-class spread, d > 0.5, same run.
3. Pruned tree response < overfit tree response (H2 confirmed or refuted with numbers).
4. RF response < single overfit-tree response; XGBoost response documented.
5. At least one injection knob (`direction`/`kind`/`ref`) produces a documented, reproducible
   change in the signal (H6).

## 11. Reuse / dependencies

- Reuses: `cluster_stats`, `attack_adv`, OPTICS params, StratifiedKFold, checkpoint/resume
  pattern from `dtree_attack_experiment/run_experiment.py`.
- New deps: **`xgboost`** (Phase 3) — add to `pyproject.toml`. RF is in scikit-learn already.
- HSJ on trees/ensembles **hangs** — reuse the subprocess-timeout runner pattern.
