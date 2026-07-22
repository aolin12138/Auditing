# Supervisor Meeting — Prep Notes

**Project:** Adversarial-example geometry as a diagnostic for dataset quality
**Status:** Metric bug fixed, pipeline extended to 2 models × 2 attacks × 2 defects, 3 experiment grids complete (1,080 runs), all committed + documented.

---

## 1. Progress since last meeting (what to report)

- **Confirmed and characterized the bug** in Aiden's `audit_tree_bias` density metric. Three stacked bugs:
  1. `np.linalg.norm((p1, p2))` = Frobenius norm of stacked points, not `‖p1−p2‖` (two identical points reported distance ≈ 4.24 instead of 0).
  2. `count / (dist + 1)` — an inverted density score, mislabeled as "adv distance."
  3. Loop index reused a cluster label, silently dropping valid pairs.
- **Decomposed the metric into three** measured on the same OPTICS clusters:
  `aiden_density` (buggy) → `density` (norm fixed, still size-confounded) → `mean_dist` (clean spread, geometry only).
- **Extended the pipeline** beyond Aiden's Tree + DecisionTreeAttack:
  - Added a black-box attack (**HopSkipJump**, decision-based) so the *same* attack runs on both Tree and RBF-SVM → separates "does the signal survive a new model" from "does it survive a new attack."
  - Added a second defect (**label noise**) alongside coverage gap.
- **Three complete grids, 360 runs each (iris, 5-fold CV):**
  `dtree_attack` (Tree+DTA, both defects), `hsj_svm` (Tree & SVM + HSJ, coverage gap), `hsj_label_noise` (Tree & SVM + HSJ, label noise).

---

## 2. Key findings (headline numbers)

**Coverage gap → clean spread rises monotonically, accuracy stays flat:**

| Combination               | mean_dist (0.1→0.9) | Cohen's d    | Test acc   |
| ------------------------- | ------------------- | ------------ | ---------- |
| Tree + DecisionTreeAttack | 0.44 → 0.56         | **+2.06**    | flat ~0.95 |
| SVM + HopSkipJump         | 0.53 → 0.59         | +0.75        | flat ~0.96 |
| Tree + HopSkipJump        | 0.62 → 0.65         | +0.38 (weak) | flat       |

→ **The model looks fine on accuracy; only adversarial geometry reveals the coverage gap.** This is the potentially publishable result.

**Aiden's buggy metric is unstable, not a reliable signal:** on his original run, his density *decreased* with bias (0.0309 → 0.0298) — which is actually *consistent* with spread increasing (density = count/(dist+1), so dist↑ → density↓). BUT re-running the same buggy formula under different settings (tree depth, OPTICS params) flips its *sign*. The Frobenius-norm bug makes it depend on absolute point position, not separation. `mean_dist` increased consistently across every setup; the buggy metric didn't. That's the real reason to replace it.

**Label noise → spread also rises, but accuracy collapses (0.96 → 0.78).** So it's accuracy-confounded — plain accuracy already flags it, geometry adds nothing model-agnostic there.

---

## 3. Limitations we found (be honest about these)

1. **Coverage-gap confound:** because bias is injected *before* the CV split, the test set itself shrinks at high bias (fewer target-class points to attack). The geometry change is partly entangled with "fewer points available." *Need to control for this.*
2. **Only iris.** 4 features, 3 classes, 150 samples. Tiny, low-dim, well-behaved. OPTICS (density clustering) degrades in high dimensions — the whole method may not survive realistic data.
3. **Tree + HSJ signal is weak (d = +0.38)** and HSJ frequently hangs on trees (piecewise-constant surface → no gradient to estimate). 12–27 runs per grid had to be killed and NaN'd.
4. **Label noise above 0.5 is unusable** — variance explodes (std 3× the mean), the earlier "density spike" we saw was sampling noise from a collapsed model, not a real signal.

---

## 4. Questions for the supervisor (where I need feedback)

1. **Is `mean_dist` (clean spread) the right primary metric?** We deliberately dropped density to remove the cluster-size confound. Do you agree spread-not-density is the correct framing, or do you want density kept as a secondary reported quantity?
2. **The coverage-gap = detectable, label-noise = accuracy-confounded split** — is that a *useful* result (method discriminates structured vs unstructured defects) or a *weak* one (method only sees what accuracy already sees)? How would you frame it?
3. **How much does the test-set-shrink confound worry you?** Options: (a) inject bias only into the training fold, keep a fixed clean test set; (b) subsample to equalize test-set size across bias levels; (c) report it as a caveat. Which do you prefer?
4. **Scope for the Part 4 report:** is iris-only enough to state the claim, or is higher-dimensional / real-world validation a hard requirement before writing up?
5. **Baselines:** Aiden's notes mention Katerina Dost's bias-detection work. Should we benchmark our signal against an existing bias-detection method, or is this purely exploratory?

---

## 5. Proposed next steps (for discussion — not committed)

**Priority A — fix the coverage-gap confound.** Re-run with bias injected into *train only*, fixed clean test set. This isolates "does the model's geometry change" from "do we have fewer test points." Cheapest high-value fix.

**Priority B — dimensionality stress test.** Add 1–2 higher-dim datasets (e.g. digits, or synthetic 10–20D). Check whether the spread signal survives when OPTICS starts failing. If it doesn't, we may need a dimension-robust measure (relative density vs. original data, Hopkins statistic, kNN-based).

**Priority C — a "clean" defect.** Design a defect that degrades *data quality* without collapsing accuracy (e.g. systematic mislabeling of one sub-region, feature corruption). This would test whether the geometry signal can *ever* fire when accuracy is blind — the strongest version of the claim.

**Lower priority:** more seeds / statistical rigor (mixed-effects model over the fold structure), neural-net + PGD as a third model/attack pair.

---

## 6. The direction question (the one to actually resolve)

**Current thesis:** "The spatial geometry of adversarial examples is a black-box diagnostic for dataset defects that accuracy cannot see."

**Honest state:** it works for *coverage gaps* on *iris* across two models. It does **not** yet clearly beat accuracy for label noise, and it's untested in high dimensions.

**The decision to bring to the supervisor:**
- **Option A — deepen (recommended):** commit to coverage-gap-as-the-detectable-defect, fix the confound (A), prove it survives dimensionality (B), and find one defect where geometry beats accuracy (C). Narrow but defensible.
- **Option B — broaden:** more defects / models / datasets, map where the signal exists and where it doesn't — a "characterization study" rather than a single strong claim.
- **Option C — pivot:** if the supervisor thinks the accuracy-confound makes the whole premise weak, reconsider the framing before investing more.

*My read: Option A. The coverage-gap result is real and the accuracy-flat property is the interesting hook — worth making bulletproof rather than spreading thin.*


some next step: feature noise, class imbalance, outlier as defects, xgboost random forestr models etc