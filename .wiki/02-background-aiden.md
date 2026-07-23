# 02 — Background: Aiden's Prior Work

## What Aiden built

A pipeline testing whether adversarial examples diagnose dataset quality, using
a **decision tree** on the **iris** dataset.

**Files** (now preserved in `aiden_original/`):
- `util.py` — core library: `audit_tree()` (clean/margin experiment),
  `audit_tree_bias()` (bias/clustering experiment), `labelencoding()`,
  synthetic data generators. An `audit_svc()` SVM path existed but was
  **commented out and never finished**.
- `bias.py` — driver for the bias experiment (iris, wine, Car Evaluation)
- `main.py` — driver for the clean experiment
- `eda.ipynb`, `eda_bias.ipynb` — exploratory analysis

**Model & attack:** `DecisionTreeClassifier` + ART's `DecisionTreeAttack`
(white-box, tree-specific — walks the tree structure to find the minimal
perturbation that flips a leaf).

## Aiden's metric

Per OPTICS cluster of adversarial points:

```
density = n_pairs / (sum_of_pairwise_distances + 1)
```

This is approximately **1 / mean_pairwise_distance** — the inverse of geometric
spread. It is pair-count-invariant (adding points at the same density doesn't
change it).

**Aiden's finding:** density *decreased* as coverage-gap bias increased on iris
(0.0309 → 0.0298). Interpreted as: adversarial points spread apart under bias.

## The limitations we identified

1. **White-box, tree-only attack.** `DecisionTreeAttack` cannot run on SVM or
   NN. No cross-model validation was possible.
2. **The signal was tiny.** A 3% change in density, visually amplified by
   z-scoring so it looked large.
3. **Accuracy confound.** His test accuracy *also* dropped (0.934 → 0.836,
   −10 pts) across the same bias range. The density change could just be
   tracking accuracy.
4. **Only coverage gap.** No other defect types tested — no specificity claim.
5. **A norm bug** in the implementation (see below) corrupted the numerical
   values, though not the basic directional trend.

## The norm bug (historical, now understood)

Aiden's code used `np.linalg.norm((p1, p2))` — the Frobenius norm of two
*stacked* points, not `‖p1 − p2‖`. Two identical points reported distance ≈ 4.24
instead of 0. This corrupted the magnitude but the pair-count-invariant form
still trended correctly. **We do not lead with "buggy metric" in the report** —
the honest framing is that his density ≈ 1/spread, a valid metric; the bug
affected magnitude, not the core direction. See [06-lessons-gotchas.md](06-lessons-gotchas.md)
for the full formula subtlety.

## What we carried forward

The clustering/spread strand — extended to a second model (SVM), a black-box
attack (HopSkipJump), and a second defect (label noise). See
[03-methodology.md](03-methodology.md).
