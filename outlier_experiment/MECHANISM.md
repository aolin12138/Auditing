# The Outlier Mechanism — Why `toward` Outliers Change Adversarial Spread

**Status:** evidence-backed (2026-08-11) · **Chains:** tree + white-box DTA · **Figures:**
`plots/viz2d_outlier_mechanism.png`, `plots/distance_geometry.png`, `plots/distance_all_models.png`

This document answers, in order, the chain of mechanism questions raised while interpreting the
distance (k) dose-response: why the signal is null at k≤4, why it fires at k≥6, why it keeps
growing with k, and why it is attack-specific and model-specific. Every step is backed by a
direct measurement (tree dumps, split candidates, adversarial displacement).

---

## 0. Setup (facts, not metaphors)

- Defect: `n_out=10` correctly-labelled outliers added to class `tc=2` (virginica) in the
  **train fold only**, at distance `k` (units of the class's mean per-feature std) along the
  unit vector from virginica's centroid toward the **nearest other class** centroid.
- Nearest other class of virginica = **versicolor** (class 1).
- Model: overfit tree (`max_depth=None`). Attack: ART **DecisionTreeAttack** (DTA).

**Geometry of the `toward` ray** (iris, petal length × width): virginica centroid ≈
(5.55, 2.03), versicolor centroid ≈ (4.26, 1.33), separation 3.69σ, versicolor mean radius
1.61σ. So along the ray the outlier: **enters** versicolor territory at k≈2.1, sits **on** the
versicolor centroid at k≈3.7, **exits** at k≈5.3, and then — because the ray is diagonal
(decreasing petal length AND width) — continues **into the setosa–versicolor gap** toward the
setosa corner (plen ≈ 2.4–2.9 at k=8). *There is no "empty space past versicolor": past
versicolor lies setosa's corner.* (`_diag_distance.py`)

## 1. What DTA's "nearest" actually means

DTA never computes a Euclidean distance. From the ART source
(`.venv/.../art/attacks/evasion/decision_tree_attack.py`):

1. Trace the input's decision path (root → its leaf).
2. Walk **up** the ancestors. At each ancestor, depth-first search the **sibling subtree** for
   any leaf of a different class.
3. The first such leaf found is the adversarial destination — "nearest" = **fewest tree-node
   traversals** (up n, then down the sibling branch).
4. Perturb the input across each split threshold along the found path:
   `x[feature] = threshold ± offset`.

Consequences:
- DTA's destinations are a property of the **tree graph**, not of spatial distance.
- **Every split threshold in the tree is a landing coordinate** for adversarial examples
  (through `threshold ± offset`).
- Anything that changes the tree's split values changes where adversarial points land,
  regardless of whether the topology (leaf count, feature order) changes.

## 2. Why the outlier changes the tree at all — coordinates are split candidates

A tree's candidate thresholds are the **midpoints of consecutive feature values of the
training points**. The 10 outliers are training points, so **their coordinates are split
candidates** no matter how few they are.

Measured (one fold, seed 42): setosa plen max = 1.90, versicolor plen min = 3.00. Candidate
thresholds near the setosa split:

- k=6 (outliers at plen 2.96–3.44): candidates …1.80, **2.43**, 2.98…
- k=8 (outliers at plen 2.40–2.91): candidates …1.80, **2.15**, 2.43, 2.47…

The k=8 root split became `plen ≤ 2.15` where 2.15 = (1.90 + 2.40)/2 — **the midpoint between
setosa's max and the outlier itself.** The outlier literally became the boundary point of the
root split. Count is irrelevant here: the location is what enters the threshold bookkeeping.

## 3. The three k regimes (the dose-response explained)

| k | outlier position | tree effect | spread |
|---|------------------|-------------|--------|
| 0–2 | near virginica | minimal | ~baseline (1.0×) |
| 3–4 | **inside versicolor** | no clean cascade (same-label points far apart, other-label points around → impurity signal muddied) | **dip, ~1.0×** |
| 5–6 | just past versicolor | **topological cascade** — class proportions at the root change → the whole tree reorganises (same leaf count, different feature order) | rises, ~1.19× |
| 6–8+ | in the setosa–versicolor gap | **topology stabilises**, but outlier coordinates keep shifting the split thresholds (root 2.43→2.15, deeper splits 0.94→0.74) | keeps rising, ~1.25× |

Why the k=3–4 dip: the outlier is *inside* the versicolor cluster but labelled virginica.
The impurity signal is contradictory (same label far from the class, different label nearby),
so the tree does not reorganise cleanly — near-baseline spread. The boundary already exists
there (it separates virginica from versicolor), so the outlier is absorbed.

Why it rises with k even after topology stabilises (k=6→8): tree topology is identical
(9 leaves, depth 6, same feature at root) and adversarial **displacement does not increase**
(0.79 → 0.74 L2) — but the **split values drift** because the outlier coordinates move.
Different thresholds → different adversarial landing coordinates → different spread. The
effect is spatial, but it travels *through the tree's threshold bookkeeping*, never through a
Euclidean search by DTA.

## 4. Why the effect is attack-specific

- **DTA (white-box):** reads the tree graph directly → every threshold shift is felt as a
  changed destination. Sensitive to both stages of §3.
- **HopSkipJump (black-box):** probes the boundary by decision queries from random starts →
  averages over the boundary's local geometry, insensitive to tree-graph reorganisation. Null
  at 10 seeds (toward 0.94×, outward 1.02×, random 0.97× — `FINDINGS_outlier.md` F1).

## 5. Why the effect is model-specific

- **Single overfit tree:** impurity minimisation at every node → the cascade of §3 exists.
- **SVM (rbf):** decision boundary from support vectors, no impurity cascade → immune
  (~1.0× at all k; only stirs at extreme k=12 where the outliers finally become support
  vectors, `phase05_svm_directions.png`).
- **RandomForest (bagging):** each tree sees a bootstrap sample; the class-proportion shift is
  diluted per tree and averaged → muted/null (≤1.13×, `FINDINGS_outlier.md` F2).

## 6. Predictions this mechanism makes (testable)

1. **Saturation:** push k→∞ and the outlier becomes the extreme point of every relevant
   feature; thresholds asymptote → spread plateaus (k=12+ should stop the rise).
2. **Axis-dependence:** outliers along a single feature (`axis='feat'`) shift only that
   feature's thresholds → weaker, feature-localised signal.
3. **Count matters only via the root:** 1 outlier can shift a root threshold (it only takes
   one point to move a midpoint), so the *onset* k≥6 should be count-independent; but the
   *reorganisation* stage (k=5–6) needs enough outliers to shift class proportions.
4. **Any tree-model attack that reads leaves is sensitive; query-based attacks are not** —
   the same split we observe between DTA and HSJ should generalise to other attack pairs.

## 7. One-sentence summary

The `toward` outlier acts not by pulling the boundary toward itself but by **entering the
tree's threshold bookkeeping** — first through the global impurity cascade (tree reorganisation)
and then through its own coordinates (split-value drift) — and DTA, whose "nearest wrong leaf"
is a tree-graph search with `threshold ± offset` landing coordinates, converts both stages
into a measurable change in adversarial spread.
