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
| 5–10 | in the **GAP** between versicolor and setosa | impurity cascade → tree reorganises; outlier isolated by one split → unbounded strip on the versicolor side | **ONSET + plateau ~1.2–1.35×** (flat within noise, 50 seeds) |
| ≥12 | **enters setosa territory** (plen 1.3 at k=12; below setosa at k≥14) | new boundary on the setosa side → outlier leaf hangs off the setosa branch → tree-FAR from versicolor leaves | **COLLAPSE back to ~1.0×** (tight CIs [0.98,1.03], definitively not noise) |

Why the k=3–4 dip: the outlier is *inside* the versicolor cluster but labelled virginica.
The impurity signal is contradictory (same label far from the class, different label nearby),
so the tree does not reorganise cleanly — near-baseline spread. The boundary already exists
there (it separates virginica from versicolor), so the outlier is absorbed.

Why it PLATEAUS after onset (k=6→10): the tree topology stabilises and adversarial
displacement is flat; split-value drift is a second-order effect. The dominant effect is the
ONSET cascade.

**Why it COLLAPSES at k≥12 (new finding, 50-seed sweep `results_kplateau.parquet`):** the
toward ray continues past versicolor INTO setosa territory (4D: outlier plen = 3.4 at k=6,
2.7 at k=8, 2.0 at k=10, **1.3 at k=12** — inside setosa's [1.0,1.9]; below at k≥14). Once
the outlier overlaps setosa, the tree must separate it from setosa → the outlier leaf hangs
off the SETOSA branch → for versicolor test points it is tree-FAR (traverse to the root and
down the other side) → DTA's nearest wrong-class leaf reverts to the normal compact virginica
leaves at the true boundary → adversarial points cluster tightly → spread returns to baseline.
So the signal is **non-monotone**: it exists only in the Goldilocks window where the outlier
sits in the EMPTY GAP between two clusters — absorbed (null) inside either cluster, firing
(high spread) only between them. Figures: `plots/kplateau.png`.

**The full mechanism chain (updated, incorporating the one-attribute insight):**
1. The outlier enters the tree's threshold bookkeeping (its coordinates are split candidates).
2. At k=3–4 it sits inside versicolor → no cascade → dip.
3. At k=5–6 it exits → impurity cascade reorganises the tree (ONSET).
4. The outlier gets isolated by a SINGLE split (pwid) → pure leaf → tree stops → the leaf is
   an UNBOUNDED STRIP (tree optimises purity, not compactness).
5. DTA sends adversarial points into the strip, perturbing only the attribute that fails the
   threshold — the other attribute stays at the original test value (measured Δ=0.000).
6. Adversarial points string out along the strip, inheriting the test set's diversity on the
   untouched axis → mean pairwise distance (spread) jumps above baseline.

So the answer to the user's question: the adversarial examples DON'T all land "on the outlier". SOME of them land at the outlier's leaf (the far-away virginica leaf created by the outlier), while others land at the normal boundary. The cloud gets torn into two far-apart groups → mean pairwise distance increases → spread up.

**The crispest formulation (one attribute ⇒ untouched other attribute):** the outlier leaf is
entered by a single split (`pwid ≤ 0.74`), so DTA perturbs ONLY the attribute that fails the
threshold along the path; the other attribute stays EXACTLY at the original test value
(measured: mean |pwid change| = 0.0000 for setosa→strip points, mean plen preserved for
versicolor→strip points). Adversarial points therefore inherit the full diversity of the test
set along the untouched axis and string out along the strip — spread up. If the leaf were a
tight box (both attributes bounded, like the baseline setosa leaf), the attack would snap both
attributes and the cloud would collapse into a compact region — baseline spread. The tree
optimises purity, not compactness, so a pure leaf containing only outliers can be a huge
unbounded strip.

### Evidence for the mechanism (all measured, fold 1, seed 42, 2D)
- All 10 outliers land in one leaf; the leaf's rule is `pwid ≤ 0.74 AND plen > 2.06` — the
  entire bottom strip, unbounded in plen (one split sufficed for purity; the tree stopped).
- 20/29 adversarial points land in that strip; 10 of them (from setosa test points) keep
  pwid EXACTLY unchanged (mean |Δpwid| = 0.000) and snap plen to 2.06; 10 (from versicolor
  test points) keep plen unchanged and snap pwid to 0.73.
- Resulting mean pairwise distance of the adversarial cloud: 1.40 (k=0) → 1.74 (k=8).

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

1. **Signal window:** the signal should vanish again when the outlier is pushed INTO setosa
   territory (k where plen drops into [1.0,1.9]) — **CONFIRMED**: collapse at k=12 with
   tight CIs (50 seeds, `results_kplateau.parquet`, `plots/kplateau.png`). The signal is a
   window, not a ramp.
2. **Axis-dependence:** outliers along a single feature (`axis='feat'`) shift only that
   feature's thresholds → weaker, feature-localised signal.
3. **Count matters only via the root:** 1 outlier can shift a root threshold (it only takes
   one point to move a midpoint), so the *onset* k≥6 should be count-independent; but the
   *reorganisation* stage (k=5–6) needs enough outliers to shift class proportions.
4. **Any tree-model attack that reads leaves is sensitive; query-based attacks are not** —
   the same split we observe between DTA and HSJ should generalise to other attack pairs.

## 7. One-sentence summary

The `toward` outlier acts by **entering the tree's threshold bookkeeping** — and the effect
is **non-monotone in distance**: absorbed (null) while the outlier sits inside either cluster
(versicolor at k≤4, setosa at k≥12), and firing only while it sits in the empty GAP between
clusters (k=5–10), where the impurity cascade re-routes DTA into an unbounded strip. DTA's
"nearest wrong leaf" is a tree-graph search with `threshold ± offset` landing coordinates,
and the one-attribute perturbation leaves the untouched axis at the test point's original
value — so adversarial points string out along the strip (spread up) only while that strip is
tree-adjacent to the attacked class.
