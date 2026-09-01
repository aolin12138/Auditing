# FINDINGS — Phase 0 train–test leakage (mode=leak) (2026-08-17)

Copy `n_leak` random **train** points into the **test** fold (exact copies, Phase 0). Standard
accuracy is inflated because the model has memorized them. Question: does the adversarial
geometry flag the memorization that accuracy hides? Sweep `n_leak ∈ {0,2,5,10,20}`, overfit tree
+ white-box DTA, 30 seeds, iris (4-D) + wine (13-D). Train fold unchanged; test tagged
leaked/clean and metrics split.

- Runner: `run_leakage.py` · data: `results_leakage.parquet` (300 cells) · fig:
  `plots/leakage/phase0.png` · fold-mean ± 95% CI over 30 seeds.

## F1 — H2 CONFIRMED (decisively): leakage inflates measured accuracy, honest accuracy is flat ✅

| n_leak | iris vacc (measured) | iris vacc_clean (honest) | wine vacc | wine vacc_clean |
|---|---|---|---|---|
| 0  | 0.943±.004 | 0.943±.004 | 0.900±.005 | 0.900±.005 |
| 5  | 0.951±.003 | 0.943±.004 | 0.913±.004 | 0.900±.005 |
| 10 | 0.957±.003 | 0.943±.004 | 0.922±.004 | 0.900±.005 |
| 20 | **0.966±.002** | 0.943±.004 | **0.936±.003** | 0.900±.005 |

`vacc` rises monotonically; `vacc_clean` (non-leaked test points only) is **exactly flat**. The
entire accuracy "gain" is the leaked points and only them — the defect is a pure accuracy
artifact. This is the clean demonstration of "accuracy actively hides the defect": a practitioner
watching only test accuracy sees the model *improve*.

## F2 — H1 NULL: leaked points have no distinct adversarial signature ❌ (for the overfit tree)

`margin_leak` (adversarial perturbation L2 for leaked points) vs `margin_clean`:

| n_leak | iris leak / clean | wine leak / clean |
|---|---|---|
| 5  | 0.738±.066 / 0.749±.056 | 1.370±.083 / 1.371±.043 |
| 20 | 0.718±.056 / 0.749±.056 | 1.331±.044 / 1.371±.043 |

CIs overlap at every dose, no trend, direction inconsistent across datasets. **The adversarial
margin does not distinguish leaked from clean points.** Global spread `m0` drifts *down* with
n_leak (iris 0.61→0.42, wine 2.68→2.31) but that is a **composition effect** — leaked points sit
on interior training points, so adding them makes the attacked set more interior and the cloud
more compact — not a memorization-locality signal.

**Mechanism (why it's null, and why it was predictable):** a `max_depth=None` tree memorizes
**all** training data into pure leaves. A leaked test point *is* a training point → routed to the
same pure leaf → its adversarial behaviour is identical to any interior point. There is no
special "over-confident pocket" for leaked points because the overfit tree is over-confident
about *everything* equally. The memorization is **uniform, not local**, so there is no locality
to detect.

## 3. Verdict against success criteria

1. `vacc` increases monotonically with `n_leak` — **MET** (F1, both datasets, CIs separated).
2. A local-geometry statistic near leaked points separates from clean regions — **NOT MET** (F2).

**Phase 0 = split verdict:** the accuracy-inflation half is textbook; the geometry half **fails**
for the overfit tree + DTA.

## 4. Why this CONFIRMS the organizing principle (PLAN §0)

The predictive rule: *a defect is visible to the adversarial-geometry probe iff it imposes a
**global, structured** boundary distortion that is **separable from accuracy**.* Leakage is the
opposite on both counts — it is **local** (a few memorized points) and, for the overfit tree, not
even locally distinct (uniform memorization). The rule therefore **predicts the null**, and the
data delivers it. Leakage is the clean counterexample: the defect where **accuracy is maximally
fooled and the geometry probe is blind** — because the distortion is not global.

## 5. Phase 1 direction (where a locality signal could exist)

A locality signal requires **selective** memorization — a model that treats leaked/duplicated
points differently from the rest:
- **`mode=dup`** (in-train replication): duplicating points forces the model to over-weight those
  regions, creating genuine over-confident pockets. More promising than `leak` for locality.
- **RandomForest** (duplicates change bagging weights) or a **pruned tree** (`max_depth` small, so
  memorization is selective and a duplicated cluster can force a dedicated pocket).
- Metric: per-point neighbourhood spread / margin near duplicated regions, or attack-success /
  confidence anomalies — not global spread.

Honest expectation: even so, the effect may be weak; leakage/duplication is fundamentally a
*local* defect and this probe is tuned for *global* distortion. The strong result here is F1 +
the principle confirmation, not a promise that Phase 1 recovers a geometry signal.

## 6. Limitations / debt

- Phase 0 = exact copies (jitter=0), `mode=leak`, overfit tree + DTA only. `dup` mode, jitter,
  RF/pruned, and black-box HSJ are Phase 1.
- The margin metric is one locality proxy; per-point confidence/margin at the leaked coordinates
  (not the adversarial displacement) is an alternative untested here.
