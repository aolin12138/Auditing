# 04 — Findings

## Headline

**Coverage-gap bias is detectable from adversarial geometry while accuracy stays
flat. Label noise is not.**

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
2. **Non-monotonic drop at bias 0.9.** Real, explainable: at bias 0.9 only 4/50
   target-class points survive. A depth-3 tree's class-0 decision *region*
   collapses (catchment drops from 9 test points to <1), so the attack targets
   shift to the **intact class-1-vs-2 boundary** in healthy, dense space →
   spread reverts down. The measurement stops probing the gap. SVM avoids this
   (RBF support vectors keep a real class-0 boundary alive). See
   [06-lessons-gotchas.md](06-lessons-gotchas.md).

---

## Finding 2 — Label noise: no independent signal

Spread also moves under label noise, but it is **not a useful diagnostic**:

- **Accuracy-confounded.** Tree+HSJ: spread d=−0.80 but accuracy collapses
  0.93→0.64. SVM+HSJ: null (d=−0.09), accuracy 0.96→0.78. When the geometry
  moves, accuracy has already flagged the problem.
- **Direction depends on class separability.** Density *decreased* on iris/wine
  (well-separated), *increased* on Car Evaluation (categorical/overlapping).
  Confirmed on synthetic 3D: well-separated → down, overlapping → slight up,
  categorical → strong up.
- **Mechanism:** well-separated data + noise → tree grows many leaves to
  memorise noise → points scatter across fragmented boundaries. Overlapping
  data → tree near ceiling → few new boundaries → points stay clustered.
- **Above noise 0.5:** variance explodes (std 3× the mean) — signal unusable.

Figure: `figures/report/p2_label_noise.png`.

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

## Metric decomposition note

Density combines cluster size and spread. On Tree+DTA coverage gap, cluster size
*grows* with bias while spread also grows, so `n_points/(mean+1)` density can go
the "wrong" way — this is why we report spread. But Aiden's actual `n_pairs/sum`
density ≈ 1/spread and is size-invariant, so it agrees with spread. See
[06-lessons-gotchas.md](06-lessons-gotchas.md) for the full story.
