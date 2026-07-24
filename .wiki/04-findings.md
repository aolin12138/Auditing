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
2. **Non-monotonic drop at bias 0.9 — it's an ATTACK effect (HSJ vs DTA), not
   purely a model effect.** Driven entirely by tc=0 (setosa depleted): spread
   0.806 → 0.627. At bias 0.9 only 4/50 setosa points survive, so <1 lands in
   the test fold — the attack has almost nothing to target there and reverts to
   the **intact class-1-vs-2 boundary** in dense space → spread drops. Why only
   HSJ shows it: **DecisionTreeAttack is deterministic** and always captures the
   far, stretched gap boundary, so **Tree+DTA keeps rising** (0.574 → 0.610).
   **HSJ must *reach* that far boundary by random-init + sampling**; when the
   depleted class runs out of test points it can't, so its points revert to the
   healthy boundary (and it fails more — 4 NaN at 0.9, zero for SVM). Note SVM+HSJ
   *also* drops for tc=0 (0.616 → 0.533) — it's just **masked in the aggregate by
   class averaging** over tc=1/tc=2, whereas Tree+HSJ's swing is large enough to
   survive averaging. So "SVM avoids it" is an aggregation artifact. See
   [06-lessons-gotchas.md](06-lessons-gotchas.md).

---

## Finding 2 — Label noise: no independent signal

Spread **rises** under label noise (it is NOT flat), but it is still **not a
useful diagnostic** for two reasons. Measured across the full 0.1–0.9 range for
all three combos (grids extended past 0.5 for this):

- **The rise is real but accuracy-CONFOUNDED.** Spread 0.1→0.5 rises with large
  effect: Tree+DTA d=+2.02, Tree+HSJ d=+1.57, SVM+HSJ d=+1.08. But test accuracy
  falls in lockstep (tree 0.93→0.64, SVM 0.96→0.78). The defect is already
  plainly visible in accuracy, so the geometry adds nothing. (Note: in *density*
  terms this reads as a *decrease* — density ≈ 1/spread; earlier notes quoting
  d=−0.80 / −0.09 were the size-sensitive density, not spread.)
- **Direction depends on class separability.** Spread *increases* on iris/wine
  (well-separated), little change / opposite on Car Evaluation (categorical).
  Confirmed on synthetic 3D.
- **Above noise 0.5: the metric destabilises entirely.** The model tends toward a
  random classifier (accuracy below the 1/3 chance level by 0.8); spread variance
  grows ~an order of magnitude (std 0.05 → 0.6–0.9) and the number of valid runs
  collapses (Tree+DTA n: 36 → 12 at 0.9; HSJ hangs more). No stable trend exists
  — both which points stay correct and where the stochastic HSJ lands are noise.
- **Mechanism:** well-separated data + noise → tree grows many leaves to
  memorise noise → every point sits next to a boundary → tiny perturbation → adv
  cloud ≈ original test cloud (compression ratio → 0.98). Past 0.5 this saturates.

Figure: `figures/report/p2_label_noise.png` (full range, 3 combos, shaded >0.5
randomness regime).

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
