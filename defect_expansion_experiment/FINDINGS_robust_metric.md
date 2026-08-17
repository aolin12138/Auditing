# FINDINGS — §8 Dimension-robust spread metric (2026-08-17)

The methodological gate: find a spread/dispersion measure that **recovers the
coverage-gap (spatial) vs imbalance (random) signal on 13-D wine** that raw OPTICS
spread lost, **while keeping the working 4-D iris case intact**.

- Runner: `run_robust_metric.py` (metrics on persisted clouds) · clouds persisted by
  `run_variance.py --save-clouds` · figs: `plots/robust_{tree+dta,svm+hsj}.png`,
  `plots/robust_concentration.png` · data: `results_robust_metric.parquet`,
  `results_robust_summary.csv`.

## 0. Same-clouds guarantee (criterion 3)

Metrics are computed on the **exact adversarial clouds** behind `FINDINGS_variance.md`:
- tree+DTA is deterministic → re-run with `--save-clouds` reproduced all 1260 parquet rows
  **identically (1260/1260 cells, max diff 0.0)** vs the backup.
- svm+HSJ is stochastic → clouds persisted in a fresh run that **reproduced the documented
  numbers to 3 decimals** (iris recall 0.444/0.673 vs documented 0.44/0.67; wine 0.477/0.900
  vs 0.48/0.90; spread 1.122× iris / 1.019× wine).

## 1. What each candidate measures, and why it could resist dimensionality

Context: raw spread = OPTICS within-cluster **mean over ALL pairwise distances**. In high
dimensions pairwise distances concentrate around their mean (std/mean 0.54 iris → 0.30 wine),
so the mean goes numb. Each candidate attacks this differently.

| metric | what it measures | why it could survive high dimension |
|---|---|---|
| **M1 kNN-distance ratio** | Each adversarial point's distance to its k-th nearest neighbour, **divided by the cloud's median** of those distances (mean ratio). Relative *local isolation*. | Distances in high-D are inflated and squeezed together; taking a **ratio cancels the global scale** — only relative contrasts survive, and those should be dimension-invariant. |
| **M2 LOF density ratio** | Local Outlier Factor: each point's **local reachability density divided by the mean density of its neighbours** (~1 in uniform regions, >1 where locally isolated). Local *density anomaly*. | Built from **ratios of local densities**: uniform distance inflation cancels by construction; the metric is dimensionless from the start. |
| **M3 PCA-then-spread** | The **same OPTICS spread, but in the cloud's top-3 PC subspace**. Same quantity, cleaner space. | Measured earlier: adversarial displacement is carried by only **~2.8/13 axes on wine**; the noise axes are what drive concentration. Projecting them out should restore the signal. |
| **M4 kNN-graph local spread** | **Mean distance from each point to its k nearest neighbours only** — local dispersion, the *bulk (far) pairwise distances excluded*. | The far-pair distances are exactly the ones that concentrate most and dominate the raw mean. Local neighbour distances retain relative variance even in high-D. |

## 2. Results (95% CIs, normalised × clean baseline, train-only)

### tree + white-box DTA (30 seeds) — the working case

| metric | wine frac .8 spat vs rand | sep | wine frac .9 spat vs rand | sep | iris frac .9 spat vs rand | sep |
|---|---|---|---|---|---|---|
| m0 raw spread (reference) | 1.040±.012 vs 0.995±.017 | yes* | 1.040±.016 vs 1.005±.015 | yes* | 1.259±.043 vs 1.120±.039 | yes |
| m1 kNN ratio | 1.007 vs 1.003 | no | 1.011 vs 1.007 | no | 0.982 vs 0.996 | no |
| m2 LOF (raw) | — | — | — | — | **numerically broken** (see §3) | — |
| m2b LOF jittered+median | 0.999 vs 1.000 | no | 1.002 vs 1.000 | no | 0.996 vs 0.997 | no |
| **m3 PCA-then-spread** | 1.039 vs 1.007 | no | **1.072±.031 vs 1.006±.029** | **yes** | **1.256±.044 vs 1.123±.047** | **yes** |
| **m4 kNN-local spread** | **1.055±.007 vs 1.012±.006** | **yes** | **1.078±.008 vs 1.032±.008** | **yes** | 1.198±.010 vs 1.112±.018 | yes |

\* raw spread on wine is *marginal*, not zero: the CIs just separate at 30 seeds, but the
effect size (1.04×) is a fifth of iris (1.26×). M4 turns the same gap into a **clean**
separation: comparable gap (0.043–0.046) at roughly **half the noise** (se ≈ 0.007 vs 0.016).

### svm + black-box HSJ (15 seeds) — the fragile case

| metric | wine frac .9 spat vs rand | sep | iris frac .9 spat vs rand | sep |
|---|---|---|---|---|
| m0 raw spread | 1.019 vs 1.021 | no | 1.122 vs 1.061 | no (never worked under HSJ) |
| **m3 PCA-then-spread** | **1.070±.051 vs 0.962±.047** | **yes** | 1.116 vs 1.066 | no |
| m4 kNN-local | 0.952 vs 0.958 | no | 1.142±.022 vs 1.087±.021 | yes |
| m4 @ frac .8 (wine) | **0.940 vs 0.977** | yes, **inverted** | 1.133 vs 1.065 | yes |

Under black-box HSJ the recovery is **partial**: m3 recovers wine at frac 0.9; m4 separates
iris at both severities and wine at 0.8, but with the **direction flipped** (spatial cloud is
locally *tighter* than random under HSJ on wine). Caveat: 15 seeds; HSJ's 10-iteration budget.

## 3. Why the losers lost (mechanistically informative)

- **M1 null everywhere (even iris)** — the kNN-ratio *removed the working signal*. The
  coverage-gap stretch is approximately **uniform inflation** of the cloud: all kNN distances
  scale together, so distance/median is invariant. The signal lives in **absolute local
  distances**, not relative isolation. Ratio-normalisation was the wrong cure: it cancels the
  signal along with the concentration.
- **M2 numerically unstable on white-box tree clouds** — DTA maps many test points to the
  *same* adversarial instance → near-duplicate points → local reachability density → ∞ → LOF
  → 0 for some points and unbounded for others (baseline norm hit 1.97e6, contaminating every
  cell). After the fix (1e-8 jitter + median instead of mean) M2b is well-behaved but **null**
  — same ratio-invariance reason as M1. LOF as-is must not be used on white-box adversarial
  clouds; the instability is a finding in itself.
- **M0's wine failure re-diagnosed** — at 30 seeds raw spread is marginal-not-dead (CIs just
  separate; 1.04×). The earlier "collapses to ~1.0×" was a fair reading of the plot at the
  time; the sharper statement is: effect size collapses (1.26× → 1.04×) and noise dominates.

## 4. Concentration check (criterion c)

std/mean of the metric's dispersion quantity on clean baseline clouds (raw-spread reference
iris 0.54 / wine 0.30, ratio 0.56):

| metric | iris | wine | wine/iris ratio | interpretation |
|---|---|---|---|---|
| m0 raw spread | 0.54 | 0.30 | 0.56 | the concentration imbalance (reference) |
| m1 kNN ratio | 0.47 | 0.22 | 0.47 | not equalised — and null anyway |
| m3 PCA-then-spread | 0.54 | 0.44 | **0.82** | **equalised — mechanism confirmed** |
| m4 kNN-local | 0.47 | 0.22 | 0.47 | **not equalised, yet separates** |

Two distinct recovery mechanisms, both real:
- **M3 fixes the space** — removing noise axes genuinely equalises concentration (0.56 → 0.82,
  1.02 on the svm arm).
- **M4 fixes the statistic** — it does *not* equalise concentration, but it excludes the bulk
  far-pair distances that concentrate, and the local distances that remain still carry the
  spatial-vs-random contrast.

## 5. Verdict against success criteria

1. **≥1 candidate with wine separation at frac ≥ 0.7 AND iris intact — MET.**
   **M4 kNN-local spread** (k = min(5, n//4)): wine 1.055 vs 1.012 (0.8) and 1.078 vs 1.032
   (0.9), iris 1.177/1.198 vs 1.074/1.112 — clean CIs, no parameters to tune.
   **M3 PCA-then-spread** (m = 3): passes at frac 0.9 on both datasets.
2. Honest-negative branch not needed.
3. **Same-clouds guarantee — MET** (§0: 1260/1260 identical for tree; svm reproduced
   documented numbers to 3 decimals).
4. Plots generated + numerically cross-checked against the summary CSV (all cells match);
   **visual check pending user** (current session has no image rendering).

**Recommendation going forward:** replace raw OPTICS mean-pairwise spread with **M4 kNN-local
spread** in future defect experiments (it is strictly better: same iris behaviour, wine
recovered, near-zero added complexity). Recall stays the most robust cross-dataset/cross-model
discriminator overall. Under black-box HSJ use **m3** for wine (only separator there).

## 6. Limitations / debt

- svm+HSJ arm = 15 seeds; the wine direction-flip under HSJ is unexplained (candidate:
  HSJ points cluster near the hole on the smooth SVM boundary — needs a mechanism study if
  pursued).
- m3 fixed at m=3 (motivated by the measured ~2.8/13 axes); no m-sweep.
- train-only protocol only (per §8 test protocol); iris/wine only — a >50-D dataset would
  stress M4 harder (debt inherited from the variance study).
- `m2_lof` in `results_robust_metric.parquet` is the unstable raw variant — exclude from
  analysis; use `m2b_lof_robust`.
