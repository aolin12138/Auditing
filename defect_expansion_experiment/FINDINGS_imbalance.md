# Class Imbalance vs Coverage Gap — Phase 0 Findings (2026-08-11)

Overfit tree (`max_depth=None`) + white-box DecisionTreeAttack. iris, 5-fold, class `tc=2`
(virginica), spatial arm sorts by feature 2 (petal length). **30 seeds.** Both arms delete the
same COUNT from the train fold; test kept clean. Data: `results_imbalance.parquet`.
Fig: `plots/imbalance/phase0_spread_vs_random.png`. Runner: `run_imbalance.py`.

## H1 — CONFIRMED: the spatial hole adds spread on top of the count effect

Normalised adversarial spread vs fraction removed (95% CI, baseline spread 0.42):

| frac | random (imbalance) | spatial (coverage gap) | CIs separate? |
|------|--------------------|------------------------|---------------|
| 0.25 | 1.03× | 1.07× | overlap |
| 0.50 | 1.07× | **1.19×** | **spatial > random** |
| 0.70 | 1.13× | **1.23×** | **spatial > random** |
| 0.85 | 1.19× | **1.34×** | **spatial > random** |
| 0.95 | 1.24× | **1.37×** | **spatial > random** |

**Both** arms raise spread, so having *fewer samples* is itself part of the coverage-gap signal
(random imbalance alone → 1.24× at 95%). But **spatial deletion is reliably higher** (CIs
separate for frac ≥ 0.5, gap widening with severity). So the coverage-gap spread signal is
**both** a count effect **and** a spatial-hole effect, the spatial hole dominating at high
severity. This is the disambiguation we wanted: the hole (directional extrapolation) is real,
not merely "fewer points".

## H2 — CONFIRMED: minority-class RECALL is the clean discriminator

| frac | random min-recall | spatial min-recall | overall acc (rand / spat) |
|------|-------------------|--------------------|---------------------------|
| 0.25 | 0.91 | 0.77 | 0.94 / 0.92 |
| 0.50 | 0.87 | 0.62 | 0.94 / 0.87 |
| 0.70 | 0.83 | 0.56 | 0.93 / 0.85 |
| 0.85 | 0.73 | 0.23 | 0.90 / 0.75 |
| 0.95 | 0.61 | **0.14** | 0.87 / 0.71 |

Coverage gap **craters** minority recall (→0.14) because it deletes a contiguous petal-length
band → test virginica in that band are systematically misclassified. Random imbalance thins the
class uniformly → coverage of the full range survives → recall degrades gently (→0.61). The
recall gap grows to **+0.50**. So the two defects are trivially separable by *per-class recall*,
even where their scalar spread values are close.

## Important nuance — train-only coverage gap is NOT accuracy-blind (revises the earlier claim)

The established coverage-gap result (`model_family_experiment/FINDINGS_coverage_gap.md`,
`.wiki/04-findings.md`) reported spread rising while **accuracy stayed flat**. That used
**before-split** injection, which removes the deleted region from the *test* set too — so there
were no test points left to misclassify (exactly the confound flagged in
`.wiki/06-lessons-gotchas.md`). With the correct **train-only** protocol here, coverage gap
**does** cost accuracy (0.95 → 0.71) and minority recall (→0.14). Takeaway: coverage gap's
"accuracy-blindness" was partly an artifact of test-set removal; under a clean test it is
visible in *both* accuracy and geometry, most sharply in minority recall.

## Verdict & next
- H1 ✅, H2 ✅, coverage-gap reproduction ✅ (spatial arm matches the known rising-spread shape).
- GO to **Phase 1**: add RF+HSJ and SVM+HSJ (does the spatial>random spread gap survive
  ensembling / black-box?), add `tc=0` (separable class) for the asymmetry check, and a
  before-split vs train-only accuracy comparison to nail the confound quantitatively.
