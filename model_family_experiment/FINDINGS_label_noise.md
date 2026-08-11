# Model-Family × Label Noise — Findings (2026-07-30)

Attempted RF + single tree on label noise via black-box HopSkipJump. iris, 5-fold,
5 noise levels × 3 seeds × 3 noise-seeds per model. Data: `results.parquet`, `defect=label_noise`.

## Headline — LARGELY INTRACTABLE: label-noise + HSJ on tree models hangs

**53 of 90 cells hung** (RF 31/45, tree 22/45) and were timed out → NaN. Mechanism: flipping
training labels forces the tree/forest to grow a **maximally fragmented boundary** (many tiny
leaves memorising the noise); HSJ's boundary-direction search degenerates on those flat facets
and loops forever — the same pathology that makes XGBoost+HSJ unusable, but worse here because
label noise *maximises* boundary fragmentation. The 37 survivors are **survivorship-biased**
(only the cells where HSJ happened to converge), so they cannot support a clean quantitative
claim.

## What the survivors show (directional only)
- RF @ noise 0.1: spread ~0.6–0.8, test acc ~0.92.
- tree @ noise 0.4–0.5: spread ~0.8–1.0 but test acc **collapsing to 0.48–0.59**.

This is consistent with the project's established result: **label noise is not an independent
geometric signal — it shows up in accuracy** (which collapses), and above noise 0.5 the metric
destabilises. The white-box Tree+DTA label-noise grids (which don't hang) already cover this.

## Recommendation
Do **not** pursue RF/tree label-noise via HSJ further — it's a hang trap with no clean payoff.
For label noise, rely on the existing white-box DTA grids. If an ensemble label-noise number is
ever needed, it requires a per-point-timeout attack rework (same as the XGB fix).
