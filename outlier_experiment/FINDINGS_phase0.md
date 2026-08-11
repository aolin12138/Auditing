# Outlier Experiment — Phase 0 Findings (2026-07-30)

Signal-check for the outlier defect. Overfit tree (`max_depth=None`) + DecisionTreeAttack,
iris, 5-fold, 3 seeds. Outliers injected into the **training fold only**; test fold clean.
Data: `results.parquet` (99 rows). Reproduce: `python run_experiment.py`. Mechanism probe:
`_diag.py`. Figures: `python plot_phase0.py`.

**Figures** (`plots/`):
- `phase0_dose_response.png` — spread vs `k` and vs `n_out`, toward vs outward, accuracy twin
  axis (the money shot: spread rises while accuracy flat; outward pinned to baseline).
- `phase0_interaction_heatmap.png` — `k` x `n_out` spread (toward); signal switches on at
  `k>=6` (a threshold: outliers must be far enough to cross into the neighbour's territory).

## Headline — the accuracy-blind signal exists (for the right injection)

At the strongest firing cell (`tc=2`, `toward`, `k=8`, `n_out=5`, seed 42):

| | tacc | vacc | spread(all) | spread_c2 |
|--|------|------|-------------|-----------|
| clean baseline | 1.000 | 0.953 | 0.361 | 0.600 |
| **toward** k=8 n=5 | 1.000 | **0.953** | **0.522 (+0.161)** | 0.696 (+0.096) |
| outward k=8 n=5 | 1.000 | 0.953 | 0.361 (+0.000) | 0.600 |

**Accuracy is exactly flat while spread rises +0.161.** This is the "defect that hurts data
quality without collapsing accuracy" the project has been looking for
(`.wiki/08-open-questions.md`). The `outward` null contrast proves the effect is mechanistic,
not a test-set artifact (test fold is identical across all three).

## Dose-response (full factorial, 3 seeds, within `toward`)

Baseline spread(all) = 0.374, vacc = 0.953.

| k (× class std) | spread | vacc |  | n_out | spread | vacc |
|-----|--------|------|--|-------|--------|------|
| 2 | 0.405 | 0.943 |  | 1 | 0.413 | 0.952 |
| 4 | 0.388 | 0.933 |  | 3 | 0.437 | 0.949 |
| 6 | 0.487 | 0.951 |  | 5 | 0.469 | 0.946 |
| 8 | 0.503 | 0.956 |  | 10 | 0.464 | 0.936 |

Spread rises with both distance `k` and count `n_out` (n_out saturating by ~5). Accuracy
stays within ~0.02 of baseline throughout — the signal is a genuine geometry-only tell.

## The decisive knob: injection **direction** (validates the user's instinct)

Main effect of direction (mean over k, n_out, seeds): **outward 0.377 ≈ baseline 0.374 (null)**
vs **toward 0.446 (signal)**.

Mechanism (from `_diag.py`, Δleaves = whether the overfit tree reacted at all):

| tc | direction | Δleaves | Δspread(all) |
|----|-----------|---------|--------------|
| 0 setosa (isolated) | outward | +0.0 | **+0.000** |
| 0 setosa | toward | +0.0 | +0.029 |
| 1 versicolor (central) | outward | +1.4 | +0.037 |
| 1 versicolor | toward | +1.8 | +0.071 |
| 2 virginica (contested edge) | outward | +0.0 | **+0.000** |
| 2 virginica | toward | +1.0 | **+0.161** |

**Two design lessons that reshape the plan:**

1. **A decision tree is blind to correctly-labeled `outward` outliers.** They land in empty
   space already labeled as the class, so the tree adds no split (Δleaves = 0) and the
   minimal-perturbation attack — which travels to the *nearest* (opposite-side) boundary —
   is unaffected. Spread is *exactly* unchanged. This is a real characterisation result:
   tree-based geometric auditing does not see low-density outward outliers.
2. **The target class must be contested.** `tc=0` (setosa, linearly separable) barely reacts
   in any direction; `tc=2`/`tc=1` (the overlapping pair) react to `toward`. My original
   Phase-0 default (`tc=0`, `outward`) was the single guaranteed-null cell — the smoke test
   caught it before we ran a null grid.

## Caveat

At extreme `toward` injection (`n_out=10`), vacc dips to ~0.936 — the outliers start landing
*inside* the neighbouring class and behave like mild contamination rather than sparse
outliers. Below that the accuracy-blind claim is clean. This boundary between "outlier" and
"contamination" is itself worth mapping (it distinguishes this defect from label noise, which
collapses accuracy outright).

## Implications for later phases

- **Direction is promoted to a Phase-0/1 factor** (was Phase 2). It is the dominant knob.
- **Phase 1 asymmetry** (`tc` sweep) should compare *contested* classes; expect setosa to be
  a near-null target — itself an asymmetry result.
- **Model comparison (Phase 3)** is now sharper: SVM/logistic have a *global* boundary, so
  they may show the `outward` signal a tree cannot — a clean model-family contrast.
