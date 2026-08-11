# Outlier Experiment — Phase 0.5 Findings (2026-07-30)

Extension of Phase 0 addressing a batch of review questions: **tree depth 10, add SVM,
normalise the spread, more seeds (tighter CIs), extended ranges, and OFAT slices with the
other factor explicitly held.**

Setup: iris, 5-fold, outliers into **training fold only**, target class **tc=2 (virginica)**.
Data: `results_extended.parquet` (1511 rows). Runner: `run_extended.py`. Figures: `plot_extended.py`.

- **Models:** `tree_d3` (pruned, DecisionTreeAttack), `tree_d10` (deep — for iris the natural
  full-tree depth is ~5, so depth-10 ≈ unlimited; DecisionTreeAttack), `svm` (rbf, HopSkipJump).
- **Normalisation:** spread ÷ that model's own clean (n_out=0) baseline. **1.0 = no effect.**
  (Absolute spread is L2 distance; normalising makes tree vs SVM comparable — their baselines
  differ: tree≈0.44, svm≈0.49.)
- **Grid:** `k ∈ {2,3,4,5,6,8}` × `n_out ∈ {1,3,5,10,15,20}` × direction {toward, outward}.
  Seeds: **10** for trees, **3** for SVM (HSJ slower; k∈{2,4,6,8}, n_out∈{5,10}).
- **OFAT slices (other factor fixed and stated):** distance sweep @ **n_out=10**; count
  sweep @ **k=6**. Full heatmap shows every combination.

## Headline numbers (strong cell k=8, n_out=10; normalised, 95% CI)

| model | toward | outward | accuracy |
|-------|--------|---------|----------|
| tree_d3 (pruned) | **1.36×** (CI clears 1.0) | 1.01× (null, CI on 1.0) | flat ~0.94 |
| tree_d10 (deep)  | **1.25×** (CI clears 1.0) | 1.00× (null) | flat ~0.95 |
| svm (rbf, 10 seeds) | 0.98× (null) | 1.05× (null, CI straddles 1.0) | flat ~0.96 |

## Findings

**F1 — The toward signal is real and dose-responsive (trees), accuracy flat.** Both trees
rise with distance and count; onset at **k ≥ 6** (see heatmap), up to 1.25–1.36× baseline
while test accuracy stays ~flat. With 10 seeds + OFAT the tree CIs now clear the baseline —
the CI-overlap concern from Phase 0 is resolved for the trees.

**F2 — Depth barely matters; direction dominates depth.** Contrary to the "deeper overfits →
stronger signal" guess, the **pruned depth-3 tree reacts slightly *more*** (1.36×) than the
deep tree (1.25×). The deep tree also needs the outlier *farther* to react (tree_d3 fires at
k=5, tree_d10 only at k≥6) — extra capacity lets it isolate a moderate outlier in a small
local pocket that barely distorts the probed boundary. So model *capacity* is a second-order
knob; **injection direction is first-order.**

**F3 — Outward is null for both trees.** Normalised spread ≈ 1.0 with CI on the baseline —
a decision tree is blind to correctly-labeled outward outliers regardless of depth (they land
in already-correct empty space; the minimal-perturbation attack never probes there).

**F4 — SVM (rbf) is genuinely robust in BOTH directions (re-run at 10 seeds, k out to 12).**
Normalised spread hovers at ~1.0 with CIs straddling baseline everywhere; the only point that
clears 1.0 is a faint `toward` bump at the absurd k=12 (1.14× [1.09,1.20]). **`outward` is
dead flat at every distance, including k=12 (1.00 [0.90,1.09]).** Mechanism: the **rbf kernel
is local** — a far outlier's Gaussian influence decays to ~0, so it never reaches the decision
boundary the attack probes (opposite of a *global*/linear boundary, which a far outlier drags).
**The n=3 "faint opposite lean" reported earlier was noise — it washed out at 10 seeds** (good
reason we re-ran). Net: trees are sensitive to `toward` outliers; rbf-SVM is not.

**F5 — Realistic-magnitude outliers are invisible to the tree.** Real outliers are usually
**~2–3σ** (classic |z|>3 rule; Tukey mild ≈2.7σ, extreme ≈4.7σ). In our grid k=2–3 (and up
to k=4) sit at ~1.0 for both trees — **no signal**. The tree only reacts to *gross* outliers
(k≥6, i.e. 6σ, more like data-entry errors). This is a real limitation of tree-based
geometric auditing and a motivation to test more sensitive / global-boundary models.

## Figures (`plots/`)
- `phase05_dose_response.png` — normalised spread vs k (n_out=10) and vs n_out (k=6), toward,
  one line per model, 95% CI.
- `phase05_direction_by_model.png` — toward vs outward at the strong cell, per model.
- `phase05_svm_directions.png` — SVM toward vs outward vs k (10 seeds), robust in both directions.
- `phase05_tree_heatmaps.png` — normalised k×n_out (toward) for both tree depths.

## Caveats / next
- SVM now at **10 seeds, k∈{2,4,6,8,12}** — robustness claim is firm; `outward` flat even at 12σ.
- **rbf vs linear:** the flat SVM result is specific to the *local* rbf kernel. A **linear-kernel
  SVM** is the clean test of whether a *global* boundary sees the outward drag — a cheap next step.
- At extreme toward (n_out=20), accuracy dips slightly (~0.93) — outliers become contamination
  rather than sparse outliers; the clean accuracy-blind regime is n_out ≤ ~10.
- RandomForest / XGBoost still pending (Phase 3) — H4 (bagging robustness) untested.
