# Session — 2026-08-16 (defect-expansion Phase 1: class imbalance across models)

Continued Thread C from the canonical next gate (`ACTIVE.md` → Phase 1 class imbalance).
Built and ran the Phase 1 experiment, produced code + parquets + three verified plots + a
findings doc with confidence intervals, updated the wiki, committed and pushed
(`2c4a3e2`, origin/master). Evidence-based throughout — every claim below is backed by a
parquet cell or a viewed figure.

## What was built
- `defect_expansion_experiment/run_p1.py` — class imbalance across models via **black-box HSJ**.
  Hang-safe subprocess-per-row timeout driver (resumable); `--out` override forwarded to the
  worker (the Attempt-11 "flag not forwarded" bug avoided by design). Grid: models {svm, tree,
  rf} × structure {random, spatial} × frac {0.25…0.95} × tc {0, 2} × seeds (10 fast / 3 rf).
- `defect_expansion_experiment/run_confound.py` — before-split vs train-only accuracy
  (overfit tree + white-box DTA, deterministic, 30 seeds).
- `plot_p1.py` → `plots/imbalance/p1_{models,asymmetry,confound}.png`.
- `FINDINGS_imbalance_p1.md`.

## Runs (all logged)
- svm 210 rows, 0 hangs, 21.4 m. tree 210 rows, **50 hung/NaN** (known single-tree+HSJ
  flat-facet pathology; 25 s timeout handled each). rf 39 rows, 3 seeds, **8 hung**, 56.7 m —
  launched **detached (PID 13544) to its own parquet** so it ran in parallel with the fast
  models without a write-race. confound 660 rows, 1.6 m. No orphan processes afterward
  (verified by `CommandLine` filter).

## Findings (with intervals)
- **Minority recall is the ROBUST, model-agnostic separator** of coverage gap (spatial) vs
  imbalance (random). SVM frac 0.85: spatial recall 0.26 ± 0.01 vs random 0.75 ± 0.03; tree
  0.24 vs 0.74; RF 0.35 vs 0.77 — decisive on every model. This is the clean Phase-1 result.
- **The scalar-spread gap survives only PARTIALLY under black-box HSJ:** RF spatial 1.50× vs
  random 1.09× (direction clear, **n = 3** → underpowered); SVM separates at frac 0.95
  (1.46 ± 0.11 vs 1.23 ± 0.09) but is borderline at 0.85; **tree+HSJ shows no spread gap**
  (1.12 vs 1.14, CIs overlap; survivorship-biased). So the white-box DTA spread signal from
  Phase 0 did **not** cleanly generalise to black-box — *recall* did.
- **Class asymmetry confirmed:** deleting from a *separable* class (tc=0 setosa) is null in
  both geometry (~1.0×) and recall (pinned 1.00); only the *contested* class (tc=2 virginica)
  moves. A defect must distort a contested boundary to register.
- **Coverage-gap accuracy confound quantified (30 seeds):** before-split injection makes test
  acc RISE to 1.000 ± 0.000 (recall 1.000) because it also deletes the hard test band;
  train-only DROPS to 0.714 ± 0.001 (recall 0.143). Same model/attack/class/deletion — pure
  protocol artifact. Confirms the flagship "accuracy stays flat" headline was partly an
  artifact of before-split injection.

## Corrections / honesty notes
- Did **not** overclaim H1: the spread gap is model-dependent and RF is n=3. Reported the
  tree+HSJ null and the SVM borderline explicitly rather than reading a trend off point
  estimates (per the `verification-protocol` interval rule).

## Debt / next
- RF underpowered (3 seeds, 8/39 hung) — firming to 10 seeds ≈ 1 h detached (optional).
- tree+HSJ survivorship bias (50/210 hung) — its spread panel is not reliable; recall is.
- **Next gate:** Phase 0 shortcut / spurious feature (`PLAN.md §3`) — the strongest
  "geometry catches what accuracy misses" candidate.

---

## Continued same day — variance study, mechanisms, SVM, tooling (2026-08-16)

After Phase 1, the session extended into a full cross-dataset variance study, mechanism
interrogation, a second model+attack, and a tooling review. All committed/pushed to
origin/master.

### Cross-dataset variance (`run_variance.py`, `run_confound.py`, `FINDINGS_variance.md`)
Re-ran imbalance-vs-coverage-gap on **wine (13-D)** as well as iris, factored by injection
protocol (train-only vs before-split), standardized, tree+DTA (30 seeds) and **svm+HSJ (15
seeds)**. Figures: `variance_{iris,wine}.png`, `variance_svm_{iris,wine}.png`,
`spread_fragility_map.png` (fragility map).
- **Recall discriminator is robust** across dataset AND model/attack (wine spatial recall
  0.48 vs random 0.90).
- **Spread signal is fragile** to *both* dimensionality (iris 1.26× → wine ~1.04×) and attack
  (svm+HSJ ~1.11× on iris). Cause = distance concentration (std/mean 0.54 iris vs 0.29 wine).
- **before-split spread move is a test-set composition artifact** (target class dropped from
  test; surviving-class spreads flat) — geometry-sibling of the accuracy confound.
- Class asymmetry + accuracy confound replicate cross-dataset.

### Mechanisms logged (obs → hypothesis → evidence → conclusion) in `FINDINGS_variance.md`
M1 random-vs-coverage-gap (directional boundary vs density; refined "pure density"); M2
before-split = composition artifact (decomposition refutes boundary hypothesis); M3
spread-fragility = dimensionality not overlap; M4 how `min_recall` is computed (= TP/(TP+FN) on
tc, clean test) and why (per-class, recall-not-precision, distance-free; naming caveat = minority
not minimum).

### Tooling / skills review (`.wiki/10-tooling-and-skills.md`)
Reviewed `nature-skills` (18 research skills). Top picks: **nature-figure** (submission figures +
QA scripts), **nature-statistics**, writing/polishing, literature search (for the Dost baseline).
Concrete viz fixes flagged: our **red/green palette is not colorblind-safe** (switch to blue/
orange + linestyle), add **vector (PDF/SVG) export**, panel labels, a shared `plotstyle.py`, and a
glyph-size floor. Not yet installed/applied — awaiting go-ahead.

### NEXT (no context lost)
**Dimension-robust spread metric** — `defect_expansion_experiment/PLAN.md §8`: implement kNN-ratio,
LOF-style density ratio, PCA-then-spread, kNN-graph local spread; recompute the wine/iris
adversarial clouds; check whether any recovers the coverage-gap-vs-imbalance signal raw spread
lost (success = wine CIs separate at frac ≥ 0.7 AND iris still separates). Deliverables:
`run_robust_metric.py`, `plot_robust_metric.py`, `FINDINGS_robust_metric.md`.
Then: Phase 0 shortcut/spurious-feature defect (PLAN §3). Optional: apply the viz optimizations /
install nature-figure for the report-figure pass.
