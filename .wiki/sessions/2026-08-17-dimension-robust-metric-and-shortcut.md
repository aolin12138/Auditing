# 2026-08-17 — §8 metric, shortcut defect, leakage defect + Thread C characterization

Thread C continued and effectively **completed the characterization** (all 3 new defects piloted).
Four experiments + a plots reorganisation + a wiki sync. Journal: Attempts 21–25 in
`~/.pi/runs/2026-07-30-outlier-model-experiments-plan/journal.md`.

## What was done

### §8 — dimension-robust spread metric (Attempt 21) → honest-negative
Persisted the adversarial clouds (`run_variance.py --save-clouds`; tree re-run reproduced all
1260 cells identically, svm re-run reproduced documented numbers to 3 dp — same-clouds guarantee).
Tested 4 candidates on the same clouds (`run_robust_metric.py`): **m1 kNN-ratio, m2 LOF,
m3 PCA-then-spread, m4 kNN-local**.
- By the letter, m4 and m3 give clean wine CIs — but **substantively an honest-negative**: the
  wine effect is a **+4% ripple** on a clean baseline already as large as the defect arms (user
  caught this). **Spread is genuinely dimension-limited; recall is the recommended diagnostic.**
- Useful negatives: ratio metrics (m1/m2) are **provably null** (the coverage-gap stretch is
  ~uniform inflation → ratios invariant); PCA fixes concentration (0.56→0.82) but not the effect;
  raw LOF is numerically unstable on white-box clouds (near-duplicate adv points → exploding LOF).
- `FINDINGS_robust_metric.md`, `plots/spread_metric/`.

### Shortcut / Clever Hans — Phase 0 tree+DTA (Attempt 22) + Phase 1 svm+HSJ (Attempt 23)
5th feature = `corr·(y−1)+N(0,1)` in train, pure noise in test. Sweep `corr {0..8}`, iris + wine.
- **Scalar spread fails** (directional defect, not dispersion). The right metric is **per-axis**:
  `spur_frac` = fraction of adversarial L2 displacement on the cheat axis.
- **H1 confirmed** (tree+DTA): spur_frac monotone 0.11→0.50 iris, 0.00→1.00 wine; first spurious
  split rises to the root; cheap-escape prediction *refuted* (displacement grows — thresholds
  recede).
- **Survives black-box HSJ** (svm): separates from control by corr=2, peaks corr=4, dips corr=8
  (survivorship). **Model-type flip**: wine leads under svm+HSJ (smooth boundary leans on the
  global cheat axis); iris led under tree+DTA (overfit tree exploits iris's real structure).
- Caveat: co-moves with accuracy — never a zero-cost detector. `FINDINGS_shortcut.md`,
  `plots/shortcut/{phase0,svm_hsj}.png`. Finding 7 in `04-findings.md`.

### Leakage — Phase 0 mode=leak (Attempt 25) → split verdict, confirms the principle
Copy `n_leak` train points into the test fold (overfit tree+DTA, iris+wine, 30 seeds).
- **H2 confirmed decisively:** measured accuracy inflates (iris 0.94→0.97, wine 0.90→0.94) while
  *honest* accuracy on non-leaked points is exactly flat — the gain is 100% fake.
- **H1 null:** leaked points have the same adversarial margin as clean (CIs overlap). The overfit
  tree memorizes ALL training data uniformly → no local pocket to detect.
- **Confirms the §0 organizing principle:** a defect is visible iff it's a GLOBAL, structured
  distortion. Leakage is LOCAL → predicted invisible → delivered. The clean counterexample
  (accuracy maximally fooled, geometry blind). `FINDINGS_leakage.md`, `plots/leakage/phase0.png`,
  Finding 8.

### Thread C characterization (now complete)
One predictive rule (§0), three defects on either side of it:
| defect | geometry | why |
|---|---|---|
| coverage gap / imbalance | spread catches it (iris) | global spatial hole |
| shortcut | per-axis catches it (survives wine + black-box) | global directional distortion |
| leakage | invisible (only fake accuracy reveals it) | local, not global |

### Plots reorganised (user feedback: names confusing)
`plots/{imbalance,variance,spread_metric,shortcut}/` with self-explanatory names + `plots/README.md`
index (plot → experiment → findings doc → script → data). All scripts write to subfolders; all
FINDINGS/PLAN/wiki references updated.

## Decisions / signals
- **User standard: no overselling.** A CI-separation "pass" on a ripple-sized effect is downgraded
  to the honest-negative; effect size vs baseline is stated next to every separation claim.
- **New principle:** the metric must match the defect's geometry — spatial-hole defects → scalar
  spread; directional defects → per-axis displacement. (This reframes "spread is fragile".)
- **User-proposed direction (open):** dimension reduction *before* the metric — but to the
  **signal-carrying** subspace, not top-variance PCA (which was m3 and only fixed concentration).
  Captured in `08-open-questions.md`.

## Open / next
- **Next session (planned): dimensionality, normalization & signal-directed reduction study**
  (`PLAN.md §9`). User pivot: investigate how dimensionality affects the spread metric on the
  VERIFIED defects (coverage gap + imbalance), whether **dimension reduction before the metric**
  and the right **feature/metric normalization** recover/stabilise the signal. Controlled design:
  **pad iris with `m` noise dims** (ground-truth signal in the original 4) to isolate pure
  dimensionality; test if reduction recovers a KNOWN signal buried in noise — and whether the same
  reduction fails on wine (disambiguating the §8 negative: dimensionality vs defect-not-distorting).
- Leakage Phase 1 (mode=dup + RF/pruned, selective memorization) — deferred, weak expectation.
- RF+HSJ shortcut arm deferred (hang-prone).

## Commits (origin/master)
`df6ebc0` (§8 metric) · `d3bc716`/`5720d7b` (§8 honest-negative downgrade) · `eff163e` (shortcut
Phase 0) · `f67962a` (shortcut spread row) · `1a9da7b` (plots reorg) · `9927ca1` (shortcut Phase 1
svm+HSJ).
