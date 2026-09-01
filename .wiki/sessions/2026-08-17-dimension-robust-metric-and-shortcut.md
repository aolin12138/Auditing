# 2026-08-17 — §8 dimension-robust spread metric + shortcut defect (Phase 0 & 1)

Thread C continued. Three experiments + a plots reorganisation. Journal: Attempts 21–23 in
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
- **Phase 0 leakage / duplication** (`PLAN.md §4`) — the accuracy-*inflating* defect; does geometry
  catch what accuracy hides? (NEXT)
- Signal-directed dimension reduction (discover the signal subspace from the cloud) — open.
- RF+HSJ shortcut arm deferred (hang-prone).

## Commits (origin/master)
`df6ebc0` (§8 metric) · `d3bc716`/`5720d7b` (§8 honest-negative downgrade) · `eff163e` (shortcut
Phase 0) · `f67962a` (shortcut spread row) · `1a9da7b` (plots reorg) · `9927ca1` (shortcut Phase 1
svm+HSJ).
