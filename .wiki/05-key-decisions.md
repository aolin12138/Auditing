# 05 — Key Decisions

A log of the design decisions made, with rationale. Newest first.

## Report metric framing: "spread, equivalently 1/density" — not "spread is better"
**Decision:** Present spread as the reported metric because a distance is more
interpretable, but state explicitly that it equals the inverse of Aiden's
density. **Why:** we initially claimed "spread beats density" — that was wrong.
Aiden's `n_pairs/sum` density ≈ 1/spread; they carry identical information. The
"density confounds size" claim only applied to our own `n_points/(mean+1)`
formula. Being honest here avoids an obvious reviewer question.

## Report format: follow UoA P4P template, no title/abstract/refs
**Decision:** 4–6 pages, single-column, Times New Roman 11pt justified, section
headings 12pt bold CAPS, sub-headings 11pt bold. No title page, abstract,
references (per the mid-year assignment spec, which differs from the generic
template). **Why:** matches the actual Canvas assignment requirements.

## Presentation: edit the original HTML deck, preserve style
**Decision:** Fix the existing HTML deck in place rather than rebuild as pptx.
**Why:** keeps the exact original aesthetic (Space Grotesk / IBM Plex, blue
#1e5eff). Added a compression-ratio slide, corrected the metric slide, fixed the
Tree+HSJ explanation.

## Figures: z-score per combination (like Aiden)
**Decision:** z-score spread within each model+attack combination on the main
finding plot. **Why:** puts the three combinations on a common scale so the
shared upward trend is visible regardless of their different absolute spreads.

## Add HopSkipJump as the second attack
**Decision:** Use HSJ (decision-based, black-box) alongside DecisionTreeAttack.
**Why:** DTA is tree-only, so cross-model comparison was impossible. HSJ needs
only labels, so the identical attack runs on tree and SVM — any signal
difference is about the model, not the attack. Cost: HSJ is weak on trees
(flat boundary) and hangs frequently (handled via subprocess timeouts).

## Add label noise as the second defect
**Decision:** Test label noise alongside coverage gap. **Why:** establishes
specificity — does the signal fire for *any* defect, or only structured ones?
Result: label noise is not independently detectable, which sharpens the claim.

## Verify with compression ratio
**Decision:** Measure adv_spread / orig_spread to check the coverage-gap signal
isn't just the changing test set. **Why:** coverage gap is injected before the
CV split, so the test set differs per bias level — a real confound. The ratio
(~0.70 vs →1.0) shows the attack genuinely reshapes the cloud.

## OPTICS over DBSCAN
**Decision:** density-adaptive clustering. **Why:** DBSCAN's fixed ε would
pre-filter the very density variation we measure. See [03-methodology.md](03-methodology.md).

## Metric: report mean_dist (spread), keep density + clust_size as columns
**Decision:** compute all three per cluster in one OPTICS pass
(`aiden_density`, `density`, `mean_dist`, `clust_size`). **Why:** lets us
decompose the signal and show what drives it, without re-running.

## Repo organisation
**Decision:** separate `aiden_original/` from our experiment folders; consolidate
plots into `plots/investigation/` and `figures/report/`; exploratory scripts in
`experiments/exploratory/`, probes in `experiments/probes/`. **Why:** peers must
immediately see what's Aiden's vs ours. See [07-repo-map.md](07-repo-map.md).
