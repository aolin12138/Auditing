# Session — 2026-07-30

Presentation preparation: fact-checking, extending label-noise experiments to
0.9, building the full-range figure, rewriting report narrative, assembling
speaker script.

## Topics resolved (evidence-driven)

### HSJ attack mechanics — verified from ART source
Read `art/attacks/evasion/hop_skip_jump.py` directly. Untargeted HSJ: (1) draws
uniform random noise until a misclassified point is found (up to 100 attempts),
(2) binary-searches inward from that random point toward the test point, stopping
just on the misclassified side of the nearest boundary, (3) walks the boundary
for `max_iter=10` iterations, estimating the boundary normal via Monte-Carlo
sampling of ~50-200 perturbations. Result: one adversarial point per
correctly-classified test point (minus failures). Endpoint is always across the
boundary (misclassified). Perturbation = ‖adv − test_point‖.

### Label noise > 0.5 — measured, not guessed
Extended both experiment grids from 0.1-0.5 to 0.1-0.9:
- `dtree_attack_experiment`: +144 rows (504 total)
- `hsj_label_noise_experiment`: +288 rows (648 total, 89 hung/skipped)

Observations across the full range:
- Spread rises from 0.1 to 0.5 with large effect sizes (d=+1.1 to +2.0 across
  combos), but test accuracy falls in lockstep — not a hidden defect.
- Above 0.5: variance grows ~10× (std 0.05→0.6-0.9), valid runs collapse
  (Tree+DTA: 36→12 at 0.9), model accuracy falls below the 1/3 chance line.
- Trees (spread→1.8) and SVM (spread→0.8) diverge above 0.5: trees memorize
  noise by growing leaves → adversarial points scatter widely; SVM's RBF
  regularization bounds the surface → spread stays constrained. Both are
  downstream of a dead model, not diagnostic.
- The previous description ">0.5 is random" was imprecise: the mean continues
  to rise but the metric is unusable because variance dominates, valid n
  collapses, and the model is below chance.
- Label noise was also tested on wine, Car Evaluation, and 3 synthetic 3D
  datasets. Direction of geometric change depends on class separability:
  well-separated → spread increases; overlapping/categorical → direction
  reverses.

### Coverage gap, bias 0.7→0.9 Tree+HSJ drop — per-class decomposition
The drop appears to be only Tree+HSJ in the aggregate plot, but per-class (tc=0,
setosa depleted) decomposition shows:

| tc=0 spread | bias 0.7 | bias 0.9 |
|-------------|----------|----------|
| Tree+HSJ | 0.806 | 0.627 (−0.18) |
| SVM+HSJ | 0.616 | 0.533 (−0.08) |
| Tree+DTA | 0.574 | 0.610 (+0.04) |

Both HSJ combinations drop for the depleted class. Tree+HSJ's drop is large
enough to survive averaging over tc=1 and tc=2; SVM+HSJ's smaller drop is
diluted by the other classes. Tree+DTA's slight rise (0.036) is within the 95%
confidence interval overlap — not a confirmed trend.

The explanation for why HSJ drops but DTA does not was initially overstated as
"DTA deterministically finds the far boundary." The per-point tracing needed to
verify this mechanism was not completed. The conservative observation: at bias
0.9, only 4 target-class points survive; almost no test setosa points are
available to attack; the pool of attackable points shifts toward the intact
classes where the nearest boundary is close → spread drops. The swerve is
visible in HSJ and within noise for DTA.

### Compression ratio — per-point evidence
`_probe_review.py` confirmed with per-point perturbation tracing (Tree+HSJ,
label noise 0.1/0.3/0.5): perturbation shrinks (1.31→0.82) because more label
noise places boundaries closer to every point. Perturbation and spread move in
opposite directions — perturbation measures distance to nearest boundary per
point; spread measures distance between adversarial points. Moving LESS (small
perturbation) means points stay at their original spread-out locations → spread
goes UP. This is consistent, not contradictory.

## Artifacts produced

- `figures/report/p2_label_noise.png` — full 0.1-0.9 range, 3 combos, shaded
  >0.5 regime, error bars, accuracy panel
- `midyear_report.docx` — §3.1 rewritten (spread rises but confounded; >0.5
  metric destabilises with variance quantification)
- `presentation/deck.html` — new slide 08 (label noise), slide 10 replaced
  (Limitations → Conclusion & Future Work), slides renumbered
- `presentation/p2_label_noise.png` — copy for deck
- `presentation/SPEAKER_SCRIPT.md` — 4-slide script, ~4 minutes, trimmed of
  speculative mechanics
- `presentation/HANDOFF.md` — self-contained briefing doc for LLM sessions
- `experiments/probes/_probe_review.py` — per-point perturbation + source-class
  decomposition probe
- Extended `.parquet` data in both label-noise grids

## Wiki corrections

- `.wiki/04-findings.md`: Tree+HSJ bias-0.9 drop reframed as "attack effect
  (HSJ vs DTA determinism)" rather than "model effect (region collapse)."
  Added per-class evidence table. Added: SVM+HSJ also drops per-class, masked
  by averaging. Label noise finding expanded to full 0.1-0.9 range with
  variance quantification.
- `.wiki/06-lessons-gotchas.md`: HSJ mechanics section added (verified from ART
  source). Bias-0.9 gotcha rewritten with per-class data. Clarified that
  coverage gap deletes one class along one feature, not all data.

## Corrections made during this session

- The phrase ">0.5 is random" was replaced with "metric destabilises: variance
  grows ~10×, valid runs collapse, model below chance."
- "Label noise trend is flat" — incorrect. Spread rises with d=+1.08 to +2.02
  up to 0.5. Correct framing: rises but confounded with accuracy.
- "SVM avoids the 0.9 drop because RBF keeps boundary alive" — incomplete.
  SVM+HSJ also drops for tc=0 (0.616→0.533); it's masked by class averaging.
- "DTA keeps rising because it's deterministic" — the rise (0.574→0.610) is
  within 95% CI overlap; the mechanism was not verified with per-point tracing.
