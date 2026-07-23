# 06 — Lessons & Gotchas

Subtle things that cost time or were initially wrong. Read before extending the
work.

## The density formula subtlety (n_pairs vs n_points)

The single most confusing issue in the project.

- **Aiden's actual formula:** `density = n_pairs / (sum_pairwise_dist + 1)`.
  Since `sum = n_pairs × mean_dist`, this is ≈ `1 / mean_dist` — **independent
  of cluster size**. It decreases cleanly as spread rises.
- **Our `dtree_attack` reconstruction accidentally used:**
  `density = n_points / (mean_dist + 1)`. This **depends on cluster size** `n`.
  When cluster sizes grow with bias (as they do for Tree+DTA: 5.5→6.5), this
  density goes *up* even though spread also goes up — the opposite of Aiden's.

**Consequence:** we saw "density increases for Tree+DTA but decreases for
SVM+HSJ" and briefly invented a story that "spread is a better, size-independent
metric." That story was wrong — the size dependence was an artifact of *our*
formula choice, not a flaw in density. Aiden's real density and spread agree
(one is 1/the other).

**Lesson:** when reproducing a metric, match the exact formula (pairs vs points,
sum vs mean). Report **spread** for interpretability, but never claim it beats
density — they're equivalent.

## "No gradient to follow" was the wrong explanation for Tree+HSJ

We initially said HSJ is weak on trees because the surface is piecewise-constant
so "there's no gradient." **Wrong** — the boundary still has a well-defined
normal on each flat facet. The correct explanation: HSJ estimates the boundary
direction by *sampling* perturbations and seeing which side they land on. On a
flat, axis-aligned facet, most perturbations stay inside the same leaf without
crossing the boundary, so the direction estimate is **noisy** and the attack
**frequently fails to converge** (many runs terminated). It's a sampling/
geometry problem, not an absence of gradient.

## Tree+HSJ non-monotonic drop at bias 0.9 is real, not noise

Spread rises 0.1→0.7 then *drops* at 0.9. Cause: at bias 0.9 only 4/50
target-class points survive. A depth-3 tree's class-0 decision *region*
collapses — its catchment falls from 9 test points to <1. So adversarial points
shift from probing the extrapolated gap boundary (high spread) to the intact
class-1-vs-2 boundary in dense space (low spread). The measurement stops
measuring the gap. **The leaf still exists (~1 leaf predicts class 0) — it's the
region volume that collapses, not the boundary.** SVM avoids this because RBF
support vectors keep a real class-0 boundary with only 4 points.

## Label noise density direction depends on data separability

Not a universal signal. Well-separated (iris/wine/synth-separated) → density
down. Overlapping/categorical (Car Eval/synth-categorical) → density up. If you
test a new dataset and get the "wrong" direction, that's expected — it's the
class geometry, not a bug.

## Above 0.5 label noise, the metric is unusable

Variance explodes (std 3× the mean). The apparent "spike" some exploratory plots
showed at noise 0.6–0.7 is sampling noise from a collapsed model, not signal.
The final grids cap label noise at 0.5.

## Coverage-gap test-set confound

Bias is injected *before* the CV split, so the test set composition changes with
bias level (fewer target-class points to attack). The compression-ratio analysis
partially addresses this (shows the signal survives), but a clean fix is to
inject bias into the **training fold only** and keep a fixed clean test set.
Not yet done — see [08-open-questions.md](08-open-questions.md).

## HSJ hangs on trees — use subprocess timeouts

HSJ frequently fails to converge on decision trees. The HSJ runners execute each
row in a fresh subprocess with a hard timeout; hung rows are recorded NaN and
skipped (12–27 per grid). Don't run HSJ-on-tree inline without a timeout.

## .gitignore quirk

Quoted filenames (`"foo bar.zip"`) do NOT work in .gitignore. Use unquoted
patterns or globs (`*.zip`). Cost us a wrong-looking commit.
