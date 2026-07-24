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

## Tree+HSJ non-monotonic drop at bias 0.9 — it's an ATTACK effect, not a model effect

Initial (incomplete) story: "the depth-3 tree's class-0 region collapses, so
SVM avoids it." **That's wrong as stated** — region collapse hits all three
combos equally (same coverage gap, same shrunk test set), so it can't be the
whole reason. The data (broken down by target class):

| tc=0 spread | bias 0.7 | bias 0.9 |
|-------------|----------|----------|
| Tree+HSJ | 0.806 | 0.627 ↓ |
| SVM+HSJ | 0.616 | 0.533 ↓ |
| Tree+DTA | 0.574 | 0.610 ↑ |

- The drop is driven entirely by **tc=0 (setosa depleted)** and appears in **both
  HSJ variants** — SVM+HSJ *also* drops for tc=0, it's just **averaged out** in
  the aggregate by tc=1/tc=2. Tree+HSJ's swing is big enough to survive averaging.
- **Tree+DTA does NOT drop** — because DecisionTreeAttack is **deterministic** and
  always finds the far, stretched gap boundary. HSJ must *reach* that boundary by
  random-init + Monte-Carlo sampling; when the depleted class runs out of
  attackable test points, HSJ reverts to the nearer healthy boundary (and fails
  more — 4 NaN at 0.9, zero for SVM). So the real distinction is **attack**
  (deterministic DTA vs stochastic HSJ), not model.
- The leaf still exists (~1 leaf predicts class 0); it's the region *volume* that
  collapses (catchment 9 test points → <1), not the boundary.

**Lesson:** always break an aggregate signal down by the looped factor (here `tc`)
before explaining it — "SVM avoids it" was an aggregation artifact.

## HSJ mechanics (verified from ART source, not memory)

`art/attacks/evasion/hop_skip_jump.py`, untargeted: for each correctly-classified
test point `x`, HSJ (1) draws **uniform random noise** points until one is
misclassified (`_init_sample`, up to `init_size=100` tries — failure mode #1 if
none found), (2) **binary-searches inward** from that random point toward `x`,
stopping just on the misclassified side of the nearest boundary, (3) **walks the
boundary** (`max_iter=10`) estimating the normal by **sampling ~50–200
perturbations** (Monte-Carlo — the stochastic/black-box part), stepping toward `x`.
Result = a misclassified point sitting just across `x`'s nearest boundary, as
close to `x` as possible. Key consequences:

- **Perturbation = ‖adv − x‖** (measured from the original test point; verified in
  `_probe_cg.py`) ≈ distance from `x` to its nearest boundary. Small perturbation
  = a boundary is right next to the test point.
- **The adv point is ALWAYS across the boundary** (misclassified). Mechanically
  HSJ moves *inward* from far away, but the right mental model for the numbers is
  "start at `x`, move out just far enough to cross."
- **n test points → ≤n adv points** (minus init failures and non-convergence hangs).
- **Non-deterministic**: random init + sampled direction → different runs give
  different points. This is why HSJ (black-box) is noisier than DTA and why it
  destabilises when boundaries are dense (label noise >0.5) or far (extreme gap).

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
