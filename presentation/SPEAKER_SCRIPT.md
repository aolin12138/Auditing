# Speaker Script — Aolin (Slides 07–10)

> Your section (partner does 01–06). ~5 minutes. Bold = say clearly.

---

## Slide 07 — Coverage Gap (~75 s)

"We varied three things: the model — decision tree or SVM. The attack — white-box
DecisionTreeAttack, or black-box HopSkipJump that only sees predicted labels.
And the defect type.

This is the coverage gap — deleting up to 90 percent of one class. Green: Tree
with white-box attack, strongest, Cohen's d of **+2.06**. Red: SVM under
black-box — **d = +0.75**, still clear. Blue: Tree under HopSkipJump, weakest —
**d = +0.38**. The tree's flat boundary makes HopSkipJump's direction estimate noisy.

Notice the blue line dips at 0.9 — almost the entire class is gone, only 4
points survive. The attack can only happen in the healthy region, so points
cluster together again.

But the headline: accuracy is **flat at 96 percent**. Model looks healthy.
The geometry gives the bias away."

---

## Slide 08 — Label Noise (~75 s)

"Now the contrast — label noise: we randomly flip training labels. Full range
to 0.9, all three combinations.

Spread **does** rise — effect sizes similar, around +1.1 to +2.0. But look at
accuracy: it falls in **lockstep**. Tree from 93 to 64 percent. SVM from 96 to
78. This defect is **not hidden** — accuracy already flags it. The geometry adds nothing.

The shaded region past 0.5: the model is below chance. Error bars blow up ten
times. Only 12 of 36 runs produce a usable cluster at noise 0.9. Just noise
from a dead model.

Even before 0.5, the **direction** depends on class separability — on overlapping
datasets the effect reverses. So when the metric moves, you can't trust which way.

Bottom line: coverage gap was a hidden defect. Label noise was never hidden.
The metric is specific, and that's a good thing."

---

## Slide 09 — Verification (~45 s)

"One concern: the coverage gap changes the test set too. Is spread just tracking that?

We checked with the **compression ratio** — adv spread divided by test spread.
Ratio = 1 means the attack did nothing.

Red, label noise: ratio → **0.98** — points barely moved. No signal.

Green, coverage gap: ratio stays at **~0.70** — the cloud is 30 percent tighter.
And perturbation **increases** with bias — points travel farther. Label noise
goes the opposite way.

The coverage-gap signal is a **real attack effect**, not an artifact."

---

## Slide 10 — Conclusion (~35 s)

"To wrap up.

Coverage gap: our metric caught a hidden defect while accuracy was blind.

Label noise: the geometry moved, but accuracy was already collapsing — never
hidden. Past 0.5 the metric destabilises.

What this tells us: different defects respond differently. No single metric
works for everything.

Next: more realistic defects — outliers, feature corruption. More models —
random forest, XGBoost. And most importantly, a **set of metrics**. One can
fail — it's much harder for all of them to fail at once."

*[Handoff to partner for Slide 11 — Takeaway]*

---

## Cheat sheet

| When you say… | Number |
|---------------|--------|
| Tree + DTA | d = +2.06 |
| SVM + HSJ | d = +0.75 |
| Tree + HSJ | d = +0.38 |
| Accuracy (coverage gap) | flat ~0.96 |
| Tree accuracy 0.1→0.5 noise | 0.93 → 0.64 |
| SVM accuracy 0.1→0.5 noise | 0.96 → 0.78 |
| Compression ratio, coverage gap | ~0.70 |
| Compression ratio, label noise | → 0.98 |
| Perturbation, coverage gap | rises |
| Perturbation, label noise | drops |
| Bias 0.9 surviving points | 4 of 50 |
| Variance above noise 0.5 | ~10× |
