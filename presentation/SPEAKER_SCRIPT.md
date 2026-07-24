# Speaker Script — Aolin (Slides 07–10)

> Your section. Partner covers opening through Pipeline + Metric.
> Read naturally. Bold = numbers to say clearly. ~4 minutes total.

---

## Slide 07 — Finding: Coverage Gap (~90 s)

"So in the experiment there are three things we varied to test the relationship between spread and the defect.

One: the model — we trained on a decision tree and an RBF-SVM. Two: the attack — one white-box, DecisionTreeAttack, which reads the tree structure directly; and one black-box, HopSkipJump, which only sees the predicted label. Three: the defect type — different defects give different signals, which is the whole point.

This plot shows all the combinations under the **coverage gap** defect. The x-axis is how much of one class we delete — going from 0.1 to 0.9, from 45 surviving points all the way down to just 4. The y-axis is spread — how far apart the adversarial points are from each other.

You can see the green line, Tree plus the white-box attack — strongest signal. Cohen's d of plus 2.06. The red line, SVM under the black-box attack — d of plus 0.75, still a clear effect. The blue line is the weakest — Tree under HopSkipJump, d of plus 0.38. There's a reason for that: the tree's boundary is a set of flat, axis-aligned facets. HopSkipJump works by sampling random perturbations to estimate the boundary direction — on a flat surface most samples stay inside the same leaf without crossing the boundary, so the estimate is noisy and the attack often fails to converge. We had to terminate 12 of those runs.

Now you might notice the blue line dips from bias 0.7 to 0.9 — the spread drops. That's because when the bias is 0.9, only 4 points of that class survive. You barely have any test points left in that class to attack. So the attack falls back onto the healthy classes — the ones that weren't touched — and their boundary is in a dense, normal region, so the adversarial points cluster closer together again. And here's the interesting part: the **white-box** attack, the green line, does NOT drop. It keeps rising. That's because DecisionTreeAttack is deterministic — even with almost no points, it still finds that far, stretched boundary in the empty gap, and crosses it, producing spread-out points. HopSkipJump, being non-deterministic, can't reliably reach that far boundary — it defaults to the easier, nearby boundary on the healthy side. So the dip is actually an attack-mechanism effect, not a model flaw.

But the punchline — look at the bottom panel. Test accuracy is basically flat, around 96 percent, for all three combinations. The model looks completely healthy. No standard check would flag anything wrong. But the geometry is shifting — the adversarial points know there's a hole in the data."

---

## Slide 08 — Label Noise (~80 s)

"Now, contrast this with a different defect — label noise. Instead of deleting a region, we randomly flip a fraction of the training labels to a different class. This is like having annotation errors in your dataset.

This is the full-range plot from 0.1 to 0.9, across all three combinations. We extended the experiment past 0.5 for this.

Now, you can see spread **does** go up here too. The effect sizes from 0.1 to 0.5 are comparable — roughly plus 1.1 to plus 2.0 across the three. So the geometry moves.

But there are three reasons this is **not** a useful signal.

**First:** look at accuracy on the bottom. It falls in lockstep with the spread. Tree accuracy drops from 93 percent to 64 percent by noise 0.5. The SVM from 96 to 78. So unlike the coverage gap, this defect is **not hidden** — accuracy already tells you something is wrong. The geometry adds no information that accuracy didn't already give you.

**Second:** the shaded region past 0.5. Here the variance — these error bars — blows up by about ten times. And the number of valid runs collapses — at noise 0.9, only 12 out of 36 runs even produce a usable cluster. The model itself is below the chance line — accuracy falls below one-third, which is the dotted line. At this point it's just a broken model thrashing. The metric isn't giving you a signal about the defect anymore — it's just measuring randomness from a dead model.

What's actually happening mechanically: with more noise, the tree creates more leaves to memorize the corrupted labels. Every test point is now surrounded by boundaries — so to flip the label, it barely has to move. Like, the adversarial point is **trapped** between the boundaries. HopSkipJump initializes from a random point and tries to inch toward the test point while staying misclassified — but any small step flips the class label again. So it can't move. Where it lands is basically set by the random initialization, and different runs give you different answers — which is why the variance explodes.

**Third:** and this is important for the bigger picture — even in the working regime before 0.5, the **direction** of the geometric change depends on class separability. On well-separated data like iris, spread goes up. On datasets where classes already overlap, the direction reverses. So even when the metric works, you can't trust which way it will go without knowing the data structure in advance.

So label noise is a negative result — but it tells us something: the coverage-gap signal is **specific**. It's not 'any defect makes the geometry shift'. It's specific to a structural hole in the data. That sharpens the whole thesis."

---

## Slide 09 — Verification (~50 s)

"One concern you might have: the coverage gap changes the test set too — deleting 90 percent of a class means fewer points to attack. Is the spread signal just tracking that, instead of a real attack effect?

We tested this with a **compression ratio** — the adversarial spread divided by the original test-point spread. If the attack does nothing, this ratio should be around 1.0 — meaning the adversarial cloud just looks like the test data.

Red is label noise: the ratio goes to **0.98**. The points barely moved — the adversarial distribution is essentially the same as the test-point distribution. No genuine attack signal.

Green is the coverage gap: the ratio stays at around **0.70** — the adversarial cloud is consistently about 30 percent tighter than the test cloud. It's genuinely compressed by the attack.

And on the right, the perturbation — how far each point travels from its original test point. For coverage gap, perturbation **increases** — points have to travel farther to cross a boundary that's been pushed into the empty region. For label noise, perturbation **decreases** — more noise means more boundaries, so the nearest boundary is closer to every point, and you barely have to move.

So together this confirms: the coverage-gap signal is a **real attack effect**, not just a mirror of the changing test set."

---

## Slide 10 — Conclusion & Future Work (~40 s)

"So to wrap up.

We verified: for coverage gap, our metric successfully revealed a hidden defect — accuracy was blind, the geometry caught it.

For label noise, the metric moved too — but accuracy was already collapsing. The defect wasn't hidden either. And above noise 0.5 the metric itself destabilises completely.

What this tells us: different defects, and different model-attack combinations, can respond very differently to the same metric. We suspect no **single** metric will work for every case.

So in the future: more realistic defects — outliers, feature corruption, things that actually happen in real datasets. More models — random forest, XGBoost — to check the geometric property generalises. And most importantly: a **set of metrics**, not just one. Because one metric can fail for some cases — but it's a lot harder for all of them to fail at the same time."

*[Handoff to partner for Slide 11 — Takeaway]*

---

## Cheat sheet: Numbers on one card

| When you say…                  | The number is…              |
| ------------------------------- | ---------------------------- |
| Tree + DTA (coverage gap)       | d = +2.06                    |
| SVM + HSJ (coverage gap)        | d = +0.75                    |
| Tree + HSJ (coverage gap)       | d = +0.38                    |
| Accuracy under coverage gap     | flat ~0.96                   |
| Tree accuracy 0.1→0.5 noise    | 0.93 → 0.64                 |
| SVM accuracy 0.1→0.5 noise     | 0.96 → 0.78                 |
| Compression ratio, coverage gap | ~0.70                        |
| Compression ratio, label noise  | → 0.98                      |
| Perturbation, coverage gap      | rises, 1.38 → 1.70          |
| Perturbation, label noise       | drops, 0.81 → 0.64          |
| Bias 0.9: surviving points      | 4 of 50                      |
| Var explodes above noise 0.5    | std ~10×, valid runs 36→12 |
| Grid scale                      | 360 runs/grid, 5-fold CV     |
