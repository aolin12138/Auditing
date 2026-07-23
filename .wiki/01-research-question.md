# 01 — Research Question

## The question

**Can the spatial geometry of adversarial examples reveal hidden defects in a
model's training data, without access to that data?**

This sits in the **data-centric AI** paradigm: instead of holding data fixed and
improving models, we hold the modelling approach fixed and try to measure
properties of the dataset itself — using the model as a probe.

## Why it matters

A model can score 96% test accuracy while its training data is missing an entire
subpopulation of a class. Accuracy says the model is fine. Every standard check
says it's fine. The bias is invisible.

If adversarial geometry can flag this, it becomes a **black-box data-quality
audit**: attack the model from the outside, watch where it breaks, and the
spatial pattern of failures reveals a blind spot that accuracy misses.

## The core intuition

Train a classifier. Generate adversarial examples (small perturbations that flip
predictions). The *geometry* of these examples is informative:

- **Margin signal:** if points need a long perturbation to flip, classes are
  well-separated there. A tiny nudge means the boundary is fragile.
- **Clustering / spread signal (the strand we pursue):** if a dataset has a
  structural hole in some region, adversarial examples generated near it behave
  differently — they scatter across an under-constrained, extrapolated boundary
  rather than concentrating at natural weak spots.

## Scope note (important)

"Bias" in this project means a **coverage gap** — a contiguous region of one
class deleted along a feature axis (a clean hole, not corrupted data). This is
*one* failure mode. Other defects (label noise, feature corruption, outliers)
are structurally different and may not produce the same signature. Establishing
**what the signal is and isn't sensitive to** is a core contribution, not a
footnote — see [04-findings.md](04-findings.md).

## Context

University of Auckland Part IV (Honours) engineering research project. Builds on
prior exploratory work by a previous contributor, **Aiden** — see
[02-background-aiden.md](02-background-aiden.md).
