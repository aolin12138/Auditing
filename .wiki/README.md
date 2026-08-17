# Project Wiki — Adversarial-Geometry Dataset Auditing

> Can you tell a model was trained on biased data just by attacking it?

This wiki is the knowledge base for the Part IV research project on detecting
dataset defects from the spatial geometry of adversarial examples.

## Start here

| Page | What's in it |
|------|--------------|
| [01-research-question.md](01-research-question.md) | The core question, motivation, why it matters |
| [02-background-aiden.md](02-background-aiden.md) | Prior work (Aiden), his pipeline, the metric, its limitations |
| [03-methodology.md](03-methodology.md) | Pipeline, defects, models, attacks, OPTICS, the metric |
| [04-findings.md](04-findings.md) | All results: coverage gap, label noise, verification, tree strategy |
| [05-key-decisions.md](05-key-decisions.md) | Decisions log — what we chose and why |
| [06-lessons-gotchas.md](06-lessons-gotchas.md) | Pitfalls, corrections, subtle bugs |
| [07-repo-map.md](07-repo-map.md) | Where everything lives in the repo |
| [08-open-questions.md](08-open-questions.md) | Limitations and future work |
| [09-planned-experiments.md](09-planned-experiments.md) | Threads A (outlier) + B (model families) DONE; C (defect expansion) — imbalance Phase 0+1 + cross-dataset variance DONE; **next = dimension-robust spread metric** (`defect_expansion_experiment/PLAN.md §8`) |
| [10-tooling-and-skills.md](10-tooling-and-skills.md) | Reusable research skills (`nature-skills`) + concrete plot/visualization + rigor optimizations |
| [sessions/](sessions/) | Dated session logs |

## One-paragraph summary

We train classifiers on deliberately defective versions of the iris dataset,
generate adversarial examples against them, cluster those points, and measure
the **geometric spread** of the clusters. **Finding:** coverage-gap bias
(an under-sampled subpopulation) is detectable from adversarial spread **while
test accuracy stays flat** — the model looks healthy but the geometry reveals
the defect. **Label noise** produces no independent signal: it either collapses
accuracy (so accuracy already flags it) or its direction depends on class
separability. Verified with a compression-ratio test that distinguishes a real
attack signal (ratio ~0.70) from an artifact (ratio → 1.0).

## Status (2026-08-16)

- Threads **A (outlier)** + **B (model families)** complete; **C (defect expansion)**: class
  imbalance **Phase 0 + Phase 1** done, plus a **cross-dataset variance study** (iris + wine,
  tree/DTA + svm/HSJ) — see `defect_expansion_experiment/FINDINGS_{imbalance,imbalance_p1,variance}.md`.
- **Key 2026-08 result:** per-class **recall** is the robust cross-dataset/cross-model separator;
  the adversarial-**spread** signal is dimension- and attack-fragile (Finding 6). **Next gate =
  dimension-robust spread metric** (`PLAN.md §8`).
- Earlier: 3 grids (360 runs each), mid-year report (`midyear_report.docx`), conference deck,
  6 report figures (`figures/report/`).
- Reviewed reusable research skills (`nature-skills`) → [10-tooling-and-skills.md](10-tooling-and-skills.md).
- **Repo:** github.com/aolin12138/Auditing (origin), upstream Aidan-Jared/Auditing

## Headline numbers

| Result | Value |
|--------|-------|
| Coverage gap, SVM+HSJ spread | Cohen's d = +0.75, accuracy flat ~0.96 |
| Coverage gap, Tree+DTA spread | d = +2.06 (strongest) |
| Coverage gap, Tree+HSJ spread | d = +0.38 (weak, unreliable) |
| Label noise, SVM+HSJ | d = −0.09 (null), accuracy 0.96→0.78 |
| Compression ratio: coverage gap | ~0.70 (real signal) |
| Compression ratio: label noise | 0.77 → 0.98 (artifact) |
