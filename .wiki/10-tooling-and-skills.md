# 10 — Tooling & Research Skills

Reusable agent skills and pipeline/visualization optimizations for this project.
Reviewed 2026-08-16 from **`nature-skills`** (https://github.com/Yuan1z0825/nature-skills,
18 research skills for AI scholars, Apache-2.0; install via `npx skills`).

## Directly usable skills (ranked for this Part IV project)

| Skill | Status | Why it's useful here |
|-------|--------|----------------------|
| **`nature-figure`** | Stable | Submission-grade scientific figures. Ships `scripts/validate_figure.py` (static QA), `scripts/audit_pdf_text.py` (flags exported glyphs < 5 pt), `scripts/figure_safety.py` (label-collision / monotone helpers), and `references/{api,chart-types,qa-contract}.md` (palette, chart selection, export QA). **Top pick for the plot/visualization work.** |
| **`nature-statistics`** | Draft | Audits statistical reporting: effect sizes, CIs, multiple comparisons, sample size, **figure statistics**, cross-panel numeric consistency. We already report 95% CIs + Cohen's d; this checks rigor before the report. |
| **`nature-writing` / `nature-polishing`** | Draft/Stable | Draft & polish the Part IV report + presentation into journal-style English (needs `nature-shared`). |
| **`nature-academic-search` / `nature-literature-pipeline` / `nature-citation`** | Beta/Stable | Find & verify the **Katerina Dost bias-detection baseline** (the open supervisor question in `08-open-questions.md`) and manage references. |
| **`nature-paper-card` / `nature-reader`** | Beta | Deep-read baseline papers (evidence chains, method logic) for related work. |
| **`nature-reviewer`** | Draft | Pre-submission reviewer simulation before handing the report to the supervisor. |
| **`nature-experiment-log`** | Draft | Standardized experiment logs — complements our `run-journal` + `.wiki/sessions/`. |
| **`nature-paper2ppt`** | Beta | Regenerate the conference deck from the report. |

### Install (documented — not auto-installed)
```bash
npx skills add Yuan1z0825/nature-skills --list                 # see frontmatter names
npx skills add Yuan1z0825/nature-skills --agent codex \
  --skill nature-figure --skill nature-shared --yes --copy      # one skill + shared support
```
**Caveat:** the CLI targets Codex (`~/.codex/skills/`) / Claude Code, not pi. To use inside pi,
clone the repo and copy the relevant `skills/<name>/` into `~/.pi/agent/skills/` (or wrap via a
slash command). Python/R runtime deps are separate. *Ask before installing* (modifies loadable
agent resources — see `agent-self-preservation`).

## Concrete visualization optimizations (apply NOW, learned from `nature-figure`)

These are actionable on our existing `plot_*.py` without installing anything:

1. **Colorblind-safe palette — real issue in our current plots.** We use **red = spatial /
   green = random** (`plot_variance.py`, `plot_p1.py`). Red-green is the most common colorblind
   confusion. Switch to an Okabe-Ito / diverging blue-orange pair (e.g. `#d55e00` vs `#0072b2`)
   **and** differentiate by marker/linestyle so the plots survive greyscale printing.
2. **Vector export for the report.** Save `.pdf`/`.svg` (editable) alongside the 140-dpi PNGs;
   the Part IV report needs vector figures, not rasterized screenshots.
3. **Glyph-size floor.** Our legends at `fontsize 7–8` can fall below ~5 pt once a multi-panel
   figure is scaled to a column width. Audit final PDFs (the `audit_pdf_text.py` idea) and keep
   real text ≥ 6 pt.
4. **Panel labels (a, b, c).** Multi-panel figures (the 3×2 variance grids) need corner labels
   for the report/manuscript.
5. **Source-data traceability.** Each figure already derives from a committed `.parquet` — good;
   emit a small companion CSV per figure (the numbers behind each line) for the report's source
   data.
6. **A shared style module.** Add `defect_expansion_experiment/plotstyle.py` (rcParams: font
   family/sizes, dpi, the colorblind palette, PNG+PDF export helper) and import it everywhere so
   every figure is consistent and submission-grade in one place.

## Pipeline / rigor optimizations (from `nature-statistics` philosophy)
- Keep reporting **CIs + effect sizes** (we do); add a consistency check that numbers in a
  figure caption match the parquet (a small assertion script), and note **multiple-comparison**
  exposure when sweeping many cells.
- Standardize the "figure contract" (core conclusion → evidence hierarchy → prototype) already
  implicit in our titles; write it into each `plot_*.py` docstring.

## Status (updated 2026-08-16 — installed + applied)
- **`nature-figure` + `nature-shared` INSTALLED** into `~/.pi/agent/skills/` (frontmatter dry-load
  verified; run `/reload` in pi to load — no restart needed). Its QA scripts
  (`validate_figure.py`, `audit_pdf_text.py`, `figure_safety.py`) are now available for the
  report-figure pass.
- **Visualization optimization DONE:** new `defect_expansion_experiment/plotstyle.py` (Okabe-Ito
  colorblind-safe palette, series differentiated by colour **and** linestyle/marker, top/right
  spines off, PNG+**vector PDF** export with editable text, panel-label helper). Retrofitted
  `plot_imbalance.py`, `plot_p1.py`, `plot_variance.py` — red/green replaced by orange/blue,
  panel labels (a–f) on multi-panel grids, all figures regenerated as `.png` + `.pdf`.
- **Future plots:** `import plotstyle as ps; ps.apply()` then use `ps.DEFECT[...]`,
  `ps.DEFECT_LS/MK`, `ps.panel_label(ax,'a')`, `ps.save(fig, path_stem)`.
- Remaining optional: install writing/statistics/literature skills; apply `plotstyle` to the
  older `outlier_experiment/` + `model_family_experiment/` plots if they go into the report.
