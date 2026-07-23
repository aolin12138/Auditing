# 07 — Repo Map

```
Auditing/
├── README.md                    Project overview (adversarial-geometry auditing)
├── .wiki/                        THIS WIKI
├── MEETING_NOTES.md             Supervisor meeting prep
├── midyear_report.docx          Mid-year technical report (generated)
├── generate_report.py           Builds the .docx from parquet data
├── generate_figures.py          Matplotlib figures (v1, cluttered)
├── generate_figures_altair.py   Altair figures (Aiden's style)
├── generate_figures_v2.py       Clean Tufte-style figures (p1–p6) ← preferred
├── master_analysis.ipynb        Aggregates all 3 experiments
├── util.py                      Shared helpers (labelencoding, eval_model)
├── pyproject.toml / uv.lock     Environment (uv, Python ≥3.12)
│
├── aiden_original/              PRIOR WORK — preserved as-is
│   ├── README.md                Lists contents + limitations
│   ├── bias.py, main.py, util.py, eda*.ipynb, plots/
│
├── dtree_attack_experiment/     GRID 1: Tree + DTA, both defects
│   ├── run_experiment.py, results.parquet (360 rows), plots/
├── hsj_svm_experiment/          GRID 2: Tree+SVM + HSJ, coverage gap
│   ├── hsj_bias_experiment.ipynb, run_experiment.py, results.parquet, plots/
├── hsj_label_noise_experiment/  GRID 3: Tree+SVM + HSJ, label noise
│   ├── run_experiment.py, results.parquet, plots/
│
├── experiments/
│   ├── eda_v2.ipynb             Tree strategy analysis (overfit vs pruned)
│   ├── hopskipjump_bias.ipynb   Early HSJ notebook
│   ├── exploratory/             Investigation-phase scripts (collect_v2, run_*, plot_*)
│   └── probes/                  _probe_move.py (label noise ratio),
│                                _probe_cg.py (coverage gap ratio)
│
├── figures/report/              REPORT FIGURES
│   ├── p1_coverage_gap.png      Coverage gap spread (z-scored, 3 combos) + accuracy
│   ├── p2_label_noise.png       Label noise spread vs accuracy
│   ├── p3_metrics.png           Metric decomposition
│   ├── p4_discriminant.png      Compression ratio + perturbation
│   ├── p5_strategy_iris.png     Overfit vs pruned
│   ├── p6_aiden_original.png    Aiden's original signal
│   └── fig*_altair_*, fig*_*    Alternate styles
│
├── presentation/                CONFERENCE DECK (HTML, original style)
│   ├── deck.html                10 slides — open in browser
│   ├── deck-stage.js, support.js
│   └── p1_coverage_gap.png, p4_discriminant.png
│
├── plots/investigation/         All historical/exploratory PNGs
└── data/                        GITIGNORED — parquets, logs (large)
    ├── data_bias.parquet        Aiden's original output
    ├── data_v2.parquet          Tree strategy experiment
    └── data_label_noise*.parquet
```

## Key data files

| File | What | Committed? |
|------|------|-----------|
| `*/results.parquet` (×3) | The three final grids | ✅ yes |
| `data/data_bias.parquet` | Aiden's original bias run | ❌ gitignored |
| `data/data_v2.parquet` | Tree strategy (overfit/pruned) | ❌ gitignored |

Columns in the final `results.parquet`: `tacc, vacc, asucc, nadv, density,
nclust, mean_dist, clust_size, aiden_density, {bias|noise}, seed/…, model`.

## How to run

```bash
uv sync                          # create .venv
# analysis only (fast, reads committed parquets):
#   open master_analysis.ipynb, select .venv kernel, Run All
# regenerate figures:
python generate_figures_v2.py
# regenerate report:
python generate_report.py
# re-run an experiment (~15-30 min, resumable):
python dtree_attack_experiment/run_experiment.py
```

## Remotes

- `origin` → github.com/aolin12138/Auditing (our fork, push here)
- `upstream` → github.com/Aidan-Jared/Auditing (Aiden's)
