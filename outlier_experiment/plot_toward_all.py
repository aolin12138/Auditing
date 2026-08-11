"""TOWARD-direction outlier across ALL models, pushed to the class-size ceiling (n_out=40).
Pulls white-box tree/SVM from results_extended.parquet and black-box RF/tree from
../model_family_experiment/results.parquet.
  Panel A: toward dose-response to the ceiling (normalised spread vs n_out, k=8).
  Panel B: all-models bar at the strong cell (k=8, n_out=10), normalised, 95% CI.
-> plots/toward_all_models.png
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'
ext = pl.read_parquet(HERE / 'results_extended.parquet')
mf = pl.read_parquet(HERE.parent / 'model_family_experiment' / 'results.parquet').filter(pl.col('defect') == 'outlier')
CEIL = 40  # class size in the 5-fold train fold


def base(frame, model):
    v = frame.filter((pl.col('model') == model) & (pl.col('n_out') == 0))['mean_dist'].drop_nans().drop_nulls()
    return float(v.mean())


def cell(frame, model, k, n, direction='toward'):
    v = frame.filter((pl.col('model') == model) & (pl.col('direction') == direction)
                     & (pl.col('k') == k) & (pl.col('n_out') == n))['mean_dist'].drop_nans().drop_nulls()
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), 1.96 * float(v.std()) / np.sqrt(len(v)), len(v)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))

# Panel A: toward dose-response to ceiling (k=8)
ax = axes[0]
SERIES = [('tree_d3', ext, 'tree d3 + DTA', '#e08214', [1, 3, 5, 10, 15, 20, 30, 40]),
          ('tree_d10', ext, 'tree d10 + DTA', '#1e5eff', [1, 3, 5, 10, 15, 20, 30, 40]),
          ('svm', ext, 'SVM + HSJ', '#2ca25f', [5, 10, 20, 30, 40]),
          ('rf', mf, 'RandomForest + HSJ', '#8e44ad', [5, 10, 20, 30, 40])]
for model, frame, lab, col, ns in SERIES:
    b = base(frame, model); xs, ms, cis = [], [], []
    for n in ns:
        m, ci, kk = cell(frame, model, 8, n)
        if kk:
            xs.append(n); ms.append(m / b); cis.append(ci / b)
    ls = '--' if model == 'rf' else '-'
    ax.errorbar(xs, ms, yerr=cis, marker='o', lw=2, ls=ls, capsize=3, color=col, label=lab)
ax.axhline(1.0, ls=':', color='0.4', lw=1.2)
ax.axvline(CEIL, ls='-', color='0.7', lw=8, alpha=0.25)
ax.text(CEIL, ax.get_ylim()[0], ' ceiling\n (=class size)', ha='right', va='bottom', fontsize=7.5, color='0.35')
ax.set_xlabel('n_out (outliers added; 40 = 100% of class)')
ax.set_ylabel('normalised spread (x baseline)')
ax.set_title('TOWARD dose-response to the class-size ceiling: trees COLLAPSE at 100%,\nSVM slowly RISES, RF stays FLAT (all 4 to n_out=40)', fontsize=9.5)
ax.legend(fontsize=8, loc='center left')

# Panel B: all-models bar at (k=8, n_out=10)
ax = axes[1]
BARS = [('tree d10\n+ DTA (WB)', ext, 'tree_d10'), ('SVM\n+ HSJ', ext, 'svm'),
        ('RandomForest\n+ HSJ', mf, 'rf'), ('single tree\n+ HSJ', mf, 'tree')]
heights, errs, ns = [], [], []
for lab, frame, model in BARS:
    b = base(frame, model); m, ci, n = cell(frame, model, 8, 10)
    heights.append(m / b if n else np.nan); errs.append(ci / b if n else 0); ns.append(n)
cols = ['#1e5eff', '#2ca25f', '#8e44ad', '#c0392b']
ax.bar(range(len(BARS)), heights, yerr=errs, capsize=5, color=cols, alpha=0.85)
ax.axhline(1.0, ls=':', color='0.4', lw=1.2)
ax.set_xticks(range(len(BARS))); ax.set_xticklabels([b[0] for b in BARS], fontsize=8)
ax.set_ylabel('normalised spread (x baseline)')
ax.set_title('TOWARD @ k=8, n_out=10 — ALL models\n(white-box tree is the ONLY strong detector)', fontsize=9.5)
for i, (h, n) in enumerate(zip(heights, ns)):
    if np.isfinite(h):
        ax.text(i, h + 0.01, f'{h:.2f}\n(n={n})', ha='center', va='bottom', fontsize=7.5)

fig.suptitle('TOWARD-direction outlier across models (10 seeds, 95% CI)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(PLOTS / 'toward_all_models.png', dpi=140)
print('wrote', PLOTS / 'toward_all_models.png')
