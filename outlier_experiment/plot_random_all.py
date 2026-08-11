"""RANDOM-direction outlier across ALL models. Pulls white-box tree/SVM from
results_extended.parquet and black-box RF/tree from ../model_family_experiment/results.parquet.
  Panel A: random dose-response (normalised spread vs n_out, k=8) for tree_d3/d10 (DTA) + SVM.
  Panel B: all-models bar at (k=8, n_out=10), normalised, 95% CI.
-> plots/random_all_models.png
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


def base(frame, model):
    v = frame.filter((pl.col('model') == model) & (pl.col('n_out') == 0))['mean_dist'].drop_nans().drop_nulls()
    return float(v.mean())


def cell(frame, model, k, n, direction='random'):
    v = frame.filter((pl.col('model') == model) & (pl.col('direction') == direction)
                     & (pl.col('k') == k) & (pl.col('n_out') == n))['mean_dist'].drop_nans().drop_nulls()
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), 1.96 * float(v.std()) / np.sqrt(len(v)), len(v)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

# Panel A: random dose-response vs n_out (k=8) for the white-box/SVM models (full sweep)
ax = axes[0]
EXT = [('tree_d3', 'tree d3 + DTA', '#e08214'), ('tree_d10', 'tree d10 + DTA', '#1e5eff'),
       ('svm', 'SVM + HSJ', '#2ca25f')]
NS = [1, 3, 5, 10, 15, 20]
for model, lab, col in EXT:
    b = base(ext, model); xs, ms, cis = [], [], []
    for n in NS:
        m, ci, k = cell(ext, model, 8, n)
        if k:
            xs.append(n); ms.append(m / b); cis.append(ci / b)
    ax.errorbar(xs, ms, yerr=cis, marker='o', lw=2, capsize=3, color=col, label=lab)
ax.axhline(1.0, ls='--', color='0.4', lw=1.2, label='baseline (=1.0)')
ax.set_xlabel('n_out (absolute)   [k=8, RANDOM direction]')
ax.set_ylabel('normalised spread (x baseline)')
ax.set_title('Random outlier dose-response (white-box tree + SVM)', fontsize=10)
ax.legend(fontsize=8)

# Panel B: all-models bar at (k=8, n_out=10), random
ax = axes[1]
BARS = [('tree d10\n+ DTA (WB)', ext, 'tree_d10'), ('SVM\n+ HSJ', ext, 'svm'),
        ('RandomForest\n+ HSJ', mf, 'rf'), ('single tree\n+ HSJ', mf, 'tree')]
xs, heights, errs, ns = [], [], [], []
for lab, frame, model in BARS:
    b = base(frame, model); m, ci, n = cell(frame, model, 8, 10)
    xs.append(lab); heights.append(m / b if n else np.nan); errs.append(ci / b if n else 0); ns.append(n)
cols = ['#1e5eff', '#2ca25f', '#8e44ad', '#c0392b']
bars = ax.bar(range(len(xs)), heights, yerr=errs, capsize=5, color=cols, alpha=0.85)
ax.axhline(1.0, ls='--', color='0.4', lw=1.2)
ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8)
ax.set_ylabel('normalised spread (x baseline)')
ax.set_title('Random outlier @ k=8, n_out=10 — ALL models\n(weak everywhere; RF the only one clearing baseline)', fontsize=9.5)
for i, (h, n) in enumerate(zip(heights, ns)):
    if np.isfinite(h):
        ax.text(i, h + 0.01, f'{h:.2f}\n(n={n})', ha='center', va='bottom', fontsize=7.5)

fig.suptitle('RANDOM-direction outlier across models (10 seeds, 95% CI)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(PLOTS / 'random_all_models.png', dpi=140)
print('wrote', PLOTS / 'random_all_models.png')
print('\nBar values @k=8,n=10 random:')
for lab, frame, model in BARS:
    b = base(frame, model); m, ci, n = cell(frame, model, 8, 10)
    print(f'  {model:>9}: {m/b:.2f}x  +/-{ci/b:.2f}  (n={n}, baseline={b:.3f})')
