"""DISTANCE dose-response across ALL models: fixed n_out, increasing k (distance-to-centroid).
The complement of toward_all_models.png (which fixes k and sweeps n_out).
Pulls tree_d3/d10 + SVM from results_extended.parquet, RF/tree(HSJ) from model_family.
  Panel A: normalised spread vs k at fixed n_out=10 (toward), all models.
  Panel B: same at fixed n_out=20 (toward).
-> plots/distance_all_models.png
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

SERIES = [('tree_d3', ext, 'tree d3 + DTA', '#e08214'),
          ('tree_d10', ext, 'tree d10 + DTA', '#1e5eff'),
          ('svm', ext, 'SVM + HSJ', '#2ca25f'),
          ('rf', mf, 'RandomForest + HSJ', '#8e44ad'),
          ('tree', mf, 'single tree + HSJ', '#c0392b')]


def base(frame, model):
    v = frame.filter((pl.col('model') == model) & (pl.col('n_out') == 0))['mean_dist'].drop_nans().drop_nulls()
    return float(v.mean())


def kcurve(frame, model, n_out, direction='toward'):
    ks = sorted(frame.filter((pl.col('model') == model) & (pl.col('direction') == direction)
                             & (pl.col('n_out') == n_out))['k'].unique().to_list())
    b = base(frame, model); xs, ms, cis = [], [], []
    for k in ks:
        v = frame.filter((pl.col('model') == model) & (pl.col('direction') == direction)
                         & (pl.col('n_out') == n_out) & (pl.col('k') == k))['mean_dist'].drop_nans().drop_nulls()
        if len(v):
            xs.append(k); ms.append(float(v.mean()) / b)
            cis.append(1.96 * float(v.std()) / np.sqrt(len(v)) / b)
    return np.array(xs), np.array(ms), np.array(cis)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
for ax, n_out in zip(axes, [10, 20]):
    for model, frame, lab, col in SERIES:
        x, m, ci = kcurve(frame, model, n_out)
        if len(x) == 0:
            continue
        ls = '--' if model in ('rf', 'tree') else '-'   # HSJ models only have k=4,8 (2 pts)
        ax.errorbar(x, m, yerr=ci, marker='o', lw=2, ls=ls, capsize=3, color=col, label=lab)
    ax.axhline(1.0, ls=':', color='0.4', lw=1.2)
    ax.set_xlabel('k  =  distance to centroid (x class std)')
    ax.set_title(f'fixed n_out = {n_out} outliers,  sweeping DISTANCE (toward)', fontsize=10)
axes[0].set_ylabel('normalised spread (x baseline)')
axes[0].legend(fontsize=8, loc='upper left')
fig.suptitle('DISTANCE dose-response across models — fixed count, increasing k '
             '(complement of the count-sweep). Trees: signal switches on at k>=6; SVM/RF flat.',
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(PLOTS / 'distance_all_models.png', dpi=140)
print('wrote', PLOTS / 'distance_all_models.png')
