"""FULL k x n_out interaction heatmaps for ALL models (toward, normalised spread).
Every combination of distance (k) and count (n_out) we have data for, one panel per model.
Trees: full grid to the ceiling (n_out=40). SVM/RF: partial (blank = not run).
-> plots/interaction_all_models.png
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

K_ALL = [2, 3, 4, 5, 6, 8, 12]
N_ALL = [1, 3, 5, 10, 15, 20, 30, 40]
PANELS = [('tree_d3', ext, 'tree d3 + DTA'), ('tree_d10', ext, 'tree d10 + DTA'),
          ('svm', ext, 'SVM + HSJ'), ('rf', mf, 'RandomForest + HSJ')]


def matrix(frame, model, direction='toward'):
    b = frame.filter((pl.col('model') == model) & (pl.col('n_out') == 0))['mean_dist'].drop_nans().mean()
    M = np.full((len(K_ALL), len(N_ALL)), np.nan)
    for i, k in enumerate(K_ALL):
        for j, n in enumerate(N_ALL):
            v = frame.filter((pl.col('model') == model) & (pl.col('direction') == direction)
                             & (pl.col('k') == float(k)) & (pl.col('n_out') == n))['mean_dist'].drop_nans().drop_nulls()
            if len(v):
                M[i, j] = float(v.mean()) / b
    return M


fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
cmap = plt.cm.viridis.copy(); cmap.set_bad('#dddddd')
im = None
for ax, (model, frame, lab) in zip(axes.ravel(), PANELS):
    M = matrix(frame, model)
    im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap, vmin=0.85, vmax=1.4)
    ax.set_xticks(range(len(N_ALL))); ax.set_xticklabels(N_ALL)
    ax.set_yticks(range(len(K_ALL))); ax.set_yticklabels(K_ALL)
    ax.set_xlabel('n_out (count)'); ax.set_ylabel('k (distance, x class std)')
    ax.set_title(lab, fontsize=10)
    for i in range(len(K_ALL)):
        for j in range(len(N_ALL)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if M[i, j] < 1.18 else 'black')
fig.suptitle('Outlier: full DISTANCE (k) x COUNT (n_out) interaction — toward, normalised. '
             'Grey = not run. Read a ROW = distance-sweep, a COLUMN = count-sweep.', fontsize=11)
fig.subplots_adjust(hspace=0.33, wspace=0.18, top=0.9, right=0.88, left=0.08, bottom=0.08)
cax = fig.add_axes([0.905, 0.12, 0.02, 0.72])
fig.colorbar(im, cax=cax, label='normalised spread (x baseline)')
fig.savefig(PLOTS / 'interaction_all_models.png', dpi=140)
print('wrote', PLOTS / 'interaction_all_models.png')
