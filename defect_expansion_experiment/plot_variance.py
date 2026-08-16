"""Phase 1 unified + variance figures (tree + white-box DTA).

Per dataset, one 3x2 figure — as the defect grows:
  rows = [test accuracy, minority-class recall, normalised adversarial spread]
  cols = [train-only (clean test), before-split (deletes the band from test too)]
  lines = spatial (coverage gap) vs random (imbalance), 30 seeds, 95% CI.
Answers, for this (model, attack): how acc changes, how spread changes, coverage-gap vs
random, and how train-only vs before-split differs — replicated on iris (4-D) and wine (13-D).
-> plots/variance_<dataset>.png  +  plots/variance_spread_summary.png
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
PLOTS = HERE / 'plots'; PLOTS.mkdir(exist_ok=True)
D = pl.read_parquet(HERE / 'results_variance.parquet')
FRAC = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
COL = {'random': '#2ca25f', 'spatial': '#c0392b'}
LAB = {'random': 'random (imbalance)', 'spatial': 'spatial (coverage gap)'}
DS_TITLE = {'iris': 'iris (4-D · tc=2 virginica · feat=petal width)',
            'wine': 'wine (13-D · tc=0 · feat=proline)'}


def base_spread(ds):
    return D.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.0))['mean_dist'].drop_nans().mean()


def series(ds, structure, protocol, col, norm=None):
    xs, ms, cis = [], [], []
    for f in FRAC:
        if f == 0.0:
            sub = D.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.0))
        else:
            sub = D.filter((pl.col('dataset') == ds) & (pl.col('frac') == f) &
                           (pl.col('structure') == structure) & (pl.col('protocol') == protocol))
        v = sub[col].drop_nans()
        if len(v):
            m = float(v.mean()); s = float(v.std()) if len(v) > 1 else 0.0
            xs.append(f); ms.append(m / norm if norm else m)
            cis.append(1.96 * s / np.sqrt(len(v)) / (norm if norm else 1))
    return np.array(xs), np.array(ms), np.array(cis)


def fig_dataset(ds):
    nb = base_spread(ds)
    rows = [('vacc', 'test accuracy', None, (0, 1.03)),
            ('min_recall', 'minority-class recall', None, (0, 1.03)),
            ('mean_dist', 'normalised adversarial spread\n(× clean baseline)', nb, None)]
    cols = ['train_only', 'before_split']
    ctitle = {'train_only': 'train-only injection (clean test)',
              'before_split': 'before-split injection (deletes test band too)'}
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharex=True)
    for i, (col, ylab, nrm, ylim) in enumerate(rows):
        for j, proto in enumerate(cols):
            ax = axes[i][j]
            for s in ['spatial', 'random']:
                x, m, ci = series(ds, s, proto, col, norm=nrm)
                ax.plot(x, m, marker='o', lw=2.2, color=COL[s], label=LAB[s])
                ax.fill_between(x, m - ci, m + ci, color=COL[s], alpha=0.15)
            if col == 'mean_dist':
                ax.axhline(1.0, ls=':', color='0.4', lw=1)
            if ylim:
                ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(ctitle[proto], fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9.5)
            if i == 2:
                ax.set_xlabel('fraction of target class removed (severity)')
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{DS_TITLE[ds]} — overfit tree + white-box DTA, 30 seeds\n'
                 'coverage gap (spatial) vs imbalance (random) · train-only vs before-split',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = PLOTS / f'variance_{ds}.png'; fig.savefig(p, dpi=135); plt.close(fig)
    print('wrote', p)


def fig_summary():
    """Cross-dataset: spatial normalised spread, train-only — does the spread signal replicate?"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    style = {'iris': ('-', '#b2182b'), 'wine': ('--', '#2166ac')}
    for ds, (ls, c) in style.items():
        nb = base_spread(ds)
        x, m, ci = series(ds, 'spatial', 'train_only', 'mean_dist', norm=nb)
        ax.plot(x, m, marker='o', ls=ls, lw=2.4, color=c, label=f'{ds} — spatial (coverage gap)')
        ax.fill_between(x, m - ci, m + ci, color=c, alpha=0.13)
        xr, mr, cir = series(ds, 'random', 'train_only', 'mean_dist', norm=nb)
        ax.plot(xr, mr, marker='.', ls=':', lw=1.5, color=c, alpha=0.7, label=f'{ds} — random (imbalance)')
    ax.axhline(1.0, ls=':', color='0.4', lw=1)
    ax.set_xlabel('fraction of target class removed'); ax.set_ylabel('normalised adversarial spread')
    ax.set_title('Does the SPREAD signal replicate across datasets? (train-only)\n'
                 'iris: spatial reaches ~1.25× · wine (13-D): spatial ~1.04× — the spread metric is dataset-fragile',
                 fontsize=9.5)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    p = PLOTS / 'variance_spread_summary.png'; fig.savefig(p, dpi=140); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    for ds in ['iris', 'wine']:
        fig_dataset(ds)
    fig_summary()
